import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ants
import numpy as np
import pandas as pd
import torch


ORDER = [
    "3DFLAIR_NCE", "3DFLAIR_CE", "2DFLAIR_NCE", "2DFLAIR_CE",
    "3DT1_NCE", "3DT1_CE", "2DT1_NCE", "2DT1_CE",
    "3DT2_CE", "3DT2_NCE",
    "2DT2_CE", "2DT2_NCE", "SWI",
    "fa_dti", "md_dti", "dwi",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def check_gpu() -> None:
    if not torch.cuda.is_available():
        sys.exit("Error: GPU not available.")
    log("✅ GPU available")


def get_id_column(df: pd.DataFrame, args: argparse.Namespace) -> str:
    if ("m_id" in df.columns) and (args.id_col == "m_id"):
        return "m_id"
    if ("patient_id" in df.columns) and (args.id_col == "patient_id"):
        return "patient_id"

    if "patient_id" in df.columns:
        log("  [INFO] Requested id_col not found, fallback to 'patient_id'")
        return "patient_id"
    if "m_id" in df.columns:
        log("  [INFO] Requested id_col not found, fallback to 'm_id'")
        return "m_id"

    raise KeyError("CSV must contain either 'patient_id' or 'm_id'.")


def get_basename(p: Path) -> str:
    name = p.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return p.stem


def clip_intensities(
    img: ants.ANTsImage,
    lower: float = 0.001,
    upper: float = 0.999,
) -> ants.ANTsImage:
    arr = img.numpy()
    lo = np.quantile(arr, lower)
    hi = np.quantile(arr, upper)
    arr = np.clip(arr, lo, hi)
    return ants.from_numpy(
        arr,
        origin=img.origin,
        spacing=img.spacing,
        direction=img.direction,
    )


def read_img_reorient(path: Path, pixeltype: Optional[str] = None) -> ants.ANTsImage:
    if pixeltype is not None:
        img = ants.image_read(str(path), pixeltype=pixeltype)
    else:
        img = ants.image_read(str(path))
    img = ants.reorient_image2(img, orientation="LPI")
    return img


def run_hdbet(input_file: Path, out_prefix: Path) -> None:
    result = subprocess.run(
        ["hd-bet", "-i", str(input_file), "-o", str(out_prefix), "-device", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        log(f"  [BET][stderr] {result.stderr.strip()}")
        raise RuntimeError(f"HD-BET failed for {input_file}")


def brain_extraction(img_path: Path, bet_dir: Path, out_prefix_name: str, skip_exist: bool = False) -> Tuple[Path, Path]:
    bet_dir.mkdir(parents=True, exist_ok=True)
    skull = bet_dir / f"{out_prefix_name}.nii.gz"
    mask = bet_dir / f"{out_prefix_name}_mask.nii.gz"

    if skip_exist and skull.exists() and mask.exists():
        log(f"  [BET] skip existing {out_prefix_name}")
        return skull, mask

    log(f"  [BET] running on {out_prefix_name}")
    run_hdbet(img_path, bet_dir / out_prefix_name)
    return skull, mask


def debias_and_reorient(src: Path, out_file: Path, skip_exist: bool = False) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if skip_exist and out_file.exists():
        return out_file

    img = ants.image_read(str(src))
    img = ants.n4_bias_field_correction(img)
    img = ants.reorient_image2(img, orientation="LPI")
    img = clip_intensities(img, 0.001, 0.999)
    ants.image_write(img, str(out_file))
    return out_file


def reorient_mask(src: Path, out_file: Path, skip_exist: bool = False) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if skip_exist and out_file.exists():
        return out_file

    img = ants.image_read(str(src), pixeltype="unsigned int")
    img = ants.reorient_image2(img, orientation="LPI")
    ants.image_write(img, str(out_file))
    return out_file


def coregister_images(
    fixed: Path,
    moving: Path,
    out_file: Path,
    transform_prefix: Path,
    skip_exist: bool = False,
) -> Tuple[Path, List[str]]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    transform_prefix.parent.mkdir(parents=True, exist_ok=True)

    affine_file = f"{transform_prefix}0GenericAffine.mat"

    if skip_exist and out_file.exists() and Path(affine_file).exists():
        log(f"  [Coreg] skip existing {out_file.name}")
        return out_file, [affine_file]

    img_f = read_img_reorient(fixed)
    img_m = read_img_reorient(moving)

    reg = ants.registration(
        fixed=img_f,
        moving=img_m,
        type_of_transform="Affine",
        outprefix=str(transform_prefix),
    )
    out = ants.apply_transforms(
        fixed=img_f,
        moving=img_m,
        transformlist=reg["fwdtransforms"],
    )
    ants.image_write(out, str(out_file))
    return out_file, reg["fwdtransforms"]


def save_bet_img(img_path: Path, mask_path: Path, out_file: Path, skip_exist: bool = False) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if skip_exist and out_file.exists():
        return out_file

    img = read_img_reorient(img_path)
    mask = read_img_reorient(mask_path, pixeltype="unsigned int")

    if not ants.image_physical_space_consistency(img, mask):
        mask = ants.resample_image_to_target(mask, img, interp_type="nearestNeighbor")

    out_img = img * mask
    out_img = clip_intensities(out_img, 0.001, 0.999)
    ants.image_write(out_img, str(out_file))
    log(f"  [BET] saved {out_file}")
    return out_file


def resolve_mask_path(df: pd.DataFrame, process_mask: bool) -> Optional[Path]:
    if not process_mask or "mask_path" not in df.columns:
        return None

    mask_rows = df[df["mask_path"].notna()]
    if mask_rows.empty:
        return None

    candidate = Path(mask_rows["mask_path"].iloc[0])
    if candidate.exists():
        return candidate

    log(f"  [WARN] mask_path does not exist: {candidate}")
    return None


def is_3d_modality(modality: str) -> bool:
    return modality.startswith("3D")


def is_2d_modality(modality: str) -> bool:
    return modality.startswith("2D")


def infer_plane_from_path(path: Path) -> str:
    s = str(path).upper()
    if "SAG" in s or "SAGITTAL" in s:
        return "SAG"
    if "AX" in s or "AXIAL" in s:
        return "AX"
    return "OTHER"


def make_output_stem(modality: str, plane_group: Optional[str], split_used: bool) -> str:
    if split_used and is_2d_modality(modality) and plane_group is not None:
        return f"{modality}_{plane_group}"
    return modality


def build_modality_to_source_path(df: pd.DataFrame) -> Dict[str, Path]:
    modality_to_src: Dict[str, Path] = {}
    for _, row in df.iterrows():
        mod = row["modality"]
        src = Path(row["non-preprocessing"])
        if src.exists():
            modality_to_src[mod] = src
        else:
            log(f"  [WARN] Missing input for {mod}: {src}")
    return modality_to_src


def split_patient_into_units(
    df: pd.DataFrame,
    split_2d_by_plane: bool,
) -> List[Tuple[str, pd.DataFrame, bool]]:
    """
    Return list of processing units:
    (plane_group, unit_df, split_used)

    If no splitting:
        [('MAIN', full_df, False)]

    If 2D-only and split enabled:
        [('SAG', sag_df, True), ('AX', ax_df, True), ('OTHER', other_df, True)]
    """
    modality_to_src = build_modality_to_source_path(df)
    present_modalities = [m for m in ORDER if m in modality_to_src]

    if not present_modalities:
        return [("MAIN", df.copy(), False)]

    has_3d = any(is_3d_modality(m) for m in present_modalities)

    if (not split_2d_by_plane) or has_3d:
        return [("MAIN", df.copy(), False)]

    sag_mods, ax_mods, other_mods = [], [], []
    for mod in present_modalities:
        plane = infer_plane_from_path(modality_to_src[mod])
        if plane == "SAG":
            sag_mods.append(mod)
        elif plane == "AX":
            ax_mods.append(mod)
        else:
            other_mods.append(mod)

    units = []
    if sag_mods:
        units.append(("SAG", df[df["modality"].isin(sag_mods)].copy(), True))
    if ax_mods:
        units.append(("AX", df[df["modality"].isin(ax_mods)].copy(), True))
    if other_mods:
        units.append(("OTHER", df[df["modality"].isin(other_mods)].copy(), True))

    if not units:
        units = [("MAIN", df.copy(), False)]

    return units


def preprocess_unit(
    unit_df: pd.DataFrame,
    patient_id: str,
    plane_group: str,
    split_used: bool,
    template: Path,
    tpl_mask: Path,
    output_dir: Path,
    skip_exist: bool,
    do_debias: bool,
    do_coreg: bool,
    do_bet: bool,
    save_bet: bool,
    do_reg2tpl: bool,
    do_apply_mask: bool,
    do_final_clip: bool,
    process_mask: bool,
    mask_modality: str,
) -> Tuple[Dict[str, Dict[str, Optional[str]]], Optional[str]]:
    """
    Process one unit and return:
      modality_records: {modality -> {'debias': ..., 'BET': ..., 'preprocessed': ..., 'plane_group': ...}}
      mask_preprocessed_path: final lesion mask path if produced by this unit
    """
    display_plane = plane_group if split_used else "MAIN"
    log(f"\n=== Unit {patient_id} [{display_plane}] ===")

    debiased_dir = output_dir / patient_id / "debiased"
    bet_dir = output_dir / patient_id / "BET"
    registered_dir = output_dir / patient_id / "preprocessed"
    coreg_dir = output_dir / patient_id / "coreg"

    lesion_mask_path = resolve_mask_path(unit_df, process_mask)
    mask_preprocessed_path: Optional[str] = None

    if lesion_mask_path:
        log(f"  Found lesion mask: {lesion_mask_path}")
    elif process_mask:
        log("  No valid lesion mask found for this unit.")

    modality_records: Dict[str, Dict[str, Optional[str]]] = {}

    # Step 1: debias / reorient
    debiased: Dict[str, Path] = {}
    output_stems: Dict[str, str] = {}

    if do_debias:
        for _, row in unit_df.iterrows():
            mod = row["modality"]
            src = Path(row["non-preprocessing"])
            if not src.exists():
                log(f"  [WARN] Missing input for {mod}: {src}")
                continue

            stem = make_output_stem(mod, plane_group if split_used else None, split_used)
            output_stems[mod] = stem
            out_file = debiased_dir / f"{stem}.nii.gz"
            debiased[mod] = debias_and_reorient(src, out_file, skip_exist)

        log(f"  [1/5] Debiased {len(debiased)} modalities")
    else:
        for _, row in unit_df.iterrows():
            mod = row["modality"]
            src = Path(row["non-preprocessing"])
            if not src.exists():
                log(f"  [WARN] Missing input for {mod}: {src}")
                continue

            stem = make_output_stem(mod, plane_group if split_used else None, split_used)
            output_stems[mod] = stem
            debiased[mod] = src

        log(f"  [1/5] Skipped debias, using {len(debiased)} original modalities")

    present = [m for m in ORDER if m in debiased]
    if not present:
        log("  No valid modalities, skip.")
        return modality_records, mask_preprocessed_path

    for mod in present:
        modality_records[mod] = {
            "plane_group": plane_group if split_used else "MAIN",
            "debias": str(debiased_dir / f"{output_stems[mod]}.nii.gz") if do_debias else str(debiased[mod]),
            "BET": None,
            "preprocessed": None,
        }

    # Mask alignment
    mask_ref_mod: Optional[str] = None
    if lesion_mask_path:
        if mask_modality in present:
            mask_ref_mod = mask_modality
            mask_reoriented_path = debiased_dir / "lesion_mask_reoriented.nii.gz"
            lesion_mask_path = reorient_mask(lesion_mask_path, mask_reoriented_path, skip_exist)

            ref_img = read_img_reorient(debiased[mask_ref_mod])
            mask_img = read_img_reorient(lesion_mask_path, pixeltype="unsigned int")

            if ref_img.shape != mask_img.shape:
                log(
                    f"  [WARN] Shape mismatch between {mask_ref_mod} {ref_img.shape} "
                    f"and mask {mask_img.shape}. Mask processing disabled."
                )
                lesion_mask_path = None
                mask_ref_mod = None
            else:
                log(f"  ✅ Lesion mask aligned with reference modality: {mask_ref_mod}")
        else:
            log(f"  [INFO] mask_modality '{mask_modality}' not present in this unit; skip mask.")
            lesion_mask_path = None
            mask_ref_mod = None

    # Step 2: coregistration
    fixed_mod = present[0]
    moved: Dict[str, Path] = {fixed_mod: debiased[fixed_mod]}
    coreg_transforms: Dict[str, List[str]] = {}

    if do_coreg and len(present) > 1:
        for m in present[1:]:
            out_file = coreg_dir / f"{output_stems[m]}.nii.gz"
            transform_prefix = coreg_dir / f"{output_stems[m]}_to_{output_stems[fixed_mod]}_"
            dst, tfm = coregister_images(
                fixed=debiased[fixed_mod],
                moving=debiased[m],
                out_file=out_file,
                transform_prefix=transform_prefix,
                skip_exist=skip_exist,
            )
            moved[m] = dst
            coreg_transforms[m] = tfm
        log(f"  [2/5] Coregistered {len(present) - 1} modalities to {fixed_mod}")
    else:
        for m in present[1:]:
            moved[m] = debiased[m]
        log("  [2/5] Skipped coregistration")

    if lesion_mask_path and mask_ref_mod and do_coreg and mask_ref_mod in coreg_transforms:
        log(f"  Applying coreg transform to lesion mask from {mask_ref_mod} -> {fixed_mod}")
        mask_img = ants.image_read(str(lesion_mask_path), pixeltype="unsigned int")
        fixed_img = ants.image_read(str(debiased[fixed_mod]))

        mask_coreg = ants.apply_transforms(
            fixed=fixed_img,
            moving=mask_img,
            transformlist=coreg_transforms[mask_ref_mod],
            interpolator="genericLabel",
        )

        coreg_mask_path = coreg_dir / "lesion_mask_coregistered.nii.gz"
        coreg_mask_path.parent.mkdir(parents=True, exist_ok=True)
        ants.image_write(mask_coreg, str(coreg_mask_path))
        lesion_mask_path = coreg_mask_path
        log("  Lesion mask coregistered.")
    elif lesion_mask_path and mask_ref_mod == fixed_mod:
        log("  Lesion mask already in fixed image space; skip mask coregistration.")

    # Step 3: BET on fixed
    skull_fixed: Optional[Path] = None
    bet_mask: Optional[Path] = None
    if do_bet:
        skull_fixed, bet_mask = brain_extraction(
            img_path=debiased[fixed_mod],
            bet_dir=bet_dir,
            out_prefix_name=output_stems[fixed_mod],
            skip_exist=skip_exist,
        )
        modality_records[fixed_mod]["BET"] = str(skull_fixed)
        log(f"  [3/5] BET done on {fixed_mod}")

        if save_bet and bet_mask is not None:
            for m, mov_path in moved.items():
                if m == fixed_mod:
                    continue
                bet_out = bet_dir / f"{output_stems[m]}.nii.gz"
                save_bet_img(mov_path, bet_mask, bet_out, skip_exist)
                modality_records[m]["BET"] = str(bet_out)
    else:
        skull_fixed = debiased[fixed_mod]
        log("  [3/5] Skipped BET")

    if do_bet and modality_records[fixed_mod]["BET"] is None:
        modality_records[fixed_mod]["BET"] = str(skull_fixed)

    # Step 4: fixed -> MNI
    tfm: Optional[List[str]] = None
    tpl_img = ants.image_read(str(template)) * ants.image_read(str(tpl_mask))

    if do_reg2tpl and skull_fixed is not None:
        reg = ants.registration(
            fixed=tpl_img,
            moving=ants.image_read(str(skull_fixed)),
            type_of_transform="Affine",
        )
        tfm = reg["fwdtransforms"]

        fixed_reg_out = registered_dir / f"{output_stems[fixed_mod]}.nii.gz"
        fixed_reg_out.parent.mkdir(parents=True, exist_ok=True)

        fixed_reg = ants.apply_transforms(
            fixed=tpl_img,
            moving=ants.image_read(str(skull_fixed)),
            transformlist=tfm,
        )

        if do_final_clip:
            fixed_reg = clip_intensities(fixed_reg, 0.001, 0.999)

        ants.image_write(fixed_reg, str(fixed_reg_out))
        modality_records[fixed_mod]["preprocessed"] = str(fixed_reg_out)
        log("  [4/5] Registered fixed to template")
    else:
        log("  [4/5] Skipped template registration")

    # Step 5: other modalities -> MNI; mask -> MNI
    if do_apply_mask and tfm:
        tpl_mask_img = ants.image_read(str(tpl_mask))
        registered_dir.mkdir(parents=True, exist_ok=True)

        n_registered = 0
        for m, mov in moved.items():
            if m == fixed_mod:
                continue

            img_m = ants.image_read(str(mov))

            if bet_mask:
                msk = ants.image_read(str(bet_mask))
                if not ants.image_physical_space_consistency(img_m, msk):
                    msk = ants.resample_image_to_target(msk, img_m, interp_type="nearestNeighbor")
                moving_input = img_m * msk
            else:
                moving_input = img_m

            out_mov_reg = ants.apply_transforms(
                fixed=tpl_img,
                moving=moving_input,
                transformlist=tfm,
            )

            if not bet_mask:
                out_mov_reg = out_mov_reg * tpl_mask_img

            if do_final_clip:
                out_mov_reg = clip_intensities(out_mov_reg, 0.001, 0.999)

            reg_out = registered_dir / f"{output_stems[m]}.nii.gz"
            ants.image_write(out_mov_reg, str(reg_out))
            modality_records[m]["preprocessed"] = str(reg_out)
            n_registered += 1

        log(f"  [5/5] Registered {n_registered} moving modalities to template")

        if lesion_mask_path:
            log("  Applying final template transform to lesion mask")
            mask_img = ants.image_read(str(lesion_mask_path), pixeltype="unsigned int")

            final_mask = ants.apply_transforms(
                fixed=tpl_img,
                moving=mask_img,
                transformlist=tfm,
                interpolator="genericLabel",
            )
            final_mask = final_mask * tpl_mask_img

            final_mask_path = registered_dir / "lesion_mask.nii.gz"
            ants.image_write(final_mask, str(final_mask_path))
            mask_preprocessed_path = str(final_mask_path)
            log(f"  Final lesion mask saved: {final_mask_path}")
    else:
        log("  [5/5] Skipped template application to moving images / mask")

    log(f"=== Done {patient_id} [{display_plane}] ===")
    return modality_records, mask_preprocessed_path


def preprocess_patient(
    df: pd.DataFrame,
    id_col: str,
    template: Path,
    tpl_mask: Path,
    output_dir: Path,
    skip_exist: bool,
    do_debias: bool,
    do_coreg: bool,
    do_bet: bool,
    save_bet: bool,
    do_reg2tpl: bool,
    do_apply_mask: bool,
    do_final_clip: bool,
    process_mask: bool,
    mask_modality: str,
    split_2d_by_plane: bool,
) -> pd.DataFrame:
    patient_id = str(df[id_col].iloc[0])

    units = split_patient_into_units(
        df=df,
        split_2d_by_plane=split_2d_by_plane,
    )

    if len(units) == 1 and units[0][2] is False:
        log(f"\n[INFO] Patient {patient_id}: standard single-unit processing.")
    else:
        names = [u[0] for u in units]
        log(f"\n[INFO] Patient {patient_id}: split into temporary units {names}")

    df_out = df.copy()
    df_out["plane_group"] = "MAIN"
    df_out["debias"] = pd.NA
    df_out["BET"] = pd.NA
    df_out["preprocessed"] = pd.NA

    final_mask_path_for_patient: Optional[str] = None

    for plane_group, unit_df, split_used in units:
        modality_records, mask_preprocessed_path = preprocess_unit(
            unit_df=unit_df,
            patient_id=patient_id,
            plane_group=plane_group,
            split_used=split_used,
            template=template,
            tpl_mask=tpl_mask,
            output_dir=output_dir,
            skip_exist=skip_exist,
            do_debias=do_debias,
            do_coreg=do_coreg,
            do_bet=do_bet,
            save_bet=save_bet,
            do_reg2tpl=do_reg2tpl,
            do_apply_mask=do_apply_mask,
            do_final_clip=do_final_clip,
            process_mask=process_mask,
            mask_modality=mask_modality,
        )

        for mod, record in modality_records.items():
            mod_rows = df_out["modality"] == mod

            # if split was used, restrict to this plane by original source path
            if split_used:
                unit_modalities = set(unit_df["modality"].tolist())
                mod_rows = mod_rows & df_out["modality"].isin(unit_modalities)

                src_series = df_out.loc[mod_rows, "non-preprocessing"].astype(str)
                if plane_group == "SAG":
                    mod_rows = mod_rows & src_series.str.upper().str.contains("SAG|SAGITTAL", regex=True)
                elif plane_group == "AX":
                    mod_rows = mod_rows & src_series.str.upper().str.contains("AX|AXIAL", regex=True)
                elif plane_group == "OTHER":
                    sag_mask = src_series.str.upper().str.contains("SAG|SAGITTAL", regex=True)
                    ax_mask = src_series.str.upper().str.contains("AX|AXIAL", regex=True)
                    mod_rows = mod_rows & (~sag_mask) & (~ax_mask)

            df_out.loc[mod_rows, "plane_group"] = record["plane_group"]
            df_out.loc[mod_rows, "debias"] = record["debias"]
            df_out.loc[mod_rows, "BET"] = record["BET"]
            df_out.loc[mod_rows, "preprocessed"] = record["preprocessed"]

        if mask_preprocessed_path is not None:
            final_mask_path_for_patient = mask_preprocessed_path

    if process_mask:
        if "mask_preprocessed" not in df_out.columns:
            df_out["mask_preprocessed"] = pd.NA
        if final_mask_path_for_patient is not None:
            df_out["mask_preprocessed"] = final_mask_path_for_patient

    return df_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with modality table")
    parser.add_argument("--dataset", default=None, type=str, help="Dataset name filter")
    parser.add_argument(
        "--template",
        default="/gpfs/data/shenlab/Jiajian/MS_Project/code/MRI-preprocessing-techniques/assets/templates/mni_icbm152_t1_tal_nlin_sym_09a.nii",
    )
    parser.add_argument(
        "--tpl_mask",
        default="/gpfs/data/shenlab/Jiajian/MS_Project/code/MRI-preprocessing-techniques/assets/templates/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--manifest_name", default="preprocessing_manifest.csv", help="Output manifest CSV filename")

    parser.add_argument("--start", type=int, default=0, help="Start patient index")
    parser.add_argument("--end", type=int, default=None, help="End patient index")

    parser.add_argument("--id_col", type=str, default="m_id", help="Preferred ID column name")
    parser.add_argument("--skip_exist", action="store_true")
    parser.add_argument("--do_debias", action="store_true")
    parser.add_argument("--do_coreg", action="store_true")
    parser.add_argument("--do_bet", action="store_true")
    parser.add_argument("--save_bet", action="store_true")
    parser.add_argument("--do_reg2tpl", action="store_true")
    parser.add_argument("--do_apply_mask", action="store_true")
    parser.add_argument("--do_final_clip", action="store_true")

    parser.add_argument(
        "--process_mask",
        action="store_true",
        help='Process lesion masks from CSV column "mask_path"',
    )
    parser.add_argument(
        "--mask_modality",
        type=str,
        default="3DFLAIR_NCE",
        help="Which modality the lesion mask is originally aligned with",
    )
    parser.add_argument(
        "--split_2d_by_plane",
        action="store_true",
        help=(
            "For 2D-only patients, temporarily split modalities into SAG / AX / OTHER "
            "processing units based on file path. Final outputs remain in the standard "
            "patient directory, with 2D filenames suffixed by plane to avoid overwrite."
        ),
    )

    args = parser.parse_args()

    if not any([
        args.do_debias,
        args.do_coreg,
        args.do_bet,
        args.save_bet,
        args.do_reg2tpl,
        args.do_apply_mask,
    ]):
        args.do_debias = True
        args.do_coreg = True
        args.do_bet = True
        args.save_bet = True
        args.do_reg2tpl = True
        args.do_apply_mask = True

    check_gpu()

    df = pd.read_csv(args.csv)
    id_col = get_id_column(df, args)

    if args.dataset:
        if "dataset" not in df.columns:
            sys.exit("Error: --dataset provided but CSV has no 'dataset' column.")
        df = df[df["dataset"] == args.dataset].copy()
        if df.empty:
            sys.exit(f"Error: no data found for dataset '{args.dataset}'.")

    required_cols = [id_col, "modality", "non-preprocessing"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        sys.exit(f"Error: CSV missing required columns: {missing_cols}")

    df = df.sort_values([id_col, "modality"]).copy()

    all_ids = df[id_col].unique()
    total_patients = len(all_ids)

    start_idx = max(0, args.start)
    end_idx = total_patients if args.end is None else min(args.end, total_patients)
    target_ids = all_ids[start_idx:end_idx]

    log(f"Processing range: [{start_idx}:{end_idx}] ({len(target_ids)}/{total_patients} patients)")

    df_subset = df[df[id_col].isin(target_ids)].copy()
    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    manifest_parts = []

    for _, grp in df_subset.groupby(id_col, sort=False):
        patient_manifest = preprocess_patient(
            df=grp,
            id_col=id_col,
            template=Path(args.template),
            tpl_mask=Path(args.tpl_mask),
            output_dir=base_out,
            skip_exist=args.skip_exist,
            do_debias=args.do_debias,
            do_coreg=args.do_coreg,
            do_bet=args.do_bet,
            save_bet=args.save_bet,
            do_reg2tpl=args.do_reg2tpl,
            do_apply_mask=args.do_apply_mask,
            do_final_clip=args.do_final_clip,
            process_mask=args.process_mask,
            mask_modality=args.mask_modality,
            split_2d_by_plane=args.split_2d_by_plane,
        )
        manifest_parts.append(patient_manifest)

    if manifest_parts:
        final_manifest = pd.concat(manifest_parts, axis=0, ignore_index=True)
    else:
        final_manifest = df_subset.copy()

    manifest_path = base_out / args.manifest_name
    final_manifest.to_csv(manifest_path, index=False)
    log(f"\n✅ Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()