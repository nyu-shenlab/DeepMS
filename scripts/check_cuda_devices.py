#!/usr/bin/env python3
"""Fail fast when a Slurm allocation contains an unusable CUDA device."""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys

import torch


def query_nvidia_smi_devices() -> list[str]:
    """Return the node-level PCI/UUID inventory reported by nvidia-smi."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,uuid,name",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(
            f"WARNING: could not query CUDA PCI inventory with nvidia-smi: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return []

    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    for row in rows:
        print(f"NVIDIA-SMI node inventory: {row}", flush=True)
    return rows


def map_visible_device_inventory(
    inventory: list[str], visible_count: int
) -> list[str]:
    """Map CUDA-local indices to physical inventory without assuming row order."""
    raw_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not raw_visible_devices or not raw_visible_devices.strip():
        reason = "CUDA_VISIBLE_DEVICES is unset"
        return [f"physical mapping unavailable ({reason})"] * visible_count

    tokens = [token.strip() for token in raw_visible_devices.split(",")]
    if len(tokens) != visible_count or any(not token for token in tokens):
        reason = (
            f"CUDA_VISIBLE_DEVICES={raw_visible_devices!r} has {len(tokens)} tokens "
            f"for {visible_count} visible devices"
        )
        return [f"physical mapping unavailable ({reason})"] * visible_count

    records: list[tuple[str, str, str, str]] = []
    for row in inventory:
        columns = [column.strip() for column in next(csv.reader([row]))]
        if len(columns) < 4:
            continue
        physical_index, pci_bus_id, uuid = columns[:3]
        records.append((physical_index, pci_bus_id, uuid, row))

    mapped: list[str] = []
    for token in tokens:
        if token.isdigit():
            matches = [record for record in records if record[0] == token]
        elif token.upper().startswith("GPU-"):
            token_upper = token.upper()
            matches = [
                record
                for record in records
                if record[2].upper().startswith(token_upper)
            ]
        else:
            # MIG identifiers cannot be safely joined to --query-gpu rows.
            matches = []

        if len(matches) == 1:
            physical_index, pci_bus_id, uuid, _ = matches[0]
            mapped.append(
                f"CUDA_VISIBLE_DEVICES token={token} -> "
                f"physical_index={physical_index} pci_bus_id={pci_bus_id} uuid={uuid}"
            )
        else:
            mapped.append(
                f"CUDA_VISIBLE_DEVICES token={token} -> physical mapping unavailable "
                f"(inventory matches={len(matches)})"
            )
    return mapped


def probe_cuda_devices(expected: int) -> None:
    """Initialize every visible device and execute a minimal CUDA operation."""
    visible_count = torch.cuda.device_count()
    print(
        "CUDA allocation: "
        f"host={socket.gethostname()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS', '<unset>')} "
        f"visible_count={visible_count} expected_count={expected}",
        flush=True,
    )
    if visible_count != expected:
        raise RuntimeError(
            f"Expected {expected} visible CUDA devices, but found {visible_count}."
        )

    inventory = query_nvidia_smi_devices()
    visible_inventory = map_visible_device_inventory(inventory, visible_count)
    for index in range(visible_count):
        inventory_row = visible_inventory[index]
        print(
            f"CUDA probe starting: cuda:{index} inventory={inventory_row}",
            flush=True,
        )
        try:
            torch.cuda.set_device(index)
            properties = torch.cuda.get_device_properties(index)
            probe = torch.ones(1, device=f"cuda:{index}")
            probe.add_(1)
            torch.cuda.synchronize(index)
        except Exception as exc:
            raise RuntimeError(
                f"cuda:{index} failed context/compute probe "
                f"({inventory_row}): {exc}"
            ) from exc
        print(
            f"CUDA probe passed: cuda:{index} "
            f"name={properties.name} memory_bytes={properties.total_memory} "
            f"inventory={inventory_row}",
            flush=True,
        )
        del probe

    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected < 1:
        print("ERROR: --expected must be a positive integer.", file=sys.stderr)
        return 2

    try:
        probe_cuda_devices(args.expected)
    except Exception as exc:
        print(f"ERROR: CUDA allocation preflight failed: {exc}", file=sys.stderr)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
