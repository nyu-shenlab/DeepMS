#!/usr/bin/env python3
"""Validate that the active Python environment matches DeepMS core pins."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PIN_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^;\s]+)$")


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _core_pins(project_root: Path) -> dict[str, str]:
    with (project_root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    pins: dict[str, str] = {}
    for requirement in project["project"]["dependencies"]:
        match = PIN_PATTERN.fullmatch(requirement)
        if match is None:
            raise ValueError(
                "Core dependencies must use exact == pins; "
                f"unsupported requirement: {requirement}"
            )
        name, version = match.groups()
        pins[_normalized_distribution_name(name)] = version
    return pins


def audit_environment(project_root: Path = REPOSITORY_ROOT) -> tuple[dict[str, str], list[str]]:
    problems: list[str] = []
    expected_python = (project_root / ".python-version").read_text(encoding="utf-8").strip()
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        problems.append(f"Python {actual_python} is active; expected {expected_python}.")

    pins = _core_pins(project_root)
    installed: dict[str, str] = {}
    for name, expected_version in sorted(pins.items()):
        try:
            actual_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"Missing distribution: {name}=={expected_version}")
            continue
        installed[name] = actual_version
        if actual_version != expected_version:
            problems.append(
                f"Version mismatch for {name}: installed {actual_version}, "
                f"expected {expected_version}."
            )

    torch_version = "unavailable"
    torch_cuda = "unavailable"
    cuda_available = "unavailable"
    if "torch" in installed:
        try:
            import torch

            torch_version = torch.__version__
            torch_cuda = str(torch.version.cuda)
            cuda_available = str(torch.cuda.is_available())
        except Exception as error:  # pragma: no cover - depends on binary runtime
            problems.append(f"PyTorch is installed but cannot be imported: {error}")

    lock_digest = hashlib.sha256((project_root / "uv.lock").read_bytes()).hexdigest()
    report = {
        "python": sys.version.split()[0],
        "environment": sys.prefix,
        "uv_lock_sha256": lock_digest,
        "torch": torch_version,
        "torch_cuda": torch_cuda,
        "cuda_available": cuda_available,
    }
    return report, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only validation failures.",
    )
    args = parser.parse_args()

    try:
        report, problems = audit_environment()
    except (KeyError, OSError, ValueError) as error:
        print(f"DeepMS environment audit could not run: {error}", file=sys.stderr)
        return 1

    if problems:
        print("DeepMS locked environment is not ready:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("DeepMS locked environment is ready.")
        for name, value in report.items():
            print(f"  {name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
