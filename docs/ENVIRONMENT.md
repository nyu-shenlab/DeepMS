# Reproducible environment

DeepMS uses `pyproject.toml` for direct dependency declarations and `uv.lock`
for the complete Linux x86_64 resolution. Commit both files. Do not commit or
share `.venv`; each user creates a local environment from the same lockfile.

## New clone

Install `uv` once under your own account using the official standalone
installer, then run from the repository root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Start a new shell if `uv` is not immediately available on PATH.
uv sync --locked --no-dev
./scripts/bootstrap_env.sh --check
```

`uv sync --locked` downloads the exact locked dependency graph and refuses a
stale lockfile. The optional helper below performs the same core sync and then
runs the repository environment audit:

```bash
./scripts/bootstrap_env.sh
```

If you installed `uv` somewhere that is not on `PATH`, point DeepMS to your own
executable:

```bash
export DEEPMS_UV_BIN=/absolute/path/to/uv
./scripts/bootstrap_env.sh
```

The Shenlab site profile resolves `uv` from each user's `PATH`. It deliberately
does not reference another user's executable or environment.

## Put large environments on scratch

The core GPU environment is several gigabytes. Sites with small home quotas
can keep both the project environment and package cache on scratch:

```bash
export UV_PROJECT_ENVIRONMENT=/path/to/scratch/deepms-venv
export UV_CACHE_DIR=/path/to/scratch/uv-cache
./scripts/bootstrap_env.sh
```

Export the same two values before submitting Slurm jobs. Slurm exports the
calling environment by default, and every DeepMS job resolves the interpreter
through `uv run`; no `.venv` symlink is required.

## Optional environments

```bash
# Structural preprocessing (ANTsPy and HD-BET)
./scripts/bootstrap_env.sh --extra preprocessing

# Weights & Biases tracking
./scripts/bootstrap_env.sh --extra tracking

# Tests and linting
./scripts/bootstrap_env.sh --dev
```

## Updating dependencies

Only maintainers should change the resolved environment. Use `uv add`,
`uv remove`, or an intentional `uv lock --upgrade-package <name>`, then commit
both `pyproject.toml` and `uv.lock`. Consumers should pull those files and run:

```bash
uv sync --locked --no-dev
```

Do not maintain a separate hand-written `requirements.txt`; it would duplicate
the dependency source of truth and can drift from `uv.lock`. Export one from
the lockfile only when a downstream platform cannot use `uv`.

## Slurm policy

Training and inference jobs use `uv run --locked --no-sync`. They never install
packages or contact an index on a compute node. Before consuming GPU resources,
the jobs verify Python 3.11, every direct pinned distribution, the lockfile
fingerprint, and a working PyTorch import. An incomplete environment therefore
fails with a setup instruction instead of launching training.

If the audit reports `ModuleNotFoundError: torch`, the environment directory
exists but was not successfully synchronized. Run `./scripts/bootstrap_env.sh`
from a login node and repeat the check before resubmitting.
