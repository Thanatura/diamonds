# Diamonds — ML Production Challenges

This repository is a learning sandbox for ML engineering and MLOps practices.

The course is split into **5 challenges**, and each challenge is delivered on its own Git branch. This `README` stays generic and gives only the common workflow used across all branches.

## Repository Structure

- `src/diamonds/`: Python package code
- `data/`: raw and preprocessed data
- `models/`: locally saved models
- `mlartifacts/`: MLflow artifacts
- `notebooks/`: exploration and experimentation notebooks
- `docs/`: challenge instructions

Current docs already available:
- `docs/01-Packaging.md`
- `docs/02-Model-Registry.md`

## Challenge Model (5 Branches)

At the end, the repository contains 5 challenge branches (one branch per challenge).

Typical flow:
1. Checkout the branch for the challenge.
2. Read the matching file in `docs/`.
3. Implement the requested tasks.
4. Commit and push your solution branch.

## Quick Start (Common to All Challenges)

From the project root:

```bash
make setup
```

This installs the project in editable mode (`pip install -e .`).

Python requirement (from `pyproject.toml`):
- Python `>=3.11,<4.0`

## Branch Workflow

```bash
git fetch origin
git checkout <challenge-branch>
git pull origin <challenge-branch>
```

If you work on your own solution branch:

```bash
git checkout -b <your-name>-<challenge>
```

## MLflow (for tracking/registry challenges)

Start MLflow server:

```bash
make launch-mlflow-server
```

Default local UI:
- `http://127.0.0.1:5000`

Optional environment variable for scripts:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

## Notes

- Keep changes focused on the current challenge branch.
- Do not assume code from another challenge branch is already merged.
- Prefer small, clear commits with explicit messages.
