# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

**MLinCP** is the course repository for *Machine Learning in Computational Physics* at the University of Vienna. It contains Jupyter notebook workshops for 9 course modules and their solutions, plus the Docker/UV environment used by students.

## People

- **rM** — Reinhard Maurer (reinhard.maurer@univie.ac.at) — lead lecturer, majority of workshops
- **aC** — Alessandro Coretti — co-lecturer (Modules V, VI, VIII, IX)
- **aS** — Alexander Spears — graduate TA, supports all workshops

## Repository structure

```
Module_I_Introduction/WS1_Introduction/
    WS1_Introduction.ipynb           ← student-facing notebook
    WS1_Introduction_solutions.ipynb ← solutions, never distributed
...
Module_VII_HDNNP/
    WS7_MACE_Embeddings/
    WS8_MACE_Fine_Tuning/            ← Module VII spans two workshops
...
ToC.md                               ← canonical schedule (dates, lecturers, workshop titles)
pyproject.toml                       ← UV dependency manifest (source of truth)
Dockerfile / docker-compose.yml      ← student container
.devcontainer/devcontainer.json      ← VS Code Dev Container config
```

`old_summerschool/` and `ML_summerschool_2025/` are gitignored — they contain prior summerschool content used as source material when building new notebooks.

## Environment

Dependencies are managed with **UV**. The `uv.lock` file pins all transitive dependencies.

```bash
uv sync              # create/update .venv from pyproject.toml + uv.lock
source .venv/bin/activate   # activate (Linux/macOS)
.venv\Scripts\activate      # activate (Windows)
```

After adding or changing a dependency in `pyproject.toml`:
```bash
uv sync              # resolves and updates uv.lock
# commit both pyproject.toml and uv.lock
```

PyTorch wheels are selected automatically by platform via `[tool.uv.sources]` in `pyproject.toml`: CUDA 12.4 on Linux x86_64 (course server), CPU-only on Windows/macOS x86_64, and native MPS wheels on Apple Silicon. `aseMolec` is installed from GitHub (`imagdau/aseMolec`) as it is not on PyPI.

## Running notebooks

```bash
# Start JupyterLab (local UV env)
jupyter lab

# Start via Docker Compose (what students use)
docker compose up
# → open http://localhost:8888

# Rebuild container after dependency changes
docker compose up --build
```

## Notebook conventions

- Each workshop has a **student notebook** (`WSN_Title.ipynb`) and a **solutions notebook** (`WSN_Title_solutions.ipynb`).
- Solutions notebooks are gitignored by default for student-facing deployments. If a solutions branch or separate repo is used, update `.gitignore` accordingly.
- Notebooks import standard libraries (`numpy`, `matplotlib`, `torch`) at the top of a Setup cell. Workshop-specific imports follow.
- The `old_summerschool/` folder is the primary reference for adapting existing content into new workshops.

## Key dependencies

| Package | Role |
|---------|------|
| `torch`, `torchvision` | Deep learning (platform-selected: CUDA/CPU/MPS) |
| `ase` | Atomic structure I/O and simulation interface |
| `mace-torch` | Equivariant MPNN potentials (Modules VII) |
| `janus-core` | Workflows on top of MACE/ASE |
| `dscribe` | Structural descriptors (SOAP, MBTR, Coulomb matrix) |
| `chemeleon` | Diffusion-based crystal generation (Module VIII) |
| `aseMolec` | Molecular property utilities (GitHub-only) |
| `openTSNE`, `umap-learn` | Dimensionality reduction (Module III) |
| `scikit-learn` | Classical ML methods (Modules III–V) |
