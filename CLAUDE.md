# Arctic Showcase — Claude Code Guide

## Project overview

Geospatial data science showcase demonstrating Prescient for arctic data management and analysis. Datasets include Sentinel-2, SAR, ArcticDEM, sea ice, SST, and AIS. The first module is `data_ingest`; additional modules live under `src/` as the project grows.

## Repo structure

```
arctic-showcase/
├── src/                  # Workspace members (one subdirectory per module)
│   └── data_ingest/      # uv init --lib layout
├── scripts/              # One-off runners, not imported as packages
├── notebooks/            # Exploratory Jupyter notebooks
├── tests/                # Pytest suite (mirrors src/ layout)
├── documentation/        # Design docs, changelog
├── pyproject.toml        # UV virtual workspace root (package = false)
├── uv.lock
└── Dockerfile
```

## UV workspace conventions

This repo uses a **UV workspace** with a virtual root. Key rules:

- **Never** run `pip install`. Always use `uv add` or `uv sync`.
- Add a new module: `uv init --lib src/<module_name>` — this creates the library layout automatically.
- Add a module-specific dep: `uv add --package <module_name> <dep>`.
- Add a shared dep (available to all modules): `uv add <dep>` at the root.
- Dev tools (ruff, mypy, pytest, ipykernel) live in the root `[dependency-groups] dev`, not per-module.
- Sync everything: `uv sync --group dev`.

## Common commands

```bash
uv sync --group dev          # Install all deps including dev tools
uv run pytest                # Run tests
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run mypy src/             # Type check
uv run jupyter lab           # Notebooks (ipykernel is in dev group)
```

## Docker

Production runtime uses `ghcr.io/osgeo/gdal` as the base image (bundles GDAL, PROJ, GEOS). Do **not** add the `gdal` Python package — rasterio and geopandas bundle their own GDAL wheels and a system `libgdal` version mismatch will break them.

## Code conventions

- `src/` modules use the library layout (`src/<name>/src/<name>/`), not flat layout.
- Type annotations required; `mypy --strict` is enforced in CI.
- Line length: 100. Formatter: ruff.
- No comments explaining *what* code does — only *why* when non-obvious.
