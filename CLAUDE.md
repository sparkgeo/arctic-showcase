# Arctic Showcase — Claude Code Guide

## Project overview

Prescient Ice is an automated Arctic sea ice concentration (SIC) mapping project. It classifies SIC into eleven discrete classes (0–10 tenths, SIGRID-3 scheme) at 320m resolution over Hudson Bay, using Sentinel-1 EW SAR imagery and the Clay v1.5 geospatial foundation model as a frozen feature extractor. The project also serves as an internal showcase for Prescient, Sparkgeo's cloud-native geospatial data management platform (STAC/PGStac, TiTiler, MapLibre).

Design documentation lives in `docs/project_plan/`. Start with `prescient_ice_index.md` for navigation; consult the relevant document before implementing any non-trivial task.

## Working practices

**Surface confusion before coding.** Many invariants in this project fail *silently* — a wrong Clay `model_size`, mismatched normalisation constants, or a feature-vector column out of order produces plausible but corrupt output, not an error. When a request is ambiguous or an assumption is load-bearing, state it or ask before implementing rather than guessing.

**Surgical changes.** Touch only what the task requires. Don't refactor, reformat, or "improve" adjacent code; match existing style even where you'd differ. Remove only the imports or names your own change orphaned — flag pre-existing dead code rather than deleting it. This mirrors the project's discuss-then-edit discipline.

**Simplicity first.** The minimum code that solves the problem; no speculative abstraction, configurability, or error handling for impossible cases.

**Verify against criteria.** Prefer verifiable goals over vague ones — for a bug, write a failing test that reproduces it, then make it pass; ensure the suite passes before and after a refactor.

## Repo structure

```
arctic-showcase/
├── src/                  # Workspace members (one subdirectory per module)
│   └── data_ingest/      # uv init --lib layout; Track A ingestion code
├── scripts/              # One-off runners, not imported as packages
├── notebooks/            # Exploratory Jupyter notebooks
├── tests/                # Pytest suite (mirrors src/ layout)
├── docs/
│   └── project_plan/     # Design documents (read-only reference)
├── pyproject.toml        # UV virtual workspace root (package = false)
├── uv.lock
└── Dockerfile
```

Track B modelling code (Clay encoding, classifier training) will live as a sibling package under `src/` when scaffolded.

## UV workspace conventions

This repo uses a **UV workspace** with a virtual root. Key rules:

- **Never** run `pip install`. Always use `uv add` or `uv sync`.
- Add a new module: `uv init --lib src/<module_name>` — this creates the library layout automatically.
- Add a module-specific dep: `uv add --package <module_name> <dep>`.
- Add a shared dep (available to all modules): `uv add <dep>` at the root.
- Dev tools (ruff, mypy, pytest, ipykernel) live in the root `[dependency-groups] dev`, not per-module.
- Sync everything: `uv sync --all-packages --group dev`.

## Common commands

```bash
uv sync --all-packages --group dev   # Install all workspace members + dev tools
uv run pytest                        # Run tests
uv run ruff check .                  # Lint
uv run ruff format .                 # Format
uv run mypy src/                     # Type check
uv run jupyter lab                   # Notebooks (ipykernel is in dev group)
```

## Docker

Production runtime builds on the OSGeo GDAL base image (see `Dockerfile`), which provides system GDAL, PROJ, and GEOS. Do **not** add the `gdal` Python package — rasterio and geopandas bundle their own GDAL wheels, and a system `libgdal` version mismatch will break them. The image builds with `uv sync --frozen --no-group dev`, so anything needed at runtime must be a non-dev dependency in the lockfile.

## Code conventions

- `src/` modules use the library layout (`src/<name>/src/<name>/`), not flat layout.
- Type annotations required; `mypy --strict` is enforced in CI.
- Line length: 100. Formatter: ruff.
- No comments explaining *what* code does — only *why* when non-obvious.

---

## Project architecture

Work is split across two semi-parallel tracks that share a common feature contract:

**Track A — Prescient datasets & inference.** Dataset ingestion into Prescient as COG/vector STAC collections, then the 2025–26 Hudson Bay inference pipeline. Code lives in `src/data_ingest/` and future sibling packages. Reads and writes S3 via Prescient's STAC interface.

**Track B — Modelling.** AI4Arctic training data assembly, Clay encoding, classifier training and evaluation. Reads and writes S3 directly; does not depend on Prescient. The co-op student owns this track.

The tracks converge at the **feature contract** — the authoritative specification in `prescient_ice_model_architecture.md` § Feature Contract. Both tracks must produce an identical five-block feature vector schema in identical column order. This is the single most important invariant in the codebase; see § Invariants below.

**Critical path:** B1 (normalisation constants) gates all Clay encoding on both tracks. B1 → B2 → B3 is the time-bound path (co-op student, ~end of August 2026). Track A ingestion tasks (A1–A7) carry no dependency on Track B and can proceed in parallel.

**Key infrastructure:** AWS S3 (asset storage), SageMaker (training, managed MLflow), AWS Batch GPU g5 (Clay encoding), Step Functions (orchestration). EPSG:3978 (NAD83 / Canada Atlas Lambert) is the analytical CRS throughout; TiTiler reprojects to Web Mercator on the fly for tile serving.

For full pipeline detail, see `prescient_ice_pipeline_architecture.md`. For the implementation task breakdown with inputs, outputs, and acceptance criteria, see `prescient_ice_implementation_plan.md`.

---

## Invariants — do not break or reinterpret

### Feature contract parity

The feature vector is five ordered blocks: raw SAR statistics → Clay patch token → AMSR2 brightness temperatures → ERA5 variables → distance-to-land index and incidence angle mean. Full column schema and ordering are specified in `prescient_ice_model_architecture.md` § Feature Contract. Both the training assembly path (B2) and the inference path (A9) must produce this schema identically. Any deviation is a silent distribution shift between training and inference — it will not raise an error.

### NERSC normalisation constants

Dataset-wide per-band mean and std of NERSC-corrected σ⁰ (dB), computed across all 513 AI4Arctic training scenes, are computed once in task B1 and stored as a versioned S3 artefact. Both B2.2 (training encoding) and A8 (inference encoding) must **consume** these constants from S3 — never re-derive them, and never substitute Clay's built-in `sentinel-1-rtc` constants. These constants also serve as the nodata substitution mean (substituted pixels become exactly zero in Clay's normalised input space).

### Clay encoder API

Load `ClayMAEModule(model_size="large", mask_ratio=0.0, shuffle=False)`. The default `model_size="base"` produces 768-dimensional output — the feature contract assumes 1024, which only `"large"` provides, and the mismatch will not raise an error. Call `module.model.encoder(batch)` where `batch` is a dict with keys `pixels` `[B, C, 256, 256]`, `time` `[B, 4]`, `latlon` `[B, 4]`, `waves`, and `gsd`. Unpack as `encoded, *_ = module.model.encoder(batch)` — the remaining tuple elements are unused at inference. Index 0 of the sequence dimension is the class token `[B, 1024]`; indices `[1:]` are the 1024 patch tokens, reshaped via:

```python
einops.rearrange(encoded[:, 1:, :], "b (h w) d -> b d h w", h=32, w=32)
```

The public quick-start documentation contains a misleading API snippet that contradicts the actual codebase — treat `claymodel/model.py` and `docs/tutorials/embeddings.ipynb` in the Clay repository as ground truth. Do not use `EmbeddingEncoder` — its default `forward` returns only the class token and discards patch tokens.

### AMSR2 is brightness temperature

The AMSR2 feature is passive-microwave brightness temperature (14 channels, ascending frequency, H before V), **not** a derived SIC value. Training reads it from the AI4Arctic `btemp_FFP` bundle; inference sources JAXA G-Portal Level-1R data. Do not interpret, rename, or reframe AMSR2 columns as concentration or ice fraction values.

### SAR band ordering

`nersc_sar_primary` = HH (index 0), `nersc_sar_secondary` = HV (index 1). HH is always first in the feature vector and in the Clay `pixels` tensor.
