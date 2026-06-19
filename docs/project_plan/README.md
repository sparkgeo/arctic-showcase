# Prescient Ice

Automated Arctic sea ice concentration mapping using Sentinel-1 EW SAR imagery and the Clay v1.5 geospatial foundation model, built on the Prescient data management platform. SIC is classified into eleven discrete classes (0–10 tenths) on a 320m grid defined by Clay's patch footprint at Sentinel-1 EW's native ~40m ground sampling distance.

## Documentation

### [Project Index](prescient_ice_index.md)
Section summaries with pointers to the standalone documents below, plus project phases, open questions and risks, and future extensions. Start here.

### [Overview](prescient_ice_overview.md)
Background and motivation, project objectives, and a summary of the six-stage pipeline.

### [Datasets](prescient_ice_datasets.md)
Per-dataset profiles covering all evaluated data sources — product selection, access paths, caveats, and suitability classification (model input, supplemental, candidate, not selected). Includes the AI4Arctic Sea Ice Challenge Dataset as the primary training and evaluation source.

### [Study Area and Temporal Scope](prescient_ice_study_area.md)
Study area definition (Hudson Bay, 95°W–75°W, 58°N–66°N), temporal scope (October 2025–January 2026 freeze-up season as a prospective evaluation dataset against AI4Arctic-trained models), analytical CRS and projection decisions (EPSG:3978), Sentinel-1 constellation status across the window, and prospective evaluation pair volume estimates.

### [Training Strategy](prescient_ice_training_strategy.md)
How chart polygon labels (CIS/DMI via AI4Arctic for primary training; CIS Hudson Bay weekly regional charts for 2025–26 prospective evaluation) are converted into 320m patch-level eleven-class targets via pure-cell extraction and area-weighted midpoint rounding; the NERSC noise correction consistency requirement between training and inference data; and how SAR acquisitions are temporally aligned to charts for the 2025–26 pipeline using a season-adaptive window.

### [Model Architecture](prescient_ice_model_architecture.md)
Two-phase modeling strategy: Phase 1 uses Clay v1.5 as a frozen feature extractor — extracting 32×32 patch token grids per chip — with downstream classifiers (Random Forest and XGBoost in parallel) trained across three feature configurations (raw HH/HV baseline, patch tokens, patch tokens + chip embedding). Phase 2 (conditional) adds end-to-end Clay fine-tuning with a classification head if Phase 1 R² falls below threshold or fails to outperform the non-embedding baseline.

### [Pipeline Architecture](prescient_ice_pipeline_architecture.md)
End-to-end pipeline across six stages — data acquisition, ingestion, embedding generation, model training, inference, and visualization — including the projection and CRS strategy, dual-asset vector pattern (GeoParquet + PMTiles), the AI4Arctic-bypasses-Prescient training path, NERSC noise correction at Sentinel-1 ingest, per-stage infrastructure recommendations, and a full STAC collection inventory.

### [Implementation Plan](prescient_ice_implementation_plan.md)
Phased task breakdown for implementation using Claude Code as a coding assistant. Covers seventeen tasks across two semi-parallel tracks — Track A (Prescient ingestion and 2025–26 inference) and Track B (AI4Arctic training assembly, Clay encoding, and classifier training) — with explicit inputs, outputs, dependencies, and acceptance criteria per task.

## Citations

This project uses the AI4Arctic Sea Ice Challenge Dataset as its primary training and evaluation source. Per the dataset terms, any work using this dataset must cite:

Buus-Hinkler, Jørgen; Wulf, Tore; Stokholm, Andreas; Korosov, Anton; Saldo, Roberto; Pedersen, Leif Toudal; Arthurs, David; Solberg, Rune; Longépé, Nicolas; and Kreiner, Matilde Brandt; (2022): AI4Arctic Sea Ice Challenge Dataset. Danish Meteorological Institute. Dataset. https://doi.org/10.11583/DTU.c.6244065.
