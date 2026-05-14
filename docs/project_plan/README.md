# Prescient Ice

Automated Arctic sea ice concentration mapping using Sentinel-1 SAR imagery and geospatial foundation models, built on the Prescient data management platform.

## Documentation

### [Project Index](prescient_ice_index.md)
Section summaries with pointers to the standalone documents below, plus project phases, open questions and risks, and future extensions. Start here.

### [Overview](prescient_ice_overview.md)
Background and motivation, project objectives, and a summary of the five-stage pipeline.

### [Datasets](prescient_ice_datasets.md)
Per-dataset profiles covering all evaluated data sources — product selection, access paths, caveats, and suitability classification (model input, supplemental, candidate, not selected).

### [Study Area and Temporal Scope](prescient_ice_study_area.md)
Study area definition (Hudson Bay, 95°W–75°W, 58°N–66°N), temporal scope (October 2025–January 2026 freeze-up season), analytical CRS and projection decisions (EPSG:3978), Sentinel-1 constellation status, and training data volume estimates.

### [Training Strategy](prescient_ice_training_strategy.md)
How USNIC polygon labels are converted into 500m pixel-level training targets (phased label preparation from pure cells through ICESat-2 anchors), and how SAR acquisitions are temporally aligned to those labels using a season-adaptive window.

### [Model Architecture](prescient_ice_model_architecture.md)
Two-phase modeling strategy: Phase 1 uses Clay as a frozen feature extractor with a downstream regression model trained on pre-computed embeddings; Phase 2 (conditional) adds end-to-end Clay fine-tuning if Phase 1 accuracy is insufficient.

### [Pipeline Architecture](prescient_ice_pipeline_architecture.md)
End-to-end pipeline across six stages — data acquisition, ingestion, embedding generation, model training, inference, and visualization — including the projection and CRS strategy, dual-asset vector pattern (GeoParquet + PMTiles), per-stage infrastructure recommendations, and a full STAC collection inventory.
