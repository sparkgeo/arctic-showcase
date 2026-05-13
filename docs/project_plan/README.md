# Prescient Ice

Automated Arctic sea ice concentration mapping using Sentinel-1 SAR imagery and geospatial foundation models, built on the Prescient data management platform.

## Documentation

### [Project Index](prescient_ice_index.md)
Section summaries with pointers to the standalone documents below, plus study area and temporal scope, project phases, and open questions and risks. Start here.

### [Overview](prescient_ice_overview.md)
Background and motivation, project objectives, and a summary of the five-stage pipeline.

### [Datasets](prescient_ice_datasets.md)
Per-dataset profiles covering all evaluated data sources — product selection, access paths, caveats, and suitability classification (model input, supplemental, candidate, not selected).

### [Training Strategy](prescient_ice_training_strategy.md)
How USNIC polygon labels are converted into 500m pixel-level training targets (phased label preparation from pure cells through ICESat-2 anchors), and how SAR acquisitions are temporally aligned to those labels using a season-adaptive window.

### [Model Architecture](prescient_ice_model_architecture.md)
Two-phase modeling strategy: Phase 1 uses Clay as a frozen feature extractor with a downstream regression model trained on pre-computed embeddings; Phase 2 (conditional) adds end-to-end Clay fine-tuning if Phase 1 accuracy is insufficient.

### [Pipeline Architecture](prescient_ice_pipeline_architecture.md)
End-to-end pipeline across six stages — data acquisition, ingestion, embedding generation, model training, inference, and visualization — with per-stage infrastructure recommendations and a full STAC collection inventory.
