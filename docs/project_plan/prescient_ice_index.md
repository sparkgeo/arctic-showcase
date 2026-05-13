# Prescient Ice: Project Index

## Project Overview

Prescient Ice is an automated Arctic sea ice concentration (SIC) mapping project producing 500m resolution gridded SIC estimates over the Canadian Arctic from Sentinel-1 SAR imagery, using a geospatial foundation model (GeoFM) as the feature extraction backbone. It serves as an internal showcase for Prescient, demonstrating end-to-end data ingestion, analytics integration, and derived product delivery across a realistic multi-source geospatial pipeline.

For background, motivation, and a full project summary, see [`prescient_ice_overview.md`](prescient_ice_overview.md).

### Modeling Approach Summary

The modeling strategy is two-phased. In Phase 1, Clay is used as a frozen feature extractor: Sentinel-1 embeddings are generated, ingested into Prescient as a derived STAC collection, and a simpler downstream model (e.g. Random Forest) is trained on the embedding features. Phase 2 — end-to-end Clay fine-tuning — is pursued only if the Phase 1 accuracy gap justifies the additional complexity and compute cost. See [Model Architecture](#model-architecture) for full detail.

---

## Datasets

The project draws on thirteen datasets spanning SAR imagery, passive microwave, lidar altimetry, climate reanalysis, optical imagery, foundation model embeddings, and vessel tracking. For full per-dataset detail — product selection, access paths, caveats, and evaluation rationale — see [`prescient_ice_datasets.md`](prescient_ice_datasets.md).

| Dataset | Category | Role | Prescient Collection |
|---|---|---|---|
| Sentinel-1 EW GRD | SAR | Model input — primary imagery | `sentinel-1-sar` |
| AMSR2 AU_SI12 | Passive microwave | Model input — SIC prior feature | `amsr2-sic` |
| ERA5 Single Levels | Climate reanalysis | Model input — atmospheric features; alignment filter | `era5-ancillary` |
| Clay embeddings | GeoFM | Model input — SAR patch embeddings (self-computed) | `clay-embeddings` |
| USNIC weekly SIGRID-3 | Ice chart | Training labels — primary | `usnic-ice-charts` |
| ICESat-2 ATL07/ATL10 | Lidar altimetry | Training labels — supplementary anchor points | `icesat2-tracks` |
| HLS L30/S30 | Optical | Validation; visual context layer | `hls-optical` |
| PM SIC CDR G02202 | Passive microwave | Showcase — long-term climate context | `pm-sic-cdr` |
| RCM ScanSAR | SAR | Candidate showcase / inference transferability | TBD |
| TESSERA | GeoFM embeddings | Candidate showcase — pre-computed embeddings | TBD |
| AlphaEarth AEF | GeoFM embeddings | Candidate showcase — pre-computed embeddings | TBD |
| SWOT KaRIn L2 Raster | Radar altimetry | Candidate — supplementary lead labels / showcase | TBD |
| AIS | Vessel tracking | Candidate showcase — shipping context layer | TBD |

---

## Training Strategy

The training strategy addresses two core challenges: converting coarse USNIC polygon labels into 500m pixel-level training targets, and ensuring temporal alignment between SAR acquisitions and the charts that label them.

Label preparation follows a phased approach. Phase 1 extracts only pure cells — grid cells that fall entirely within a single chart polygon — prioritising label quality over volume. Phase 2 expands coverage to mixed boundary cells using area-weighted concentration averaging, to be adopted only if empirical evaluation shows improved performance over Phase 1. Phase 3 integrates ICESat-2 anchor points along coincident tracks, providing physically-grounded near-0% and near-100% labels at the extremes of the concentration spectrum. A more advanced weak/aggregate label training strategy — calibrating model predictions at polygon scale rather than pixel scale — is noted as a future consideration after baselines are established.

Temporal alignment uses a 24-hour baseline window between SAR acquisition date and chart validity date, refined adaptively using ERA5 surface temperature and wind speed: stable winter conditions tolerate the full 24-hour window, while dynamic freeze-up and break-up periods tighten the requirement to 6–12 hours.

For full detail on label preparation phases, temporal alignment logic, and alignment edge cases, see [`prescient_ice_training_strategy.md`](prescient_ice_training_strategy.md).

---

## Model Architecture

The architecture is built around Clay, a geospatial foundation model, following a two-phase strategy. Phase 1 (primary) uses Clay as a frozen feature extractor: SAR patches are passed through Clay's encoder to produce patch embeddings, which are appended with AMSR2 and ERA5 ancillary features and ingested into Prescient as a derived STAC collection. A downstream regression model — initially Random Forest — is trained on these pre-computed embeddings. Decoupling embedding generation from downstream training enables rapid experimentation without re-running the encoder. Phase 2 (conditional) adds an end-to-end Clay fine-tuning step with a SIC regression head; it is pursued only if Phase 1 accuracy falls materially short of a defined threshold, and requires GPU compute on AWS.

For full detail on embedding generation, downstream model choices, ancillary feature integration, fine-tuning strategy, and Phase 2 compute requirements, see [`prescient_ice_model_architecture.md`](prescient_ice_model_architecture.md).

---

## Pipeline Architecture

The pipeline runs in six stages with Prescient as the shared data layer throughout. Stage 1 acquires source data from upstream providers. Stage 2 converts it to COG/PMTiles and registers it as STAC collections in Prescient. Stage 3 runs Clay's frozen encoder over Sentinel-1 scenes to produce patch embeddings, which are re-ingested into Prescient as a derived collection — decoupling the compute-intensive encoding step from downstream training iteration. Stage 4 assembles temporally aligned (embedding, label) pairs and trains the downstream regression model. Stage 5 applies the trained model to new SAR acquisitions and re-ingests the resulting 500m SIC grids. Stage 6 serves all layers through TiTiler and a MapLibre web viewer. AWS infrastructure spans Lambda and Batch for compute, Step Functions for orchestration, and S3 for all asset storage backing Prescient.

For full stage-by-stage detail — format conversions, ingestion workflows, embedding serialisation and re-ingestion, training dataset assembly, inference triggers, visualisation layers, STAC collection definitions, and infrastructure mapping by stage — see [`prescient_ice_pipeline_architecture.md`](prescient_ice_pipeline_architecture.md).

---

## Study Area and Temporal Scope

### Recommended Study Area

To be defined, but should include a region of the Canadian Arctic that experiences:
- Seasonal ice variability (freeze-up and break-up)
- A mix of consolidated pack ice and marginal ice zone
- Sufficient Sentinel-1 coverage
- Relevance to shipping or northern communities (strengthens the narrative)

Candidates include the Northwest Passage approaches, Hudson Bay, or the Beaufort Sea coast.

### Temporal Scope

For the initial showcase, a focused temporal window is recommended rather than a multi-year analysis:
- A single freeze-up or break-up season (approximately 3–4 months) provides enough temporal variability to demonstrate the model while keeping data volumes manageable
- A full annual cycle would be a stretch goal
- Multi-year trend analysis is out of scope for the showcase but could be framed as a future direction

---

## Project Phases

### Phase 1: Data Ingestion and Platform Setup
- Define study area and temporal scope
- Set up STAC collections in Prescient
- Build ingestion pipelines for each data source (format conversion, metadata creation)
- Validate data accessibility through TiTiler and MapLibre

### Phase 2: Training Data Preparation
- Rasterize USNIC ice charts to 500m grid
- Implement pure-cell label extraction
- Implement temporal alignment filtering (24-hour baseline)
- Match SAR acquisitions with labels and ancillary data
- Assemble and quality-check the training dataset

### Phase 3: Model Development
- Generate Clay embeddings from Sentinel-1 scenes and ingest into Prescient
- Train Phase 1 downstream model (Random Forest) on embedding features with AMSR2/ERA5 ancillary inputs
- Evaluate performance against AMSR2 baseline and USNIC chart-scale validation
- Iterate on label strategy (add mixed cells, ICESat-2 anchors) and downstream model choice
- If accuracy gap warrants it, proceed to Phase 2: end-to-end Clay fine-tuning on GPU
- Use cloud-free HLS imagery as independent validation where available

### Phase 4: Pipeline Integration
- Migrate inference to AWS
- Build the end-to-end pull → infer → ingest pipeline
- Ingest model outputs as a new STAC collection in Prescient
- Set up MapLibre visualization with multi-layer display

### Phase 5: Demo and Showcase
- Prepare demo environment showing the full data-to-product workflow
- Demonstrate multi-source data browsing in Prescient
- Show SIC output overlaid with source data and validation layers
- Document results and methodology

---

## Open Questions and Risks

| Item | Status | Notes |
|---|---|---|
| USNIC/CIS data overlap | **Confirmed** | USNIC weekly Arctic charts incorporate CIS analysis for Canadian waters. USNIC alone provides complete Canadian Arctic coverage; CIS need not be ingested separately as a label source for overlapping regions. |
| USNIC source imagery metadata | To investigate | Determine if charts include metadata about which SAR scenes were used per polygon |
| Clay input format for frozen extraction | To confirm | Confirm Clay's expected input patch size, normalization, and how AMSR2/ERA5 ancillary features are appended to the embedding vector |
| Phase 1 accuracy threshold | To define | Establish what accuracy gap (e.g. vs AMSR2 baseline) would trigger progression to Phase 2 fine-tuning |
| TESSERA coverage alignment | To evaluate | Check if pre-computed embeddings cover the study area and timeframe |
| AlphaEarth access model | Confirmed | Pre-computed embeddings only via GEE or Source Cooperative; model weights not available. Annual temporal resolution rules it out as a model input. Candidate showcase only. |
| RCM data access and licensing | To confirm | Determine availability and any restrictions for Radarsat Constellation Mission data |
| Prescient vector capabilities | To confirm | Validate PMTiles serving and any limitations for ice chart and AIS vector data |
| Prescient Zarr support | To confirm | Determine if Zarr is supported for ERA5 or other gridded datasets |

---

## Future Extensions

**Arctic Maritime Domain Awareness**: Extend the platform to include AIS vessel tracking data and SAR-based vessel detection, overlaid with SIC output to assess navigability and detect dark (non-broadcasting) vessels. This reuses much of the SAR and ice infrastructure built for the SIC project.

**Permafrost and Coastal Change Detection**: Use optical and SAR time series to map thermokarst development and coastal erosion along northern communities, leveraging Prescient's temporal query capabilities.

**Operational Near-Real-Time Delivery**: Optimize the inference pipeline for sub-6-hour latency from SAR acquisition to published SIC map, targeting operational use for Arctic navigation.
