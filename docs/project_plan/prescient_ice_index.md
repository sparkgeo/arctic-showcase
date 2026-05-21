# Prescient Ice: Project Index

## Project Overview

Prescient Ice is an automated Arctic sea ice concentration (SIC) mapping project producing 320m resolution gridded SIC class estimates over Hudson Bay from Sentinel-1 EW SAR imagery, using Clay v1.5 — a geospatial foundation model — as the feature extraction backbone. SIC is treated as an eleven-class classification problem on the 0–10 tenths scheme (0 = open water, 10 = full ice cover), matching the SIGRID-3 chart scheme directly. The 320m grid resolution is defined by Clay's patch footprint at Sentinel-1 EW's ~40m native ground sampling distance, representing the honest effective resolution of the model rather than a chosen round-number target. The project serves as an internal showcase for Prescient, demonstrating end-to-end data ingestion, analytics integration, and derived product delivery across a realistic multi-source geospatial pipeline.

For background, motivation, and a full project summary, see [`prescient_ice_overview.md`](prescient_ice_overview.md).

### Modeling Approach Summary

The modeling strategy is two-phased. In Phase 1, Clay v1.5 is used as a frozen feature extractor: Sentinel-1 EW HH/HV scenes are tiled into 256×256 chips and passed through Clay's encoder to extract 32×32 patch token grids per chip (each token covering ~320m × 320m on the ground at EW ~40m GSD) plus a class token serving as the chip embedding. Downstream classifiers — Random Forest and XGBoost evaluated in parallel — are trained on patch token vectors with AMSR2 and ERA5 ancillary features appended. Three feature configurations are evaluated to isolate the value Clay adds over raw features and the value of chip-level spatial context over patch-level features alone: a raw HH/HV backscatter baseline; patch tokens alone; and patch tokens combined with the class token chip embedding. Phase 2 — end-to-end Clay fine-tuning with a classification head — is pursued only if Phase 1 R² falls below a threshold T (to be calibrated against the AutoICE results distribution) or if Phase 1 with Clay embeddings does not meaningfully outperform the raw backscatter baseline. See [Model Architecture](#model-architecture) below for full detail.

---

## Datasets

The project draws on fourteen datasets spanning SAR imagery, passive microwave, lidar altimetry, climate reanalysis, optical imagery, foundation model embeddings, and vessel tracking — including the AI4Arctic Sea Ice Challenge Dataset as the primary training and evaluation source. For full per-dataset detail — product selection, access paths, caveats, and suitability classification — see [`prescient_ice_datasets.md`](prescient_ice_datasets.md).

| Dataset | Category | Role | Prescient Collection |
|---|---|---|---|
| AI4Arctic Sea Ice Challenge Dataset | Model input | Primary training and evaluation dataset (bundled SAR + AMSR2 + ERA5 + CIS/DMI charts) | Not ingested — consumed directly |
| Sentinel-1 EW GRD | Model input | 2025–26 Hudson Bay imagery; NERSC noise correction applied at ingest | `sentinel-1-sar` |
| AMSR2 AU_SI12 | Model input | SIC prior feature; via AI4Arctic for training, via JAXA for 2025–26 | `amsr2-sic` |
| ERA5 Single Levels | Model input | Atmospheric feature; via AI4Arctic for training, via CDS for 2025–26; also drives temporal alignment filter | `era5-ancillary` |
| Clay v1.5 patch tokens | Model input | SAR patch token grids (self-computed, 32×32 × 1024 per chip; class token chip embedding stored alongside) | `clay-embeddings` |
| USNIC weekly SIGRID-3 | Supplemental | 2025–26 evaluation label source; Prescient showcase ingest | `usnic-ice-charts` |
| ICESat-2 ATL07/ATL10 | Supplemental | Visualisation validation overlay; candidate 2025–26 supplementary labels | `icesat2-tracks` |
| HLS L30/S30 | Supplemental | Optical validation (seasonal); visual context layer | `hls-optical` |
| PM SIC CDR G02202 | Supplemental | Long-term climate context; Prescient showcase | `pm-sic-cdr` |
| RCM ScanSAR | Candidate | Prescient showcase / inference transferability | TBD |
| TESSERA | Candidate | Pre-computed embeddings showcase | TBD |
| AlphaEarth AEF | Candidate | Pre-computed embeddings showcase | TBD |
| SWOT KaRIn L2 Raster | Candidate | Supplementary lead labels / showcase | TBD |
| AIS | Candidate | Shipping context layer | TBD |

---

## Training Strategy

The training strategy addresses two core challenges: converting coarse polygon labels (CIS/DMI charts via AI4Arctic for primary training; USNIC charts for 2025–26 prospective evaluation) into 320m patch-level training targets under an eleven-class classification framing, and ensuring temporal alignment between SAR acquisitions and chart products for the parts of the pipeline where matching is the project's responsibility. AI4Arctic ships pre-curated scene-to-label alignment, so temporal alignment is a concern only for the 2025–26 Hudson Bay pipeline.

Label preparation follows a phased approach. Phase 1 extracts only pure cells — patches that fall entirely within a single chart polygon — prioritising label quality over volume; the patch's CT code is assigned as its class label. Phase 2 expands coverage to mixed boundary cells using a midpoint-rounding scheme: class midpoints are used as proxy fractions for area weighting, with the resulting fraction rounded to the nearest tenth to produce a discrete class label; SIGRID-3 range codes are resolved to their midpoints before weighting. A known edge case is cells that are overwhelmingly open water with a small ice presence rounding to class 0, which is documented as expected behaviour. ICESat-2 anchor points (lead detections → class 0, consolidated freeboard → class 10), previously a Phase 3 supplementary label source for primary training, are now reserved for the 2025–26 retraining scenario if it becomes warranted, with their primary project role being independent physical validation in the visualisation layer. A more advanced weak/aggregate label training strategy is noted as a future consideration after baselines are established.

For the 2025–26 prospective evaluation pipeline, temporal alignment uses a 24-hour baseline window between SAR acquisition date and chart validity date, refined adaptively using ERA5 surface temperature and wind speed: stable winter conditions tolerate the full 24-hour window, while dynamic freeze-up and break-up periods tighten the requirement to 6–12 hours. NERSC noise correction must be applied consistently to 2025–26 Sentinel-1 scenes to match the NERSC-corrected AI4Arctic training data and avoid input distribution shift.

For full detail on label preparation phases, temporal alignment logic, the NERSC consistency requirement, and the revised ICESat-2 role, see [`prescient_ice_training_strategy.md`](prescient_ice_training_strategy.md).

---

## Model Architecture

The architecture is built around Clay v1.5, following a two-phase strategy. Phase 1 (primary) uses Clay as a frozen feature extractor: Sentinel-1 EW HH/HV chips are passed through Clay's encoder, which produces a `[batch, 1025, 1024]` tensor — the first sequence element is the class token (chip embedding), the remaining 1024 are patch tokens reshaped into a 32×32 spatial grid. Each patch token is spatially registered to its ~320m footprint and treated as an individual feature vector, giving the downstream classifier approximately 1,024 training samples per chip rather than one. A custom `sentinel-1-ew` Clay metadata entry handles the mismatch between Clay's built-in IW VV/VH calibration and the project's EW HH/HV inputs, specifying the correct ~40m GSD and NERSC-derived normalisation statistics. Downstream classifiers — Random Forest and XGBoost evaluated in parallel — are trained on three parallel feature configurations: a raw HH/HV backscatter baseline; patch tokens alone (1024-dim); and patch tokens concatenated with the class token chip embedding (2,048-dim feature vector). AMSR2 and ERA5 ancillary features are appended in all configurations. Primary evaluation metric is R² on the 0–10 class scale (matching AutoICE), with an ordinal penalty metric as secondary.

Phase 2 (conditional) adds an end-to-end Clay fine-tuning step with an eleven-class classification head. It is triggered if either Phase 1 R² falls below a threshold T (to be calibrated against the AutoICE results distribution before Phase 1 evaluation) or Phase 1 with Clay embeddings does not meaningfully outperform the raw backscatter baseline. Fine-tuning requires GPU compute on AWS.

For full detail on the Clay v1.5 input specification, the EW metadata handling, patch token extraction mechanics, feature configurations, downstream classifier choices, the two-part Phase 2 trigger, and Phase 2 compute requirements, see [`prescient_ice_model_architecture.md`](prescient_ice_model_architecture.md).

---

## Pipeline Architecture

The pipeline runs in six stages. Prescient is the shared data layer for the 2025–26 Hudson Bay prospective evaluation and inference pipeline; the primary training pipeline consumes AI4Arctic directly, since AI4Arctic ships as a pre-curated scene-co-registered NetCDF dataset whose scene-to-label alignment would be lost if re-ingested as separate per-source STAC collections.

Stage 1 acquires source data: AI4Arctic via DTU Data or TorchGeo for training; Sentinel-1 from CDSE, AMSR2 from JAXA, ERA5 from CDS, USNIC charts from the USNIC archive, ICESat-2 from NSIDC, and HLS from NASA Earthdata for the 2025–26 pipeline. Stage 2 ingests the 2025–26 source files into Prescient, converting rasters to EPSG:3978 COGs and vectors to a dual-asset pattern (GeoParquet in EPSG:3978 for analytical use, PMTiles in Web Mercator for visualisation); the NERSC noise correction is applied to Sentinel-1 EW scenes during ingestion. Stage 3 runs Clay v1.5's frozen encoder over Sentinel-1 chips to produce 32×32 patch token grids per chip (1024-dim per token) plus a class token chip embedding, serialised as COGs with 32×32 spatial × 1024 bands; for the 2025–26 pipeline, these are re-ingested into Prescient as a derived collection. Stage 4 trains the downstream classifier: the AI4Arctic path is primary, with the 2025–26 path available for retraining if needed. Stage 5 applies the trained classifier to new 2025–26 Sentinel-1 scenes, producing 320m SIC class COGs that are re-ingested into Prescient. Stage 6 serves all layers through TiTiler and a MapLibre web viewer. AWS infrastructure spans Lambda and Batch for compute, Step Functions for orchestration, and S3 for all asset storage backing Prescient.

For full stage-by-stage detail — format conversions, NERSC preprocessing, ingestion workflows, projection and CRS strategy, dual-asset vector pattern, patch token serialisation and re-ingestion, AI4Arctic and 2025–26 training paths, inference workflow, visualisation layers, STAC collection definitions, and infrastructure mapping by stage — see [`prescient_ice_pipeline_architecture.md`](prescient_ice_pipeline_architecture.md).

---

## Study Area and Temporal Scope

The study area is Hudson Bay (main body), bounded approximately by 95°W–75°W, 58°N–66°N, with an analytical CRS of EPSG:3978 (NAD83 / Canada Atlas Lambert) and a 320m output grid defined by the Clay v1.5 patch footprint at Sentinel-1 EW ~40m GSD. The 2025–26 Hudson Bay window (October 2025 – January 2026, a single freeze-up season) serves as a prospective evaluation dataset, testing model generalisation from AI4Arctic 2018–2021 training data to recent unseen Hudson Bay conditions. Hudson Bay was selected for its complete annual ice cycle, confirmed Sentinel-1 EW coverage, strong CIS-derived label availability through both AI4Arctic and USNIC, and direct operational relevance to Arctic navigation and northern communities. The exact bounding box may be refined slightly once the patch-aligned 320m grid is defined in projected coordinates.

For full detail on study area rationale, Sentinel-1 constellation status across the prospective evaluation window, evaluation pair volume estimates, the AI4Arctic native-projection treatment, and CRS and projection decisions, see [`prescient_ice_study_area.md`](prescient_ice_study_area.md).

---

## Project Phases

### Phase 1: Prescient Platform Setup and Showcase Ingestion
- Set up STAC collections in Prescient for showcase datasets (PM SIC CDR, HLS, USNIC historical)
- Build ingestion pipelines including format conversion, metadata creation, and the dual-asset vector pattern
- Validate data accessibility through TiTiler and MapLibre

### Phase 2: AI4Arctic Training Pipeline
- Acquire AI4Arctic dataset (raw version preferred)
- Implement Clay v1.5 frozen-encoder embedding pipeline with the custom `sentinel-1-ew` metadata entry
- Derive NERSC-based normalisation statistics from AI4Arctic NERSC-corrected scenes
- Implement label preparation: pure-cell extraction (Phase 1); area-weighted mixed cells with midpoint rounding (Phase 2)
- Train RF and XGBoost classifiers across the three feature configurations (raw backscatter baseline, patch tokens, patch tokens + chip embedding)
- Evaluate on held-out AI4Arctic test scenes and a Hudson Bay subset; primary metric R² on 0–10 class scale, secondary ordinal penalty metric
- If Phase 1 accuracy threshold or non-embedding baseline triggers are hit, proceed to Phase 2 Clay fine-tuning on GPU

### Phase 3: 2025–26 Hudson Bay Ingestion and Prospective Evaluation
- Acquire 2025–26 Sentinel-1 EW, AMSR2, ERA5, USNIC, ICESat-2, and HLS data for the study area
- Apply NERSC noise correction to Sentinel-1 EW scenes during ingestion
- Ingest into Prescient as STAC collections (rasters as COGs in EPSG:3978; vectors with the dual-asset pattern)
- Run Clay embedding pipeline over 2025–26 Sentinel-1 scenes; re-ingest patch token COGs into Prescient
- Apply trained classifier to produce 320m SIC class COGs; re-ingest into Prescient
- Conduct prospective evaluation against USNIC charts using the temporal alignment framework
- If prospective evaluation warrants retraining, incorporate 2025–26 data (and ICESat-2 anchor points as supplementary labels)

### Phase 4: Pipeline Integration on AWS
- Migrate Stage 2 ingestion, Stage 3 embedding, and Stage 5 inference to AWS Lambda, Batch (CPU and GPU), and Step Functions
- Set up MapLibre visualization with multi-layer display, including SIC class output styling, source SAR, USNIC charts, ICESat-2 tracks, and HLS optical
- Validate end-to-end latency from SAR acquisition to published SIC class output

### Phase 5: Demo and Showcase
- Prepare demo environment showing the full data-to-product workflow
- Demonstrate multi-source data browsing in Prescient
- Show SIC class output overlaid with source data, USNIC chart comparison, and validation layers
- Document results, methodology, and Phase 1 vs Phase 2 outcome

---

## Open Questions and Risks

| Item | Status | Notes |
|---|---|---|
| USNIC/CIS data overlap | **Confirmed** | USNIC weekly Arctic charts incorporate CIS analysis for Canadian waters. USNIC alone provides complete Canadian Arctic coverage; CIS need not be ingested separately as a label source for overlapping regions. |
| Study area bounds | **Direction confirmed** | Hudson Bay, 95°W–75°W, 58°N–66°N. Exact bounds subject to minor refinement once the 320m patch-aligned grid is defined in EPSG:3978. Verify Sentinel-1 EW scene availability against CDSE catalog before ingestion. |
| Clay v1.5 input format for frozen extraction | **Resolved** | 256×256 chips, 8×8 patches, encoder output `[batch, 1025, 1024]` — class token + 32×32 patch token grid. Custom `sentinel-1-ew` metadata entry handles HH/HV bands and ~40m GSD; normalisation from NERSC-corrected AI4Arctic scenes. AMSR2/ERA5 ancillary features appended to patch token vectors after encoding. |
| Clay v1.5 patch token API | To confirm | Confirm the specific Clay v1.5 codebase API for accessing the full `[batch, 1025, 1024]` encoder output tensor and the canonical convention that the first sequence element is the class token. Empirical testing on Sentinel-2 inputs has confirmed the tensor shape and apparent ordering; final source check pending. Implementation detail rather than design uncertainty. |
| Phase 2 trigger threshold T | To define | Set after reviewing the full AutoICE results distribution (Stokholm et al. 2024) before Phase 1 evaluation begins. T calibrates the absolute-performance arm of the two-part Phase 2 trigger; the non-embedding baseline comparison is the second arm. |
| Feature configuration × classifier evaluation scope | Open | Whether to run all six combinations of three feature configurations and two classifiers (RF, XGBoost), or to narrow on one axis first. Depends on implementation difficulty of cleanly parameterising the training loop. |
| Hudson Bay subset of AI4Arctic | To investigate | Identify the Hudson Bay subset of AI4Arctic training scenes for domain-specific validation alongside the full pan-Arctic evaluation. |
| USNIC source imagery metadata | To investigate | Determine if 2025–26 charts include metadata identifying which SAR scenes were used per polygon — would enable stronger alignment than date-based matching for the 2025–26 evaluation pipeline. |
| TESSERA coverage alignment | To evaluate | Check if pre-computed embeddings cover the study area and timeframe. |
| AlphaEarth access model | **Confirmed** | Pre-computed embeddings only via GEE or Source Cooperative; model weights not available. Annual temporal resolution rules it out as a model input. Candidate showcase only. |
| RCM data access and licensing | To confirm | Determine availability and any restrictions for Radarsat Constellation Mission data. |
| Prescient vector capabilities | To confirm | Validate PMTiles serving and GeoParquet analytical asset support for ice chart and AIS vector data. |
| Prescient Zarr support | To confirm | Determine if Zarr is supported for ERA5 or other gridded datasets. |

---

## Future Extensions

**Arctic Maritime Domain Awareness**: Extend the platform to include AIS vessel tracking data and SAR-based vessel detection, overlaid with SIC output to assess navigability and detect dark (non-broadcasting) vessels. This reuses much of the SAR and ice infrastructure built for the SIC project.

**Permafrost and Coastal Change Detection**: Use optical and SAR time series to map thermokarst development and coastal erosion along northern communities, leveraging Prescient's temporal query capabilities.

**Operational Near-Real-Time Delivery**: Optimize the inference pipeline for sub-6-hour latency from SAR acquisition to published SIC map, targeting operational use for Arctic navigation.

**Pan-Arctic Expansion**: Extend the study area to a pan-Arctic scope using EPSG:3995 (WGS 84 / Arctic Polar Stereographic) as the analytical CRS, aligning with NSIDC and AMSR2 native grid conventions. Hudson Bay serves as the development and validation testbed for this extension.

**DGGS Integration**: Discrete Global Grid Systems (H3, rHEALPix) were evaluated as a potential showcase component and deferred due to integration complexity relative to value within the current scope. Could be revisited as a future extension if a clear analytical or visualisation use case emerges.
