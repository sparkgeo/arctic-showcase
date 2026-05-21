# Prescient Ice: Pipeline Architecture

## Overview

The Prescient Ice pipeline moves data from satellite acquisition through processing, model training, inference, and delivery in six stages. Prescient — the project's cloud-native STAC data management platform — sits at the centre of most stages, acting as the shared data layer between source data, derived products, and visualisation.

One important exception sits outside Prescient: the primary training dataset, AI4Arctic, is consumed directly by the training pipeline rather than being ingested into Prescient as a STAC collection. AI4Arctic ships as a pre-curated, scene-co-registered NetCDF dataset bundling Sentinel-1 EW SAR, AMSR2, ERA5, and CIS/DMI chart labels per scene — re-ingesting it into Prescient as separate per-source collections would discard the alignment that makes it useful and provides no analytical benefit. The Prescient-managed pipeline begins at the 2025–26 Hudson Bay prospective evaluation data, where Sentinel-1, AMSR2, ERA5, and USNIC charts are acquired independently from their providers and ingested as separate STAC collections to be matched up by the project pipeline.

The pipeline is structured to make the Phase 1 modeling approach — Clay v1.5 as a frozen feature extractor with a separate downstream classifier — explicit as an architectural pattern. Embedding generation and embedding ingestion are distinct stages, not implementation details. This separation is what enables rapid downstream classifier iteration without re-running the encoder on every training run.

```
STAGE 1: DATA ACQUISITION
  AI4Arctic NetCDF      ──┐ (direct to training pipeline; bypasses Prescient)
                          │
  Sentinel-1 EW GRD     ──┐
  ERA5 Single Levels    ──┤
  AMSR2 AU_SI12         ──┤──→  STAGE 2: INGESTION  ──→  PRESCIENT
  USNIC SIGRID-3        ──┤      (NERSC noise correction       (STAC catalog)
  ICESat-2 ATL07/10     ──┤       applied to Sentinel-1)             │
  HLS L30/S30           ──┘                                           │
                                                                      ▼
                                                            STAGE 3: EMBEDDING
                                                            (SAR → Clay encoder
                                                             → 32×32 patch token
                                                               COGs per chip
                                                             → re-ingest to Prescient)
                                                                      │
                          ┌──────────────────────────────────────────┘
                          │   ┌── (AI4Arctic — direct training path)
                          ▼   ▼
                  STAGE 4: TRAINING
                  (patch tokens + labels
                   → downstream classifier)
                          │
                          ▼
                  STAGE 5: INFERENCE
                  (2025–26 SAR → embeddings
                   → 320m SIC class grid → re-ingest)
                          │
                          ▼
                  STAGE 6: VISUALIZATION
                  (TiTiler → MapLibre)
```

---

## Projection and CRS Strategy

All spatial data in the Prescient-managed pipeline follows a consistent CRS convention that separates analytical storage, STAC cataloguing, and visualisation concerns.

**Analytical CRS — EPSG:3978 (NAD83 / Canada Atlas Lambert).** All COG raster products are stored in EPSG:3978. All spatial operations — label rasterisation, area-weighted polygon averaging, spatial joins, grid alignment — are performed in EPSG:3978. This is an equal-area projection well-suited to the Hudson Bay study area, and equal-area properties are a correctness requirement for the area-weighted label rasterisation step. If the project scope expands to a pan-Arctic extent, EPSG:3995 (WGS 84 / Arctic Polar Stereographic) would be the appropriate replacement, aligning with NSIDC and AMSR2 native grid conventions.

**STAC bounding boxes — EPSG:4326 (WGS84).** Per the STAC specification, all item and collection bounding boxes are expressed in WGS84 decimal degrees. This is projection-agnostic and independent of the analytical CRS.

**Serving CRS — EPSG:3857 (Web Mercator).** MapLibre expects tile data in Web Mercator. TiTiler reprojects raster tiles from the native EPSG:3978 COG to Web Mercator on the fly at serve time; no raster data is stored in Web Mercator. PMTiles vector tiles are pre-generated in Web Mercator at ingest time using `tippecanoe` (which accepts WGS84 GeoJSON input and handles the projection internally).

**Analytical operations must never use Web Mercator or WGS84 degrees.** Area calculations in geographic coordinates are distorted at high latitudes. Any operation that involves polygon areas, cell areas, or distance-based spatial joins must be performed after reprojecting to EPSG:3978.

AI4Arctic scenes are delivered in their own native scene-projected coordinate systems and are not reprojected into EPSG:3978 for training; reprojection at this stage would introduce resampling artefacts that the AI4Arctic authors specifically avoided by keeping each scene in its native projection. The training pipeline operates on AI4Arctic data in its native form. EPSG:3978 applies to the Prescient-managed 2025–26 data and to model outputs.

---

## Stage 1: Data Acquisition

Each data source is pulled from its upstream provider on an as-needed basis. For the initial build-out, acquisition is batch: a defined study area and temporal window are set, and all available data within those bounds is pulled. Acquisition scripts produce raw files in their native provider formats.

**AI4Arctic Sea Ice Challenge Dataset** — downloaded from DTU Data (DOI `10.11583/DTU.c.6244065.v2`) or accessed via TorchGeo on Hugging Face. 533 NetCDF files (513 training, 20 test) covering January 2018 – December 2021. Each file bundles Sentinel-1 EW HH/HV SAR, AMSR2, ERA5, and CIS/DMI ice chart labels per scene. The raw version is preferred over the RTT (ready-to-train) version; see `prescient_ice_datasets.md` for rationale. AI4Arctic does not flow through the Prescient ingestion stage — it is consumed directly by Stage 4 training.

**Sentinel-1 EW GRD** (2025–26 Hudson Bay) — pulled from the Copernicus Data Space Ecosystem (CDSE) via STAC API or the OData interface. Scenes are filtered by study area footprint intersection and acquisition date. HH/HV dual-polarisation GRD products are the target.

**ERA5 Single Levels** (2025–26) — downloaded from the Copernicus Climate Data Store (CDS) using the `cdsapi` Python client. Variables: 2m air temperature, 10m u/v wind components, mean sea level pressure. Downloaded as NetCDF or GRIB, regridded to the study area.

**AMSR2 AU_SI12** (2025–26) — pulled from JAXA's G-Portal. Daily 12.5km SIC composites in HDF5 format.

**USNIC weekly Arctic SIGRID-3** (2025–26) — downloaded from the USNIC Arctic archive. Vector data (polygons) in ESRI Shapefile or GeoJSON format, with concentration attributes per polygon.

**ICESat-2 ATL07/ATL10** — pulled from the NSIDC DAAC via `icepyx` or the Earthdata STAC API. Granule selection filtered to the study area and temporal window. Used as a visualisation overlay layer and as a candidate supplementary label source for 2025–26 retraining.

**HLS L30/S30** — pulled from NASA Earthdata (LP DAAC) via STAC API. Used for validation and visual context only; cloud filtering should be applied at acquisition time.

**Infrastructure**: Acquisition scripts run locally or on a lightweight compute instance. AWS Lambda is appropriate for triggered or scheduled acquisition once the pipeline is operational. All downloaded files are staged to S3 before ingestion.

---

## Stage 2: Ingestion

Ingestion converts heterogeneous source files into Prescient-compatible formats and registers them as STAC collections. The output of this stage is a fully populated STAC catalog with all 2025–26 source data, ICESat-2 tracks, and HLS imagery queryable by spatial footprint, temporal range, and collection. AI4Arctic does not pass through this stage.

### Sentinel-1 Preprocessing — NERSC Noise Correction

Sentinel-1 EW HH/HV scenes acquired for the 2025–26 pipeline must have the NERSC additional noise correction applied during ingestion, before COG conversion. AI4Arctic provides NERSC-corrected data as a packaged option; for 2025–26 scenes acquired directly from CDSE, the project pipeline must apply the same correction to maintain input distribution consistency between training and inference. The HV channel is where this matters most because residual noise is closest to typical ice/water backscatter levels there, and HV is the channel most informative for ice/water discrimination. See `prescient_ice_training_strategy.md` and `prescient_ice_datasets.md` for further context on the consistency requirement.

The NERSC correction is implemented as a pre-COG-conversion step in the Sentinel-1 ingestion workflow: ESA-corrected GRD pixels are read in, the NERSC algorithm is applied, and the corrected output is written to the EPSG:3978 COG.

### Dual-Asset Pattern for Vector Data

Vector datasets (USNIC ice charts, ICESat-2 tracks) are stored with two assets on each STAC item, separating analytical and visualisation concerns:

- **`data` asset** — GeoParquet format, EPSG:3978 (analytical CRS). This is the asset consumed by the project pipeline for label rasterisation, area-weighted polygon averaging, and spatial joins. All area and distance calculations are performed against this asset.
- **`visual` asset** — PMTiles format, Web Mercator (EPSG:3857). This is the asset served to MapLibre for display. Generated by `tippecanoe` from WGS84 GeoJSON at ingest time.

Both assets are registered on the same STAC item with the same spatiotemporal metadata. This pattern avoids duplicating catalog structure while making the appropriate asset for each use case unambiguous. It should be applied consistently across all vector collections.

### Format Conversions

| Source | Input Format | `data` Asset | `visual` Asset | Notes |
|---|---|---|---|---|
| Sentinel-1 | SAFE / GeoTIFF | COG (EPSG:3978) | — | Apply NERSC noise correction before COG conversion; ensure radiometric calibration |
| ERA5 | NetCDF / GRIB | COG (EPSG:3978) | — | Regrid to study area; one COG per variable per timestep |
| AMSR2 | HDF5 | COG (EPSG:3978) | — | Reproject from native polar stereographic grid |
| USNIC | Shapefile / GeoJSON | GeoParquet (EPSG:3978) | PMTiles (EPSG:3857) | Preserve CT codes and other concentration attributes in both assets |
| ICESat-2 | HDF5 | GeoParquet (EPSG:3978) | PMTiles (EPSG:3857) | Convert transect points/lines to GeoJSON as intermediate step |
| HLS | COG (already) | COG (EPSG:3978) | — | Reproject/clip to study area if needed |

### Ingestion Workflow

Each source has its own ingestion workflow:

1. **Convert** — run format conversion. For rasters: GDAL to produce EPSG:3978 COGs, with Sentinel-1 receiving NERSC noise correction prior to conversion. For vectors: reproject source to EPSG:3978 and write GeoParquet (`data` asset); convert to WGS84 GeoJSON and run `tippecanoe` for the PMTiles `visual` asset.
2. **Validate** — verify output geometry, CRS, nodata values, and COG/PMTiles/GeoParquet compliance.
3. **Create STAC item** — generate a STAC item JSON with spatial and temporal metadata, asset hrefs pointing to S3 for both assets (where applicable), and any source-specific properties (e.g., Sentinel-1 polarisation and noise correction applied, USNIC chart validity date). Bounding box in WGS84.
4. **Register** — POST the STAC item to Prescient's STAC Transaction API (or insert directly into PGStac if bulk loading).
5. **Upload** — copy converted assets to the S3 bucket backing the Prescient catalog.

**Infrastructure**: Lambda handles lightweight conversions (ERA5, HLS, STAC item creation). Batch handles heavy conversions (Sentinel-1 GRD processing with NERSC noise correction, USNIC PMTiles generation via `tippecanoe` for large Arctic-wide charts). Step Functions orchestrates each per-source workflow with retry logic and pipeline state visibility.

For initial development, all of this runs locally as Python scripts. The GDAL, `tippecanoe`, `geopandas`, `pystac`, and NERSC noise correction tooling all run without AWS dependencies; migrating to Lambda/Batch is straightforward once the conversion logic is stable.

---

## Stage 3: Embedding Generation and Ingestion

This stage is distinct from both ingestion and training. Its purpose is to run the Clay v1.5 encoder over Sentinel-1 scenes and persist the resulting patch token grids back into Prescient as a derived STAC collection, so that downstream training runs and inference can retrieve pre-computed embeddings without re-running the encoder.

For Phase 1 training, this stage operates on AI4Arctic scenes directly (via the training pipeline's data loader, with embeddings cached but not necessarily ingested into Prescient). For 2025–26 prospective evaluation and operational inference, the stage operates on Prescient-managed Sentinel-1 COGs and the resulting patch token grids are re-ingested into Prescient as a derived STAC collection.

### Embedding Pipeline

1. **Query** — retrieve Sentinel-1 EW HH/HV scenes from Prescient (2025–26 pipeline) or from the AI4Arctic data loader (training pipeline).
2. **Tile** — divide each scene into 256×256-pixel chips at Clay's expected input size. At EW ~40m GSD, each chip covers approximately 10.2 km × 10.2 km.
3. **Encode** — pass each chip through Clay v1.5's frozen encoder (no gradient computation), using the custom `sentinel-1-ew` metadata entry with HH/HV band names, ~40m GSD, and NERSC-derived normalisation statistics (see `prescient_ice_model_architecture.md`). The encoder produces a `[batch, 1025, 1024]` tensor per batch: the first sequence element is the class token (chip embedding); the remaining 1024 elements are patch tokens, reshaped into a 32×32 grid where each token is spatially registered to a ~320m × 320m footprint.
4. **Serialise** — write the patch token grid as a COG per chip with 32 × 32 spatial dimensions and 1024 bands (one band per embedding dimension). The class token (chip embedding) is stored separately per chip — either as a sidecar STAC asset or as an additional band/property on the patch token item, to be decided at implementation time. The COG preserves spatial registration of each patch token to its 320m footprint in EPSG:3978. Alternative serialisation (e.g., Zarr or a custom layout) may be considered if the 1024-band COG approach proves operationally awkward; the architectural commitment is to the 32×32 spatial grid, not to a specific storage format.
5. **Re-ingest** (2025–26 pipeline) — create a STAC item for each embedding COG and register it in Prescient under the `clay-embeddings` collection. STAC item metadata references the source SAR scene and the Clay model version used.

The re-ingestion step completes the Prescient round-trip: embeddings are a derived analytical product managed through the same STAC interface as source data, discoverable by the spatial and temporal bounds of the source SAR scene.

**Infrastructure**: Clay inference requires GPU compute for practical throughput (CPU-only is feasible for small volumes but slow). AWS Batch with a GPU-enabled instance (g5 family) is appropriate. The batch job pulls SAR data (from Prescient via STAC API for 2025–26 scenes, or directly from the AI4Arctic data loader for training), runs inference, writes patch token COGs to S3, and (for 2025–26) registers them via the STAC Transaction API. SageMaker batch transform is an alternative if Clay is deployed as a SageMaker model.

---

## Stage 4: Training

The training stage assembles (patch token, label) pairs and trains the downstream classifier. The primary training source is AI4Arctic; the 2025–26 Hudson Bay data is reserved for prospective evaluation and potential retraining.

### Training Dataset Assembly (AI4Arctic Path)

1. **Scene iteration** — iterate over AI4Arctic training scenes via the TorchGeo data loader or direct NetCDF reads.
2. **Chip extraction** — divide each scene into 256×256 chips aligned to Clay's input size. Extract HH/HV pixels for the encoder, and co-registered AMSR2 SIC and ERA5 surface variables for ancillary features.
3. **Embedding** — pass each chip through Clay v1.5 to produce the 32×32 patch token grid (see Stage 3).
4. **Label rasterisation** — rasterise the AI4Arctic CIS/DMI chart polygons onto the 320m patch grid in the scene's native projection. Phase 1: extract pure cells (fully within a single polygon). Phase 2: include mixed cells with area-weighted class labels using the midpoint-rounding approach (see `prescient_ice_training_strategy.md`).
5. **Patch-to-label spatial join** — each patch token is spatially joined to the rasterised class label at its 320m footprint. Joining is per-token rather than per-chip, producing approximately 1,024 (token, label) pairs per chip.
6. **Ancillary feature attachment** — append AMSR2 SIC value and ERA5 variables (sampled at the patch centroid) to each patch token vector.
7. **Dataset assembly** — write the assembled (feature_vector, class_label) pairs as a flat training dataset (numpy arrays or a columnar format) to S3.

### Training Dataset Assembly (2025–26 Retraining Path)

The same logic applies if the 2025–26 Hudson Bay data is used for retraining, with two differences. First, Sentinel-1, AMSR2, ERA5, and USNIC chart data are queried from Prescient rather than pulled from a co-registered NetCDF — temporal alignment must be performed during pair assembly (24-hour baseline window, ERA5-adaptive tightening; see `prescient_ice_training_strategy.md`). Second, ICESat-2 anchor points become a candidate supplementary label source: where coincident tracks are available within a 2–4 hour window of the SAR acquisition, retrieve the GeoParquet `data` asset for the track item and extract anchor point labels (lead detections → class 0, consolidated freeboard → class 10).

### Downstream Classifier Training

The assembled dataset is used to train the downstream SIC classifier. Phase 1 candidates: Random Forest and XGBoost, evaluated in parallel across three feature configurations (raw HH/HV backscatter baseline; patch tokens alone; patch tokens + chip embedding). See `prescient_ice_model_architecture.md` for the full feature configuration and evaluation framework.

The training loop is lightweight — no GPU required — and can run locally or on a standard CPU instance. Training inputs are the assembled feature vectors; training targets are the per-patch class labels (eleven classes, 0–10).

Model artefacts (trained model, feature importance outputs, validation metrics including R² on the 0–10 class scale and the ordinal penalty metric) are stored in S3. If Phase 1 accuracy is insufficient or fails to outperform the non-embedding baseline, Phase 2 initiates end-to-end Clay fine-tuning on GPU — see `prescient_ice_model_architecture.md` for detail on the Phase 2 architecture and infrastructure.

**Infrastructure**: Lambda or a lightweight EC2/Batch job for dataset assembly (primarily I/O-bound, pulling from AI4Arctic or Prescient). Local or CPU-only Batch for downstream classifier training. GPU Batch or SageMaker Training Job for Phase 2 fine-tuning if triggered.

---

## Stage 5: Inference

The inference pipeline applies the trained classifier to new Sentinel-1 scenes and delivers 320m SIC class grids back into Prescient as a derived product. Inference is the primary use of the 2025–26 Hudson Bay data: trained on AI4Arctic, the model is run on 2025–26 Sentinel-1 scenes to produce SIC outputs, with prospective evaluation comparing those outputs to USNIC charts for the same period.

### Inference Workflow

1. **Trigger** — a new Sentinel-1 scene over the study area is ingested into Prescient with NERSC noise correction applied (manual trigger for the showcase; EventBridge rule monitoring the STAC catalog for an operational deployment).
2. **Input assembly** — pull the SAR COG from Prescient. Retrieve the closest ERA5 and AMSR2 data within appropriate time windows.
3. **Embedding** — run Clay v1.5 over the new scene (frozen weights, same procedure as Stage 3). Output is a 32×32 patch token grid per chip across the scene.
4. **Feature assembly** — for each patch token, construct the feature vector matching the trained configuration (raw baseline, patch tokens, or patch tokens + chip embedding) with AMSR2 and ERA5 ancillary features appended.
5. **Prediction** — apply the trained downstream classifier to each patch feature vector to produce per-patch class predictions (and per-class probabilities, if useful for the visualisation layer). For each chip, the result is a 32×32 prediction grid.
6. **Rasterisation** — assemble per-chip 32×32 prediction grids into a scene-wide 320m SIC class COG in EPSG:3978. The output COG carries integer class values 0–10 with appropriate nodata handling.
7. **Post-processing** — mask land areas, apply any QA flags. Write the final COG with nodata values and overviews. Optionally write a parallel COG of per-class probabilities (multi-band, one band per class) for downstream uncertainty visualisation.
8. **Re-ingestion** — create a STAC item for the SIC output, referencing the source SAR scene, NERSC noise correction status, Clay model version, and downstream classifier version. Register under the `sic-output` collection in Prescient. Bounding box in WGS84.

**Infrastructure**: AWS Batch (GPU instance) for the Clay encoding step. Lambda or CPU Batch for the downstream classifier inference step (lightweight, fast). Step Functions orchestrates the trigger → encode → predict → ingest sequence with retry logic. The full cycle from SAR availability to SIC publication is expected to take on the order of minutes for a single scene once the pipeline is operational.

---

## Stage 6: Visualization

All data — source imagery, labels, embeddings, and derived products — is served through Prescient's TiTiler tiling server and displayed in a MapLibre-powered web viewer. TiTiler reprojects raster COGs from EPSG:3978 to Web Mercator on the fly for tile requests; PMTiles vector layers are served directly without reprojection.

The visualization interface displays:

- **SIC output** (primary) — the model-predicted 320m SIC class grid, styled with a discrete eleven-step colour ramp (e.g. blue for class 0 / open water through white for class 10 / full ice cover). Optional per-class probability layers for uncertainty visualisation.
- **Source SAR** — the Sentinel-1 backscatter imagery underlying the prediction, enabling analysts to cross-check the model's output against the raw input signal.
- **USNIC ice charts** — the polygon-based operational charts, served from the PMTiles `visual` asset, providing a direct comparison product for prospective evaluation.
- **HLS optical** — cloud-free optical context imagery for seasons and regions where it is available.
- **ICESat-2 tracks** — track transects served from the PMTiles `visual` asset, overlaid as an independent physical validation reference. Lead detections and significant freeboard measurements provide point-level corroboration of model predictions at the extremes of the class spectrum.

Layer toggling, opacity control, and temporal navigation (stepping through dates) are standard MapLibre capabilities that should be surfaced in the viewer.

**Infrastructure**: TiTiler serves tiles from S3-backed COGs on demand. The MapLibre viewer is a static web application hosted on S3 or CloudFront, with no server-side rendering required.

---

## STAC Collections

| Collection | Format | CRS | Description |
|---|---|---|---|
| `sentinel-1-sar` | COG | EPSG:3978 | Sentinel-1 EW GRD scenes over study area, NERSC noise correction applied |
| `usnic-ice-charts` | GeoParquet + PMTiles | EPSG:3978 / EPSG:3857 | USNIC weekly Arctic ice concentration polygons (dual asset) |
| `era5-ancillary` | COG | EPSG:3978 | ERA5 surface variables (temperature, wind, pressure) |
| `amsr2-sic` | COG | EPSG:3978 | AMSR2 passive microwave SIC daily composites |
| `icesat2-tracks` | GeoParquet + PMTiles | EPSG:3978 / EPSG:3857 | ICESat-2 freeboard and lead detection transects (dual asset) |
| `hls-optical` | COG | EPSG:3978 | Harmonized Landsat Sentinel-2 optical imagery |
| `clay-embeddings` | COG (32×32 × 1024 bands) | EPSG:3978 | Clay v1.5 patch token grids derived from Sentinel-1 scenes; class token chip embedding stored alongside |
| `sic-output` | COG | EPSG:3978 | Model-predicted 320m SIC class grids (derived product) |

All STAC item bounding boxes are expressed in WGS84 (EPSG:4326) per the STAC specification, regardless of the native asset CRS.

AI4Arctic is not represented as a Prescient STAC collection. It is consumed by the training pipeline directly via its native NetCDF distribution and the TorchGeo data loader. The rationale is detailed in `prescient_ice_datasets.md`.

---

## AWS Infrastructure Summary

| Component | Service | Notes |
|---|---|---|
| Asset storage | S3 | Buckets backing Prescient's STAC catalog |
| STAC catalog | PGStac on RDS | Managed by Prescient |
| Tile serving | TiTiler on ECS/Lambda | Reprojects EPSG:3978 COGs to Web Mercator on demand; serves PMTiles directly |
| Ingestion — lightweight | Lambda | ERA5, HLS, STAC item creation |
| Ingestion — heavy | Batch (CPU) | Sentinel-1 processing including NERSC noise correction; GeoParquet and PMTiles generation for vector sources |
| Embedding generation | Batch (GPU) | Clay v1.5 encoder inference; g5 instance family |
| Model training (Phase 1) | Batch (CPU) or local | Random Forest and XGBoost classifiers |
| Model training (Phase 2) | Batch (GPU) or SageMaker | End-to-end Clay fine-tuning, if triggered |
| Inference — encoding | Batch (GPU) | Clay v1.5 encoder on new scenes |
| Inference — prediction | Lambda or Batch (CPU) | Downstream classifier, fast |
| Orchestration | Step Functions | Per-stage state machines with retry logic |
| Monitoring | CloudWatch | Pipeline health and latency |
| Web viewer | S3 + CloudFront | Static MapLibre application |

For initial development, all pipeline stages except GPU-dependent steps can run locally. The recommended sequence is: build and validate locally → containerise → deploy to Lambda/Batch → wrap in Step Functions.
