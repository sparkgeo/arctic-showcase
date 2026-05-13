# Prescient Ice: Pipeline Architecture

## Overview

The Prescient Ice pipeline moves data from satellite acquisition through processing, model training, inference, and delivery in six stages. Prescient — the project's cloud-native STAC data management platform — sits at the centre, acting as the shared data layer between all stages. Every input dataset, intermediate analytical product, and model output is registered as a STAC collection in Prescient, making the full data lineage visible and all assets queryable through a common interface.

The pipeline is structured to make the Phase 1 modeling approach — Clay as a frozen feature extractor with a separate downstream model — explicit as an architectural pattern. Embedding generation and embedding ingestion are distinct stages, not implementation details. This separation is what enables rapid downstream model iteration without re-running the encoder on every training run.

```
STAGE 1: DATA ACQUISITION
  Sentinel-1 EW GRD  ──┐
  ERA5 Single Levels  ──┤
  AMSR2 AU_SI12       ──┤──→  STAGE 2: INGESTION  ──→  PRESCIENT
  USNIC SIGRID-3      ──┤                               (STAC catalog)
  ICESat-2 ATL07/10   ──┤                                    │
  HLS L30/S30         ──┘                                    │
                                                             ▼
                                                   STAGE 3: EMBEDDING
                                                   (SAR → Clay encoder
                                                    → COG embeddings
                                                    → re-ingest to Prescient)
                                                             │
                                                             ▼
                                                   STAGE 4: TRAINING
                                                   (embeddings + labels
                                                    → downstream model)
                                                             │
                                                             ▼
                                                   STAGE 5: INFERENCE
                                                   (new SAR → embeddings
                                                    → SIC grid → re-ingest)
                                                             │
                                                             ▼
                                                   STAGE 6: VISUALIZATION
                                                   (TiTiler → MapLibre)
```

---

## Stage 1: Data Acquisition

Each data source is pulled from its upstream provider on an as-needed basis. For the initial build-out, acquisition is batch: a defined study area and temporal window are set, and all available data within those bounds is pulled. Acquisition scripts produce raw files in their native provider formats.

**Sentinel-1 EW GRD** — pulled from the Copernicus Data Space Ecosystem (CDSE) via STAC API or the OData interface. Scenes are filtered by study area footprint intersection and acquisition date. HH/HV dual-polarisation GRD products are the target.

**ERA5 Single Levels** — downloaded from the Copernicus Climate Data Store (CDS) using the `cdsapi` Python client. Variables: 2m air temperature, 10m u/v wind components, mean sea level pressure. Downloaded as NetCDF or GRIB, regridded to the study area.

**AMSR2 AU_SI12** — pulled from JAXA's G-Portal. Daily 12.5km SIC composites in HDF5 format.

**USNIC weekly Arctic SIGRID-3** — downloaded from the USNIC Arctic archive. Vector data (polygons) in ESRI Shapefile or GeoJSON format, with concentration attributes per polygon.

**ICESat-2 ATL07/ATL10** — pulled from the NSIDC DAAC via `icepyx` or the Earthdata STAC API. Granule selection filtered to the study area and temporal window.

**HLS L30/S30** — pulled from NASA Earthdata (LP DAAC) via STAC API. Used for validation and visual context only; cloud filtering should be applied at acquisition time.

**Infrastructure**: Acquisition scripts run locally or on a lightweight compute instance. AWS Lambda is appropriate for triggered or scheduled acquisition once the pipeline is operational. All downloaded files are staged to S3 before ingestion.

---

## Stage 2: Ingestion

Ingestion converts heterogeneous source files into Prescient-compatible formats and registers them as STAC collections. The output of this stage is a fully populated STAC catalog with all source data queryable by spatial footprint, temporal range, and collection.

### Format Conversions

| Source | Input Format | Output Format | Notes |
|---|---|---|---|
| Sentinel-1 | SAFE / GeoTIFF | COG | Apply radiometric calibration if not pre-processed |
| ERA5 | NetCDF / GRIB | COG | Regrid to study area, one COG per variable per timestep |
| AMSR2 | HDF5 | COG | Reproject to study area CRS |
| USNIC | Shapefile / GeoJSON | PMTiles | `tippecanoe` for tile generation; preserve concentration attributes |
| ICESat-2 | HDF5 | PMTiles | Convert transect points/lines to GeoJSON, then tile |
| HLS | COG (already) | COG | Reproject/clip to study area if needed |

### Ingestion Workflow

Each source has its own ingestion workflow:

1. **Convert** — run format conversion (GDAL for rasters, `tippecanoe` for vectors).
2. **Validate** — verify output geometry, CRS, nodata values, and COG/PMTiles compliance.
3. **Create STAC item** — generate a STAC item JSON with spatial and temporal metadata, asset hrefs pointing to S3, and any source-specific properties (e.g., Sentinel-1 polarisation, USNIC chart validity date).
4. **Register** — POST the STAC item to Prescient's STAC Transaction API (or insert directly into PGStac if bulk loading).
5. **Upload** — copy converted assets to the S3 bucket backing the Prescient catalog.

**Infrastructure**: Lambda handles lightweight conversions (ERA5, HLS, STAC item creation). Batch handles heavy conversions (Sentinel-1 GRD processing, USNIC PMTiles generation via `tippecanoe` for large Arctic-wide charts). Step Functions orchestrates each per-source workflow with retry logic and pipeline state visibility.

For initial development, all of this runs locally as Python scripts. The GDAL, `tippecanoe`, and `pystac` tooling all run without AWS dependencies; migrating to Lambda/Batch is straightforward once the conversion logic is stable.

---

## Stage 3: Embedding Generation and Ingestion

This stage is distinct from both ingestion and training. Its purpose is to run the Clay encoder over Sentinel-1 scenes and persist the resulting embeddings back into Prescient as a derived STAC collection, so that downstream training runs can retrieve pre-computed embeddings without re-running the encoder.

### Embedding Pipeline

1. **Query** — retrieve Sentinel-1 COGs from Prescient for the study area and period.
2. **Tile** — divide each scene into patches at Clay's expected input size.
3. **Encode** — pass each patch through Clay's frozen encoder (no gradient computation). Append AMSR2 and ERA5 ancillary features to the embedding vector for each patch.
4. **Serialise** — write patch embeddings as a COG or structured array, one item per SAR scene, with spatial metadata preserving the patch grid geometry.
5. **Re-ingest** — create a STAC item for each embedding output and register it in Prescient under the `clay-embeddings` collection.

The re-ingestion step is what completes the Prescient round-trip: embeddings are a derived analytical product managed through the same STAC interface as source data, discoverable by the spatial and temporal bounds of the source SAR scene.

**Infrastructure**: Clay inference requires GPU compute for practical throughput (CPU-only is feasible for small volumes but slow). AWS Batch with a GPU-enabled instance (g5 family) is appropriate. The batch job pulls SAR COGs from S3 via Prescient's STAC API, runs inference, writes embedding COGs to S3, and registers them via the STAC Transaction API. SageMaker batch transform is an alternative if Clay is deployed as a SageMaker model.

---

## Stage 4: Training

The training stage assembles temporally aligned (embedding, label) pairs from Prescient and trains the downstream regression model.

### Training Dataset Assembly

1. **Pair assembly** — query Prescient for Clay embeddings and USNIC ice charts within the study area and period. For each embedding item, find chart items within the temporal alignment window (24-hour baseline, ERA5-adaptive tightening during dynamic periods — see [`prescient_ice_training_strategy.md`](prescient_ice_training_strategy.md)).
2. **Label rasterisation** — rasterise USNIC SIGRID-3 polygons onto the 500m target grid. Phase 1: extract pure cells (fully within a single polygon). Phase 2: include mixed cells with area-weighted labels.
3. **ICESat-2 augmentation** — where coincident ICESat-2 tracks are available within a 2–4 hour window of the SAR acquisition, extract anchor point labels from ATL07/ATL10 and add them to the training set.
4. **Dataset assembly** — join embedding vectors with corresponding cell labels. Write as a flat training dataset (numpy arrays or a columnar format) to S3.

### Downstream Model Training

The assembled dataset is used to train the downstream SIC regression model. Phase 1 target: Random Forest (or gradient boosting as an alternative). The training loop is lightweight — no GPU required — and can run locally or on a standard CPU instance. Training inputs are the assembled (embedding + ancillary) feature vectors; training targets are the rasterised SIC values per cell.

Model artefacts (trained model, feature importance outputs, validation metrics) are stored in S3. If Phase 1 accuracy is insufficient, Phase 2 initiates end-to-end Clay fine-tuning on GPU — see [`prescient_ice_model_architecture.md`](prescient_ice_model_architecture.md) for detail on the Phase 2 architecture and infrastructure.

**Infrastructure**: Lambda or a lightweight EC2/Batch job for dataset assembly (primarily I/O-bound, pulling from Prescient and S3). Local or CPU-only Batch for downstream model training. GPU Batch or SageMaker Training Job for Phase 2 fine-tuning if triggered.

---

## Stage 5: Inference

The inference pipeline applies the trained model to new SAR acquisitions and delivers SIC grids back into Prescient as a derived product.

### Inference Workflow

1. **Trigger** — a new Sentinel-1 scene over the study area is ingested into Prescient (manual trigger for the showcase; EventBridge rule monitoring the STAC catalog for an operational deployment).
2. **Input assembly** — pull the SAR COG from Prescient. Retrieve the closest ERA5 and AMSR2 data within appropriate time windows.
3. **Embedding** — run the Clay encoder over the new scene (frozen weights, same procedure as Stage 3). Append ancillary features.
4. **Prediction** — apply the trained downstream model to the patch embeddings to produce per-cell SIC predictions. Output is a 500m gridded SIC field covering the scene footprint, values in [0, 1].
5. **Post-processing** — mask land areas, clip values to [0, 1], apply any QA flags. Write as a COG with CRS, nodata values, and overviews.
6. **Re-ingestion** — create a STAC item for the SIC output, referencing the source SAR scene and model version. Register under the `sic-output` collection in Prescient.

**Infrastructure**: AWS Batch (GPU instance) for the Clay encoding step. Lambda or CPU Batch for the downstream model inference step (lightweight, fast). Step Functions orchestrates the trigger → encode → predict → ingest sequence with retry logic. The full cycle from SAR availability to SIC publication is expected to take on the order of minutes for a single scene once the pipeline is operational.

---

## Stage 6: Visualization

All data — source imagery, labels, embeddings, and derived products — is served through Prescient's TiTiler tiling server and displayed in a MapLibre-powered web viewer.

The visualization interface displays:

- **SIC output** (primary) — the model-predicted 500m SIC grid, styled with a continuous colour ramp (e.g. blue for 0% through white for 100%).
- **Source SAR** — the Sentinel-1 backscatter imagery underlying the prediction, enabling analysts to cross-check the model's output against the raw input signal.
- **USNIC ice charts** — the polygon-based operational charts, serving as both training label context and a direct comparison product.
- **HLS optical** — cloud-free optical context imagery for seasons and regions where it is available.
- **ICESat-2 tracks** — track transects overlaid as a validation layer where available.

All layers are served directly from Prescient via TiTiler using COG range requests (for raster layers) and PMTiles (for vector layers). Layer toggling, opacity control, and temporal navigation (stepping through dates) are standard MapLibre capabilities that should be surfaced in the viewer.

**Infrastructure**: TiTiler serves tiles from S3-backed COGs on demand. The MapLibre viewer is a static web application hosted on S3 or CloudFront, with no server-side rendering required.

---

## STAC Collections

| Collection | Format | Description |
|---|---|---|
| `sentinel-1-sar` | COG | Sentinel-1 EW GRD scenes over study area |
| `usnic-ice-charts` | PMTiles | USNIC weekly Arctic ice concentration polygons |
| `era5-ancillary` | COG | ERA5 surface variables (temperature, wind, pressure) |
| `amsr2-sic` | COG | AMSR2 passive microwave SIC daily composites |
| `icesat2-tracks` | PMTiles | ICESat-2 freeboard and lead detection transects |
| `hls-optical` | COG | Harmonized Landsat Sentinel-2 optical imagery |
| `clay-embeddings` | COG | Clay encoder embeddings derived from Sentinel-1 scenes |
| `sic-output` | COG | Model-predicted 500m SIC grids (derived product) |

---

## AWS Infrastructure Summary

| Component | Service | Notes |
|---|---|---|
| Asset storage | S3 | Buckets backing Prescient's STAC catalog |
| STAC catalog | PGStac on RDS | Managed by Prescient |
| Tile serving | TiTiler on ECS/Lambda | Dynamic COG and PMTiles serving |
| Ingestion — lightweight | Lambda | ERA5, HLS, STAC item creation |
| Ingestion — heavy | Batch (CPU) | Sentinel-1 processing, `tippecanoe` for PMTiles |
| Embedding generation | Batch (GPU) | Clay encoder inference; g5 instance family |
| Model training (Phase 1) | Batch (CPU) or local | Random Forest / gradient boosting |
| Model training (Phase 2) | Batch (GPU) or SageMaker | End-to-end Clay fine-tuning, if triggered |
| Inference — encoding | Batch (GPU) | Clay encoder on new scenes |
| Inference — prediction | Lambda or Batch (CPU) | Downstream model, fast |
| Orchestration | Step Functions | Per-stage state machines with retry logic |
| Monitoring | CloudWatch | Pipeline health and latency |
| Web viewer | S3 + CloudFront | Static MapLibre application |

For initial development, all pipeline stages except GPU-dependent steps can run locally. The recommended sequence is: build and validate locally → containerise → deploy to Lambda/Batch → wrap in Step Functions.
