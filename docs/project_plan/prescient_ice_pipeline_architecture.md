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

## Projection and CRS Strategy

All spatial data in the pipeline follows a consistent CRS convention that separates analytical storage, STAC cataloguing, and visualisation concerns.

**Analytical CRS — EPSG:3978 (NAD83 / Canada Atlas Lambert).** All COG raster products are stored in EPSG:3978. All spatial operations — label rasterisation, area-weighted polygon averaging, spatial joins, grid alignment — are performed in EPSG:3978. This is an equal-area projection well-suited to the Hudson Bay study area, and equal-area properties are a correctness requirement for the area-weighted label rasterisation step. If the project scope expands to a pan-Arctic extent, EPSG:3995 (WGS 84 / Arctic Polar Stereographic) would be the appropriate replacement, aligning with NSIDC and AMSR2 native grid conventions.

**STAC bounding boxes — EPSG:4326 (WGS84).** Per the STAC specification, all item and collection bounding boxes are expressed in WGS84 decimal degrees. This is projection-agnostic and independent of the analytical CRS.

**Serving CRS — EPSG:3857 (Web Mercator).** MapLibre expects tile data in Web Mercator. TiTiler reprojects raster tiles from the native EPSG:3978 COG to Web Mercator on the fly at serve time; no raster data is stored in Web Mercator. PMTiles vector tiles are pre-generated in Web Mercator at ingest time using `tippecanoe` (which accepts WGS84 GeoJSON input and handles the projection internally).

**Analytical operations must never use Web Mercator or WGS84 degrees.** Area calculations in geographic coordinates are distorted at high latitudes. Any operation that involves polygon areas, cell areas, or distance-based spatial joins must be performed after reprojecting to EPSG:3978.

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

### Dual-Asset Pattern for Vector Data

Vector datasets (USNIC ice charts, ICESat-2 tracks) are stored with two assets on each STAC item, separating analytical and visualisation concerns:

- **`data` asset** — GeoParquet format, EPSG:3978 (analytical CRS). This is the asset consumed by the training pipeline for label rasterisation, area-weighted polygon averaging, and spatial joins. All area and distance calculations are performed against this asset.
- **`visual` asset** — PMTiles format, Web Mercator (EPSG:3857). This is the asset served to MapLibre for display. Generated by `tippecanoe` from WGS84 GeoJSON at ingest time.

Both assets are registered on the same STAC item with the same spatiotemporal metadata. This pattern avoids duplicating catalog structure while making the appropriate asset for each use case unambiguous. It should be applied consistently across all vector collections.

### Format Conversions

| Source | Input Format | `data` Asset | `visual` Asset | Notes |
|---|---|---|---|---|
| Sentinel-1 | SAFE / GeoTIFF | COG (EPSG:3978) | — | Apply radiometric calibration if not pre-processed |
| ERA5 | NetCDF / GRIB | COG (EPSG:3978) | — | Regrid to study area; one COG per variable per timestep |
| AMSR2 | HDF5 | COG (EPSG:3978) | — | Reproject from native polar stereographic grid |
| USNIC | Shapefile / GeoJSON | GeoParquet (EPSG:3978) | PMTiles (EPSG:3857) | Preserve concentration attributes in both assets |
| ICESat-2 | HDF5 | GeoParquet (EPSG:3978) | PMTiles (EPSG:3857) | Convert transect points/lines to GeoJSON as intermediate step |
| HLS | COG (already) | COG (EPSG:3978) | — | Reproject/clip to study area if needed |

### Ingestion Workflow

Each source has its own ingestion workflow:

1. **Convert** — run format conversion. For rasters: GDAL to produce EPSG:3978 COGs. For vectors: reproject source to EPSG:3978 and write GeoParquet (`data` asset); convert to WGS84 GeoJSON and run `tippecanoe` for the PMTiles `visual` asset.
2. **Validate** — verify output geometry, CRS, nodata values, and COG/PMTiles/GeoParquet compliance.
3. **Create STAC item** — generate a STAC item JSON with spatial and temporal metadata, asset hrefs pointing to S3 for both assets (where applicable), and any source-specific properties (e.g., Sentinel-1 polarisation, USNIC chart validity date). Bounding box in WGS84.
4. **Register** — POST the STAC item to Prescient's STAC Transaction API (or insert directly into PGStac if bulk loading).
5. **Upload** — copy converted assets to the S3 bucket backing the Prescient catalog.

**Infrastructure**: Lambda handles lightweight conversions (ERA5, HLS, STAC item creation). Batch handles heavy conversions (Sentinel-1 GRD processing, USNIC PMTiles generation via `tippecanoe` for large Arctic-wide charts). Step Functions orchestrates each per-source workflow with retry logic and pipeline state visibility.

For initial development, all of this runs locally as Python scripts. The GDAL, `tippecanoe`, `geopandas`, and `pystac` tooling all run without AWS dependencies; migrating to Lambda/Batch is straightforward once the conversion logic is stable.

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
2. **Label rasterisation** — retrieve the GeoParquet `data` asset for each USNIC chart item. Rasterise SIGRID-3 polygons onto the 500m target grid in EPSG:3978. Phase 1: extract pure cells (fully within a single polygon). Phase 2: include mixed cells with area-weighted labels. All spatial operations are performed in EPSG:3978.
3. **ICESat-2 augmentation** — where coincident ICESat-2 tracks are available within a 2–4 hour window of the SAR acquisition, retrieve the GeoParquet `data` asset for the track item and extract anchor point labels from ATL07/ATL10.
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
4. **Prediction** — apply the trained downstream model to the patch embeddings to produce per-cell SIC predictions. Output is a 500m gridded SIC field in EPSG:3978 covering the scene footprint, values in [0, 1].
5. **Post-processing** — mask land areas, clip values to [0, 1], apply any QA flags. Write as a COG in EPSG:3978 with nodata values and overviews.
6. **Re-ingestion** — create a STAC item for the SIC output, referencing the source SAR scene and model version. Register under the `sic-output` collection in Prescient. Bounding box in WGS84.

**Infrastructure**: AWS Batch (GPU instance) for the Clay encoding step. Lambda or CPU Batch for the downstream model inference step (lightweight, fast). Step Functions orchestrates the trigger → encode → predict → ingest sequence with retry logic. The full cycle from SAR availability to SIC publication is expected to take on the order of minutes for a single scene once the pipeline is operational.

---

## Stage 6: Visualization

All data — source imagery, labels, embeddings, and derived products — is served through Prescient's TiTiler tiling server and displayed in a MapLibre-powered web viewer. TiTiler reprojects raster COGs from EPSG:3978 to Web Mercator on the fly for tile requests; PMTiles vector layers are served directly without reprojection.

The visualization interface displays:

- **SIC output** (primary) — the model-predicted 500m SIC grid, styled with a continuous colour ramp (e.g. blue for 0% through white for 100%).
- **Source SAR** — the Sentinel-1 backscatter imagery underlying the prediction, enabling analysts to cross-check the model's output against the raw input signal.
- **USNIC ice charts** — the polygon-based operational charts, served from the PMTiles `visual` asset, serving as both training label context and a direct comparison product.
- **HLS optical** — cloud-free optical context imagery for seasons and regions where it is available.
- **ICESat-2 tracks** — track transects served from the PMTiles `visual` asset, overlaid as a validation layer where available.

Layer toggling, opacity control, and temporal navigation (stepping through dates) are standard MapLibre capabilities that should be surfaced in the viewer.

**Infrastructure**: TiTiler serves tiles from S3-backed COGs on demand. The MapLibre viewer is a static web application hosted on S3 or CloudFront, with no server-side rendering required.

---

## STAC Collections

| Collection | Format | CRS | Description |
|---|---|---|---|
| `sentinel-1-sar` | COG | EPSG:3978 | Sentinel-1 EW GRD scenes over study area |
| `usnic-ice-charts` | GeoParquet + PMTiles | EPSG:3978 / EPSG:3857 | USNIC weekly Arctic ice concentration polygons (dual asset) |
| `era5-ancillary` | COG | EPSG:3978 | ERA5 surface variables (temperature, wind, pressure) |
| `amsr2-sic` | COG | EPSG:3978 | AMSR2 passive microwave SIC daily composites |
| `icesat2-tracks` | GeoParquet + PMTiles | EPSG:3978 / EPSG:3857 | ICESat-2 freeboard and lead detection transects (dual asset) |
| `hls-optical` | COG | EPSG:3978 | Harmonized Landsat Sentinel-2 optical imagery |
| `clay-embeddings` | COG | EPSG:3978 | Clay encoder embeddings derived from Sentinel-1 scenes |
| `sic-output` | COG | EPSG:3978 | Model-predicted 500m SIC grids (derived product) |

All STAC item bounding boxes are expressed in WGS84 (EPSG:4326) per the STAC specification, regardless of the native asset CRS.

---

## AWS Infrastructure Summary

| Component | Service | Notes |
|---|---|---|
| Asset storage | S3 | Buckets backing Prescient's STAC catalog |
| STAC catalog | PGStac on RDS | Managed by Prescient |
| Tile serving | TiTiler on ECS/Lambda | Reprojects EPSG:3978 COGs to Web Mercator on demand; serves PMTiles directly |
| Ingestion — lightweight | Lambda | ERA5, HLS, STAC item creation |
| Ingestion — heavy | Batch (CPU) | Sentinel-1 processing; GeoParquet and PMTiles generation for vector sources |
| Embedding generation | Batch (GPU) | Clay encoder inference; g5 instance family |
| Model training (Phase 1) | Batch (CPU) or local | Random Forest / gradient boosting |
| Model training (Phase 2) | Batch (GPU) or SageMaker | End-to-end Clay fine-tuning, if triggered |
| Inference — encoding | Batch (GPU) | Clay encoder on new scenes |
| Inference — prediction | Lambda or Batch (CPU) | Downstream model, fast |
| Orchestration | Step Functions | Per-stage state machines with retry logic |
| Monitoring | CloudWatch | Pipeline health and latency |
| Web viewer | S3 + CloudFront | Static MapLibre application |

For initial development, all pipeline stages except GPU-dependent steps can run locally. The recommended sequence is: build and validate locally → containerise → deploy to Lambda/Batch → wrap in Step Functions.
