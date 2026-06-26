# Prescient Ice: Pipeline Architecture

## Overview

The Prescient Ice pipeline moves data from satellite acquisition through processing, model training, inference, and delivery in six stages. Prescient — the project's cloud-native STAC data management platform — sits at the centre of most stages, acting as the shared data layer between source data, derived products, and visualisation.

One important exception sits outside Prescient: the primary training dataset, AI4Arctic, is consumed directly by the training pipeline rather than being ingested into Prescient as a STAC collection. AI4Arctic ships as a pre-curated, scene-co-registered NetCDF dataset bundling Sentinel-1 EW SAR, AMSR2, ERA5, and CIS/DMI chart labels per scene — re-ingesting it into Prescient as separate per-source collections would discard the alignment that makes it useful and provides no analytical benefit. The Prescient-managed pipeline begins at the 2025–26 Hudson Bay prospective evaluation data, where Sentinel-1, AMSR2, ERA5, and CIS charts are acquired independently from their providers and ingested as separate STAC collections to be matched up by the project pipeline.

The pipeline is structured to make the Phase 1 modeling approach — Clay v1.5 as a frozen feature extractor with a separate downstream classifier — explicit as an architectural pattern. Embedding generation and embedding ingestion are distinct stages, not implementation details. This separation is what enables rapid downstream classifier iteration without re-running the encoder on every training run.

The two diagrams below summarise these pipelines separately: the first covers the training pipeline from AI4Arctic through to the trained model artefact; the second covers the inference and evaluation pipeline from data acquisition through to visualisation and prospective evaluation.

```
TRAINING PIPELINE

AI4ARCTIC SEA ICE CHALLENGE DATASET
(533 NetCDF scenes — 513 train, 20 test; Sentinel-1 EW HH/HV + AMSR2 + ERA5 + CIS/DMI charts)
                          │
                          ▼
              ┌─────────────────────┐
              │  CHIP EXTRACTION    │
              │  256×256px chips    │
              │  aligned to Clay    │
              │  input size         │
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  CLAY v1.5 ENCODER  │
              │  (frozen weights)   │
              │  → [B, 1025, 1024]  │
              │  → 32×32 patch      │
              │    token grid       │
              └─────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌──────────────────┐           ┌──────────────────────┐
│ LABEL            │           │ ANCILLARY FEATURES   │
│ RASTERISATION    │           │ AMSR2 Tb + ERA5      │
│ CIS/DMI charts   │           │ (from AI4Arctic       │
│ → 320m patch     │           │  NetCDF bundle)      │
│   grid           │           └──────────────────────┘
└──────────────────┘                      │
          │                               │
          └───────────────┬───────────────┘
                          ▼
              ┌─────────────────────┐
              │  FEATURE ASSEMBLY   │
              │  patch token +      │
              │  label fractions +  │
              │  ancillary → two    │
              │  GeoParquet tables  │
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  DOWNSTREAM         │
              │  CLASSIFIER         │
              │  TRAINING           │
              │  RF + XGBoost       │
              │  3 feature configs  │
              └─────────────────────┘
                          │
                          ▼
                  Trained model → S3
```

```
INFERENCE & EVALUATION PIPELINE

STAGE 1: ACQUISITION
  Sentinel-1 EW GRD  ──┐
  ERA5 Single Levels ──┤
  AMSR2 L1R Tb       ──┤
  CIS SIGRID-3      ──┤
  ICESat-2 ATL07/10  ──┤
  HLS L30/S30        ──┘
                          │
                          ▼
              STAGE 2: INGESTION → PRESCIENT
              (NERSC correction on Sentinel-1;         (STAC catalog)
               format conversion to COG/GeoParquet/
               PMTiles; STAC item registration)
                          │
                          ▼
              STAGE 3: EMBEDDING → PRESCIENT
              (Sentinel-1 COGs tiled into 256×256       (STAC catalog)
               chips → Clay v1.5 encoder → 32×32
               patch token COGs per chip → re-ingested
               as clay-embeddings collection)
                          │
                          ▼
              STAGE 5: INFERENCE  ←──  trained model (see Training Pipeline)
              (patch token COGs + ancillary data
               → downstream classifier
               → 320m SIC class grid
               → re-ingested as sic-output collection)
                          │
                    ┌─────┴──────┐
                    ▼            ▼
            STAGE 6:        PROSPECTIVE
            VISUALIZATION   EVALUATION
            (TiTiler →      (SIC outputs vs
             MapLibre)       CIS charts)
```

---

## Projection and CRS Strategy

All spatial data in the Prescient-managed pipeline follows a consistent CRS convention that separates analytical storage, STAC cataloguing, and visualisation concerns.

**Analytical CRS — EPSG:3978 (NAD83 / Canada Atlas Lambert).** All COG raster products are stored in EPSG:3978. All spatial operations — label rasterisation, area-weighted polygon averaging, spatial joins, grid alignment — are performed in EPSG:3978. This is an equal-area projection well-suited to the Hudson Bay study area, and equal-area properties are a correctness requirement for the area-weighted label rasterisation step. If the project scope expands to a pan-Arctic extent, EPSG:3995 (WGS 84 / Arctic Polar Stereographic) would be the appropriate replacement, aligning with NSIDC and AMSR2 native grid conventions.

**STAC bounding boxes — EPSG:4326 (WGS84).** Per the STAC specification, all item and collection bounding boxes are expressed in WGS84 decimal degrees. This is projection-agnostic and independent of the analytical CRS.

**Serving CRS — EPSG:3857 (Web Mercator).** MapLibre expects tile data in Web Mercator. TiTiler reprojects raster tiles from the native EPSG:3978 COG to Web Mercator on the fly at serve time; no raster data is stored in Web Mercator. PMTiles vector tiles are pre-generated in Web Mercator at ingest time using `tippecanoe` (which accepts WGS84 GeoJSON input and handles the projection internally).

**Analytical operations must never use Web Mercator or WGS84 degrees.** Area calculations in geographic coordinates are distorted at high latitudes. Any operation that involves polygon areas, cell areas, or distance-based spatial joins must be performed after reprojecting to EPSG:3978.

AI4Arctic scenes are delivered in their own native scene-projected coordinate systems and are not reprojected into EPSG:3978 for training; reprojection at this stage would introduce resampling artefacts that the AI4Arctic authors specifically avoided by keeping each scene in its native projection. The training pipeline operates on AI4Arctic data in its native form. EPSG:3978 applies to the Prescient-managed 2025–26 data and to model outputs.

Clay's coordinate inputs — the `latlon` tensor carrying sin/cos pairs for latitude and longitude, and the `gsd` scalar in metres — are expressed in geographic coordinates (WGS84) and are independent of the CRS of the underlying raster data. The projection of a Sentinel-1 scene, whether EPSG:3978 for the 2025–26 pipeline or a native scene projection for AI4Arctic, has no bearing on how Clay constructs its positional encoding. Scene-centre latitude and longitude can be derived from any CRS without reprojection.

---

## Stage 1: Data Acquisition

Each data source is pulled from its upstream provider on an as-needed basis. For the initial build-out, acquisition is batch: a defined study area and temporal window are set, and all available data within those bounds is pulled. Acquisition scripts produce raw files in their native provider formats.

**AI4Arctic Sea Ice Challenge Dataset** — downloaded from DTU Data (DOI `10.11583/DTU.c.6244065.v2`) or accessed via TorchGeo on Hugging Face. 533 NetCDF files (513 training, 20 test) covering January 2018 – December 2021. Each file bundles Sentinel-1 EW HH/HV SAR, AMSR2, ERA5, and CIS/DMI ice chart labels per scene. The raw version is preferred over the RTT (ready-to-train) version; see `prescient_ice_datasets.md` for rationale. AI4Arctic does not flow through the Prescient ingestion stage — it is consumed directly by Stage 4 training.

**Sentinel-1 EW GRD** (2025–26 Hudson Bay) — pulled from the Copernicus Data Space Ecosystem (CDSE) via STAC API or the OData interface. Scenes are filtered by study area footprint intersection and acquisition date. HH/HV dual-polarisation GRD products are the target.

**ERA5 Single Levels** (2025–26) — downloaded from the Copernicus Climate Data Store (CDS) using the `cdsapi` Python client. Variables: 2m air temperature, 10m u/v wind components, mean sea level pressure. Downloaded as NetCDF or GRIB, regridded to the study area.

**AMSR2 L1R brightness temperature** (2025–26) — accessed from JAXA's G-Portal, the same product family AI4Arctic uses for training. HDF5 Level-1R swaths carrying all seven AMSR2 bands; which channels are retained as features is deferred to implementation (informed by feature-importance analysis). Stored on a fixed 2 km canonical grid matching AI4Arctic Table 1 and resampled to the patch grid at inference.

**CIS Hudson Bay weekly regional SIGRID-3** (2025–26) — the dedicated Hudson Bay regional chart (`SGRDRHB`), obtained from the NSIDC G02171 archive (historical; may lag the live window) or directly from CIS for the current window (exact access path to confirm at implementation). Vector data (polygons) in SIGRID-3 shapefile format, with total concentration and other egg-code attributes per polygon; rasterised to a single-band eleven-class COG at ingest (see Format Conversions).

**ICESat-2 ATL07/ATL10** — pulled from the NSIDC DAAC via `icepyx` or the Earthdata STAC API. Granule selection filtered to the study area and temporal window. Used as a visualisation overlay layer and as a candidate supplementary label source for 2025–26 retraining.

**HLS L30/S30** — pulled from NASA Earthdata (LP DAAC) via STAC API. Used for validation and visual context only; cloud filtering should be applied at acquisition time.

**Infrastructure**: Acquisition scripts run locally or on a lightweight compute instance. AWS Lambda is appropriate for triggered or scheduled acquisition once the pipeline is operational. All downloaded files are staged to S3 before ingestion.

---

## Stage 2: Ingestion

Ingestion converts heterogeneous source files into Prescient-compatible formats and registers them as STAC collections. The output of this stage is a fully populated STAC catalog with all 2025–26 source data, ICESat-2 tracks, and HLS imagery queryable by spatial footprint, temporal range, and collection. AI4Arctic does not pass through this stage.

### Sentinel-1 Preprocessing — NERSC Noise Correction

Sentinel-1 EW HH/HV scenes acquired for the 2025–26 pipeline must have the NERSC additional noise correction applied during ingestion, before COG conversion. AI4Arctic provides NERSC-corrected data as a packaged option; for 2025–26 scenes acquired directly from CDSE, the project pipeline must apply the same correction to maintain input distribution consistency between training and inference. The HV channel is where this matters most because residual noise is closest to typical ice/water backscatter levels there, and HV is the channel most informative for ice/water discrimination. See `prescient_ice_training_strategy.md` and `prescient_ice_datasets.md` for further context on the consistency requirement.

The NERSC correction is implemented as a pre-COG-conversion step in the Sentinel-1 ingestion workflow: ESA-corrected GRD pixels are read in, the NERSC algorithm is applied, and the corrected output is written to the EPSG:3978 COG.

Land masking is deliberately not applied to the Sentinel-1 COG at ingestion. The source SAR COG is preserved as a faithful representation of what Sentinel-1 acquired; land masking is applied downstream at chip preparation (Stage 3) and again at final SIC output (Stage 5), sourced from the `land-mask` STAC collection (see below). This means the land mask can be revised or replaced without re-ingesting source SAR data, and the per-patch valid-fraction sidecar produced at Stage 3 captures land coverage in the same channel as data nodata.

### Land Mask — One-Time Ingest

A static land/water mask is ingested into Prescient once at project setup as the `land-mask` collection. It is derived from NRCan CanVec Land Features (Shoreline and Island entities, 1:50,000 scale; Open Government Licence – Canada), rasterised to a two-band COG at the Sentinel-1 EW native 40m pixel spacing in EPSG:3978 over the Hudson Bay study area extent. Band 1 encodes a binary land/water mask (land = `1`, water = `0`, nodata outside the study area extent). Band 2 encodes a distance-to-land index (integer 0–41) following the AI4Arctic Table 7 scheme (Buus-Hinkler et al., 2022): `0` = land; `1`–`41` = water at increasing distance in nonlinear bins from 0–0.5 km (index 1) to 300+ km (index 41). Band 2 is computed by applying a Euclidean distance transform to the inverted Band 1 mask, converting pixel distances to kilometres using the 40m pixel spacing, and binning against the Table 7 thresholds. One STAC item references the COG, with metadata recording the CanVec product version, the NTS tile set used, the rasterisation parameters, and the Table 7 index encoding. See `prescient_ice_datasets.md` § Pipeline Infrastructure for the source selection rationale and full band specifications.

This collection is used at two distinct points in the pipeline: at Stage 3, the mask is sampled at SAR pixel resolution to flag land pixels as nodata before Clay encoding; at Stage 5, the same mask is applied to the final SIC output COG so that the published product distinguishes land from predicted open water. AI4Arctic training data uses the bundled `distance_map` variable directly and does not consume this collection — keeping training and inference symmetric in their treatment of land at Stage 3, just sourced from different rasters. See `prescient_ice_model_architecture.md` § Input Preparation for the rationale.

### Dual-Asset Pattern for Vector Data

Vector datasets (ICESat-2 tracks) are stored with two assets on each STAC item, separating analytical and visualisation concerns:

- **`data` asset** — GeoParquet format, EPSG:3978 (analytical CRS). This is the asset consumed by the project pipeline for label rasterisation, area-weighted polygon averaging, and spatial joins. All area and distance calculations are performed against this asset.
- **`visual` asset** — PMTiles format, Web Mercator (EPSG:3857). This is the asset served to MapLibre for display. Generated by `tippecanoe` from WGS84 GeoJSON at ingest time.

Both assets are registered on the same STAC item with the same spatiotemporal metadata. This pattern avoids duplicating catalog structure while making the appropriate asset for each use case unambiguous. It should be applied consistently across all vector collections.

### Format Conversions

| Source | Input Format | `data` Asset | `visual` Asset | Notes |
|---|---|---|---|---|
| Sentinel-1 | SAFE / GeoTIFF | COG (EPSG:3978) | — | Apply NERSC noise correction before COG conversion; ensure radiometric calibration |
| ERA5 | NetCDF / GRIB | COG (EPSG:3978) | — | Store on a fixed canonical ~31 km grid (native ERA5 0.25° resolution, one grid shared across all six variables), clipped to study area; resample to the 320m patch grid at inference feature-assembly; one COG per variable per timestep |
| AMSR2 | HDF5 (L1R swath) | COG (EPSG:3978, fixed 2 km canonical grid matching AI4Arctic Table 1) | — | Resample L1R brightness-temperature swaths to a regular coarse grid at ingest; resample to the 320 m patch grid at inference feature-assembly (Gaussian-weighted, matching AI4Arctic) |
| CIS ice charts | SIGRID-3 shapefile | COG (EPSG:3978, 320m) | — | Rasterise CT to eleven-class (0–10) on the canonical 320m grid via the pure-cell / area-weighted midpoint method (see `prescient_ice_training_strategy.md`); single band; retain raw SIGRID-3 in S3 |
| ICESat-2 | HDF5 | GeoParquet (EPSG:3978) | PMTiles (EPSG:3857) | Convert transect points/lines to GeoJSON as intermediate step |
| HLS | COG (already) | COG (EPSG:3978) | — | Reproject/clip to study area if needed |

### Ingestion Workflow

Each source has its own ingestion workflow. The initial ingestion is a bulk operation over the fixed 2025–26 study window; ongoing or incremental ingest is not currently in scope.

1. **Create STAC collection** (once per dataset) — define and register a STAC collection for each data source before ingesting items. Each dataset maps to one collection; items are registered as members of that collection. Collection metadata includes spatial and temporal extent, license, and a description of the data source.
2. **Convert** — run format conversion. For rasters: GDAL to produce EPSG:3978 COGs, with Sentinel-1 receiving NERSC noise correction prior to conversion. For vectors: reproject source to EPSG:3978 and write GeoParquet (`data` asset); convert to WGS84 GeoJSON and run `tippecanoe` for the PMTiles `visual` asset.
3. **Validate** — verify output geometry, CRS, nodata values, and COG/PMTiles/GeoParquet compliance.
4. **Create STAC item** — generate a STAC item JSON with spatial and temporal metadata, asset hrefs pointing to S3 for both assets (where applicable), and any source-specific properties (e.g., Sentinel-1 polarisation and noise correction applied, CIS chart validity date). Bounding box in WGS84. A thumbnail asset (PNG, visualised representation of the data) should also be generated and registered on each item; Prescient supports thumbnail assets for catalog browsing purposes.
5. **Register** — register the STAC item and its collection with Prescient. The exact registration mechanism (API, direct database insertion, or other tooling) is to be confirmed once the Prescient workflow is better understood; the output of this step is a STAC item correctly associated with its parent collection in the Prescient catalog.
6. **Upload** — copy converted assets to the S3 bucket backing the Prescient catalog.

**Infrastructure**: Lambda handles lightweight conversions (ERA5, HLS, CIS chart rasterisation, STAC item creation). Batch handles heavy conversions (Sentinel-1 GRD processing with NERSC noise correction). Step Functions orchestrates each per-source workflow with retry logic and pipeline state visibility.

For initial development, all of this runs locally as Python scripts. The GDAL, `tippecanoe`, `geopandas`, `pystac`, and NERSC noise correction tooling all run without AWS dependencies; migrating to Lambda/Batch is straightforward once the conversion logic is stable.

---

## Stage 3: Embedding Generation and Ingestion

This stage runs the Clay v1.5 encoder over Sentinel-1 scenes to produce the 32×32 patch token grids that feed the downstream classifier. The encoding logic is shared between the training and inference paths, but the two paths handle the resulting embeddings differently, and the difference is deliberate.

On the **inference path** (2025–26 prospective evaluation and operational inference), the stage operates on Prescient-managed Sentinel-1 COGs and persists the resulting patch token grids back into Prescient as a derived STAC collection (`clay-embeddings`), so that the SIC product assembly and any re-runs can retrieve pre-computed embeddings without re-running the encoder. This round-trip is also one of the project's Prescient showcase goals — a derived analytical product managed through the same STAC interface as source data.

On the **training path** (Phase 1 AI4Arctic assembly), the embeddings are *not* serialised to COG and are *not* registered in Prescient. They are transient intermediates: the encoder is called inline as part of the single-pass feature assembly (Stage 4), where each chip's patch tokens flow directly into the GeoParquet feature table and are never written to an intermediate raster. Serialising training embeddings to COGs and then re-reading them into the feature table would mean two passes over the corpus and two stored copies of the same transient data; the single-pass design avoids both. The durable training artefact is the assembled feature table, not an embedding store. See Stage 4 § Training Dataset Assembly.

### Embedding Pipeline

1. **Query** — retrieve Sentinel-1 EW HH/HV scenes from Prescient (2025–26 pipeline) or from the AI4Arctic data loader (training pipeline).
2. **Tile** — divide each scene into 256×256-pixel chips at Clay's expected input size. At EW ~40m GSD, each chip covers approximately 10.2 km × 10.2 km. Scene dimensions do not in general divide evenly into 256 pixels; the final chip in each row and column is shifted backward so that its trailing edge aligns with the scene edge, overlapping its predecessor by `256 - (scene_dim mod 256)` pixels. This covers every source pixel without introducing padded or synthetic content into Clay's inputs. Before encoding, the land mask is applied (sourced from the `land-mask` STAC collection for 2025–26 scenes, or the AI4Arctic `distance_map` variable for training), and land pixels and genuine SAR nodata pixels are both substituted with the per-band mean (post-normalisation zero). A per-patch valid-fraction is computed and retained for downstream filtering. Chips that contain no valid SAR pixels (entirely nodata or entirely land) are skipped — no encoding, no STAC item. See `prescient_ice_model_architecture.md` § Input Preparation for the rationale.
3. **Encode** — pass each chip through Clay v1.5's frozen encoder (no gradient computation), using the custom `sentinel-1-ew` metadata entry with HH/HV band names, ~40m GSD, and NERSC-derived normalisation statistics (see `prescient_ice_model_architecture.md`). The encoder produces a `[batch, 1025, 1024]` tensor per batch: the first sequence element is the class token (chip embedding); the remaining 1024 elements are patch tokens, reshaped into a 32×32 grid where each token is spatially registered to a ~320m × 320m footprint.
4. **Serialise** *(inference path only)* — write the patch token grid as a COG per chip with 32 × 32 spatial dimensions and 1024 bands (one band per embedding dimension). Alongside, write a single-band 32 × 32 valid-fraction COG holding the per-patch fraction of valid (non-nodata, non-land) source SAR pixels — used downstream to filter patches by data quality without re-deriving the mask from the source SAR. Both COGs are registered on the same STAC item with consistent spatiotemporal bounds. The class token (chip embedding) is stored separately per chip — either as a sidecar STAC asset or as an additional band/property on the patch token item, to be decided at implementation time. The patch token COG preserves spatial registration of each patch token to its 320m footprint in EPSG:3978. Alternative serialisation (e.g., Zarr or a custom layout) may be considered if the 1024-band COG approach proves operationally awkward or if the per-scene chip count — approximately 1,280 chips per Sentinel-1 EW scene, each producing a patch token COG, a valid-fraction COG, and a STAC item — makes catalog management or visualisation impractical at scale. The architectural commitment is to the 32×32 spatial grid plus the per-patch valid-fraction sidecar, not to a specific storage format. On the training path this serialisation step does not occur: the patch tokens, class token, and per-patch valid-fraction are passed in memory to the single-pass feature assembly (Stage 4), where they become columns in the GeoParquet feature table rather than raster bands.
5. **Re-ingest** *(inference path only)* — create a STAC item for each embedding COG and register it in Prescient under the `clay-embeddings` collection. STAC item metadata references the source SAR scene and the Clay model version used.

The re-ingestion step completes the Prescient round-trip: embeddings are a derived analytical product managed through the same STAC interface as source data, discoverable by the spatial and temporal bounds of the source SAR scene.

**Infrastructure**: Clay inference requires GPU compute for practical throughput (CPU-only is feasible for small volumes but slow). AWS Batch with a GPU-enabled instance (g5 family) is appropriate. On the inference path the batch job pulls SAR data from Prescient via the STAC API, runs encoding, writes patch token COGs to S3, and registers them in Prescient. On the training path the encoder runs inline within the single-pass feature assembly job (Stage 4) over the AI4Arctic corpus, writing GeoParquet feature rows rather than embedding COGs. SageMaker batch transform is an alternative if Clay is deployed as a SageMaker model.

---

## Stage 4: Training

The training stage assembles (patch token, label) pairs and trains the downstream classifier. The primary training source is AI4Arctic; the 2025–26 Hudson Bay data is reserved for prospective evaluation and potential retraining.

### Training Dataset Assembly (AI4Arctic Path)

Training dataset assembly is a **single pass** over the AI4Arctic corpus. A harness iterates the scenes; for each chip it calls the Clay encoder (Stage 3) inline, prepares the patch labels, assembles the per-patch feature rows, and appends them to the GeoParquet output — all without writing an intermediate embedding artefact. The encoding, label preparation, and feature assembly concerns are kept as separate, independently testable modules, but they execute together in one pass so the corpus is read once and the transient embeddings are stored once, as feature columns. The output is partitioned by scene so a failure mid-run leaves completed scenes persisted.

The pass covers all 533 AI4Arctic scenes — the 513 training scenes and the 20 held-out test scenes — and tags each row with a `split` value (train, validation, or test) derived from a scene-level assignment, so downstream consumers select a split by filtering the table rather than maintaining separate datasets. The test scenes' labels, released after the AutoICE challenge as a separate file, are joined at load time (the loader, Stage 3 / B2) so label preparation runs identically across all three splits. The train/validation/test split and its rationale are defined in `prescient_ice_training_strategy.md` § Dataset Splits.

1. **Scene iteration** — iterate over all 533 AI4Arctic scenes (513 train/validation, 20 test) via direct NetCDF reads, tagging each scene with its `split` and joining the separate test-scene label file where applicable.
2. **Chip extraction** — divide each scene into 256×256 chips aligned to Clay's input size (Stage 3 tiling rule). Extract HH/HV pixels for the encoder, and co-registered AMSR2 brightness temperatures and ERA5 surface variables for ancillary features.
3. **Embedding** — pass each chip through Clay v1.5 inline to produce the 32×32 patch token grid and the class token (see Stage 3). The embeddings are held in memory for assembly into the feature row; they are not serialised to COG.
4. **Label preparation** — rasterise the AI4Arctic CIS/DMI chart polygons onto the 320m patch grid in the scene's native projection, then compute per-patch label statistics. For each patch, the SIGRID-3 CT codes of the intersecting polygons are resolved to the eleven-class scale (0–10) and a per-class area-fraction vector is computed: `frac_sic0` … `frac_sic10`, each the fraction of the patch's *valid-class* pixels falling in that class (so the eleven values sum to 1.0 over the labelled area). A separate `valid_class_fraction` records the fraction of the patch's pixels that carry a valid class at all (excluding pixels outside any polygon and pixels in unknown/not-filled/glacier codes — the SIGRID-3 255 bucket). The discrete `label` column is assigned for every patch by the area-weighted collapse (Σ class × fraction, rounded), pure and mixed alike; a pure cell (a single class at fraction 1.0 over its valid-class pixels) is the degenerate case and resolves to that class directly. A boolean `is_pure` column flags the pure patches so that the pure-only-versus-all-patches training choice is a read-time filter. Because the fraction vector is retained, any later revision of the mixed-cell labelling rule is an in-place column update with no re-rasterisation. See `prescient_ice_training_strategy.md` § Label Preparation and § Label Storage on the Feature Table for the method and the rationale.
5. **Ancillary feature attachment** — append the AMSR2 brightness-temperature channels and ERA5 variables (sampled at the patch centroid) to each patch feature row.
6. **Dataset assembly** — append the assembled rows to the two GeoParquet tables in S3, partitioned by `scene_id`. The tables are split by grain to avoid storing the chip embedding redundantly. The **patch table** holds one row per patch token: the 1024-dimensional patch token vector, the raw HH/HV backscatter statistics (mean, standard deviation, and ratio), the appended AMSR2 and ERA5 ancillary features, the distance-to-land index, the per-class label fraction vector (`frac_sic0` … `frac_sic10`), `valid_class_fraction`, the discrete `label` (populated for every patch by the area-weighted collapse), the `is_pure` flag, the patch footprint geometry, a `scene_id`, a `chip_id`, the per-patch SAR `valid_fraction`, and the scene's `split` (train / validation / test). The **chip table** holds one row per chip: the 1024-dimensional class token (chip embedding), the chip footprint geometry, a `scene_id`, the matching `chip_id`, and the scene's `split`. Because the chip embedding is identical for all ~1,024 patches within a chip, storing it on its own ~1,024×-smaller table rather than repeating it across every patch row roughly halves the stored width of the embedding data. The two tables are joined on `chip_id` only when Feature Configuration 3 (patch tokens + chip embedding) is trained; Configurations 1 and 2 read the patch table alone. The chip-level fallback (`prescient_ice_model_architecture.md` § Feature Configurations) requires no additional stored columns: its features are aggregations of the patch table joined to the chip table's class token, derivable at read time.

**Patch retention.** The patch table stores the fuller patch set rather than only the patches that pass a quality or label filter. Patches are retained down to a permissive `valid_fraction` floor (the exact floor is an open design question — see `prescient_ice_index.md`), with `valid_fraction`, `valid_class_fraction`, the label fraction vector, and `is_pure` all stored. This keeps filtering and labelling as read-time operations: the per-consumer `valid_fraction` threshold (training versus final SIC product), the pure-only-versus-all-patches training choice (a filter on `is_pure`), and any later revision of the mixed-cell labelling rule (an in-place column update over the stored fractions) are all applied as predicates or column updates against the stored table, never as a re-encode of the corpus. The cost is storage of patches that a given consumer may discard; the benefit is that the expensive GPU encoding pass is never repeated to revise a threshold or a labelling scheme. A permissive floor additionally protects the chip-level fallback, whose aggregates are computed over the retained patches. Storing the label fraction vector also banks the option for a future weak/aggregate-label training strategy, which consumes per-patch class distributions directly.

**Format rationale.** GeoParquet is chosen over plain numpy or HDF5 because the classifier iteration loop repeatedly reads the same assembled data with column projection (each feature configuration selects only the columns it needs) and row-group predicate pushdown (the scene-level train/validation split filters on `scene_id`, and the valid-fraction and label filters are read-time predicates), all of which Parquet serves natively and efficiently on S3-backed object storage; the GeoParquet variant additionally carries the EPSG:3978 CRS and patch/chip geometry needed for the scene-level split, spatial QA, and the reserved Hudson Bay geographic validation subset, consistent with the project's vector `data` asset convention. The classifier never consumes the geometry columns — they are projected out of the feature matrix at load time. The AMSR2 and ERA5 ancillary features are kept on the patch table rather than split to a coarser-grain table, despite their coarse resolution (AMSR2 brightness temperature at roughly 5–25 km depending on channel, ERA5 at ~31 km) making their per-patch values highly repetitive within and across chips. The redundancy is cheap: these are roughly ten float32 columns against the patch table's ~1024, so a separate table would save on the order of one percent of width, and Parquet's dictionary and run-length encoding already compress the repeated values at the storage layer. More decisively, ancillary features are appended in all three feature configurations (including the raw baseline, so the comparison isolates Clay's contribution rather than the ancillary inputs), so moving them off the patch table would force a join on every training run rather than only for Configuration 3 — the opposite of the chip embedding's access pattern. See `prescient_ice_model_architecture.md` § Feature Configurations for the per-configuration feature composition.

### Training Dataset Assembly (2025–26 Retraining Path)

The same logic applies if the 2025–26 Hudson Bay data is used for retraining, with two differences. First, Sentinel-1, AMSR2, ERA5, and CIS chart data are queried from Prescient rather than pulled from a co-registered NetCDF — temporal alignment must be performed during pair assembly (24-hour baseline window, ERA5-adaptive tightening; see `prescient_ice_training_strategy.md`). Second, ICESat-2 anchor points become a candidate supplementary label source: where coincident tracks are available within a 2–4 hour window of the SAR acquisition, retrieve the GeoParquet `data` asset for the track item and extract anchor point labels (lead detections → class 0, consolidated freeboard → class 10).

### Downstream Classifier Training

The assembled dataset is used to train the downstream SIC classifier. Phase 1 candidates: Random Forest and XGBoost, evaluated in parallel across three feature configurations (raw HH/HV backscatter baseline; patch tokens alone; patch tokens + chip embedding). See `prescient_ice_model_architecture.md` for the full feature configuration and evaluation framework.

The training loop is lightweight — no GPU required — and can run locally or on a standard CPU instance. Training inputs are the assembled feature vectors; training targets are the per-patch class labels (eleven classes, 0–10).

Model artefacts (trained model, feature importance outputs, validation metrics including R² on the 0–10 class scale and the ordinal penalty metric) are stored in S3. If Phase 1 accuracy is insufficient or fails to outperform the non-embedding baseline, Phase 2 initiates end-to-end Clay fine-tuning on GPU — see `prescient_ice_model_architecture.md` for detail on the Phase 2 architecture and infrastructure.

**Infrastructure**: Lambda or a lightweight EC2/Batch job for dataset assembly (primarily I/O-bound, pulling from AI4Arctic or Prescient). Local or CPU-only Batch for downstream classifier training. GPU Batch or SageMaker Training Job for Phase 2 fine-tuning if triggered.

---

## Stage 5: Inference

The inference pipeline applies the trained classifier to new Sentinel-1 scenes and delivers 320m SIC class grids back into Prescient as a derived product. Inference is the primary use of the 2025–26 Hudson Bay data: trained on AI4Arctic, the model is run on 2025–26 Sentinel-1 scenes to produce SIC outputs, with prospective evaluation comparing those outputs to CIS charts for the same period.

### Inference Workflow

1. **Trigger** — a new Sentinel-1 scene over the study area is ingested into Prescient with NERSC noise correction applied (manual trigger for the showcase; EventBridge rule monitoring the STAC catalog for an operational deployment).
2. **Input assembly** — pull the SAR COG from Prescient. Retrieve the closest ERA5 and AMSR2 data within appropriate time windows.
3. **Embedding** — run Clay v1.5 over the new scene (frozen weights, same procedure as Stage 3). Output is a 32×32 patch token grid per chip across the scene.
4. **Feature assembly** — for each patch token, construct the feature vector matching the trained configuration (raw baseline, patch tokens, or patch tokens + chip embedding) with AMSR2 and ERA5 ancillary features appended.
5. **Prediction** — apply the trained downstream classifier to each patch feature vector to produce per-patch class predictions (and per-class probabilities, if useful for the visualisation layer). For each chip, the result is a 32×32 prediction grid.
6. **Rasterisation** — assemble per-chip 32×32 prediction grids into a scene-wide 320m SIC class COG in EPSG:3978. Where chips overlap at scene boundaries (see `prescient_ice_model_architecture.md` § Input Preparation), each patch in the overlap region is assigned to the chip in which it sits more interior to the chip footprint (further from the nearest chip edge), giving a deterministic, single-source class for every output cell. Patches whose valid-fraction sidecar value falls below the configured threshold are written as nodata in the output. The output COG carries integer class values 0–10 with appropriate nodata handling.
7. **Post-processing** — apply the `land-mask` STAC collection to mark land pixels as nodata in the output, and apply any additional QA flags. The land mask used here is the same authoritative source applied at Stage 3 chip preparation, ensuring consistency between what Clay saw as land and what the published product reports as land. Write the final COG with nodata values and overviews. Optionally write a parallel COG of per-class probabilities (multi-band, one band per class) for downstream uncertainty visualisation.
8. **Re-ingestion** — create a STAC item for the SIC output, referencing the source SAR scene, NERSC noise correction status, Clay model version, and downstream classifier version. Register under the `sic-output` collection in Prescient. Bounding box in WGS84.

**Infrastructure**: AWS Batch (GPU instance) for the Clay encoding step. Lambda or CPU Batch for the downstream classifier inference step (lightweight, fast). Step Functions orchestrates the trigger → encode → predict → ingest sequence with retry logic. The full cycle from SAR availability to SIC publication is expected to take on the order of minutes for a single scene once the pipeline is operational.

---

## Stage 6: Visualization

All data — source imagery, labels, embeddings, and derived products — is served through Prescient's TiTiler tiling server and displayed in a MapLibre-powered web viewer. TiTiler reprojects raster COGs from EPSG:3978 to Web Mercator on the fly for tile requests; PMTiles vector layers are served directly without reprojection.

The visualization interface displays:

- **SIC output** (primary) — the model-predicted 320m SIC class grid, styled with a discrete eleven-step colour ramp (e.g. blue for class 0 / open water through white for class 10 / full ice cover). Optional per-class probability layers for uncertainty visualisation.
- **Source SAR** — the Sentinel-1 backscatter imagery underlying the prediction, enabling analysts to cross-check the model's output against the raw input signal.
- **CIS ice charts** — the rasterised chart product, served as a COG via TiTiler, providing a direct comparison product for prospective evaluation on the same grid as the SIC output.
- **HLS optical** — cloud-free optical context imagery for seasons and regions where it is available.
- **ICESat-2 tracks** — track transects served from the PMTiles `visual` asset, overlaid as an independent physical validation reference. Lead detections and significant freeboard measurements provide point-level corroboration of model predictions at the extremes of the class spectrum.

Layer toggling, opacity control, and temporal navigation (stepping through dates) are standard MapLibre capabilities that should be surfaced in the viewer.

**Infrastructure**: TiTiler serves tiles from S3-backed COGs on demand. The MapLibre viewer is a static web application hosted on S3 or CloudFront, with no server-side rendering required.

---

## STAC Collections

| Collection | Format | CRS | Description |
|---|---|---|---|
| `sentinel-1-sar` | COG | EPSG:3978 | Sentinel-1 EW GRD scenes over study area, NERSC noise correction applied |
| `cis-ice-charts` | COG | EPSG:3978 | CIS Hudson Bay weekly regional charts, rasterised to eleven-class SIC (0–10) on the canonical 320m grid |
| `era5-ancillary` | COG | EPSG:3978 | ERA5 surface variables (temperature, wind, pressure) |
| `amsr2` | COG | EPSG:3978 | AMSR2 passive-microwave brightness temperature (JAXA L1R), stored on a fixed 2 km canonical grid matching AI4Arctic Table 1; resampled to the 320m patch grid at inference feature-assembly |
| `icesat2-tracks` | GeoParquet + PMTiles | EPSG:3978 / EPSG:3857 | ICESat-2 freeboard and lead detection transects (dual asset) |
| `hls-optical` | COG | EPSG:3978 | Harmonized Landsat Sentinel-2 optical imagery |
| `land-mask` | COG (2-band) | EPSG:3978 | Static land/water mask and distance-to-land index over the Hudson Bay study area, derived from NRCan CanVec Land Features at 1:50,000 (Open Government Licence – Canada); Band 1: binary uint8 land/water; Band 2: distance-to-land index uint8 0–41 (AI4Arctic Table 7 encoding); 40m; applied at Stage 3 chip preparation and Stage 5 post-processing |
| `clay-embeddings` | COG (32×32 × 1024 bands) + valid-fraction sidecar COG (32×32 × 1 band) | EPSG:3978 | Clay v1.5 patch token grids derived from 2025–26 Sentinel-1 inference scenes; class token chip embedding stored alongside; per-patch valid-fraction registered on the same STAC item. Inference path only — training embeddings are not serialised here (they flow into the GeoParquet feature table; see Stage 4) |
| `sic-output` | COG | EPSG:3978 | Model-predicted 320m SIC class grids (derived product) |

All STAC item bounding boxes are expressed in WGS84 (EPSG:4326) per the STAC specification, regardless of the native asset CRS.

AI4Arctic is not represented as a Prescient STAC collection. It is consumed by the training pipeline directly via its native NetCDF distribution, read by a purpose-built loader (B2) rather than TorchGeo. The rationale is detailed in `prescient_ice_datasets.md`.

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
