# Prescient Ice — Implementation Plan

## Purpose

A phased, sequenced task breakdown to drive implementation. Tasks carry explicit inputs, outputs, dependencies, and acceptance criteria so each maps cleanly onto a development session. This is an engineering work breakdown for human developers (using Claude Code as a coding assistant), not a set of agent instructions.

Work splits into two semi-parallel tracks:

- **Track A — Prescient datasets & inference.** Dataset ingestion into Prescient (COG/vector + STAC registration) and the 2025–26 Hudson Bay inference pipeline.
- **Track B — Modelling.** AI4Arctic training assembly, Clay encoding, classifier training and evaluation. Runs independently of Prescient, reading and writing S3 directly.

The two tracks share the feature contract (`prescient_ice_model_architecture.md` § Feature Contract) as their common specification: Track B builds the training side, Track A builds the inference side, and both must produce an identical feature schema. See § Cross-track dependencies and sequencing.

---

## Phase 0 — Shared foundations

The minimal scaffolding both tracks depend on. Precedes substantive work on either track.

### P0.1 — S3 prefix layout

- **Inputs:** Existing project S3 bucket (record the actual bucket name here); collection inventory from `prescient_ice_pipeline_architecture.md` collections table.
- **Outputs:** Documented prefix convention recording the bucket name and covering: the AI4Arctic raw training dataset; source datasets (`land-mask`, `sentinel-1`, `amsr2`, `era5`, `cis-ice-charts`); derived products (`clay-embeddings`); and training artefacts (normalisation statistics, GeoParquet tables, trained models).
- **Dependencies:** None.
- **Done when:** A written prefix scheme exists, names the bucket, and lets both tracks resolve where any artefact reads from or writes to without ambiguity.
- **Notes:** Both tracks touch S3 — Track A for Prescient assets, Track B for AI4Arctic data, embeddings, tables, and models — so a single agreed layout prevents the two tracks from diverging on paths. The AI4Arctic dataset is included because the student reads it directly from S3.

### P0.2 — AWS resource inventory and GPU compute target

- **Inputs:** Current infrastructure (S3 and SageMaker provisioned; AWS Batch and Lambda not yet provisioned); Clay encoding compute requirement (GPU, g5 family).
- **Outputs:** A recorded inventory of established resources — AWS account, SageMaker Studio domain/profile, and the S3 bucket (cross-referenced from P0.1) — plus a recorded decision on the GPU compute target for the Clay encoding passes (B3 training-side, A8 inference-side): AWS Batch with a g5 environment, or a SageMaker batch transform / training job.
- **Dependencies:** None.
- **Done when:** The established resources are documented and the GPU compute target is chosen, so B3 and A8 begin with no provisioning or account ambiguity.
- **Notes:** The inventory makes the plan a complete reference for a developer starting cold. The GPU decision is not on the critical path for early ingestion or normalisation work, but should be settled before the first encoding pass. CPU-only encoding is feasible at small volume but too slow for the ~1,600-chips-per-scene throughput Clay encoding implies.

---

## Track A — Prescient datasets & inference

Recommended order. A1 establishes the canonical grid; A2–A3 establish the ingestion-and-registration scaffolding on a low-risk dataset; A4–A7 ingest the substantive 2025–26 datasets; A8–A9 run inference.

### A1 — Canonical study-area grid

- **Inputs:** Study area (95°W–75°W, 58°N–66°N); EPSG:3978; Sentinel-1 EW native 40m pixel spacing.
- **Outputs:** A versioned grid specification — exact study-area bounds snapped to a 40m grid in EPSG:3978 (origin, pixel size, dimensions, CRS) — that all on-grid raster ingests target.
- **Dependencies:** P0.1.
- **Done when:** A grid spec exists that any raster ingest can target to produce pixel-aligned output, removing the need to resample co-registered rasters later.
- **Notes:** Track A only — Track B's embeddings inherit each AI4Arctic scene's native grid, so the common grid is purely the Prescient/inference-side raster grid. It is the storage grid for `land-mask`, Sentinel-1, and the final SIC product, and the **resample target** for AMSR2 and ERA5 (which store at native resolution and resample to it at inference, A9). Establishing exact bounds is non-trivial and underpins every downstream raster task, so it comes first. The coarser canonical grids for AMSR2 and ERA5 are separate, dataset-specific grids defined within A6 and A7 — not this 40m analytical grid.

### A2 — STAC collection templates

- **Inputs:** Prescient STAC catalogue interface; a current, widely-used STAC validation utility (`stac-validator` for schema; `stac-check` for best-practice linting); dual-asset pattern and format-conversion table from `prescient_ice_pipeline_architecture.md`.
- **Outputs:** Reusable STAC collection JSON templates for raster (single COG), vector (dual-asset GeoParquet + PMTiles), and derived-embedding collections, each passing schema validation.
- **Dependencies:** P0.1.
- **Done when:** A collection definition validates clean against the chosen utility and can be registered into Prescient by filling collection-specific fields only.
- **Notes:** Track A only — Track B does not register anything in Prescient. The registration interface is not assumed to be PGStac (see § Open questions); validation is tool-based and interface-agnostic so the templates hold regardless of how Prescient ingests them.

### A3 — `land-mask` COG ingest and STAC registration

- **Inputs:** NRCan CanVec Land Features (Shoreline and Island entities, 1:50,000; OGL–Canada); the canonical grid (A1); AI4Arctic Table 7 distance bins.
- **Outputs:** A two-band COG on the canonical grid — Band 1 binary mask (land = 1, water = 0, nodata 255 outside extent), Band 2 distance-to-land index (uint8, 0–41 per Table 7). One STAC item registered in the `land-mask` collection, with metadata recording CanVec product version, NTS tile set, rasterisation parameters, and the Table 7 encoding.
- **Dependencies:** P0.1, A1, A2.
- **Done when:** The COG is retrievable through TiTiler and renders both bands in MapLibre; Band 2 values spot-check correctly against Table 7 thresholds at known coastal and offshore points; the STAC item is queryable by spatial extent through the Prescient STAC API. This also resolves the open Prescient registration-mechanism question.
- **Notes:** The deliberate "hello world" — exercises the full download → reproject → rasterise Band 1 → Euclidean distance transform (`scipy.ndimage.distance_transform_edt`) → bin Band 2 → write COG → register sequence on a static, low-risk dataset before Sentinel-1. Documents are complete and ready. Confirming the registration mechanism through this round-trip is intentional: the static dataset de-risks the procedure before it is relied on for time-varying source data.

### A4 — Sentinel-1 EW ingest (2025–26 Hudson Bay)

- **Inputs:** CDSE Sentinel-1 EW GRD scenes over the study area for the elapsed October 2025 → spring 2026 window; NERSC noise-correction implementation (Korosov et al., 2022); the canonical grid (A1); `sentinel-1` collection template (A2).
- **Outputs:** One COG per scene on the canonical grid with NERSC-corrected σ⁰, registered as a STAC item in the `sentinel-1` collection. Land masking deliberately not applied at ingest.
- **Dependencies:** P0.1, A1, A2, A3 (registration procedure validated).
- **Done when:** CDSE scene availability over the study window is confirmed; a representative scene is NERSC-corrected, written to COG on the canonical grid, registered, and renders in MapLibre; the corrected HV channel is sanity-checked against expected backscatter ranges.
- **Notes:** NERSC correction is a pre-COG step (read ESA-corrected GRD → apply NERSC → write COG), required so the 2025–26 inference input distribution matches the AI4Arctic training distribution; HV is where residual noise matters most. Land masking is left to Stage 3 / Stage 5 so the source COG stays a faithful record and the mask can be revised without re-ingest. CDSE catalogue verification is the first sub-step — it gates the rest of the task.

### A5 — CIS ice chart ingest

- **Inputs:** CIS Hudson Bay weekly regional charts (SIGRID-3) over the study window; the canonical grid (A1); the label-rasterisation method (`prescient_ice_training_strategy.md`); raster collection template (A2).
- **Outputs:** Per-chart STAC items in the `cis-ice-charts` collection, each a single-band COG (EPSG:3978, canonical 320m grid) carrying total concentration as the eleven-class value (0–10 tenths); the raw SIGRID-3 source retained in S3.
- **Dependencies:** P0.1, A1, A2, A3.
- **Done when:** A chart rasterises to a COG on the canonical 320m grid via the pure-cell / area-weighted midpoint method; the COG is spatially valid in EPSG:3978 and aligns cell-for-cell with the SIC product grid; it renders in MapLibre via TiTiler; CT values spot-check correctly against the source polygons; the current-data access path (NSIDC G02171 archive vs direct CIS) is confirmed for the 2025–26 window.
- **Notes:** Charts are rasterised at ingest rather than stored as vectors so the evaluation reference shares the SIC product's canonical 320m grid, making prospective evaluation a direct cell-to-cell comparison with no spatial join; this reuses the label-rasterisation logic from the training path (`prescient_ice_training_strategy.md`). The dual-asset vector pattern is exercised instead by the ICESat-2 track collection. CIS charts are the primary evaluation reference for the inference product.

### A6 — AMSR2 L1R ingest

- **Inputs:** GCOM-W/AMSR2 L1R brightness-temperature swaths from JAXA G-Portal over the study window; `amsr2` collection template (A2).
- **Outputs:** STAC items in the `amsr2` collection holding all 14 channels (seven frequencies × H/V) co-registered onto a fixed 2 km canonical grid in EPSG:3978 — matching AI4Arctic's Table 1 AMSR2 grid — stable across acquisitions.
- **Dependencies:** P0.1, A2, A3.
- **Done when:** All 14 channels resample (Gaussian-weighted, `pyresample`) from the L1R swaths onto the fixed 2 km canonical grid, write to a single multi-band COG, register, and render; the grid is identical across acquisition dates; channel inventory is confirmed and 7.3 GHz availability is verified (the open item from the feature contract).
- **Notes:** The 2 km grid matches AI4Arctic exactly (Table 1: all AMSR2 channels resampled to a 2 km grid via Gaussian-weighted `pyresample` interpolation), so storing here on a fixed 2 km canonical grid reproduces the training-side representation. A fixed grid across dates gives stable multi-band rasters for display and a regular grid for inference sampling. The resample to the 320m patch grid happens at A9. The two-step path (swath → 2 km → patch) is not a parity liability but the parity-faithful choice: AI4Arctic's own pipeline is itself two-step (swath → 2 km grid, then that 2 km field read at the finer SAR/patch grain), so reproducing both steps at inference matches training more closely than a single swath → patch interpolation would. L1R brightness temperature is the same product family AI4Arctic bundled.

### A7 — ERA5 ingest

- **Inputs:** CDS ERA5 download over the study window; the six feature-contract variables (`u10m_rotated`, `v10m_rotated`, `t2m`, `skt`, `tcwv`, `tclw`); per-scene Sentinel-1 heading angle at acquisition (for wind rotation); `era5` collection template (A2).
- **Outputs:** STAC items in the `era5` collection — one COG per variable per timestep, on a single fixed canonical EPSG:3978 grid at ERA5's native ~31 km resolution (all six variables share the one grid, since they are co-resolution), clipped to the study area and stable across dates.
- **Dependencies:** P0.1, A2, A3, A4 (heading angle is read from the corresponding Sentinel-1 acquisition).
- **Done when:** The six variables ingest in native units with no normalisation, on the fixed canonical ~31 km grid identical across dates; wind components are rotated to Sentinel-1 flight direction using the acquisition heading angle, matching the AI4Arctic bundle convention; values are queryable through the STAC API.
- **Notes:** A single canonical grid is near-free here because all six variables are co-resolution (0.25°), and double interpolation (native → canonical → patch, at A9) is negligible: ERA5 surface fields are spatially smooth, so an intermediate grid at native spacing barely smooths them. Pre-regridding onto the 40m analytical grid would instead be hugely redundant and would bake the interpolation choice into storage. `mslp` is not in scope — it was never in the AI4Arctic bundle and was removed from the docs as an error. Wind rotation is the one non-trivial transform: AI4Arctic delivers `u10m_rotated`/`v10m_rotated` already rotated, so the 2025–26 path must apply the same rotation or it breaks parity.

### A8 — Inference Clay encoding (2025–26 scenes)

- **Inputs:** Ingested Sentinel-1 COGs (A4); the `land-mask` collection (A3); the frozen NERSC normalisation constants (B1); the custom `sentinel-1-ew` Clay metadata entry (B3); GPU compute target (P0.2).
- **Outputs:** Per-chip patch-token COGs (32×32×1024) plus per-chip valid-fraction sidecar COGs and class-token assets, registered in the `clay-embeddings` collection, referencing the source SAR scene and Clay model version.
- **Dependencies:** A4, A3, P0.2, **B1, B3** (must reuse the identical constants and metadata entry built on the training side).
- **Done when:** A 2025–26 scene is chipped (regular 256-pixel grid with overlapping edge chips), land/nodata-substituted, encoded, and re-ingested into Prescient; embeddings are discoverable by the source scene's spatial/temporal bounds.
- **Notes:** This is the Prescient round-trip for derived products — embeddings managed through the same STAC interface as source data. Parity with training is enforced by construction: A8 must consume B1's constants and B3's metadata entry rather than re-deriving either. `mask_ratio=0.0` and `shuffle=False` enforced at model load.

### A9 — Inference feature assembly, classification, and SIC product

- **Inputs:** Patch-token / class-token embeddings (A8); AMSR2 on its 2 km canonical grid (A6) and ERA5 on its ~31 km canonical grid (A7); distance-to-land (A3, Band 2); the canonical 40m grid (A1) as patch-grid basis and resample target; the trained classifier selected from B5; valid-fraction grids (A8).
- **Outputs:** A per-scene SIC product COG on the canonical grid at 320m (eleven-class ordinal, 0–10 tenths), land-masked at Stage 5, registered in Prescient for MapLibre display and comparison against CIS charts (A5). Where overlapping edge-chip patches claim the same canonical cell, predictions are reconciled per the reconciliation rule (see § Open questions).
- **Dependencies:** A8, A6, A7, A3, A5, A1, **B5** (trained model), and the feature contract (shared with B4).
- **Done when:** AMSR2 is resampled from its 2 km grid to the 320m patch grid (Gaussian-weighted, matching AI4Arctic's 2 km → SAR-grain step) and ERA5 from its canonical grid to the patch grid; the inference feature vector reproduces the feature-contract schema and column order exactly (the same assembly B4 produces on the training side); a scene classifies end-to-end; the SIC product renders and aligns cell-for-cell with the corresponding CIS chart COG; sub-threshold valid-fraction patches are excluded.
- **Notes:** Both AMSR2 and ERA5 are resampled from their own canonical coarse grids to the patch grid here — the single place the 40m analytical grid is applied to those two datasets. The per-patch valid-fraction threshold is configurable and decoupled from encoding, so it can be tuned without re-running A8.

---

## Track B — Modelling

Reads and writes S3 directly; does not depend on Prescient. Depends on the shared AWS baseline (P0.2) and S3 layout (P0.1). The co-op student already has a working Python environment, the Clay repository, and the AI4Arctic training data, so this track starts at implementation rather than setup.

### B1 — NERSC σ⁰ normalisation statistics

- **Inputs:** All 513 AI4Arctic training scenes (NERSC-corrected σ⁰, NetCDF); band naming `nersc_sar_primary` = HH (index 0), `nersc_sar_secondary` = HV (index 1).
- **Outputs:** Dataset-wide mean and standard deviation of NERSC-corrected σ⁰ in dB, per band (HH and HV separately), persisted as a versioned artefact in S3.
- **Dependencies:** P0.1.
- **Done when:** The per-band constants are computed across all 513 scenes and stored at a fixed S3 location that both B3 (training encoding) and A8 (inference encoding) read, so the two paths normalise identically.
- **Notes:** Early implementation deliverable — required before any Clay encoding begins. These constants double as the nodata-substitution means (substituted pixels become exactly zero in Clay's normalised input space). The built-in `sentinel-1-rtc` constants must not be reused.

### B2 — AI4Arctic data loaders and chip/patch preparation

- **Inputs:** AI4Arctic NetCDF scenes (σ⁰ HH/HV, `btemp_FFP` AMSR2, ERA5 variables, `distance_map`, `sar_grid_latitude`/`sar_grid_longitude` GCPs, `sar_grid_points`); the bundled ice charts as labels.
- **Outputs:** A loader producing prepared 256-pixel chips on a regular grid with overlapping edge chips, with land treated as nodata (via `distance_map` code 0), nodata substituted by the per-band mean (B1), and a per-patch valid-fraction grid (valid pixels / 64 per 8×8 patch) computed before substitution. Fully-invalid chips skipped.
- **Dependencies:** P0.1, B1 (substitution mean).
- **Done when:** A known scene reproduces the documented chip placement and expected chip count; the valid-fraction grid matches a hand-checked patch; `time`/`latlon` metadata is interpolated correctly from the GCPs (GCP count read dynamically from `sar_grid_points`).
- **Notes:** Chip = 256 px ≈ 10.2 km at 40m GSD; patch = 8 px = 320m footprint → 32×32 = 1024 patches per chip. Training reads land and distance-to-land from the bundled `distance_map`, not the Prescient `land-mask` collection — symmetric treatment, different source raster.

### B3 — Custom `sentinel-1-ew` Clay metadata entry and encoding pass

- **Inputs:** Prepared chips (B2); normalisation constants (B1); Clay v1.5 source (verified against `claymodel/model.py`, `claymodel/module.py`, not the public quick-start); GPU compute target (P0.2).
- **Outputs:** A custom `sentinel-1-ew` Clay metadata entry (HH index 0, HV index 1; NERSC mean/std from B1; GSD 40m). Per-chip patch-token COGs (32×32×1024), per-chip valid-fraction sidecars, and per-chip class-token Parquet (`chip_id`, `scene_id`, acquisition timestamp, chip geometry, 1024-dim class token), written to S3.
- **Dependencies:** B1, B2, P0.2.
- **Done when:** The encoder returns `[B, 1025, 1024]`; the EW HH/HV embedding dimension is empirically confirmed at 1024 (resolving the open question); index 0 is taken as the class token and indices 1–1024 reshaped via `einops.rearrange(encoded[:, 1:, :], "b (h w) d -> b d h w", h=32, w=32)`; outputs land in S3 at the agreed prefixes.
- **Notes:** `mask_ratio=0.0` and `shuffle=False` enforced at model load. The built-in `sentinel-1-rtc` entry (VV/VH, RTC, global stats) cannot be reused — hence the custom entry. Training-side encoding writes to S3 only; Prescient registration of embeddings belongs to the inference path (A8).

### B4 — Two-table GeoParquet feature assembly

- **Inputs:** Patch tokens and class tokens (B3); raw HH/HV statistics (from B2/B3); AMSR2 `btemp_FFP`, ERA5 variables, and distance-to-land index (from the AI4Arctic NetCDF); per-patch class labels rasterised from the bundled ice charts (per `prescient_ice_training_strategy.md`).
- **Outputs:** A patch-grain GeoParquet table (patch token, raw HH/HV stats, AMSR2 channels, ERA5 variables, distance-to-land, class label 0–10, patch geometry, `scene_id`, `chip_id`, valid-fraction) and a chip-grain table (class token, chip geometry, `scene_id`, `chip_id`), in S3.
- **Dependencies:** B3.
- **Done when:** Both tables carry the feature-contract column schema and order; the `chip_id` join produces the correct Configuration 3 width (2069); the scene-level train/validation split filters cleanly on `scene_id`; geometry projects out at load time without entering the feature matrix.
- **Notes:** Tables join on `chip_id` only for Configuration 3; Configurations 1 and 2 read the patch table alone. Label rasterisation (mixed-polygon handling, ordinal 0–10) follows `prescient_ice_training_strategy.md`. Feature dimensions: Config 1 = 26, Config 2 = 1045, Config 3 = 2069.

### B5 — Classifier training loop and evaluation

- **Inputs:** Assembled GeoParquet tables (B4); SageMaker compute and managed MLflow (P0.2).
- **Outputs:** Trained Random Forest and XGBoost models across the three feature configurations, logged to MLflow (parameters, metrics, confusion matrices, feature importances, git commit, model artefacts), with the selected model registered for inference (A9).
- **Dependencies:** B4, P0.2.
- **Done when:** SageMaker managed MLflow tracking is enabled; the configured combinations are trained and logged; primary R² is computed on the 0–10 integer class scale (comparable to the AutoICE leaderboard) and the secondary ordinal-penalty metric is reported; a best model is registered for A9 to consume.
- **Notes:** Enabling managed MLflow tracking is a prerequisite step within this task, not a separate task — MLflow is used only here. Open question — whether to run all six configuration × classifier combinations or narrow one axis first — resolves at implementation against how cleanly the loop parameterises (trivial config flips → run all six; meaningful per-variant rework → fix one axis first). Configuration 1 is the raw-backscatter baseline against which Clay's contribution is measured; if Clay does not beat it, the frozen-extractor approach triggers a Phase 2 rethink.

---

## Cross-track dependencies and sequencing

The hard inter-track edges:

- **B1 → B3** — the normalisation constants must exist before any encoding.
- **B1, B3 → A8** — inference encoding must reuse the *identical* constants (B1) and custom metadata entry (B3) built on the training side. This is the parity-by-construction edge: A8 consumes these artefacts rather than re-deriving them.
- **B4 ⟺ A9 (feature contract)** — B4 (training-side) and A9 (inference-side) must produce the same column schema and order. The feature contract is the shared spec; neither path is free to diverge.
- **B5 → A9** — the trained classifier is consumed by inference.

Looser, informational edges (not blocking): the ingestion logic you work out in A4/A6/A7 — NERSC correction, AMSR2 and ERA5 resampling, ERA5 wind rotation — directly informs the parity requirements B2/B4 must honour, which is the intended benefit of running the inference-dataset work alongside the modelling.

**Critical path and timeline.** The time-bound path is the co-op student's B1 → B5, which must complete within the ~2-month horizon (student departs ~end of August 2026). Track A's inference tasks (A8, A9) depend on B1, B3, and B5, so they necessarily tail the modelling work — but because the October 2025 → spring 2026 inference window has already elapsed and there is no seasonal gate, A8/A9 can extend past the student's departure without external time pressure. Track A's grid and ingestion tasks (A1–A7) carry no dependency on Track B and can proceed in parallel from the start; a delay in the student's workstream stalls the sea-ice training dataset but not the other Prescient datasets.

**Suggested early sequencing:** P0.1 and P0.2 first (both tracks blocked on S3 layout / compute target). Then A1 → A2 → A3 on Track A and B1 → B2 on Track B in parallel. A1 (canonical grid) is foundational to every Track A raster task, and B1 gates both B3 and A8 — both should be prioritised early.

---

## Open questions

- **Does Prescient use PGStac?** Unconfirmed. Affects the registration interface assumed by A2/A3; A3's land-mask round-trip will surface the actual mechanism. STAC collection validation (A2) is deliberately tool-based and interface-agnostic so the templates hold either way. Mirror this question into `prescient_ice_index.md` Open Questions when committing.
- **ERA5 ancillary resampling method.** The inference-time resampling of ERA5 to the patch grid (A9) must match the method AI4Arctic used to co-register it onto the SAR grid, or train/inference parity is lost. AMSR2 is well-pinned — AI4Arctic used Gaussian-weighted `pyresample` interpolation onto a 2 km grid (manual Table 1), which A6/A9 reproduce. ERA5's regridding method (likely bilinear) is to be confirmed against the AI4Arctic dataset manual. Resolving this may warrant pinning the method explicitly in the feature contract.
- **Edge-patch prediction reconciliation rule.** The overlapping-edge-chip placement (the final chip in each row and column shifted backward to align with the scene edge) produces patch predictions that can claim the same canonical 320m cell when the SIC product is assembled (A9 / Stage 5). The reconciliation rule (last-write-wins, mean, or highest valid-fraction) is unspecified and must be fixed before product assembly. Mirror this question into `prescient_ice_index.md` Open Questions.

---

## Related documents

- `prescient_ice_index.md` — navigation hub, Open Questions, Changelog
- `prescient_ice_datasets.md` — dataset inventory, Pipeline Infrastructure (land mask, AMSR2, format conversions)
- `prescient_ice_model_architecture.md` — Feature Contract, Input Preparation, Clay encoder details, Feature Configurations
- `prescient_ice_pipeline_architecture.md` — Stage 2 (land-mask ingest), Stage 3 (chip prep), Stage 4 (training assembly), Stage 5 (inference / post-processing), collections table
- `prescient_ice_training_strategy.md` — label preparation, temporal alignment
