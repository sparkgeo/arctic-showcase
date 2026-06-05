# Prescient Ice: Datasets

This document describes all datasets selected for or under active evaluation for the project, organised by their role. A summary table is provided below. Datasets that were evaluated and not selected are not listed here; see the dataset profiles document for full evaluation notes.

Each dataset entry is tagged with a suitability category:

- **Model input** — used as model input and/or training/evaluation data.
- **Supplemental** — confirmed for Prescient ingest as visualisation, validation, or showcase context. Not a model input.
- **Candidate** — under consideration, not yet committed. Each candidate has a specific open question that must be resolved before development effort is allocated.

## Dataset Summary

| Dataset | Category | Role | Prescient Collection |
|---|---|---|---|
| AI4Arctic Sea Ice Challenge Dataset | Model input | Primary training and evaluation dataset (bundled SAR + AMSR2 + ERA5 + CIS/DMI charts) | Not ingested — consumed directly |
| Sentinel-1 EW GRD | Model input | 2025–26 Hudson Bay imagery; NERSC noise correction applied at ingest | `sentinel-1-sar` |
| AMSR2 L1R brightness temperature | Model input | Passive-microwave ancillary feature; via AI4Arctic bundle for training, via JAXA G-Portal for 2025–26 | `amsr2` |
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

## Model Inputs

These datasets are direct inputs to the SIC classifier — either as primary imagery from which embeddings are extracted, as ancillary feature channels appended to the patch token vectors, or as the bundled training package.

### AI4Arctic Sea Ice Challenge Dataset

**Category:** Model input. Primary training and evaluation dataset.

The AI4Arctic Sea Ice Challenge Dataset (Buus-Hinkler et al., 2022) is a curated machine-learning dataset assembled to support the AutoICE challenge — the most significant published benchmark for SAR-based sea ice classification. It bundles Sentinel-1 EW HH/HV SAR imagery with co-registered AMSR2 brightness temperature, ERA5 surface variables, and CIS/DMI ice chart labels per scene, packaged as NetCDF files. The pre-curated co-registration eliminates the temporal alignment problem that would otherwise dominate the training pipeline: scenes and labels are already paired and validated by the dataset authors.

The dataset comprises 533 scenes (513 training, 20 held-out test) covering January 2018 – December 2021 across the Canadian and Greenlandic Arctic, including Hudson Bay. SIC labels use the 0–10 tenths eleven-class scheme that the project adopts directly — no class remapping is needed (see `prescient_ice_training_strategy.md`).

AI4Arctic distributes Sentinel-1 EW data with a single noise correction applied: the **NERSC algorithm** described in Korosov et al. (2022), *IEEE Transactions on Geoscience and Remote Sensing*, vol. 60, doi:10.1109/TGRS.2021.3131036. The user manual (Buus-Hinkler et al., 2022, document version 1.1, Section 2.1) describes the NERSC correction as state-of-the-art for Sentinel-1 EW, surpassing the standard ESA-applied correction, and notes that it is not distributed as a parallel option — both raw and RTT inherit the same NERSC correction, so the noise-correction choice is not a discriminator between dataset versions. The correction matters because it addresses residual scalloping (periodic intensity variations across the swath from antenna gain patterns) and incidence-angle-dependent biases that the ESA correction does not fully remove. These artefacts are most consequential on the HV channel, where the noise floor is closer to typical ice/water backscatter levels than on HH, and where HV is also the channel most informative for ice/water discrimination. The SAR data are exposed as the `nersc_sar_primary` (HH polarisation) and `nersc_sar_secondary` (HV polarisation) variables in the source netCDFs, giving backscatter coefficient (σ⁰) in dB. The same NERSC noise correction must be applied to 2025–26 Sentinel-1 scenes acquired directly from CDSE for prospective evaluation and inference, to maintain end-to-end consistency with the training data (see `prescient_ice_pipeline_architecture.md` for ingest-pipeline integration).

Two distribution forms are available. The **raw version** preserves the SAR data at its native 40m pixel spacing along with the original SIGRID-3 (Sea Ice GeoReferenced Information and Data) polygon codes and lookup tables, with all variables in the source netCDFs unaltered apart from the NERSC noise correction (which is applied to both versions). The **RTT (ready-to-train) version** applies several pre-processing steps documented in the user manual (Buus-Hinkler et al., 2022) and Stokholm et al. (2024), *The Cryosphere*, 18, 3471–3494: a 2×2 averaging kernel downsamples SAR from 40m to 80m pixel spacing (with a 2×2 max kernel on charts at the same scale); per-channel min-max normalisation to the [-1, 1] range using statistics derived from training data; conversion of SOD and FLOE polygon attributes into class maps under a 65% dominant-class threshold; and sentinel value assignments for masked pixels. SIC in the RTT version is binned into 11 classes at 10% increments — mapping SIGRID-3 codes 0, 1, 2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 91, and 92 into the 11-class scheme, with the 9+/10 special code (SIGRID-3 91) folded into the 100% class. The misc/ folder distributes both min/max and mean/standard-deviation statistics per channel, so either normalisation approach can be applied to the raw version when working outside RTT.

The **raw version is preferred** for this project. Three arguments support this. First and decisively, normalisation: the RTT version is distributed already scaled to the [-1, 1] range using global per-channel min-max statistics, whereas the project's custom Clay `sentinel-1-ew` metadata entry expects backscatter in dB standardised with the NERSC-derived mean and standard deviation it specifies. Working from raw σ⁰ in dB lets the pipeline apply exactly that normalisation, and keeps the training data consistent with the 2025–26 CDSE scenes, which are processed from raw through the same path; consuming RTT's pre-scaled values would feed Clay a different transform than its metadata entry assumes and break that consistency. Second, the project's eleven-class SIGRID-3 SIC encoding can be obtained directly from raw via the `convert_raw_icechart.py` script distributed in the AutoICE get-started tools (`github.com/astokholm/AI4ArcticSeaIceChallenge`), which produces SIC, SOD, and FLOE class maps at the original 40m pixel resolution using the same conversion table as RTT — so working from raw does not forfeit AutoICE-compatible label encoding. Third, output resolution: RTT's 2×2 downsampling to 80m would place Clay's 8×8 patch footprint at 640m and the SIC output grid at 640m, coarser than the project's 320m target; raw preserves the ~40m GSD that yields the 320m footprint. This is a choice about output spatial detail, not a constraint imposed by Clay — Clay is sensor-agnostic and designed for multi-scale inputs well outside its 10m IW Sentinel-1 pre-training data, and handles the EW GSD correctly provided the metadata entry's GSD field is set accordingly, so 40m versus 80m does not in itself disadvantage the encoder. A consequence to note: the project's output operates at a 320m minimum mapping unit (the 8×8 patch footprint), a different spatial scale from RTT-based AutoICE submissions. R² values computed on held-out scenes remain numerically comparable since the metric is scale-aware, so the quantitative comparison to the published top-five distribution is preserved despite the scale difference. All three of the top-five teams whose solutions are described in Stokholm et al. (2024) used RTT; the University of Waterloo winning team additionally downsampled RTT by a further factor of 10 to increase model field of view (Chen et al., 2024, *The Cryosphere*, 18, 1621–1632). Our use of raw is the opposite trade — preserving native resolution to fit Clay's patch geometry.

A constellation note: AI4Arctic covers Sentinel-1A and 1B (2018–2021). 2025–26 inference scenes will come from Sentinel-1A, 1C, and 1D. These satellites are well-calibrated relative to each other, but it is worth monitoring during prospective evaluation as a potential source of input distribution shift.

AI4Arctic also serves as a benchmarking reference: published AutoICE challenge results provide a meaningful comparison point for Phase 1 R² evaluation, and the distribution of submitted scores (not just the winning entry) is the intended basis for setting the Phase 2 trigger threshold T (see `prescient_ice_model_architecture.md`). The full results paper to reference is Stokholm et al. (2024), *The Cryosphere*, 18, 3471–3494.

AI4Arctic is consumed directly by the training pipeline rather than ingested into Prescient as a STAC collection. Re-ingesting the dataset as separate per-source collections would discard the per-scene co-registration that is its primary value, and the data is already analysis-ready in its native form. The rationale is detailed in `prescient_ice_pipeline_architecture.md`.

**Products:** 533 NetCDF files (513 training, 20 test); raw version preferred over RTT  
**Access:** DTU Data DOI `10.11583/DTU.c.6244065.v2`; also available via TorchGeo on Hugging Face. Starter toolkit at `github.com/astokholm/AI4ArcticSeaIceChallenge`  
**Prescient collection:** Not ingested — consumed directly by training pipeline

> **Citation required.** The dataset terms require that users who publish work using AI4Arctic cite: Buus-Hinkler, Jørgen; Wulf, Tore; Stokholm, Andreas; Korosov, Anton; Saldo, Roberto; Pedersen, Leif Toudal; Arthurs, David; Solberg, Rune; Longépé, Nicolas; and Kreiner, Matilde Brandt; (2022): AI4Arctic Sea Ice Challenge Dataset. Danish Meteorological Institute. Dataset. https://doi.org/10.11583/DTU.c.6244065.

---

### Sentinel-1 SAR

**Category:** Model input. Source imagery for the 2025–26 Hudson Bay prospective evaluation and operational inference pipeline.

Sentinel-1 is a C-band synthetic aperture radar operated by ESA as part of the Copernicus programme. It images through cloud cover and polar darkness, making it the only sensor suitable for year-round Arctic sea ice monitoring — and the same input platform used by both CIS and USNIC to produce the ice charts that serve as training labels (in AI4Arctic) and prospective evaluation references (for 2025–26).

The relevant acquisition mode is **EW (Extra Wide Swath) GRD**, which provides 400 km swath width at ~20–40m resolution with HH+HV dual polarization. EW is the operationally mandated Sentinel-1 mode for sea ice monitoring and is what CIS and USNIC use for their analyses. IW (Interferometric Wide Swath) acquisitions over Hudson Bay during freeze-up are not systematically acquired and would require multiple scenes to cover the area that one EW scene covers. The project commits to EW HH/HV throughout. The Clay v1.5 metadata mismatch with EW HH/HV — Clay's `sentinel-1-rtc` entry is calibrated for IW VV/VH — is handled via a custom `sentinel-1-ew` metadata entry (see `prescient_ice_model_architecture.md`).

2025–26 scenes acquired directly from CDSE must have the **NERSC noise correction** applied during ingestion before COG conversion, to maintain consistency with the NERSC-corrected AI4Arctic training data. This is implemented as a pre-COG-conversion step in the Sentinel-1 ingestion workflow.

For temporal scope planning: Sentinel-1B failed in December 2021 and was not replaced until Sentinel-1C became fully operational in May 2025, reducing the archive to single-satellite (12-day revisit) during that interval. Sentinel-1D launched in November 2025. The 2025–26 Hudson Bay window therefore spans a three-satellite operational period (1A, 1C, 1D), which gives good revisit cadence over Hudson Bay but represents a different constellation configuration than the 1A/1B period captured in AI4Arctic.

**Products:** EW GRD HH+HV (NERSC noise-corrected at ingest)  
**Access:** Copernicus Data Space Ecosystem (CDSE) STAC/OData; ASF DAAC (NASA ecosystem)  
**Prescient collection:** `sentinel-1-sar` (COG)

---

### AMSR2 Passive Microwave Brightness Temperature

**Category:** Model input. Coarse-resolution passive-microwave brightness-temperature features, appended to patch token feature vectors.

AMSR2 is a passive microwave radiometer aboard JAXA's GCOM-W1 satellite, providing all-weather, Arctic-wide microwave brightness-temperature observations across seven frequency bands (6.9–89.0 GHz) at V and H polarisation, continuously from 2012 to present. The radiometer footprint is coarse and channel-dependent (from roughly 25 km at the lowest frequencies to a few kilometres at 89 GHz). The brightness temperatures for each patch footprint are appended directly to the patch token feature vector as ancillary inputs, and the downstream classifier learns their relationship to ice concentration alongside Clay's SAR-derived features.

AMSR2 enters the project via two paths that resolve to the same underlying product family. For **primary training and evaluation on AI4Arctic**, AMSR2 brightness temperatures are bundled into each NetCDF scene file by the dataset authors (the `btemp_FFP` variables), already resampled from JAXA G-Portal Level-1R swaths onto the 2 km Sentinel-1 grid by Gaussian-weighted interpolation and selected within a seven-hour window of the SAR acquisition. No separate AMSR2 acquisition or alignment is required for the training phase. For the **2025–26 Hudson Bay prospective evaluation and inference pipeline**, AMSR2 Level-1R brightness temperature is acquired from JAXA's G-Portal — the same product AI4Arctic itself used — and ingested into Prescient as a STAC collection at its native coarse grid (resampled from the L1R swaths to a regular EPSG:3978 grid, not the 320 m patch grid). Resampling to the patch grid happens at inference feature-assembly, using the same Gaussian-weighted interpolation, so the stored asset stays compact and the patch-grid registration is performed identically to the training path.

Passive microwave brightness temperature carries a strong, weather-independent ice signal: open water and sea ice differ markedly in microwave emissivity, and the polarisation and frequency structure of that emission is the basis for operational ice concentration retrievals. As an ancillary feature it gives the classifier a coarse but robust regional anchor that complements Clay's fine-resolution SAR texture — most useful where SAR alone is ambiguous, such as wind-roughened open water that mimics the backscatter of new ice. Its coarse, channel-dependent footprint (tens of kilometres against the 320 m target grid) means it informs rather than determines the per-patch prediction.

**Products:** GCOM-W/AMSR2 Level-1R brightness temperature (JAXA), all seven frequency bands (6.9–89.0 GHz) at V and H. Which channels are retained as model features is an implementation decision informed by feature-importance analysis; the low-frequency bands (6.9–10.7 GHz) are weakly sensitive to sea ice and are candidates for exclusion, but this is not fixed here.  
**Access:** JAXA G-Portal (HTTPS/FTP, free, registration required); HDF5 L1R swaths. For training, the same data is read directly from the AI4Arctic NetCDF bundles (`btemp_FFP`).  
**Prescient collection:** `amsr2` (COG) — for the 2025–26 inference pipeline

---

### ERA5 Atmospheric Reanalysis

**Category:** Model input. Atmospheric context features appended to patch token vectors. Also used for temporal alignment filtering in the 2025–26 pipeline.

ERA5 is ECMWF's fifth-generation global atmospheric reanalysis, providing hourly estimates of surface and near-surface atmospheric variables on a ~31 km grid from 1940 to present. It is produced by the Copernicus Climate Change Service and is freely available.

ERA5 serves two distinct purposes in this project. First, selected variables — 2m air temperature, 10m wind components (u, v), and mean sea level pressure — are used as ancillary features appended to the patch token vectors, providing atmospheric physical context that aids SAR interpretation (e.g. wind-roughened open water can produce high SAR backscatter that mimics young ice signatures, and temperature distinguishes active melt from stable frozen conditions). Second, ERA5 temperature and wind data is used to drive the season-adaptive temporal alignment filter in the 2025–26 evaluation pipeline (see `prescient_ice_training_strategy.md`).

For **primary training on AI4Arctic**, ERA5 data is bundled into each scene NetCDF by the dataset authors, pre-co-registered to the SAR footprint. For the **2025–26 pipeline**, ERA5 is acquired directly from the Copernicus Climate Data Store and ingested into Prescient as a STAC collection. ERA5 is distributed on a regular 0.25° lat-lon grid (EPSG:4326) and is reprojected to the project's analytical CRS (EPSG:3978) during ingestion.

Note that ERA5 also includes a sea ice concentration variable derived from passive microwave observations. It is not used as a model feature here — it is not among the ERA5 variables provided in the AI4Arctic bundle, and including it would break train/inference feature parity — though it remains a candidate comparison layer for the Prescient showcase viewer.

**Products:** ERA5 Single Levels (hourly or daily, 2m temperature, 10m wind u/v, MSLP)  
**Access:** Copernicus Climate Data Store (CDS) via `cdsapi`; NetCDF output with server-side spatial/temporal subsetting. For training, accessed via AI4Arctic NetCDF bundles.  
**Prescient collection:** `era5-ancillary` (COG; Zarr as alternative pending Prescient format support)

---

### Clay v1.5 (Geospatial Foundation Model)

**Category:** Model input. Feature extractor producing SAR patch tokens that serve as the primary representation for downstream SIC classification.

Clay is an open-source geospatial foundation model pre-trained via self-supervised learning on Sentinel-1, Sentinel-2, Landsat, and NAIP imagery. Its Sentinel-1 pre-training is the key selection criterion for this project: the model has already learned to represent SAR backscatter patterns, texture, and spatial context in ways directly relevant to sea ice interpretation, without requiring labeled sea ice data during pre-training.

The project uses **Clay v1.5**. Its built-in `sentinel-1-rtc` platform entry is calibrated for IW RTC VV/VH data; Prescient Ice uses EW GRD HH/HV. A custom `sentinel-1-ew` metadata entry is created, specifying HH/HV band names, ~40m GSD (correct for EW), and normalisation statistics derived from NERSC-corrected AI4Arctic scenes. See `prescient_ice_model_architecture.md` for the full custom metadata specification and the rationale.

In Phase 1, Clay's encoder is used as a frozen feature extractor. Sentinel-1 scenes are divided into 256×256 chips and passed through the encoder; the encoder produces a `[batch, 1025, 1024]` tensor per batch, where the first sequence element is a learned class token (the chip embedding) and the remaining 1024 elements are patch tokens reshaped into a 32×32 grid, each spatially registered to a ~320m × 320m footprint. The patch token vectors — appended with AMSR2 and ERA5 ancillary features — form the input to the downstream classifier (Random Forest and XGBoost evaluated in parallel). Clay's weights are not updated in Phase 1.

The patch token grids generated from project imagery are treated as a derived data product. For the 2025–26 pipeline, they are ingested into Prescient as a STAC collection (`clay-embeddings`, with 32×32 spatial structure × 1024 bands per chip). This serves two purposes: it separates the compute-intensive encoding step from downstream classifier iteration (patch tokens can be reused across multiple classifier configurations without re-running Clay), and it demonstrates Prescient's ability to manage AI-derived geospatial products alongside source imagery.

In Phase 2, end-to-end fine-tuning of Clay with a task-specific classification head is explored, conditional on Phase 1 R² falling below a defined threshold or failing to outperform the non-embedding SAR baseline (see `prescient_ice_model_architecture.md`).

**Products:** Self-computed patch token grids from project Sentinel-1 imagery using Clay v1.5 model weights  
**Access:** Open source, model weights and code available on GitHub (`clay-foundation/model`)  
**Prescient collection:** `clay-embeddings` (COG, 32×32 spatial × 1024 bands per chip)

---

## Supplemental — Reference, Validation, and Visualization

These datasets are not used as model inputs or training labels. They are ingested into Prescient as STAC collections to support visualisation, independent validation, or showcase narrative.

### USNIC Weekly Arctic Ice Charts

**Category:** Supplemental. Label source for 2025–26 Hudson Bay prospective evaluation and a Prescient showcase ingest example. Not the primary training label source for model development.

The US National Ice Center produces operational sea ice analyses and forecasts for Arctic waters as a fully integrated multi-agency partnership (US Navy, NOAA, US Coast Guard). The core USNIC analysis is the weekly hemispheric Arctic chart, produced through manual interpretation of SAR imagery — primarily Sentinel-1 and RADARSAT — and distributed in SIGRID-3 shapefile format. Each polygon carries total ice concentration, partial concentrations by ice type, stage of development, and ice form attributes.

A confirmed key finding: **USNIC weekly Arctic charts incorporate CIS (Canadian Ice Service) analysis for Canadian territorial waters.** CIS analysts produce regional sea ice charts for Canadian waters, which USNIC imports, checks for discrepancies against its own analysis, and integrates into the hemispheric product. This means USNIC alone provides complete Canadian Arctic coverage for the purposes of label generation — it is not necessary to separately ingest CIS charts for the same time period. The two products are not independent for overlapping Canadian waters.

USNIC's role in the project has been revised. AI4Arctic is the primary training and evaluation dataset, and AI4Arctic's CIS/DMI chart labels — which incorporate the same analyst pipeline — supersede USNIC for primary training purposes. USNIC remains relevant for two purposes:

1. **2025–26 Hudson Bay prospective evaluation.** USNIC weekly Arctic charts covering October 2025 – January 2026 are used as labels for evaluating model predictions on the prospective Hudson Bay dataset, and as a candidate label source if that data is incorporated into a retraining run.
2. **Prescient showcase ingest.** The USNIC chart ingestion pipeline demonstrates Prescient's dual-asset vector handling (GeoParquet for analytical use, PMTiles for visualisation), Arctic-wide PMTiles tiling, and CIS-derived attribute preservation.

For label generation, the weekly SIGRID-3 vector charts (NSIDC archive G10013) are the appropriate product, not the 10 km gridded derivative product (G10033), which discards the spatial detail of the original vector data. Known data quality issues exist in some historical charts (erroneous polygon attribute codes); these should be filtered during label preparation.

A temporal note: USNIC transitioned from weekly to bi-weekly publishing frequency for the SIGRID-3 archive product in April 2022. The 2025–26 window is in the bi-weekly era; chart coverage cadence is therefore approximately fortnightly rather than weekly.

**Products:** Weekly/bi-weekly Arctic analysis, SIGRID-3 shapefiles (NSIDC G10013, 2003–present)  
**Access:** NSIDC via FTP/HTTPS or `earthaccess`; current charts directly from `usicecenter.gov`  
**Prescient collection:** `usnic-ice-charts` (GeoParquet + PMTiles, dual asset)

---

### ICESat-2 Altimetry

**Category:** Supplemental. Prospective validation overlay layer in the visualisation; candidate supplementary label source for 2025–26 retraining if needed.

ICESat-2 carries the ATLAS photon-counting lidar instrument, providing precise along-track surface elevation measurements up to 88°N. For sea ice applications, the relevant products are ATL07 (sea ice surface heights and lead classification) and ATL10 (sea ice freeboard), both at along-track resolution of approximately 17–200m. Lead detections identify open water within the ice pack with high confidence; high-freeboard measurements identify thick consolidated ice. Under the project's eleven-class framing, lead detections map cleanly to class 0 (open water) and consolidated freeboard measurements map to class 10 (full ice cover).

ICESat-2's role in the project has been revised. The original plan was to use ICESat-2 as supplementary training labels — physically-grounded anchor points at the extremes of the concentration spectrum, complementing analyst-derived polygon labels. With AI4Arctic's 513 training scenes now the primary training source, the anchor-point motivation (boosting label confidence in a small training set) is materially weakened. ICESat-2 is retained for two revised purposes:

1. **Prospective validation overlay.** ICESat-2 tracks overlaid on SIC output in the MapLibre viewer provide an independent, physically-grounded reference for analysts reviewing model predictions. Where a track crosses a model-predicted class-10 region and ATL10 shows significant freeboard, the prediction is corroborated; disagreements flag investigation. This is the primary use of ICESat-2 in the project as currently scoped.
2. **Candidate 2025–26 supplementary labels.** If prospective evaluation reveals shortcomings that warrant incorporating 2025–26 Hudson Bay data into a retraining run, ICESat-2 anchor points become attractive supplementary labels alongside USNIC charts, given that labelled volume from charts alone is likely modest. Lead detections → class 0, consolidated freeboard → class 10. Tight temporal coincidence window (2–4 hours) given ICESat-2's UTC-precise timestamps.

ICESat-2's primary limitation is spatial sparsity: it provides transect observations, not spatially continuous coverage. On any given day, only a narrow set of ground tracks pass over the study area. Coverage accumulates over the 91-day repeat cycle. Lidar also cannot penetrate cloud cover, reducing useful Arctic coverage to below ~40% during and after spring melt onset.

ICESat-2 is also an interesting Prescient showcase dataset in its own right: its along-track point cloud data structure is distinct from the gridded and vector datasets elsewhere in the project, demonstrating the platform's ability to handle diverse geospatial data types.

**Products:** ATL07 (sea ice height and lead classification, v6); ATL10 (sea ice freeboard, v6)  
**Access:** NSIDC via `earthaccess` or `icepyx` Python library; HDF5 format  
**Prescient collection:** `icesat2-tracks` (GeoParquet + PMTiles, dual asset)

---

### Harmonized Landsat Sentinel-2 (HLS)

**Category:** Supplemental. Independent optical validation (seasonal); visual context layer in the MapLibre interface.

HLS is a NASA-produced product that harmonizes Landsat 8/9 and Sentinel-2A/B/C into a single analysis-ready surface reflectance dataset at 30m resolution. The combined five-satellite constellation achieves sub-1.4-day global revisit on average. Coverage extends from 2013 to present.

HLS is not a model input for this project. SAR is the core input for its all-weather, year-round reliability; optical imagery is excluded from the model because cloud cover and polar darkness make it unavailable for the majority of the Arctic year. HLS is included in the project for two purposes. First, it provides independent validation during cloud-free summer periods — optical imagery enables direct visual verification of ice conditions and supports melt pond detection, a period when SAR interpretation is ambiguous and where model performance may degrade. Second, it serves as a contextual visualization layer in the MapLibre interface, providing an intuitive natural-color reference for analysts reviewing SIC output.

On the Prescient side, HLS is an interesting ingest example: it is hosted as COGs in the NASA Earthdata Cloud (AWS us-west-2) with a CMR-STAC catalog, making it a candidate for federated catalog access rather than full local ingestion — demonstrating Prescient's ability to reference externally-hosted STAC datasets alongside locally-managed collections.

**Products:** L30 (Landsat 8/9-derived) and S30 (Sentinel-2-derived) surface reflectance at 30m  
**Access:** NASA LP DAAC via `earthaccess` or CMR-STAC; COGs in NASA Earthdata Cloud (AWS us-west-2)  
**Prescient collection:** `hls-optical` (COG; potentially federated rather than fully ingested)

---

### NOAA/NSIDC Passive Microwave SIC CDR

**Category:** Supplemental. Prescient showcase of multi-decade climate data record ingest. Long-term sea ice concentration reference for narrative context.

The NOAA/NSIDC Sea Ice Concentration Climate Data Record (CDR) provides a bias-corrected, multi-sensor passive microwave sea ice concentration time series from October 1978 to present on a 25 km polar stereographic grid. It is one of the most widely cited datasets in Arctic science and provides the long-term baseline against which recent ice loss is measured. The CDR blends two well-established retrieval algorithms (NASA Team and NASA Bootstrap) and applies inter-sensor calibration across multiple passive microwave instruments to maintain consistency across the full record.

For this project, the CDR is not used as a model input or training label — its 25 km resolution and retrospective processing cycle make it unsuitable for either role, and the AMSR2 brightness-temperature features already supply the passive-microwave ancillary signal at better resolution and latency. Its inclusion as a supplemental dataset is motivated by the long-term context it provides: visualizing four-plus decades of Arctic sea ice decline as a backdrop to the project's high-resolution SAR-derived output strengthens the scientific narrative considerably, and it demonstrates Prescient's ability to ingest and serve coarse gridded climate data records alongside fine-resolution derived products.

**Products:** Final CDR G02202 v6 (daily and monthly SIC, 25 km, 1978–present)  
**Access:** NSIDC via `earthaccess`; NetCDF on NSIDC Sea Ice Polar Stereographic grids (EPSG:3411)  
**Prescient collection:** `pm-sic-cdr` (COG)

---

## Candidate Datasets

The following datasets have been profiled and are under active evaluation, but are not yet committed to the project. Each has a specific open question that must be resolved before development effort is allocated.

**Radarsat Constellation Mission (RCM)** — Canada's operational C-band SAR constellation, operated by CSA. The primary users of RCM data are CIS, making it directly tied to the project's operational context. RCM is not required as a model input (Sentinel-1 and RCM are both C-band sensors with broadly similar sea ice backscatter characteristics, so marginal modeling value is low), but it is a strong candidate as a Prescient showcase layer, and running the trained model on RCM data would be a compelling transferability demonstration. The blocking question is data access: public EODMS access is limited to 16m resolution or coarser, and higher-resolution access requires a formal vetted-user application with CSA security screening. This should be investigated before any ingest effort is scoped.

**TESSERA** — A pixel-level geospatial foundation model from Cambridge pre-trained on Sentinel-1/2 time series. A global pre-computed embedding map for 2024 is freely available via the `geotessera` Python library, with additional years in progress. TESSERA's SAR pre-training makes it thematically relevant and it would demonstrate Prescient's ability to manage pre-computed AI-derived products from external sources. Two limitations apply: TESSERA produces annual embeddings (a single embedding per pixel summarizing a full year of observations), which cannot capture sea ice temporal dynamics and rule it out as a model input; and Arctic coverage of the pre-computed products is not guaranteed — the project study area may need to be specifically requested. Inclusion as a showcase dataset should be confirmed once Arctic coverage availability is verified.

**AlphaEarth Foundations (AEF)** — Annual global embedding layers from Google DeepMind, available via Google Earth Engine and Source Cooperative. Model weights are not public. Same annual resolution limitation as TESSERA rules out model input use. Lower priority than TESSERA for showcase purposes: TESSERA's SAR-specific pre-training is more directly relevant to the project theme, and TESSERA is more accessible outside the GEE ecosystem. Worth revisiting if TESSERA Arctic coverage proves unavailable.

**SWOT** — NASA/CNES wide-swath Ka-band interferometric altimeter providing 250m-resolution surface height and backscatter across a 120 km swath. Emerging research applications to sea ice suggest it can detect open-water leads and thin ice areas within the swath — a spatially wider version of the along-track lead detection capability ICESat-2 provides. SWOT has a hard 78°N latitude cap, which does not affect the current Hudson Bay study area (max 66°N) but would constrain coverage for any pan-Arctic extension. The principal constraint is research-grade maturity of sea ice applications, with key validation papers only published 2025–2026; inclusion as a showcase dataset should be reassessed once that literature has had time to settle.

**AIS (Automatic Identification System)** — Maritime vessel tracking data. Not a model input; the appeal is narrative: overlaying Arctic shipping traffic on SIC model output directly illustrates the real-world relevance of accurate sea ice information for navigation. The access constraint is that comprehensive open-ocean Arctic coverage requires satellite AIS, and no free global S-AIS archive exists. The decision to include AIS depends on whether a suitable data source can be identified for the study area (e.g. a published research dataset covering the Northwest Passage or Beaufort Sea).


