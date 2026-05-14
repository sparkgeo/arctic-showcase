# Prescient Ice: Datasets

## Datasets

This section describes all datasets selected for or under active evaluation for the project, organised by their role. A summary table is provided at the end. Datasets that were evaluated and not selected are not listed here; see the dataset profiles document for full evaluation notes.

---

### Model Inputs

These datasets are direct inputs to the SIC model — either as primary imagery from which embeddings are extracted, or as ancillary feature channels appended to the per-patch embedding vector.

#### Sentinel-1 SAR

**Role:** Primary model input. Source imagery for Clay embedding generation.

Sentinel-1 is a C-band synthetic aperture radar operated by ESA as part of the Copernicus programme. It images through cloud cover and polar darkness, making it the only sensor suitable for year-round Arctic sea ice monitoring — and the same input platform used by both CIS and USNIC to produce the ice charts that serve as training labels. This scene-to-label sensor consistency is a key reason Sentinel-1 is the backbone of this project.

The relevant acquisition mode is **EW (Extra Wide Swath) GRD**, which provides 400 km swath width at ~20–40m resolution with HH+HV dual polarization. EW is the operational ice monitoring mode used by the ice services, maximising consistency with training labels. IW (Interferometric Wide Swath) GRD at ~10m resolution is a secondary consideration where higher spatial detail is needed, but its 250 km swath is narrower. SLC products are not required.

For temporal scope planning: Sentinel-1B failed in December 2021 and was not replaced until Sentinel-1C became fully operational in May 2025, reducing the archive to single-satellite (12-day revisit) during that interval. Study periods either before December 2021 or after May 2025 avoid this coverage reduction.

**Products:** EW GRD HH+HV (primary); IW GRD VV+VH (secondary)  
**Access:** Copernicus Data Space Ecosystem (CDSE) STAC/OData; ASF DAAC (NASA ecosystem)  
**Prescient collection:** `sentinel-1-sar` (COG)

---

#### AMSR2 Passive Microwave SIC

**Role:** Ancillary model input. Coarse-resolution physics-based SIC prior.

AMSR2 is a passive microwave radiometer aboard JAXA's GCOM-W1 satellite. It provides daily, all-weather Arctic-wide sea ice concentration derived from microwave brightness temperature retrieval algorithms, available continuously from 2012 to present at 12.5–25 km resolution. In this project it serves as a physics-based regional SIC prior: rather than asking the model to estimate ice concentration from SAR texture alone, the AMSR2 value for the corresponding grid cell is appended as an ancillary feature, providing a coarse but physically-grounded starting estimate that the model refines to 500m resolution.

AMSR2 is explicitly not used as a training label. Its coarse resolution (25× the 500m target grid) and known accuracy degradation during summer melt — due to melt ponds and changing surface emissivity — make it unsuitable as a pixel-level supervision signal. Its value is as a contextual prior that complements SAR texture, not as ground truth.

**Products:** AU_SI12 (12.5 km daily SIC, NASA Team 2 algorithm) — preferred; AU_SI25 (25 km) as fallback  
**Access:** NSIDC via `earthaccess`; HDF-EOS5 format on NSIDC polar stereographic grids (EPSG:3411)  
**Prescient collection:** `amsr2-sic` (COG)

---

#### ERA5 Atmospheric Reanalysis

**Role:** Ancillary model input. Atmospheric context features. Also used for temporal alignment filtering.

ERA5 is ECMWF's fifth-generation global atmospheric reanalysis, providing hourly estimates of surface and near-surface atmospheric variables on a ~31 km grid from 1940 to present. It is produced by the Copernicus Climate Change Service and is freely available.

ERA5 serves two distinct purposes in this project. First, selected variables — 2m air temperature, 10m wind components (u, v), and mean sea level pressure — are used as ancillary features appended to the per-patch embedding vector, providing atmospheric physical context that aids SAR interpretation (e.g. wind-roughened open water can produce high SAR backscatter that mimics young ice signatures, and temperature distinguishes active melt from stable frozen conditions). Second, ERA5 temperature data is used to drive the season-adaptive temporal alignment filter in the training data preparation pipeline (see Training Strategy).

ERA5 is distributed on a regular 0.25° lat-lon grid (EPSG:4326) and requires reprojection to the project's target CRS (EPSG:3413). Note that ERA5 includes its own sea ice concentration variable derived from passive microwave observations; this is not the same as the AMSR2 SIC product used elsewhere in this project and should not be used in its place.

**Products:** ERA5 Single Levels (hourly or daily, 2m temperature, 10m wind u/v, MSLP)  
**Access:** Copernicus Climate Data Store (CDS) via `cdsapi`; NetCDF output with server-side spatial/temporal subsetting  
**Prescient collection:** `era5-ancillary` (COG or Zarr, pending Prescient format confirmation)

---

#### Clay (Geospatial Foundation Model)

**Role:** Model backbone. Feature extractor producing SAR patch embeddings that serve as the primary representation for downstream SIC regression.

Clay is an open-source geospatial foundation model pre-trained via self-supervised learning on Sentinel-1, Sentinel-2, Landsat, and NAIP imagery. Its Sentinel-1 pre-training is the key selection criterion for this project: the model has already learned to represent SAR backscatter patterns, texture, and spatial context in ways directly relevant to sea ice interpretation, without requiring labeled sea ice data during pre-training.

In Phase 1 (see Model Architecture), Clay's encoder is used as a frozen feature extractor. Sentinel-1 scenes are divided into patches, passed through the encoder, and the resulting embedding vectors — combined with spatially-matched AMSR2 and ERA5 ancillary features — form the feature set for downstream model training (Random Forest or equivalent). Clay's weights are not updated in Phase 1; no GPU-intensive end-to-end training is required.

The embeddings generated from project imagery are treated as a derived data product and ingested into Prescient as a STAC collection. This serves two purposes: it separates the compute-intensive encoding step from downstream training iteration (embeddings can be reused across multiple training runs without re-running the encoder), and it demonstrates Prescient's ability to manage AI-derived geospatial products alongside source imagery.

In Phase 2, end-to-end fine-tuning of Clay with a task-specific regression head is explored, conditional on the Phase 1 accuracy gap warranting the additional complexity and compute cost.

**Products:** Self-computed embeddings from project Sentinel-1 imagery using Clay model weights  
**Access:** Open source, model weights and code available on GitHub (`clay-foundation/model`)  
**Prescient collection:** `clay-embeddings` (COG)

---

### Training Labels

These datasets provide the supervision signal for model training. They are not model inputs at inference time — they are used only during training data preparation.

#### USNIC Weekly Arctic Ice Charts

**Role:** Primary training labels and validation data.

The US National Ice Center produces operational sea ice analyses and forecasts for Arctic waters as a fully integrated multi-agency partnership (US Navy, NOAA, US Coast Guard). The core USNIC analysis is the weekly hemispheric Arctic chart, produced through manual interpretation of SAR imagery — primarily Sentinel-1 and RADARSAT — and distributed in SIGRID-3 shapefile format. Each polygon carries total ice concentration, partial concentrations by ice type, stage of development, and ice form attributes.

A confirmed key finding: **USNIC weekly Arctic charts incorporate CIS (Canadian Ice Service) analysis for Canadian territorial waters.** CIS analysts produce regional sea ice charts for Canadian waters, which USNIC imports, checks for discrepancies against its own analysis, and integrates into the hemispheric product. This means USNIC alone provides complete Canadian Arctic coverage for the purposes of training label generation — it is not necessary to separately ingest CIS charts for the same time period. The two products are not independent for overlapping Canadian waters.

For label generation, the weekly SIGRID-3 vector charts (NSIDC archive G10013) are the appropriate product, not the 10 km gridded derivative product (G10033), which discards the spatial detail of the original vector data. Known data quality issues exist in some historical charts (erroneous polygon attribute codes); these should be filtered during label preparation.

A temporal note: USNIC transitioned from weekly to bi-weekly publishing frequency for the SIGRID-3 archive product in April 2022. Study periods before that date have denser label coverage.

**Products:** Weekly Arctic analysis, SIGRID-3 shapefiles (NSIDC G10013, 2003–present)  
**Access:** NSIDC via FTP/HTTPS or `earthaccess`; current charts directly from `usicecenter.gov`  
**Prescient collection:** `usnic-ice-charts` (PMTiles)

---

#### ICESat-2 Altimetry

**Role:** Supplementary training labels. High-confidence anchor points for open water leads and consolidated pack ice.

ICESat-2 carries the ATLAS photon-counting lidar instrument, providing precise along-track surface elevation measurements up to 88°N. For sea ice applications, the relevant products are ATL07 (sea ice surface heights and lead classification) and ATL10 (sea ice freeboard), both at along-track resolution of approximately 17–200m. Lead detections identify open water within the ice pack with high confidence; high-freeboard measurements identify thick consolidated ice. These provide physically-grounded pixel-level labels at the extremes of the SIC range — near 0% concentration at leads, near 100% at thick pack — that complement and anchor the coarser polygon labels from USNIC charts.

ICESat-2's primary limitation as a label source is spatial sparsity: it provides transect observations, not spatially continuous coverage. On any given day, only a narrow set of ground tracks pass over the study area. Coverage accumulates over the 91-day repeat cycle, so ICESat-2 labels are used as a spatial sample along matched transects, not as an area-covering label layer. Lidar also cannot penetrate cloud cover, reducing useful Arctic coverage to below ~40% during and after spring melt onset.

ICESat-2 is also an interesting Prescient showcase dataset in its own right: its along-track point cloud data structure is distinct from the gridded and vector datasets elsewhere in the project, demonstrating the platform's ability to handle diverse geospatial data types.

**Products:** ATL07 (sea ice height and lead classification, v6); ATL10 (sea ice freeboard, v6)  
**Access:** NSIDC via `earthaccess` or `icepyx` Python library; HDF5 format  
**Prescient collection:** `icesat2-tracks` (PMTiles)

---

### Reference and Validation

These datasets are not used as model inputs or training labels. They provide independent validation evidence and contextual reference layers in the Prescient/MapLibre visualization interface.

#### Harmonized Landsat Sentinel-2 (HLS)

**Role:** Independent optical validation (seasonal); visual context layer in the Prescient interface.

HLS is a NASA-produced product that harmonizes Landsat 8/9 and Sentinel-2A/B/C into a single analysis-ready surface reflectance dataset at 30m resolution. The combined five-satellite constellation achieves sub-1.4-day global revisit on average. Coverage extends from 2013 to present.

HLS is not a model input for this project. SAR is the core input for its all-weather, year-round reliability; optical imagery is excluded from the model because cloud cover and polar darkness make it unavailable for the majority of the Arctic year. HLS is included in the project for two purposes. First, it provides independent validation during cloud-free summer periods — optical imagery enables direct visual verification of ice conditions and supports melt pond detection, a period when SAR interpretation is ambiguous and where model performance may degrade. Second, it serves as a contextual visualization layer in the MapLibre interface, providing an intuitive natural-color reference for analysts reviewing SIC output.

On the Prescient side, HLS is an interesting ingest example: it is hosted as COGs in the NASA Earthdata Cloud (AWS us-west-2) with a CMR-STAC catalog, making it a candidate for federated catalog access rather than full local ingestion — demonstrating Prescient's ability to reference externally-hosted STAC datasets alongside locally-managed collections.

**Products:** L30 (Landsat 8/9-derived) and S30 (Sentinel-2-derived) surface reflectance at 30m  
**Access:** NASA LP DAAC via `earthaccess` or CMR-STAC; COGs in NASA Earthdata Cloud (AWS us-west-2)  
**Prescient collection:** `hls-optical` (COG; potentially federated rather than fully ingested)

---

### Prescient Showcase Datasets

These datasets are ingested into Prescient for platform demonstration purposes — to illustrate the breadth of data types the platform can manage, and to provide narrative context for the project's scientific framing. None are model inputs or training labels.

#### NOAA/NSIDC Passive Microwave SIC CDR

**Role:** Long-term sea ice concentration reference. Prescient showcase of multi-decade climate data record ingest.

The NOAA/NSIDC Sea Ice Concentration Climate Data Record (CDR) provides a bias-corrected, multi-sensor passive microwave sea ice concentration time series from October 1978 to present on a 25 km polar stereographic grid. It is one of the most widely cited datasets in Arctic science and provides the long-term baseline against which recent ice loss is measured. The CDR blends two well-established retrieval algorithms (NASA Team and NASA Bootstrap) and applies inter-sensor calibration across multiple passive microwave instruments to maintain consistency across the full record.

For this project, the CDR is not used as a model input or training label — its 25 km resolution and retrospective processing cycle make it unsuitable for either role; the AMSR2 AU_SI12 product fills the SIC prior role with better resolution and latency. Its inclusion as a showcase dataset is motivated by the long-term context it provides: visualizing four-plus decades of Arctic sea ice decline as a backdrop to the project's high-resolution SAR-derived output strengthens the scientific narrative considerably, and it demonstrates Prescient's ability to ingest and serve coarse gridded climate data records alongside fine-resolution derived products.

**Products:** Final CDR G02202 v6 (daily and monthly SIC, 25 km, 1978–present)  
**Access:** NSIDC via `earthaccess`; NetCDF on NSIDC Sea Ice Polar Stereographic grids (EPSG:3411)  
**Prescient collection:** `pm-sic-cdr` (COG)

---

### Datasets Under Evaluation

The following datasets have been profiled and are under active evaluation, but are not yet committed to the project. Each has a specific open question that must be resolved before development effort is allocated.

**Radarsat Constellation Mission (RCM)** — Canada's operational C-band SAR constellation, operated by CSA. The primary users of RCM data are CIS, making it directly tied to the project's operational context. RCM is not required as a model input (Sentinel-1 and RCM are both C-band sensors with broadly similar sea ice backscatter characteristics, so marginal modeling value is low), but it is a strong candidate as a Prescient showcase layer, and running the trained model on RCM data would be a compelling transferability demonstration. The blocking question is data access: public EODMS access is limited to 16m resolution or coarser, and higher-resolution access requires a formal vetted-user application with CSA security screening. This should be investigated before any ingest effort is scoped.

**TESSERA** — A pixel-level geospatial foundation model from Cambridge pre-trained on Sentinel-1/2 time series. A global pre-computed embedding map for 2024 is freely available via the `geotessera` Python library, with additional years in progress. TESSERA's SAR pre-training makes it thematically relevant and it would demonstrate Prescient's ability to manage pre-computed AI-derived products from external sources. Two limitations apply: TESSERA produces annual embeddings (a single embedding per pixel summarizing a full year of observations), which cannot capture sea ice temporal dynamics and rule it out as a model input; and Arctic coverage of the pre-computed products is not guaranteed — the project study area may need to be specifically requested. Inclusion as a showcase dataset should be confirmed once Arctic coverage availability is verified.

**AlphaEarth Foundations (AEF)** — Annual global embedding layers from Google DeepMind, available via Google Earth Engine and Source Cooperative. Model weights are not public. Same annual resolution limitation as TESSERA rules out model input use. Lower priority than TESSERA for showcase purposes: TESSERA's SAR-specific pre-training is more directly relevant to the project theme, and TESSERA is more accessible outside the GEE ecosystem. Worth revisiting if TESSERA Arctic coverage proves unavailable.

**SWOT** — NASA/CNES wide-swath Ka-band interferometric altimeter providing 250m-resolution surface height and backscatter across a 120 km swath. Emerging research applications to sea ice suggest it can detect open-water leads and thin ice areas within the swath — a spatially wider version of the along-track lead detection capability ICESat-2 provides. Two constraints apply: a hard 78°N latitude cap (which may exclude portions of the study area depending on its geographic extent) and the research-grade maturity of sea ice applications (key validation papers only published 2025–2026). Viability should be reassessed once the study area northern extent is defined.

**AIS (Automatic Identification System)** — Maritime vessel tracking data. Not a model input; the appeal is narrative: overlaying Arctic shipping traffic on SIC model output directly illustrates the real-world relevance of accurate sea ice information for navigation. The access constraint is that comprehensive open-ocean Arctic coverage requires satellite AIS, and no free global S-AIS archive exists. The decision to include AIS depends on whether a suitable data source can be identified for the study area (e.g. a published research dataset covering the Northwest Passage or Beaufort Sea).

---

### Dataset Summary

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

