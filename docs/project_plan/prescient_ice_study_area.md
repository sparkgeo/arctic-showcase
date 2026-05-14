# Prescient Ice: Study Area and Temporal Scope

## Study Area

The study area is Hudson Bay, bounded approximately by 95°W to 75°W longitude and 58°N to 66°N latitude. This encompasses the main body of the bay from the Manitoba and Ontario shorelines in the south and west to the Quebec coast in the east, and north to the approaches of Foxe Basin. James Bay and Hudson Strait are excluded from the initial bounding box; the former is a shallower, lower-salinity sub-basin with somewhat different ice dynamics, and the latter introduces open-ocean fetch conditions that differ from the enclosed bay environment. These exclusions keep the study area coherent and the data volume manageable. The exact bounding box may be refined slightly once the analytical grid is defined in projected coordinates (see Coordinate Reference System, below).

### Why Hudson Bay

Hudson Bay offers a combination of properties that make it an unusually clean choice for an initial sea ice concentration mapping project.

**Complete, well-characterised annual ice cycle.** Hudson Bay freezes over completely each winter and becomes entirely ice-free each summer — a full cryogenic cycle repeated predictably year after year. Freeze-up typically begins in the northwest in late October and progresses southward, with complete ice cover established by January. Break-up runs roughly in reverse, with the bay clearing by early August. This regularity means the seasonal dynamics are well understood and the training window can be selected with confidence.

**Confirmed Sentinel-1 EW coverage.** Published sea ice research using Sentinel-1 EW data specifically over Hudson Bay confirms consistent acquisition during freeze-up periods, including multi-year datasets covering October–May windows. The Canadian Ice Service operationally monitors Hudson Bay and its analysis feeds directly into USNIC weekly Arctic charts, creating tight alignment between the SAR acquisition program and the label source.

**Strong label availability.** USNIC weekly Arctic charts incorporate CIS analysis for Hudson Bay, producing weekly polygon-level ice concentration coverage with fine delineation in this operationally important region. The Churchill shipping corridor and the southern margin of the bay receive particularly detailed chart treatment.

**Operational and community relevance.** Churchill, Manitoba — Canada's only Arctic deep-water port — sits on the southwest shore of Hudson Bay. The port's operating season is constrained directly by sea ice: grain shipments, resupply voyages, and the emerging cruise tourism sector all depend on timely, accurate ice information. Northern and Indigenous communities along the Manitoba and Ontario coasts rely on sea ice for travel, hunting, and cultural practice, and changes to ice timing and extent have direct impacts on community safety and food security. This context makes the showcase genuinely useful rather than merely technically illustrative.

**Research precedent.** Hudson Bay is a well-represented region in the published SAR sea ice machine learning literature. The AutoICE challenge dataset, the most significant benchmark for SAR-based sea ice ML, includes CIS-labelled Sentinel-1 EW scenes from the Canadian Arctic with coverage that includes Hudson Bay freeze-up periods. Existing published studies provide methodological reference points and validation comparisons.

---

## Temporal Scope

The initial temporal window is **October 2025 through January 2026**, covering a single freeze-up season. This window was selected as the most recent complete freeze-up season at the time of project initiation, maximising data currency and ensuring access to the most recent Sentinel-1 acquisition archive.

### Why Freeze-Up

The freeze-up season is preferred over break-up for the initial model for several reasons. Freeze-up presents a progression from open water through new ice, young ice, and first-year ice formation — a wider range of ice concentration values and surface types across the season than the more abrupt and spatially heterogeneous break-up. The SAR backscatter signatures of new and young ice formation are among the more scientifically interesting and practically important targets for a GeoFM-based approach. The freeze-up window also aligns well with the period of highest operational demand for ice information, as the navigation season closes and communities begin relying on ice for over-ice travel.

### Sentinel-1 Constellation Status Note

The October 2025–January 2026 window spans a transition in the Sentinel-1 constellation. Sentinel-1A has operated as the sole satellite since the loss of Sentinel-1B in December 2021, with Sentinel-1C launched in December 2024. Sentinel-1D launched in November 2025, partially overlapping with the study window. The implications are:

- **October–November 2025:** Sentinel-1A and 1C operational; 1D in commissioning. Revisit cadence over Hudson Bay determined by the 1A/1C two-satellite configuration.
- **December 2025–January 2026:** Sentinel-1D progressively entering operational service, potentially increasing coverage frequency toward the end of the window.

The practical effect is that the earlier part of the temporal window may have somewhat lower scene frequency than the later part. This is not expected to be a material constraint — even with a two-satellite constellation, Sentinel-1 EW revisit at the latitude of Hudson Bay is sufficient for weekly-or-better coverage — but the exact scene count across the window should be verified against the CDSE catalog before finalising the training dataset assembly plan. A query of the CDSE STAC API for the study bounding box, the date range, and EW GRD product type will confirm scene availability and identify any gaps.

### Training Data Volume Estimate

As a rough estimate: at approximately 58–66°N, a two-satellite Sentinel-1 constellation provides EW coverage at intervals of a few days over the study bounding box. Over the four-month window, this suggests on the order of 30–60 scenes covering all or part of the study area, against 16–17 weekly USNIC chart publications. Not every scene will align temporally with a chart within the 24-hour baseline window, and scenes must spatially intersect the ice-covered portion of the bay (early October may show minimal ice). A practical estimate of 20–40 usable scene–chart pairs for the initial training dataset is reasonable, with ICESat-2 anchor points providing supplementary labels where coincident passes are available.

This is a modest training volume for the Phase 1 frozen-extractor approach, which is appropriate: the Random Forest downstream model is not data-hungry, and the embedding features from Clay carry substantial pre-trained representational capacity. If volume proves limiting, the temporal window can be extended backward to include the October 2024–January 2025 freeze-up season as additional training data with minimal additional ingestion complexity.

---

## Coordinate Reference System

The analytical CRS for the study area is **EPSG:3978 — NAD83 / Canada Atlas Lambert**. This is the standard Canadian government equal-area projection, well-suited to a Hudson Bay study area: the bay sits comfortably within the projection's zone of least distortion, and the Lambert Conformal Conic base provides good shape preservation across the study extent. Equal-area properties ensure that grid cell areas are consistent across the study area, which is a correctness requirement for the area-weighted label rasterisation in the training pipeline.

All COG raster products are stored in EPSG:3978. All spatial operations — label rasterisation, area-weighted polygon averaging, spatial joins — are performed in EPSG:3978. STAC bounding boxes are expressed in WGS84 (EPSG:4326) per the STAC specification. Vector data is stored in two asset formats per STAC item: GeoParquet in EPSG:3978 for analytical use, and PMTiles in Web Mercator (EPSG:3857) for MapLibre visualisation. TiTiler reprojects raster tiles from EPSG:3978 to Web Mercator on demand at serve time; no raster data is stored in Web Mercator.

If the project scope expands to a pan-Arctic extent in future, EPSG:3995 (WGS 84 / Arctic Polar Stereographic) would be the appropriate replacement, as it is the standard projection for pan-Arctic sea ice data products (AMSR2, NSIDC passive microwave CDR) and avoids the seams of Lambert projections at high latitudes. The transition would require re-ingesting all COG products but would not require changes to the model architecture or training pipeline logic.

### Output Grid

The 500m SIC output grid is defined in EPSG:3978 coordinates. The precise grid origin, extent, and cell dimensions will be finalised once the bounding box is projected into EPSG:3978 and snapped to a round-number origin. This definition will be documented in the STAC collection template for the `sic-output` collection and used consistently across all raster products in the pipeline to ensure spatial alignment.

---

## Summary

| Parameter | Value |
|---|---|
| Study area | Hudson Bay (main body) |
| Bounding box (WGS84) | 95°W–75°W, 58°N–66°N (approximate; subject to minor refinement) |
| Analytical CRS | EPSG:3978 (NAD83 / Canada Atlas Lambert) |
| Output grid resolution | 500m |
| Temporal scope | October 2025 – January 2026 (freeze-up season) |
| Extension option | Add October 2024 – January 2025 if training volume is insufficient |
| Expected training pairs | ~20–40 scene–chart pairs (to be verified against CDSE catalog) |
| Primary label source | USNIC weekly Arctic charts (CIS analysis for Canadian waters) |
| Sentinel-1 mode | EW GRD, HH+HV dual polarisation |
