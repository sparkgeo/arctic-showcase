# Prescient Ice: Study Area and Temporal Scope

## Study Area

The study area is Hudson Bay, bounded approximately by 95°W to 75°W longitude and 58°N to 66°N latitude. This encompasses the main body of the bay from the Manitoba and Ontario shorelines in the south and west to the Quebec coast in the east, and north to the approaches of Foxe Basin. James Bay and Hudson Strait are excluded from the initial bounding box; the former is a shallower, lower-salinity sub-basin with somewhat different ice dynamics, and the latter introduces open-ocean fetch conditions that differ from the enclosed bay environment. These exclusions keep the study area coherent and the data volume manageable. The exact bounding box may be refined slightly once the 320m analytical grid is defined in projected coordinates (see Coordinate Reference System, below).

### Why Hudson Bay

Hudson Bay offers a combination of properties that make it an unusually clean choice for the project's prospective evaluation region.

**Complete, well-characterised annual ice cycle.** Hudson Bay freezes over completely each winter and becomes entirely ice-free each summer — a full cryogenic cycle repeated predictably year after year. Freeze-up typically begins in the northwest in late October and progresses southward, with complete ice cover established by January. Break-up runs roughly in reverse, with the bay clearing by early August. This regularity means the seasonal dynamics are well understood and the evaluation window can be selected with confidence.

**Confirmed Sentinel-1 EW coverage.** Published sea ice research using Sentinel-1 EW data specifically over Hudson Bay confirms consistent acquisition during freeze-up periods, including multi-year datasets covering October–May windows. The Canadian Ice Service operationally monitors Hudson Bay and its analysis feeds directly into USNIC weekly Arctic charts, creating tight alignment between the SAR acquisition program and the prospective evaluation label source.

**Strong label availability.** USNIC weekly Arctic charts incorporate CIS analysis for Hudson Bay, producing polygon-level ice concentration coverage with fine delineation in this operationally important region. The Churchill shipping corridor and the southern margin of the bay receive particularly detailed chart treatment.

**Operational and community relevance.** Churchill, Manitoba — Canada's only Arctic deep-water port — sits on the southwest shore of Hudson Bay. The port's operating season is constrained directly by sea ice: grain shipments, resupply voyages, and the emerging cruise tourism sector all depend on timely, accurate ice information. Northern and Indigenous communities along the Manitoba and Ontario coasts rely on sea ice for travel, hunting, and cultural practice, and changes to ice timing and extent have direct impacts on community safety and food security. This context makes the showcase genuinely useful rather than merely technically illustrative.

**Research precedent and AI4Arctic coverage.** Hudson Bay is a well-represented region in the published SAR sea ice machine learning literature. The AI4Arctic Sea Ice Challenge Dataset — the project's primary training and evaluation source (see `prescient_ice_datasets.md`) — includes CIS-labelled Sentinel-1 EW scenes from the Canadian Arctic with coverage that includes Hudson Bay freeze-up periods. A Hudson Bay subset of AI4Arctic should be identified and used for domain-specific validation alongside the full pan-Arctic evaluation, confirming Phase 1 performance on the project's target region specifically. Existing published studies, including the AutoICE results paper (Stokholm et al., 2024, *The Cryosphere*, 18, 3471–3494), provide methodological reference points and benchmarking targets.

---

## Temporal Scope

The 2025–26 Hudson Bay window covers **October 2025 through January 2026** — a single freeze-up season. Its role in the project has been revised in light of AI4Arctic becoming the primary training and evaluation dataset: rather than serving as the project's primary training period, the 2025–26 Hudson Bay window is a **prospective evaluation dataset**, used to test model generalisation to unseen recent data after training on AI4Arctic's 2018–2021 scenes. If prospective evaluation reveals shortcomings warranting it, 2025–26 data can subsequently be incorporated into a retraining run; see `prescient_ice_training_strategy.md` for the retraining label preparation strategy.

The window was selected as the most recent complete freeze-up season at the time of project initiation, maximising the temporal distance between the training data (2018–2021) and the prospective evaluation data. This is exactly the kind of generalisation test the prospective evaluation is designed to expose: differences in SAR calibration between 1A/1B (the AI4Arctic constellation) and 1A/1C/1D (the 2025–26 constellation); changes in ice conditions relative to the training period given accelerating Arctic change; and any other distribution shifts that surface only against recent data.

### Why Freeze-Up

The freeze-up season is preferred over break-up for prospective evaluation for several reasons. Freeze-up presents a progression from open water through new ice, young ice, and first-year ice formation — a wider range of ice concentration classes and surface types across the season than the more abrupt and spatially heterogeneous break-up. The SAR backscatter signatures of new and young ice formation are among the more scientifically interesting and practically important targets for a GeoFM-based approach. The freeze-up window also aligns well with the period of highest operational demand for ice information, as the navigation season closes and communities begin relying on ice for over-ice travel.

### Sentinel-1 Constellation Status Note

The October 2025–January 2026 window spans a transition in the Sentinel-1 constellation. Sentinel-1B failed in December 2021 and was not replaced until Sentinel-1C became fully operational in May 2025. Sentinel-1D launched in November 2025, partially overlapping with the study window. The implications are:

- **October–November 2025:** Sentinel-1A and 1C operational; 1D in commissioning. Revisit cadence over Hudson Bay determined by the 1A/1C two-satellite configuration.
- **December 2025–January 2026:** Sentinel-1D progressively entering operational service, potentially increasing coverage frequency toward the end of the window.

The practical effect is that the earlier part of the temporal window may have somewhat lower scene frequency than the later part. This is not expected to be a material constraint for prospective evaluation — even with a two-satellite constellation, Sentinel-1 EW revisit at the latitude of Hudson Bay is sufficient for weekly-or-better coverage — but the exact scene count across the window should be verified against the CDSE catalog before finalising the prospective evaluation plan. A query of the CDSE STAC API for the study bounding box, the date range, and EW GRD product type will confirm scene availability and identify any gaps.

The constellation configuration also represents a meaningful difference from the AI4Arctic training data, which covers the 2018–2021 1A/1B period. Sentinel-1A, 1C, and 1D are designed to be well-calibrated relative to one another, but inter-satellite calibration consistency is one of the distribution shift factors that the prospective evaluation will help characterise.

### Prospective Evaluation Volume Estimate

As a rough estimate: at approximately 58–66°N, a two- or three-satellite Sentinel-1 constellation provides EW coverage at intervals of a few days over the study bounding box. Over the four-month window, this suggests on the order of 30–60 scenes covering all or part of the study area. The corresponding USNIC chart coverage is approximately fortnightly (USNIC transitioned from weekly to bi-weekly SIGRID-3 publication in April 2022), so 8–9 charts cover the window. Each scene can be evaluated against the nearest temporally-aligned chart within the 24-hour baseline window (ERA5-adaptive tightening during dynamic transition periods; see `prescient_ice_training_strategy.md`), giving a practical estimate of 20–40 usable scene–chart evaluation pairs.

This is sufficient for a meaningful prospective evaluation. The classifier is trained on AI4Arctic's 513 scenes, not on 2025–26 data, so the evaluation set need not be data-hungry — its purpose is to characterise generalisation rather than to drive training. If the prospective evaluation reveals shortcomings that warrant retraining with 2025–26 data, the same scene–chart pair pool can be used as a retraining set, supplemented by ICESat-2 anchor points where coincident tracks are available (see `prescient_ice_training_strategy.md` and `prescient_ice_datasets.md` for the ICESat-2 role under the retraining scenario).

---

## Coordinate Reference System

The analytical CRS for the project's Prescient-managed data is **EPSG:3978 — NAD83 / Canada Atlas Lambert**. This is the standard Canadian government equal-area projection, well-suited to a Hudson Bay study area: the bay sits comfortably within the projection's zone of least distortion, and the Lambert Conformal Conic base provides good shape preservation across the study extent. Equal-area properties ensure that grid cell areas are consistent across the study area, which is a correctness requirement for the area-weighted label rasterisation in the evaluation and retraining pipeline.

All COG raster products in the Prescient-managed pipeline are stored in EPSG:3978. All spatial operations — label rasterisation, area-weighted polygon averaging, spatial joins — are performed in EPSG:3978. STAC bounding boxes are expressed in WGS84 (EPSG:4326) per the STAC specification. Vector data is stored in two asset formats per STAC item: GeoParquet in EPSG:3978 for analytical use, and PMTiles in Web Mercator (EPSG:3857) for MapLibre visualisation. TiTiler reprojects raster tiles from EPSG:3978 to Web Mercator on demand at serve time; no raster data is stored in Web Mercator.

AI4Arctic scenes are processed in their native scene-projected coordinate systems during training, rather than being reprojected into EPSG:3978; reprojection at training time would introduce resampling artefacts that the AI4Arctic authors specifically avoided by keeping each scene in its native projection. See `prescient_ice_pipeline_architecture.md` for the full projection strategy.

If the project scope expands to a pan-Arctic extent in future, EPSG:3995 (WGS 84 / Arctic Polar Stereographic) would be the appropriate replacement, as it is the standard projection for pan-Arctic sea ice data products (AMSR2, NSIDC passive microwave CDR) and avoids the seams of Lambert projections at high latitudes. The transition would require re-ingesting all COG products but would not require changes to the model architecture or training pipeline logic.

### Output Grid

The 320m SIC output grid is defined by the Clay v1.5 patch footprint at Sentinel-1 EW's native ~40m ground sampling distance: Clay's 8×8 pixel patch size at 40m GSD produces a 320m × 320m output cell. This is the honest effective resolution of the model — it is defined by the architecture rather than chosen as a round-number target, and no resampling step is applied to align it to a different grid. The full mechanics of the patch token spatial structure are documented in `prescient_ice_model_architecture.md`.

In EPSG:3978, the 320m grid origin, extent, and cell dimensions will be finalised once the study bounding box is projected into EPSG:3978 and the patch-aligned cell scheme is established. This grid definition will be documented in the STAC collection template for the `sic-output` collection and used consistently across all 2025–26 raster products in the pipeline to ensure spatial alignment.

---

## Summary

| Parameter | Value |
|---|---|
| Study area | Hudson Bay (main body) |
| Bounding box (WGS84) | 95°W–75°W, 58°N–66°N (approximate; subject to minor refinement) |
| Analytical CRS | EPSG:3978 (NAD83 / Canada Atlas Lambert) |
| Output grid resolution | 320m (defined by Clay v1.5 patch footprint at Sentinel-1 EW ~40m GSD) |
| Temporal scope | October 2025 – January 2026 (freeze-up season) |
| Role | Prospective evaluation dataset (primary training is AI4Arctic 2018–2021) |
| Expected evaluation pairs | ~20–40 scene–chart pairs (to be verified against CDSE catalog) |
| Evaluation label source | USNIC weekly Arctic charts (bi-weekly cadence; CIS analysis for Canadian waters) |
| Sentinel-1 mode | EW GRD, HH+HV dual polarisation, NERSC noise-corrected |
