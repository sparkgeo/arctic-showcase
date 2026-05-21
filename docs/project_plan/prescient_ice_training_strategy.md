# Prescient Ice: Training Strategy

## Overview

Training a sea ice concentration classifier from SAR imagery presents a core challenge: the available labels — analyst-produced ice charts — are polygon-based products with spatial resolution on the order of tens of kilometres, while the target output is a 320m gridded SIC class field. Bridging this resolution gap requires careful label preparation that balances data quality against training volume. Separately, when the training pipeline relies on independent SAR acquisitions and chart products, temporal misalignment between them introduces noise that degrades performance, particularly during dynamic transition seasons.

The primary training and evaluation dataset for Prescient Ice is the AI4Arctic Sea Ice Challenge Dataset (Buus-Hinkler et al., 2022), which packages Sentinel-1 EW HH/HV SAR, co-registered AMSR2, ERA5 surface variables, and CIS/DMI chart labels together per scene. Because AI4Arctic delivers scene-to-label alignment pre-computed and validated by the dataset authors, the temporal alignment problem that dominates the USNIC-based pipeline does not arise for the primary training data. Temporal alignment becomes relevant again for the 2025–26 Hudson Bay prospective evaluation dataset, where Sentinel-1 acquisitions, USNIC charts, AMSR2, and ERA5 are pulled from their respective providers and must be matched up by the project pipeline.

This document covers both concerns: how training labels are constructed from chart polygons under the eleven-class classification framing, and how SAR acquisitions are matched to charts in time for the parts of the pipeline where that matching is the project's responsibility.

---

## Label Preparation

SIGRID-3 ice charts — both the CIS/DMI charts packaged in AI4Arctic and the USNIC weekly Arctic charts used for the 2025–26 prospective evaluation pipeline — assign each polygon a total ice concentration code (CT) from the discrete set 0, 1, 2, …, 10 (tenths). The 0–10 tenths scheme aligns directly with the project's eleven-class classification framing — no remapping is needed. Converting these polygon labels into 320m patch-level training targets is the central label preparation problem. The approach is phased, starting from the cleanest possible labels and progressively expanding coverage.

### Phase 1 — Pure Cell Extraction

The 320m target grid (defined by the Clay patch footprint at EW ~40m GSD; see `prescient_ice_study_area.md`) is overlaid onto chart polygons. Only grid cells that fall entirely within a single polygon are retained; cells that straddle polygon boundaries are discarded. The polygon's CT code is assigned as the cell's class label.

This produces clean, unambiguous labels: each training cell corresponds to exactly one analyst-assigned class with no boundary ambiguity. The cost is reduced training volume — boundary cells, which may constitute a significant fraction of cells in regions with complex polygon geometry, are excluded. For the initial model, this trade-off is appropriate: foundation model feature extractors trained on large corpora benefit more from clean signal than from increased volume at the expense of label fidelity. AI4Arctic's 513 training scenes substantially reduce the data-volume pressure that would otherwise push toward more permissive label preparation.

### Phase 2 — Area-Weighted Mixed Cells

The second phase expands the training set to include cells that intersect multiple polygons. Under the classification framing, the area-weighting computation uses class midpoints as proxy fractions, with the resulting weighted average rounded to the nearest tenth to produce a discrete class label.

A worked example: a cell is 50% covered by a polygon with CT code 7 and 50% covered by a polygon with CT code 1. The midpoint-weighted fraction is (0.5 × 0.7) + (0.5 × 0.1) = 0.4, which rounds to class 4. The cell is assigned class 4 as its training label.

SIGRID-3 CT codes can include range codes (e.g. a polygon labelled "5–7") expressing analyst uncertainty about which specific tenth applies. These are resolved to their midpoint (6 in this example) before area-weighting.

A known edge case: cells that are overwhelmingly open water with a small ice presence may round to class 0 — for example, a cell 90% in a CT=0 polygon and 10% in a CT=8 polygon yields (0.9 × 0) + (0.1 × 0.8) = 0.08, which rounds to class 0 despite the small ice fraction. This is documented as expected behaviour rather than a bug. The cell is mostly open water; treating it as class 0 reflects that. The alternative — preserving the small ice fraction by rounding up — would create a class label that significantly overstates the cell's ice content.

Whether to include these mixed-cell labels depends on whether the additional training volume improves or degrades performance relative to the Phase 1 baseline. The risk is that area-weighted labels encode analyst polygon delineation choices into pixel-level targets: if an analyst drew a boundary between two ice regimes slightly differently, the mixed-cell label for a specific cell changes, and the model sees a confidently wrong target. This effect should average out over a large training set, but individual cell labels can be misleading. Phase 2 should be evaluated empirically rather than adopted unconditionally.

### Future Consideration — Weak/Aggregate Label Training

A more sophisticated alternative to pixel-level label assignment is aggregate label training: rather than assigning a class to each pixel and minimising pixel-level loss, the model is trained such that the distribution of its predictions across all cells within a given chart polygon approximates the polygon's stated concentration. This allows the model to learn sub-polygon spatial variability from SAR texture while remaining calibrated to chart-scale estimates.

The appeal is that this approach does not require the model to reproduce the analyst's polygon delineation decisions — only to produce predictions that are consistent with the chart at the polygon scale. The known failure mode is that a model with low SAR-to-SIC discriminative ability may converge to predicting the polygon mode everywhere within each polygon, which is not useful. This approach should be considered only after the Phase 1 and Phase 2 baselines are established and their limitations are understood.

---

## Training Data Consistency: NERSC Noise Correction

AI4Arctic provides Sentinel-1 EW data in two noise-corrected versions: the ESA-applied correction and an additional NERSC noise correction. The NERSC version is the project's preferred input for training because the additional correction removes residual scalloping and incidence-angle effects that can otherwise bias HV-channel statistics. The same NERSC noise correction must be applied to Sentinel-1 EW scenes acquired for the 2025–26 prospective evaluation and operational inference. If training inputs are NERSC-corrected and inference inputs are not, the model sees a different input distribution at inference time than it was trained on, with a systematic effect on the HV channel that is exactly the channel most informative for ice/water discrimination.

The NERSC noise correction must therefore be incorporated as a standard preprocessing step in the Sentinel-1 ingestion pipeline (see `prescient_ice_pipeline_architecture.md`). Normalisation statistics for Clay's custom `sentinel-1-ew` metadata entry should also be derived from NERSC-corrected data to maintain consistency end to end.

---

## Temporal Alignment

The temporal alignment problem applies to the 2025–26 Hudson Bay pipeline, where Sentinel-1 scenes, USNIC charts, AMSR2, and ERA5 are acquired independently from their providers and must be matched up by the project. It does not apply to AI4Arctic training, where the dataset authors have already paired SAR scenes with chart labels and validated the alignment.

USNIC weekly Arctic charts are published approximately once per week, with each chart nominally valid for a snapshot of conditions at the time of analysis. The analysis draws on SAR imagery, passive microwave, and analyst experience, but the precise SAR scenes used are not necessarily documented in the chart metadata. Matching a 2025–26 Sentinel-1 scene to a chart requires working from publication dates and nominal validity windows.

Ice conditions can change substantially within 24 hours in the marginal ice zone — particularly during freeze-up and break-up, when surface temperatures hover near freezing and wind-driven ice motion is active. Temporal misalignment between a SAR acquisition and the chart used to label or evaluate against it introduces systematic noise: the model is evaluated against a chart that reflects ice conditions at a different time, which can show up as model error that is in fact alignment error.

### Baseline Window

The baseline temporal alignment requirement is that the SAR acquisition date and the chart's nominal validity date are within 24 hours of one another. This is a coarse criterion that will filter out the worst misalignment cases while retaining a practical volume of evaluation pairs.

### Season-Adaptive Refinement

A tighter, ERA5-informed window is planned as a refinement. ERA5 surface air temperature and 10m wind speed over the study area are used to characterise the dynamism of the ice environment at the time of each SAR acquisition:

- **Stable conditions** (deep winter, surface temperatures well below freezing, low winds): the 24-hour baseline window is acceptable. Ice surfaces are consolidated and change slowly; a chart from the previous day accurately represents the conditions seen in a SAR acquisition taken the following morning.
- **Dynamic conditions** (freeze-up and break-up periods, surface temperatures near 0°C, significant winds): the alignment window is tightened to 6–12 hours. The marginal ice zone during these periods can see substantial change in hours, and temporal misalignment is most damaging precisely where the SIC product is most operationally useful.

The ERA5-informed window is applied as a pre-processing step during evaluation dataset assembly, not as a hard threshold: each (SAR, chart) pair is tagged with the ERA5-derived dynamism score and temporal offset, and the tightening decision is made as a configurable parameter rather than hard-coded. This preserves flexibility to evaluate the effect of different alignment thresholds on prospective evaluation outcomes.

### Additional Alignment Considerations

**USNIC source imagery metadata.** If USNIC charts include metadata identifying which specific SAR acquisitions were used to derive each polygon, this enables filtering to cases where the chart and the Sentinel-1 scene are confirmed to be derived from the same acquisition. This would be a materially stronger form of alignment than date-based matching. Whether this metadata is available in the SIGRID-3 files is an open question to investigate.

**AMSR2 temporal averaging.** AMSR2 daily SIC composites inherently smooth over the day's acquisitions, introducing a form of temporal averaging that is distinct from the point-in-time SAR observation. Since AMSR2 is used as an ancillary input feature rather than a label, this averaging is acceptable — it provides a coarse physics-based prior that is robust to short-term surface changes — but it should be noted when interpreting model sensitivity to the AMSR2 feature.

---

## ICESat-2 — Role Change

ICESat-2 was previously planned as a Phase 3 supplementary label source for primary model training, with ATL07 lead detections and ATL10 freeboard measurements providing physically-grounded anchor points at the extremes of the concentration spectrum (near-0% at leads, near-100% at consolidated pack). Under the classification framing, these would map to class 0 and class 10 cleanly.

With AI4Arctic as the primary training source (513 scenes), the anchor-point motivation — boosting label confidence at the concentration extremes in a small training set — is materially weakened. ICESat-2's role in the project is accordingly revised:

1. **Prospective validation layer in the visualisation.** ICESat-2 tracks overlaid on SIC output in the MapLibre viewer provide an independent, physically-grounded reference for analysts reviewing model predictions. Where a track crosses a model-predicted class-10 region and ATL10 shows significant freeboard, the prediction is corroborated; disagreements flag investigation. This is the primary value of ICESat-2 in the project as currently scoped.
2. **Candidate supplementary label source for 2025–26 retraining.** If prospective evaluation on the 2025–26 Hudson Bay dataset reveals shortcomings that warrant incorporating that season's data into a retraining run, and if the labelled volume from USNIC charts proves modest, ICESat-2 anchor points become attractive as supplementary training labels in that retraining configuration. The integration would follow the original Phase 3 plan: lead detections → class 0, consolidated freeboard → class 10, tight temporal coincidence window (2–4 hours, given ICESat-2's UTC-precise timestamps).

ICESat-2 is still ingested into Prescient as a STAC collection (`icesat2-tracks`) with the dual-asset pattern (GeoParquet `data` for analytical use, PMTiles `visual` for display). The Prescient ingestion pipeline does not change with the role revision — only the use of the ingested data in the analytics pipeline changes.

---

## Open Questions

- **USNIC source imagery metadata.** Whether SIGRID-3 files contain metadata identifying the SAR acquisitions used to derive each polygon. If available, this enables a materially stronger form of (SAR, chart) alignment for the 2025–26 evaluation pipeline than date-based matching.
- **Hudson Bay subset of AI4Arctic.** AI4Arctic spans the Canadian and Greenlandic Arctic, of which Hudson Bay is a subset. A Hudson Bay-specific subset of the training data should be identified and used for domain-specific validation alongside the full pan-Arctic evaluation, to confirm performance on the project's target region specifically.
- **2025–26 chart source choice.** USNIC weekly Arctic charts are the assumed label source for 2025–26 prospective evaluation. Whether CIS regional charts for Hudson Bay (which feed into USNIC) offer additional temporal resolution or detail worth pursuing directly is worth investigating, though the USNIC-CIS overlap means parallel ingestion is likely not warranted.
