# Prescient Ice: Training Strategy

## Overview

Training a sea ice concentration model from SAR imagery presents a core challenge: the available labels — USNIC weekly ice charts — are polygon-based analyst products with spatial resolution on the order of tens of kilometres, while the target output is a continuous 500m gridded SIC field. Bridging this resolution gap requires careful label preparation that balances data quality against training volume. Separately, temporal misalignment between SAR acquisitions and chart publication dates introduces noise that degrades model performance, particularly during the dynamic transition seasons when accurate SIC estimates are most operationally valuable.

This document covers both concerns: how training labels are constructed from the available sources, and how SAR acquisitions are matched to those labels in time.

---

## Label Preparation

USNIC weekly SIGRID-3 ice charts are the primary training label source. Each chart covers the Arctic basin with polygon features, each assigned a total ice concentration value (0–10 in tenths, equivalent to 0–100%). Converting these polygon labels into 500m pixel-level training targets is the central label preparation problem. The approach is phased, starting from the cleanest possible labels and progressively expanding coverage.

### Phase 1 — Pure Cell Extraction

The 500m target grid is overlaid onto USNIC chart polygons. Only grid cells that fall entirely within a single polygon are retained; cells that straddle polygon boundaries are discarded. The polygon's concentration value is assigned as the cell label.

This produces clean, unambiguous labels: each training cell corresponds to exactly one analyst-assigned concentration value with no boundary ambiguity. The cost is reduced training volume — boundary cells, which may constitute a significant fraction of cells in regions with complex polygon geometry, are excluded. For the initial model, this trade-off is appropriate: foundation model feature extractors trained on large corpora benefit more from clean signal than from increased volume at the expense of label fidelity.

### Phase 2 — Area-Weighted Mixed Cells

The second phase expands the training set to include cells that intersect multiple polygons. For each such cell, an area-weighted average concentration is computed: if a cell is 60% covered by a polygon assigned 100% concentration and 40% covered by a polygon assigned 25% concentration, the resulting label is (0.6 × 1.0) + (0.4 × 0.25) = 0.70.

Whether to include these mixed-cell labels depends on whether the additional training volume improves or degrades performance relative to the Phase 1 baseline. The risk is that area-weighted labels encode analyst polygon delineation choices into pixel-level targets: if an analyst drew a boundary between two ice regimes slightly differently, the mixed-cell label for a specific cell changes, and the model sees a confidently wrong target. This effect should average out over a large training set, but individual cell labels can be misleading. Phase 2 should be evaluated empirically rather than adopted unconditionally.

### Phase 3 — ICESat-2 Anchor Points

ICESat-2 ATL07 (sea ice surface height) and ATL10 (sea ice freeboard) products provide physically-grounded, point-level observations along satellite ground tracks. Lead detections along the track indicate near-zero ice concentration; significant freeboard measurements indicate consolidated ice (near-100% concentration). Where ICESat-2 tracks are coincident with SAR acquisitions, these observations provide pixel-level labels at the extremes of the concentration spectrum.

These anchor points are integrated as additional training samples alongside the chart-derived labels. Their value is not in volume — ICESat-2 tracks cover a small fraction of any given scene — but in providing ground truth that is independent of analyst interpretation and physically calibrated. They are particularly useful for constraining model predictions at 0% and 100%, where SAR texture is most distinctive and where label noise from chart polygon generalisation is least tolerable.

ICESat-2 measurements have precise acquisition timestamps (UTC, sub-second), enabling tight temporal matching with SAR scenes. The coincidence window for anchor points can be tighter than for chart-based labels — a 2–4 hour window is reasonable, given that leads and consolidated ice surfaces are relatively stable on those timescales in winter conditions, and that longer windows risk using observations where the surface has changed between the two sensors.

### Future Consideration — Weak/Aggregate Label Training

A more sophisticated alternative to pixel-level label assignment is aggregate label training: rather than assigning a label to each pixel and minimising pixel-level loss, the model is trained such that the mean of its predictions across all cells within a given chart polygon approximates the polygon's stated concentration. This allows the model to learn sub-polygon spatial variability from SAR texture while remaining calibrated to chart-scale estimates.

The appeal is that this approach does not require the model to reproduce the analyst's polygon delineation decisions — only to produce predictions that are consistent with the chart at the polygon scale. The known failure mode is that a model with low SAR-to-SIC discriminative ability may converge to predicting the polygon mean everywhere within each polygon, which is not useful. This approach should be considered only after the Phase 1 and Phase 2 baselines are established and their limitations are understood.

---

## Temporal Alignment

USNIC weekly Arctic charts are published approximately once per week, with each chart nominally valid for a snapshot of conditions at the time of analysis. The analysis itself draws on SAR imagery, passive microwave, and analyst experience, but the precise SAR scenes used are not necessarily documented in the chart metadata. Matching a training SAR scene to a chart requires working from publication dates and nominal validity windows.

Ice conditions can change substantially within 24 hours in the marginal ice zone — particularly during freeze-up and break-up, when surface temperatures hover near freezing and wind-driven ice motion is active. Temporal misalignment between the SAR acquisition used for training and the chart that provides the label introduces systematic noise: the model may be trained to associate a particular SAR texture with a concentration that reflects ice conditions at a different time.

### Baseline Window

The baseline temporal alignment requirement is that the SAR acquisition date and the chart's nominal validity date are within 24 hours of one another. This is a coarse criterion that will filter out the worst misalignment cases while retaining a practical volume of training pairs.

### Season-Adaptive Refinement

A tighter, ERA5-informed window is planned as a refinement. ERA5 surface air temperature and 10m wind speed over the study area are used to characterise the dynamism of the ice environment at the time of each SAR acquisition:

- **Stable conditions** (deep winter, surface temperatures well below freezing, low winds): the 24-hour baseline window is acceptable. Ice surfaces are consolidated and change slowly; a chart from the previous day accurately represents the conditions seen in a SAR acquisition taken the following morning.
- **Dynamic conditions** (freeze-up and break-up periods, surface temperatures near 0°C, significant winds): the alignment window is tightened to 6–12 hours. The marginal ice zone during these periods can see substantial change in hours, and temporal misalignment is most damaging precisely where the SIC product is most operationally useful.

The ERA5-informed window is applied as a pre-processing step during training dataset assembly, not as a hard threshold: each training pair is tagged with the ERA5-derived dynamism score and temporal offset, and the tightening decision is made as a configurable parameter rather than hard-coded. This preserves flexibility to evaluate the effect of different alignment thresholds on model performance.

### Additional Alignment Considerations

**USNIC source imagery metadata.** If USNIC/CIS charts include metadata identifying which specific SAR acquisitions were used to derive each polygon, this enables filtering to cases where the chart and the training SAR scene are confirmed to be derived from the same acquisition. This would be a materially stronger form of alignment than date-based matching. Whether this metadata is available in the SIGRID-3 files is an open question to investigate.

**AMSR2 temporal averaging.** AMSR2 daily SIC composites inherently smooth over the day's acquisitions, introducing a form of temporal averaging that is distinct from the point-in-time SAR observation. Since AMSR2 is used as an ancillary input feature rather than a label, this averaging is acceptable — it provides a coarse physics-based prior that is robust to short-term surface changes — but it should be noted when interpreting model sensitivity to the AMSR2 feature.

**ICESat-2 temporal precision.** ICESat-2 measurements have precise UTC timestamps, enabling tight temporal matching with SAR acquisitions for the supplementary label pathway. A 2–4 hour coincidence window is appropriate, with the specific threshold treated as a parameter to tune.
