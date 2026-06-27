# Prescient Ice: Training Strategy

## Overview

Training a sea ice concentration classifier from SAR imagery presents a core challenge: the available labels — analyst-produced ice charts — are polygon-based products with spatial resolution on the order of tens of kilometres, while the target output is a 320m gridded SIC class field. Bridging this resolution gap requires careful label preparation that balances data quality against training volume. Separately, when the training pipeline relies on independent SAR acquisitions and chart products, temporal misalignment between them introduces noise that degrades performance, particularly during dynamic transition seasons.

The primary training and evaluation dataset for Prescient Ice is the AI4Arctic Sea Ice Challenge Dataset (Buus-Hinkler et al., 2022), which packages Sentinel-1 EW HH/HV SAR, co-registered AMSR2, ERA5 surface variables, and CIS/DMI chart labels together per scene. Because AI4Arctic delivers scene-to-label alignment pre-computed and validated by the dataset authors, the temporal alignment problem that dominates the CIS-chart-based 2025–26 pipeline does not arise for the primary training data. Temporal alignment becomes relevant again for the 2025–26 Hudson Bay prospective evaluation dataset, where Sentinel-1 acquisitions, CIS charts, AMSR2, and ERA5 are pulled from their respective providers and must be matched up by the project pipeline.

This document covers both concerns: how training labels are constructed from chart polygons under the eleven-class classification framing, and how SAR acquisitions are matched to charts in time for the parts of the pipeline where that matching is the project's responsibility.

---

## Dataset Splits

AI4Arctic ships as 533 scenes: 513 designated training scenes and 20 designated test scenes (the held-out set the original AutoICE challenge scored via submission to the AI4EO platform). Labels for the 20 test scenes were withheld during the challenge but have since been released, distributed as a separate file that joins to the rest of the scene data cube on the scene identifier. The 20 are therefore now a fully labelled, scoreable held-out set rather than a submission-only set.

The project uses a three-way split:

- **Train and validation** are drawn from the **513 labelled training scenes**, partitioned at the scene level (the split is a filter on `scene_id`, so no chip or patch crosses the boundary). Training fits the downstream classifiers; validation drives model selection across the feature-configuration × classifier sweep (B3) and is the set on which the modelling Phase 2 go/no-go trigger is evaluated.
- **Test** is the **20 held-out test scenes**, scored once on the finally selected model(s). This is the headline, leaderboard-comparable result.

The trigger-on-validation, report-on-test discipline is deliberate. If the modelling Phase 2 decision were made on the 20 and modelling Phase 2 then fired and was re-reported on the same 20, the test scenes would have informed a development choice and the final number would no longer be a clean held-out result. Making the trigger decision on validation keeps the test report genuinely held-out regardless of whether modelling Phase 2 is invoked. (Here and below, "modelling Phase 1/Phase 2" refers to the frozen-extractor versus fine-tuning strategy in `prescient_ice_model_architecture.md`.)

The 20 test scenes are the same scenes the published AutoICE leaderboard scored, so the project's test-set SIC-R² is directly comparable to the published distribution — with two caveats carried throughout: the comparison is against the **SIC-R² component** of the combined AutoICE metric (which also weights SOD and FLOE F1, parameters the project does not produce), and the project's 320m minimum mapping unit differs from the RTT-based submissions' scale (see `prescient_ice_datasets.md`).

Normalisation statistics (the NERSC σ⁰ mean and standard deviation that calibrate the custom Clay `sentinel-1-ew` metadata entry) are computed over the 513 train-and-validation scenes only; the 20 test scenes are excluded to avoid leakage of test-set statistics into the encoder's input normalisation. A strict train-only basis (excluding validation as well) would be marginally more conservative, but the train-plus-validation pool is the operative basis here and the leakage from including validation in a global mean/std is negligible.

The single assembly pass (B2) processes all 533 scenes and tags each with its split, so a downstream consumer selects train, validation, or test by filtering a `split` column rather than maintaining separate tables. The test scenes' labels are joined from their separate label file at load time (B2.1) so that label preparation runs identically across all three splits.

---

## Label Preparation

SIGRID-3 ice charts — both the CIS/DMI charts packaged in AI4Arctic and the CIS Hudson Bay weekly regional charts used for the 2025–26 prospective evaluation pipeline — assign each polygon a total ice concentration code (CT) from the discrete set 0, 1, 2, …, 10 (tenths). The 0–10 tenths scheme aligns directly with the project's eleven-class classification framing — no remapping is needed. Converting these polygon labels into 320m patch-level training targets is the central label preparation problem. A single method is used: each cell's class is the area-weighted average of the concentrations of the polygons it intersects, rounded to the nearest tenth. Cells that fall entirely within one polygon — *pure cells* — are the degenerate case of this method, taking that polygon's class directly; cells that straddle a boundary — *mixed cells* — take the weighted average. Every labelled cell therefore receives a discrete class at assembly time, and a stored `is_pure` flag records which cells are pure, so that training on pure cells alone or on the full set is a read-time choice rather than a re-derivation (see § Label Storage on the Feature Table).

### Area-Weighted Cell Labelling

The 320m target grid (defined by the Clay patch footprint at EW ~40m GSD; see `prescient_ice_study_area.md`) is overlaid onto chart polygons, and each cell is assigned the area-weighted average of the CT codes of the polygons it intersects, rounded to the nearest tenth to produce a discrete class. A cell falling entirely within a single polygon is the degenerate case: the weighted average returns that polygon's class exactly, so pure cells are labelled directly with no boundary ambiguity. A cell straddling a boundary is a mixed cell, and the weighting resolves its competing polygon classes into one label.

A worked example: a cell is 50% covered by a polygon with CT code 7 and 50% covered by a polygon with CT code 1. The midpoint-weighted fraction is (0.5 × 0.7) + (0.5 × 0.1) = 0.4, which rounds to class 4. The cell is assigned class 4 as its training label.

SIGRID-3 CT codes can include range codes (e.g. a polygon labelled "5–7") expressing analyst uncertainty about which specific tenth applies. These are resolved to their midpoint (6 in this example) before area-weighting.

A known edge case: cells that are overwhelmingly open water with a small ice presence may round to class 0 — for example, a cell 90% in a CT=0 polygon and 10% in a CT=8 polygon yields (0.9 × 0) + (0.1 × 0.8) = 0.08, which rounds to class 0 despite the small ice fraction. This is documented as expected behaviour rather than a bug. The cell is mostly open water; treating it as class 0 reflects that. The alternative — preserving the small ice fraction by rounding up — would create a class label that significantly overstates the cell's ice content.

Pure and mixed cells differ in label confidence, not only in composition. Pure-cell labels are unambiguous: each corresponds to exactly one analyst-assigned class. Mixed-cell labels carry a known risk — area-weighting encodes analyst polygon delineation choices into cell-level targets, so if an analyst drew a boundary between two ice regimes slightly differently, the mixed-cell label for a specific cell changes and the model may see a confidently wrong target. This averages out over a large training set, but individual cell labels can be misleading. Whether mixed cells help or hurt patch-level performance — clean signal versus greater volume — is therefore an empirical question, evaluated in B3 by filtering on the stored `is_pure` flag rather than settled here. Foundation-model feature extractors often benefit more from clean signal than from volume at the expense of label fidelity, and AI4Arctic's 513 training scenes substantially reduce the data-volume pressure that would otherwise push towards admitting every mixed cell; the pure-only configuration is thus a natural first comparison.

### Label Storage on the Feature Table

The feature table does not store only the final discrete label. For each patch it stores the full per-class area-fraction vector — `frac_sic0` through `frac_sic10`, each the fraction of the patch's *valid-class* pixels assigned to that class, so the eleven values sum to 1.0 over the labelled area — alongside a separate `valid_class_fraction` recording how much of the patch carries a valid class at all (pixels outside any polygon, or in unknown/not-filled/glacier codes, are excluded from the valid-class count). The discrete `label` column is derived from this vector by the area-weighted collapse (Σ class × fraction, rounded) and is populated for every patch, pure and mixed alike. A boolean `is_pure` column records whether the patch is pure — one class at fraction 1.0 over its valid-class pixels — so that the pure-only-versus-all-patches training choice is a read-time filter, not a re-derivation. Purity is defined over valid-class pixels and is therefore orthogonal to coverage: a patch can be pure yet largely nodata or land, with that low coverage captured separately by `valid_class_fraction` and the SAR `valid_fraction`.

This storage choice decouples the expensive operation from the revisable one. Rasterising chart polygons onto the patch grid and histogramming each patch by class is the full-corpus pass, done once. Collapsing the fraction vector to a discrete label is cheap arithmetic. Because the fraction vector is retained, the discrete `label` is always re-derivable: revising the mixed-cell labelling rule — to a different weighting, or to admit or exclude particular cells — is a vectorised in-place column update over the stored fractions, never a re-rasterisation of the corpus or a re-encode of embeddings. The area-weighted midpoint rule is the committed default; alternative mixed-label rules remain a cheap read-time experiment.

It also separates the two distinct notions of patch coverage that the table tracks. The SAR `valid_fraction` (non-nodata, non-land source pixels) governs whether a patch is usable as a *feature*; `valid_class_fraction` governs how completely a patch is *labelled*. A patch can be fully valid as a feature yet only partially labelled, or vice versa, and the two are filtered independently by downstream consumers. Because the class fractions are normalised over valid-class pixels, two patches with very different label coverage can share an identical fraction vector — the vector describes the composition of the labelled area, and `valid_class_fraction` separately records its extent; together they are lossless. Storing the distribution rather than only a collapsed label is also the prerequisite for the chip-level aggregation below and for the weak/aggregate-label strategy noted thereafter, both of which consume per-patch class distributions directly.

### Chip-Level Label Aggregation

If patch-level prediction proves unsatisfactory, the project's first fallback — ahead of any fine-tuning — is chip-level prediction at the ~10 km chip grain (see `prescient_ice_model_architecture.md` § Feature Configurations and § Chip-Level Prediction Fallback). Chip-level labels are derived from the same stored fraction vectors, not from a second rasterisation pass: each patch's expected concentration (Σ class × fraction) is computed from its fraction vector, these are averaged across the chip's patches, and the result is rounded once to a discrete chip class. The single rounding preserves sub-patch detail that aggregating the already-rounded discrete `label` would discard, and because the fraction vector is populated for every patch the aggregate is well-defined even over patches a consumer would have filtered out at the patch grain.

This is distinct from the weak/aggregate-label training described next. Chip-level prediction trains a separate classifier on aggregated chip labels at a coarser output resolution; weak/aggregate-label training is a patch-grain loss formulation that keeps the 320m output but calibrates predictions to polygon-scale concentrations. The chip-level aggregation rule is documented now; its implementation is deferred until patch-level results show whether it is needed.

### Future Consideration — Weak/Aggregate Label Training

A more sophisticated alternative to pixel-level label assignment is aggregate label training: rather than assigning a class to each pixel and minimising pixel-level loss, the model is trained such that the distribution of its predictions across all cells within a given chart polygon approximates the polygon's stated concentration. This allows the model to learn sub-polygon spatial variability from SAR texture while remaining calibrated to chart-scale estimates.

The appeal is that this approach does not require the model to reproduce the analyst's polygon delineation decisions — only to produce predictions that are consistent with the chart at the polygon scale. The known failure mode is that a model with low SAR-to-SIC discriminative ability may converge to predicting the polygon mode everywhere within each polygon, which is not useful. This approach should be considered only after the patch-level and chip-level baselines are established and their limitations are understood.

---

## Training Data Consistency: NERSC Noise Correction

AI4Arctic distributes Sentinel-1 EW data with the NERSC noise correction (Korosov et al., 2022) applied — the only correction supplied; both raw and RTT inherit it (see `prescient_ice_datasets.md`). The NERSC correction removes residual scalloping and incidence-angle effects that the standard ESA correction does not fully eliminate, and these artefacts matter most on the HV channel — the channel most informative for ice/water discrimination. The same NERSC noise correction must be applied to Sentinel-1 EW scenes acquired for the 2025–26 prospective evaluation and operational inference. If training inputs are NERSC-corrected and inference inputs are not, the model sees a different input distribution at inference time than it was trained on, with a systematic effect concentrated on the HV channel.

The NERSC noise correction must therefore be incorporated as a standard preprocessing step in the Sentinel-1 ingestion pipeline (see `prescient_ice_pipeline_architecture.md`). Normalisation statistics for Clay's custom `sentinel-1-ew` metadata entry should also be derived from NERSC-corrected data to maintain consistency end to end.

---

## Temporal Alignment

The temporal alignment problem applies to the 2025–26 Hudson Bay pipeline, where Sentinel-1 scenes, CIS charts, AMSR2, and ERA5 are acquired independently from their providers and must be matched up by the project. It does not apply to AI4Arctic training, where the dataset authors have already paired SAR scenes with chart labels and validated the alignment.

CIS Hudson Bay weekly regional charts are published approximately once per week, with each chart nominally valid for a snapshot of conditions at the time of analysis. The analysis draws on SAR imagery, passive microwave, and analyst experience, but the precise SAR scenes used are not necessarily documented in the chart metadata. Matching a 2025–26 Sentinel-1 scene to a chart requires working from publication dates and nominal validity windows.

Ice conditions can change substantially within 24 hours in the marginal ice zone — particularly during freeze-up and break-up, when surface temperatures hover near freezing and wind-driven ice motion is active. Temporal misalignment between a SAR acquisition and the chart used to label or evaluate against it introduces systematic noise: the model is evaluated against a chart that reflects ice conditions at a different time, which can show up as model error that is in fact alignment error.

### Baseline Window

The baseline temporal alignment requirement is that the SAR acquisition date and the chart's nominal validity date are within 24 hours of one another. This is a coarse criterion that will filter out the worst misalignment cases while retaining a practical volume of evaluation pairs.

### Season-Adaptive Refinement

A tighter, ERA5-informed window is planned as a refinement. ERA5 surface air temperature and 10m wind speed over the study area are used to characterise the dynamism of the ice environment at the time of each SAR acquisition:

- **Stable conditions** (deep winter, surface temperatures well below freezing, low winds): the 24-hour baseline window is acceptable. Ice surfaces are consolidated and change slowly; a chart from the previous day accurately represents the conditions seen in a SAR acquisition taken the following morning.
- **Dynamic conditions** (freeze-up and break-up periods, surface temperatures near 0°C, significant winds): the alignment window is tightened to 6–12 hours. The marginal ice zone during these periods can see substantial change in hours, and temporal misalignment is most damaging precisely where the SIC product is most operationally useful.

The ERA5-informed window is applied as a pre-processing step during evaluation dataset assembly, not as a hard threshold: each (SAR, chart) pair is tagged with the ERA5-derived dynamism score and temporal offset, and the tightening decision is made as a configurable parameter rather than hard-coded. This preserves flexibility to evaluate the effect of different alignment thresholds on prospective evaluation outcomes.

### Additional Alignment Considerations

**CIS source imagery metadata.** If CIS charts include metadata identifying which specific SAR acquisitions were used to derive each polygon, this enables filtering to cases where the chart and the Sentinel-1 scene are confirmed to be derived from the same acquisition. This would be a materially stronger form of alignment than date-based matching. Whether this metadata is available in the SIGRID-3 files is an open question to investigate.

**AMSR2 temporal matching.** Each scene is matched to the AMSR2 acquisition closest in time to the SAR acquisition, within a bounded window (the AI4Arctic training bundle uses a seven-hour window). Since AMSR2 is an ancillary input feature rather than a label, a small time offset is acceptable — it provides coarse, physically-grounded context robust to short-term surface change — but the matching window should be kept consistent between the training and inference paths so the feature is constructed comparably on both.

---

## ICESat-2 — Role Change

ICESat-2 was previously planned as a Phase 3 supplementary label source for primary model training, with ATL07 lead detections and ATL10 freeboard measurements providing physically-grounded anchor points at the extremes of the concentration spectrum (near-0% at leads, near-100% at consolidated pack). Under the classification framing, these would map to class 0 and class 10 cleanly.

With AI4Arctic as the primary training source (513 scenes), the anchor-point motivation — boosting label confidence at the concentration extremes in a small training set — is materially weakened. ICESat-2's role in the project is accordingly revised:

1. **Prospective validation layer in the visualisation.** ICESat-2 tracks overlaid on SIC output in the MapLibre viewer provide an independent, physically-grounded reference for analysts reviewing model predictions. Where a track crosses a model-predicted class-10 region and ATL10 shows significant freeboard, the prediction is corroborated; disagreements flag investigation. This is the primary value of ICESat-2 in the project as currently scoped.
2. **Candidate supplementary label source for 2025–26 retraining.** If prospective evaluation on the 2025–26 Hudson Bay dataset reveals shortcomings that warrant incorporating that season's data into a retraining run, and if the labelled volume from CIS charts proves modest, ICESat-2 anchor points become attractive as supplementary training labels in that retraining configuration. The integration would follow the original Phase 3 plan: lead detections → class 0, consolidated freeboard → class 10, tight temporal coincidence window (2–4 hours, given ICESat-2's UTC-precise timestamps).

ICESat-2 is still ingested into Prescient as a STAC collection (`icesat2-tracks`) with the dual-asset pattern (GeoParquet `data` for analytical use, PMTiles `visual` for display). The Prescient ingestion pipeline does not change with the role revision — only the use of the ingested data in the analytics pipeline changes.

---

## Experiment Tracking

The training strategy involves multiple feature configurations and model types, with further iteration likely on label rasterisation, temporal alignment windows, and hyperparameters. Tracking these experiments systematically — rather than relying on ad-hoc notes and spreadsheets — is worthwhile from the outset. MLflow is the chosen tool.

**What MLflow captures.** Each training run logs parameters (feature configuration, model type, hyperparameters), metrics (validation accuracy, per-class precision/recall, confusion matrices), code version (git commit), and artefacts (the trained model, feature importances, evaluation plots). Autologging integrations for scikit-learn (Random Forest) and XGBoost capture most of this with minimal code changes — typically a `mlflow.autolog()` call and a `with mlflow.start_run()` context manager around training.

**Deployment.** If training runs occur in Amazon SageMaker (the likely environment from early in the project), SageMaker's fully managed MLflow capability is the natural choice. Tracking server compute and metadata storage are hosted in the SageMaker service account, with artefacts stored in an S3 bucket in our own AWS account; setup is a few-clicks operation through SageMaker Studio with minimal ops overhead. The serverless "MLflow Apps" variant scales automatically and is the current preferred offering. If training ever needs to happen outside SageMaker, MLflow's local-file mode remains a viable fallback for individual development.

**Relationship to the pipeline architecture.** The trained model artefact lives in S3, as already documented in the pipeline architecture. MLflow's model registry tracks model versions and lineage during development; the inference pipeline references the deployed model by its S3 path or MLflow URI. STAC catalogs the data and the model outputs (SIC class grids in the `sic-output` collection), not the model itself — STAC and MLflow have clearly separated responsibilities. A useful integration: models registered in SageMaker's managed MLflow automatically appear in the SageMaker Model Registry, unifying experiment tracking with model deployment metadata.

**What this is not.** MLflow is not a replacement for version control of code (git) or data (the STAC catalog plus AI4Arctic's fixed release versioning). It complements both by linking each training run to a specific code state and data inputs, making any historical run reproducible from its logged metadata.

---

## Open Questions

Open questions and their current status are tracked in `prescient_ice_index.md` § Open Questions and Risks.
