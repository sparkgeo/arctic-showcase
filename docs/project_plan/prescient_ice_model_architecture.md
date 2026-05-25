# Prescient Ice: Model Architecture

## Overview

The model architecture is structured around Clay v1.5, a geospatial foundation model pre-trained on multi-sensor earth observation imagery. Rather than training a SIC model from scratch — which would require large volumes of high-quality labeled data — the approach leverages Clay's pre-trained representations as a starting point. Clay has already learned to encode SAR texture, surface structure, and spatial context from self-supervised training on large EO datasets; the downstream task is to learn the mapping from those representations to discrete ice concentration classes.

The task is framed as eleven-class classification on the 0–10 tenths concentration scheme (0 = open water, 10 = full ice cover), matching the SIGRID-3 total concentration code (CT) directly and aligning with the AutoICE / AI4Arctic benchmark. This reflects the actual discrete nature of analyst-assigned chart labels and avoids the artificial precision of a continuous regression target derived from inherently discrete source data.

The architecture follows a two-phase strategy. Phase 1 uses Clay as a frozen feature extractor with no weight updates to the foundation model, and trains lightweight downstream classifiers (Random Forest and XGBoost in parallel) on the resulting features. Phase 2 — end-to-end fine-tuning — is pursued only if Phase 1 performance falls materially short of a defined threshold, or if Phase 1 does not meaningfully outperform a non-embedding SAR baseline. This structure limits unnecessary compute and complexity while providing clear decision points for escalation.

---

## Phase 1 — Clay as Frozen Feature Extractor

Phase 1 is the primary approach. Clay's encoder is loaded with its pre-trained weights and held frozen throughout; no gradients flow back into the foundation model during training. A simpler supervised classifier is trained on top of the frozen features to predict per-patch ice concentration class.

### Clay v1.5 Input Specification

Clay v1.5 expects input chips of 256 × 256 pixels with a patch size of 8 × 8 pixels. The encoder produces a 32 × 32 grid of patch tokens per chip, each token a 1024-dimensional embedding vector representing the content of one 8 × 8 pixel patch. At the Sentinel-1 EW native ground sampling distance of approximately 40m, this means:

- Each 256 × 256 chip covers approximately 10.2 km × 10.2 km on the ground.
- Each 8 × 8 patch covers approximately 320m × 320m on the ground.
- The 32 × 32 token grid is spatially registered to the chip footprint, with each token corresponding to a known 320m × 320m cell.

Clay's expected inputs are a dict containing: a pixel tensor `pixels` of shape `[batch, bands, height, width]`, a wavelengths tensor `waves` (μm) describing the spectral content of each band, a temporal encoding `time` of shape `[batch, 4]` carrying sin/cos pairs for week-of-year and hour-of-day, a spatial encoding `latlon` of shape `[batch, 4]` carrying sin/cos pairs for latitude and longitude, and a scalar `gsd` tensor in metres. Zeros are acceptable for `time` and `latlon` where temporal or spatial position is not used. Normalisation is per-band `(pixels - mean) / std` using values supplied in the model's metadata configuration. Clay is sensor-agnostic and accepts custom sensor metadata entries with custom band names, GSD, wavelengths, and normalisation statistics.

### EW HH/HV Mismatch and Custom Metadata

Clay v1.5's built-in `sentinel-1-rtc` platform entry is calibrated for Sentinel-1 IW RTC VV/VH data. Prescient Ice uses Sentinel-1 EW GRD HH/HV imagery — a different acquisition mode and polarisation pair, packaged via the AI4Arctic dataset for training and acquired directly from CDSE for prospective evaluation and inference. Using Clay's built-in metadata directly would apply IW-derived normalisation to EW data, mis-scale the position encoding via incorrect GSD, and feed mis-named bands to the model.

The mitigation is a custom `sentinel-1-ew` metadata entry in Clay's metadata format, specifying:

- **Band names:** `hh` and `hv`, rather than the built-in `vv` and `vh`.
- **GSD:** approximately 40m, matching EW GRD native resolution. This is the most important field for getting Clay's GSD-scaled positional encoding correctly calibrated for the data.
- **Wavelengths:** nominal C-band SAR wavelength placeholders, consistent with how Clay handles SAR wavelength metadata for its built-in entries.
- **Normalisation mean and std:** computed from study area data, or derived from the AI4Arctic dataset's NERSC noise-corrected scenes. The NERSC-corrected version is preferred because the same noise correction is applied consistently across training and inference inputs.

The same NERSC noise correction must be applied to Sentinel-1 EW scenes used for prospective evaluation and inference to avoid systematic input distribution shift between training and inference data. See `prescient_ice_datasets.md` for details on the AI4Arctic NERSC noise correction and its operational implications.

### Patch Token Extraction

Clay's encoder output is a `[batch, sequence, dimension]` tensor with sequence length 1025 and embedding dimension 1024. The first element of the sequence dimension is a learned **class token** that attends to all patch tokens during encoding and serves as a global chip-level representation; the remaining 1024 elements are the patch tokens themselves, one per 8 × 8 pixel patch. Standard Clay usage often takes only the class token as a chip-level embedding for downstream tasks, discarding the per-patch tokens. For Prescient Ice, both are extracted: the class token serves as the chip embedding in Feature Configuration 3, and the 1024 patch tokens are reshaped into the 32 × 32 spatial grid that gives the downstream classifier per-patch features.

The motivation is spatial fidelity. A single 10.2 km × 10.2 km chip in Hudson Bay during freeze-up can contain large gradients in ice concentration — for example, a chip straddling the ice edge may include open water, new ice, and consolidated first-year ice within its footprint. A single chip-level embedding compresses these signatures into one representation. The patch tokens preserve the within-chip spatial structure, giving the downstream classifier approximately 1,024 training samples per chip (one per 320m cell) rather than one.

Each patch token is spatially registered to its ~320m footprint in EPSG:3978 and paired with the SIC class label of the chart polygon it falls within. Mixed-polygon patches and the label preparation strategy are detailed in `prescient_ice_training_strategy.md`.

### Feature Configurations

Three parallel feature configurations are evaluated. The intent is to isolate two distinct questions: does Clay add value over raw backscatter features, and does chip-level context add value over patch tokens alone.

**Configuration 1 — Raw HH/HV backscatter baseline.** Per-patch statistics computed from the underlying SAR pixels: mean HH, mean HV, standard deviation HH, standard deviation HV, and HV/HH ratio. Five-dimensional feature vector per patch. This is the non-embedding baseline against which Clay's contribution is measured. If Phase 1 with Clay embeddings does not meaningfully outperform this configuration, the frozen-extractor approach is not adding value and Phase 2 is triggered.

**Configuration 2 — Patch tokens alone.** The 1024-dimensional Clay patch token vector for each patch. This is the core test of the foundation model approach: whether Clay's pre-trained representations of SAR texture and local spatial context are informative for SIC classification.

**Configuration 3 — Patch tokens + chip embedding.** The 1024-dimensional patch token concatenated with the chip-level class token from Clay's encoder output (also 1024-dimensional). Total feature dimension 2,048, with equal local/global weighting. The chip embedding is identical for all 1,024 patch tokens within a chip, so it adds no within-chip discriminative power; its value is in contextualising patches across chips and scenes. This is most likely to matter in the ice edge regime, where two patches with similar local texture may correspond to different concentration classes depending on their broader spatial context — pack interior vs marginal ice zone, proximity to fast ice, prevailing ice setting.

In all three configurations, AMSR2 SIC and ERA5 surface variables (2m air temperature, 10m wind components, mean sea level pressure) are appended to the feature vector after Clay encoding, providing physics-based ancillary context. AMSR2 contributes a coarse but physically-grounded SIC prior; ERA5 contributes thermodynamic and atmospheric context that helps disambiguate SAR signatures (e.g., wind-roughened open water producing high backscatter that can otherwise mimic new ice). The ancillary features are appended in all configurations including the raw baseline, so the comparison isolates the contribution of Clay's features rather than the ancillary inputs.

Implementation cost for Configuration 3 over Configuration 2 is negligible — the class token is the first element (index 0) of the same `[batch, 1025, 1024]` encoder output that already contains the patch tokens — so both Clay-based configurations can be evaluated in parallel without significant additional cost.

### Downstream Classifiers

Two classifiers are evaluated in parallel on the same feature configurations: Random Forest and XGBoost. Both handle tabular feature inputs natively, produce interpretable feature importance outputs, run on CPU, and generalise well with moderate hyperparameter sensitivity. XGBoost tends to outperform Random Forest in tabular benchmarks but requires more careful regularisation; running both lets the comparison be empirical rather than assumed.

The task is eleven-class ordinal classification (classes 0 through 10). Ordinality matters: confusing class 8 with class 9 is a smaller error than confusing class 0 with class 9, and the evaluation framework reflects this.

**Primary evaluation metric:** R² computed on the 0–10 integer class scale, matching the AutoICE challenge metric directly. This makes Phase 1 results directly comparable to the published AutoICE leaderboard.

**Secondary evaluation metric:** An ordinal penalty metric weighting misclassifications by their class distance from the true class. Misclassifying class 5 as class 6 incurs a small penalty; misclassifying class 5 as class 10 incurs a much larger one. This complements R² by giving a more directly interpretable summary of how far the model's mistakes are from the truth.

Open question: whether to evaluate all three feature configurations against both classifiers (six combinations) or to narrow on one axis first. This depends on the implementation difficulty of swapping feature configuration and classifier in the training loop. If both are trivial config flips, running all six is the cleanest evaluation. If either dimension requires meaningful per-variant rework, fixing one (e.g., evaluating all three feature configurations under XGBoost first, then revisiting RF for the winning configuration) is the more practical path.

### Embedding Store as a Derived Product

Generated patch token grids are not discarded after training. They are serialised as COGs with the 32 × 32 token grid as the spatial structure of each chip's output, and ingested into Prescient as a derived STAC collection (`clay-embeddings`). This decouples the computationally significant encoding step from downstream training iteration: subsequent experiments with different downstream classifiers, label strategies, or hyperparameter configurations can retrieve pre-computed patch tokens from Prescient rather than re-running Clay on every iteration.

This also demonstrates a concrete data management pattern that is one of the project's showcase goals: a derived analytical product feeding subsequent pipeline stages, discoverable and accessible through the same STAC interface as the source data.

### Phase 1 Advantages and Risks

The frozen extractor approach has several practical advantages for this project. Compute requirements are modest — embedding generation requires a single forward pass per scene, and training the downstream classifier on pre-computed features is fast. No GPU is required for the downstream training loop itself (only for the initial Clay encoding pass, which is a one-time cost per scene). The patch token granularity gives roughly 1,024 training samples per chip rather than one, substantially increasing effective training volume from a given set of scenes.

The principal risk is that Clay's frozen representations may not optimally encode the specific SAR texture features most relevant for ice concentration classification. Clay was pre-trained across a diverse range of earth observation imagery, including SAR — but its Sentinel-1 pre-training corpus is IW RTC VV/VH, not EW HH/HV. The custom metadata entry mitigates the worst effects of this mismatch, but cannot guarantee that the frozen representations are well-aligned with sea ice features. If this risk materialises, the Phase 1 accuracy gap will be the signal to proceed to Phase 2.

---

## Phase 2 — End-to-End Clay Fine-Tuning

Phase 2 is conditional. It is triggered if either of the following holds after Phase 1 evaluation:

1. **Absolute performance shortfall.** Phase 1 R² on held-out scenes falls below 80% (AutoICE percentage convention; equivalent to R² = 0.80 in the conventional 0–1 form). This threshold is calibrated against the AutoICE SIC R² results reported in Stokholm et al. (2024), *The Cryosphere*, 18, 3471–3494, and the winning team's follow-up paper Chen et al. (2024), *The Cryosphere*, 18, 1621–1632, where the top-five teams cluster in the 87–92% range — the University of Waterloo winning team reporting 92.0%, the rank-4 sim team reporting 87.2%, and a top-five standard deviation of 1.8 percentage points. The 80% floor sits roughly seven points below the top-five band, absorbing Phase 1's architectural disadvantage: Phase 1 uses frozen Clay v1.5 embeddings with downstream RF and XGBoost classifiers, whereas AutoICE submissions trained multi-task U-Nets from scratch with custom downscaling, spatial-temporal encoding, and tuned loss functions (Chen et al., 2024). Below 80%, the gap to the published range is large enough that the frozen-embedding approach has clearly failed to capture what task-specific models capture, and Phase 2 fine-tuning is warranted. Above 80%, the non-embedding baseline comparison (second arm below) becomes the decisive criterion. Evaluation is against held-out AI4Arctic test scenes and a Hudson Bay subset; both should clear the threshold for Phase 2 to be avoided. R² in project reporting follows the AutoICE percentage convention for comparability with the published distribution.
2. **Insufficient lift over the non-embedding baseline.** Phase 1 with Clay embeddings does not meaningfully outperform the Configuration 1 raw HH/HV backscatter baseline. This indicates that Clay's frozen representations are not encoding useful information for this task beyond what is available from raw backscatter, and that the value the foundation model is supposed to be adding is not in fact present.

A diagnostic note rather than a formal trigger: if Phase 1 underperformance is concentrated in dynamic transition periods (freeze-up onset, break-up), the appropriate response is generally more training data covering those periods rather than fine-tuning. Fine-tuning addresses representation quality; data scarcity addresses coverage of the relevant regime.

### Architecture

Phase 2 replaces the decoupled embedding + downstream classifier structure with an end-to-end trainable model. A SIC classification head is attached directly to Clay's encoder — typically a small MLP applied per patch token to produce eleven-class logits — and the full model (encoder + head) is fine-tuned on the labeled dataset.

Two fine-tuning strategies are common: initialising with the encoder frozen and progressively unfreezing layers as training proceeds (which stabilises early training and reduces the risk of catastrophic forgetting of pre-trained representations), or training end-to-end from the start with a reduced learning rate on the encoder weights. The progressive unfreezing approach is generally preferred for fine-tuning large pre-trained models on limited labeled data.

AMSR2 and ERA5 ancillary features can be incorporated either by appending them to the embedding after the encoder (as in Phase 1) or by conditioning the encoder's attention mechanism — the latter being more architecturally sophisticated and likely unnecessary unless Phase 2 itself shows a significant gap when ancillary features are appended post-encoder.

### Compute Requirements

Fine-tuning Clay end-to-end requires GPU compute. A multi-hour training run on a p3 or g5 instance family on AWS is expected. For initial iteration, SageMaker Studio or a notebook environment is more practical; migration to AWS Batch with a Docker container is appropriate once the training loop is stable and reproducible.

---

## Advantages of the GeoFM Approach

Using a geospatial foundation model — whether frozen or fine-tuned — provides several structural advantages over a task-specific model trained from scratch.

**Reduced label requirements.** Clay's self-supervised pre-training means the encoder already captures rich SAR texture representations. The downstream task only needs to learn the final mapping from those representations to SIC classes, requiring substantially less labeled data than building a SAR-to-SIC model from scratch.

**Embedding reuse.** Ingesting patch tokens into Prescient decouples the compute-intensive encoding step from downstream training iteration. Different downstream classifiers, feature configurations, label strategies, and hyperparameter configurations can all be evaluated against the same pre-computed patch tokens without re-running the foundation model.

**Contextual reasoning.** Clay's attention-based encoder considers spatial context within each patch and across the patch grid, rather than processing pixels in isolation. This is useful for resolving ambiguous ice signatures where local texture is similar between ice types (e.g., new ice and wind-roughened open water) but broader spatial context — position relative to the ice edge, proximity to land — is informative. The Configuration 3 evaluation directly tests whether this contextual reasoning, surfaced via the class token chip embedding, contributes additional discriminative power beyond the patch tokens alone.

---

## Open Questions

- **Clay patch token API call. Resolved.** Confirmed against the Clay v1.5 source code (`claymodel/model.py`) and the official `docs/tutorials/embeddings.ipynb` tutorial. The canonical inference pattern loads `ClayMAEModule` with `mask_ratio=0.0` and `shuffle=False`, then calls `module.model.encoder(batch)` where `batch` is a dict with keys `pixels` (shape `[B, C, 256, 256]`), `time` (`[B, 4]`, sin/cos pairs for week and hour), `latlon` (`[B, 4]`, sin/cos pairs for lat and lon), `waves` (per-band wavelengths in μm), and `gsd` (scalar tensor in metres). The encoder returns a tuple; the first element is the encoded sequence of shape `[B, 1025, 1024]`, the remaining three (`unmasked_indices`, `masked_indices`, `masked_matrix`) are unused at inference and conventionally discarded with `encoded, *_ = module.model.encoder(batch)`. Index 0 of the sequence dimension is the class token (chip-level embedding); indices 1 to 1024 are the patch tokens in row-major order over the 32 × 32 spatial grid, reshaped via `einops.rearrange(encoded[:, 1:, :], "b (h w) d -> b d h w", h=32, w=32)`. Note that `claymodel.finetune.embedder.factory.EmbeddingEncoder` provides a related interface with masking pre-disabled, but its default `forward` returns only the class token (`[B, 1024]`); the lower-level `ClayMAEModule` path matches the tutorial precedent and is preferred.
- **Clay v1.5 embedding dimension on Sentinel-1 EW.** Empirical testing on Sentinel-2 inputs has confirmed the embedding dimension at 1024 for both patch tokens and the class token chip embedding. The same dimension is expected for Sentinel-1 EW inputs under the custom `sentinel-1-ew` metadata entry, since Clay v1.5 is sensor-agnostic and shares the encoder embedding space across sensors, but this should be empirically confirmed on EW HH/HV inputs once the custom metadata entry is in place.
- **Normalisation statistics source.** Whether to compute custom HH/HV normalisation statistics from Hudson Bay study-area data, derive them from the AI4Arctic NERSC-corrected scenes, or use a hybrid (e.g., AI4Arctic statistics for the bulk dataset, Hudson Bay statistics as a sanity check). Defer until initial Clay inference runs are working with placeholder statistics.
- **Feature configuration × classifier evaluation scope.** Whether to run all six combinations of three feature configurations and two classifiers, or to narrow on one axis first. Depends on the implementation difficulty of cleanly parameterising the training loop. If swapping feature configuration and classifier are both trivial config flips, run all six; if either dimension requires meaningful per-variant rework, fix one and vary the other first.
- **Phase 2 threshold T.** Resolved at 80% R² (AutoICE percentage convention), calibrated against the top-five SIC R² band of 87–92% reported in Stokholm et al. (2024) and Chen et al. (2024). See the Phase 2 trigger conditions in the architecture section for the calibration argument.
