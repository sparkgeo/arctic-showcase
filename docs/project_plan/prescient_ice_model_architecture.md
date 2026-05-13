# Prescient Ice: Model Architecture

## Overview

The model architecture is structured around Clay, a geospatial foundation model pre-trained on multi-sensor earth observation imagery. Rather than training a SIC regression model from scratch — which would require large volumes of high-quality labeled data — the approach leverages Clay's pre-trained representations as a starting point. Clay has already learned to encode SAR texture, surface structure, and spatial context from self-supervised training on large EO datasets; the downstream task is to learn the mapping from those representations to continuous ice concentration values.

The architecture follows a two-phase strategy. Phase 1 uses Clay as a frozen feature extractor, with no weight updates to the foundation model. Phase 2 — end-to-end fine-tuning — is pursued only if Phase 1 performance falls materially short of the target accuracy threshold. This structure limits unnecessary compute and complexity while providing a clear decision point for escalation.

---

## Phase 1 — Clay as Frozen Feature Extractor

Phase 1 is the primary approach. Clay's encoder is loaded with its pre-trained weights and held frozen throughout; no gradients flow back into the foundation model during training. A simpler supervised model is trained on top of the frozen embeddings to predict SIC.

### Embedding Generation

Sentinel-1 EW GRD scenes are tiled into patches at the spatial scale Clay expects (typically 256 × 256 pixels at the native SAR resolution, depending on Clay's patch size requirements — this needs to be confirmed against Clay's input specification). Each patch is passed through Clay's frozen encoder to produce a feature embedding vector representing that patch's content.

Ancillary features are appended to the embedding vector per patch. The current plan includes AMSR2 daily SIC composites (resampled to the patch footprint, providing a coarse passive microwave prior) and ERA5 surface variables (2m air temperature, 10m wind speed, surface pressure — providing atmospheric and thermodynamic context). Appending these as additional feature dimensions is the simplest integration approach; the downstream model learns the appropriate weighting.

Generated embeddings are not discarded after training. They are serialised and ingested into Prescient as a derived STAC collection (`clay-embeddings`), stored as COGs. This decouples the computationally intensive encoding step from downstream training iteration: subsequent experiments with different downstream models or hyperparameters can retrieve pre-computed embeddings from Prescient rather than re-running the foundation model encoder.

### Downstream Model

The downstream model is trained on the assembled (embedding + ancillary features) → SIC regression task. The target is a continuous value in [0, 1] representing fractional ice concentration per 500m grid cell, derived from rasterised USNIC chart labels.

The initial downstream model is a Random Forest regressor. Random Forest is appropriate here for several reasons: it handles tabular feature inputs (a flat embedding vector) natively; it produces interpretable feature importance outputs, which are useful for diagnosing which dimensions of the Clay embedding are most predictive; it does not require a GPU; and it generalises well with moderate hyperparameter sensitivity. Gradient boosting (XGBoost or LightGBM) and shallow MLPs are natural alternatives to evaluate once a baseline is established — both can be trained on the same embedding features with minimal changes to the training pipeline.

The training objective is mean squared error (MSE) or mean absolute error (MAE) against the rasterised labels. Both metrics are appropriate for continuous regression; MAE may be preferable given that SIC distributions are often bimodal (concentrated near 0% and 100%) and large errors at the extremes carry disproportionate weight under MSE.

### Advantages

The frozen extractor approach has several practical advantages for this project. Compute requirements are modest — embedding generation requires a single forward pass per scene, and training the downstream model on pre-computed embeddings is fast. No GPU is required for the training loop itself (only for embedding generation, and Clay inference can run on CPU for offline batch processing, albeit slowly). The embedding store in Prescient also demonstrates a concrete data management pattern: a derived analytical product that feeds subsequent pipeline stages, discoverable and accessible through the same STAC interface as the source data.

The principal risk is that Clay's frozen representations may not optimally encode the specific SAR texture features most relevant for ice concentration regression. Clay was pre-trained across a diverse range of earth observation imagery and tasks; its encoder may compress or discard information that is important for distinguishing ice types but not salient for its pre-training objectives. If this is the case, the Phase 1 accuracy gap will be the signal to proceed to Phase 2.

---

## Phase 2 — End-to-End Clay Fine-Tuning

Phase 2 is conditional. It is pursued only if Phase 1 accuracy falls materially short of the target threshold, and the gap is large enough to justify the additional complexity and compute cost. A concrete threshold should be defined before Phase 1 evaluation begins — for example, if Phase 1 validation RMSE exceeds the AMSR2 passive microwave baseline by more than a defined margin, Phase 2 is triggered.

### Architecture

Phase 2 replaces the decoupled embedding + downstream model structure with an end-to-end trainable model. A SIC regression head is attached directly to Clay's encoder — typically a small MLP applied per patch to the encoder's output embedding — and the full model (encoder + head) is fine-tuned on the labeled dataset.

Two fine-tuning strategies are common: initialising with the encoder frozen and progressively unfreezing layers as training proceeds (which stabilises early training and reduces the risk of catastrophic forgetting of the pre-trained representations), or training end-to-end from the start with a reduced learning rate on the encoder weights. The progressive unfreezing approach is generally preferred for fine-tuning large pre-trained models on limited labeled data.

AMSR2 and ERA5 ancillary features can be incorporated either by appending them to the embedding after the encoder (as in Phase 1) or by conditioning the encoder's attention mechanism — the latter being more architecturally sophisticated and likely unnecessary unless Phase 2 itself shows a significant gap when ancillary features are appended post-encoder.

### Compute Requirements

Fine-tuning Clay end-to-end requires GPU compute. A multi-hour training run on a p3 or g5 instance family on AWS is expected. For initial iteration, SageMaker Studio or a notebook environment is more practical; migration to AWS Batch with a Docker container is appropriate once the training loop is stable and reproducible.

---

## Advantages of the GeoFM Approach

Using a geospatial foundation model — whether frozen or fine-tuned — provides several structural advantages over a task-specific model trained from scratch.

**Reduced label requirements.** Clay's self-supervised pre-training means the encoder already captures rich SAR texture representations. The downstream task only needs to learn the final mapping from those representations to SIC values, which requires substantially less labeled data than building a SAR-to-SIC model from scratch.

**Embedding reuse.** Ingesting embeddings into Prescient decouples the compute-intensive encoding step from downstream training iteration. Different downstream models, label strategies, and hyperparameter configurations can all be evaluated against the same pre-computed embeddings without re-running the foundation model.

**Contextual reasoning.** Clay's attention-based encoder considers spatial context within each patch rather than processing pixels in isolation. This is useful for resolving ambiguous ice signatures where local texture is similar between ice types (e.g., new ice and open water in high-wind conditions) but broader spatial context — position relative to the ice edge, proximity to land — is informative.
