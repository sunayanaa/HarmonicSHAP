# HarmonicSHAP: Semantic Coalition Attribution for Explainable Music Genre Classification
### Sridharan Sankaran (sridharan.sankaran@ieee.org)

This repository contains the complete execution pipeline for the experiments described in the manuscript. The codebase covers shared infrastructure, data ingestion, feature extraction, model training, and four experiments spanning baseline attribution comparison, ablation analysis, genre-level attribution profiling, and cross-dataset HAC robustness evaluation.

---

## Core Design Principle

All experiment scripts import from a single shared module (`harmonicshap_core.py`) that centralises the CRNN architecture, semantic masking logic, Shapley attribution computation, and HAC metric evaluation. This ensures consistency across all experiments and eliminates code duplication. The module must be uploaded to Google Drive once before any experiment script is run.

---

## Datasets

| Dataset | Role | Tracks | Clip Length | Source |
|---------|------|--------|-------------|--------|
| FMA-Small | Primary training and evaluation (Experiments 1, 2, 3) | 8,000 | 30 s | Defferrard et al., ISMIR 2017 |
| GiantSteps+ | Cross-dataset HAC robustness (Experiment 4) | 600 | First 30 s of 2-min clips | Knees et al., ISMIR 2015 / Zenodo |
| Ballroom | Cross-dataset HAC robustness (Experiment 4) | 698 | 30 s | Extended Ballroom dataset |
| Jamendo | Cross-dataset HAC robustness (Experiment 4) | 200 | 30 s (from offset 30 s) | MTG-Jamendo |

---

## Execution Pipeline

Scripts must be executed in the order listed. Each script copies its required inputs from Google Drive at startup and saves its outputs to Google Drive for downstream use. All scripts are designed to run in Google Colab with a T4 GPU.

---

### Shared Infrastructure

**`harmonicshap_core.py`**
The shared core module. Contains the `MultiBranchCRNN` architecture, all semantic masking functions, exact Shapley game computation, and HAC metric evaluation. Upload to your Google Drive folder once before running any other script. All subsequent scripts copy it automatically at startup.

Key components:
- `MultiBranchCRNN`: three-branch CNN-GRU classifier (Mel, CQT, Chroma inputs)
- `extract_track_entities`: extracts beat frames, structural sections, timbral Mel bins, and harmonic CQT/chroma masks for a single track
- `apply_semantic_mask`: applies player-level time-frequency masking with training-mean baseline
- `compute_shapley_game`: exact $2^4 = 16$ coalition Shapley computation
- `compute_hac_for_track`: HAC evaluation with cosine distance and prediction invariance filter
- `format_tensor`: pads/trims feature arrays and applies optional log1p transform

---

### Data Preparation and Model Training

**`01_fma_data_prep_and_features.py`** (v2.0, GPU not required)
Copies FMA-Small audio and metadata from Google Drive, extracts multi-resolution features (Mel, CQT, Chroma, MFCC) for all 8,000 tracks, saves split-labelled chunk files to Google Drive, and computes the training-set per-bin mean baseline used by all masking operations.

**`02_fma_train_models.py`** (v2.2, GPU required)
Trains the XGBoost MFCC baseline and the `MultiBranchCRNN` on FMA-Small using memory-efficient lazy chunk loading. Applies log1p transform to Mel and CQT features at load time. Recomputes the training-mean baseline with log-transformed features before training begins to ensure the masking baseline matches the inference representation.

---

### Experiment 1: Baseline Attribution Comparison

**`03_exp1_attributions_and_visualization.py`** (v2.0, GPU required)
Generates the three-panel centerpiece figure comparing Vanilla SHAP on XGBoost MFCC features, Grad-CAM acoustic saliency, and the HarmonicSHAP temporal semantic attribution profile for a representative correctly-classified FMA-Small track.

**`04_exp1_quantitative_metrics.py`** (v2.0, GPU required)
Computes Deletion AUC and Insertion AUC for all four attribution methods across 200 FMA-Small training samples. Implements checkpoint-and-resume via Google Drive every 5 samples. Runs the Wilcoxon signed-rank test comparing HarmonicSHAP against Standard Acoustic SHAP on per-sample Deletion AUC.

---

### Experiment 2: Ablation Study

**`05_exp2_ablation.py`** (v2.0, GPU required)
Evaluates the contribution of each semantic player by computing HAC, Deletion AUC, and Insertion AUC across six configurations: full HarmonicSHAP, Ablation-H, Ablation-R, Ablation-T, Ablation-CQT, and Ablation-Flat. Uses 100 fixed FMA-Small samples with deterministic seeding. Handles degenerate conditions (where model confidence collapses below 0.05) transparently by reporting N/A rather than suppressing the result. Checkpoint-and-resume every 5 samples.

---

### Experiment 3: Genre-Level Attribution Profiles

**`09_genre_attribution_profiles.py`** (v1.0, GPU required)
Computes full-track mean Shapley attribution profiles for all 8 FMA-Small genres using up to 50 correctly-classified tracks per genre. Generates an 8×4 heatmap showing mean Shapley values per genre-player combination using a diverging colormap centered at zero. Green cells indicate players that increase genre confidence; orange/red cells indicate players that suppress it.

---

### Experiment 4: HAC Robustness Study

**`07_exp3_data_prep.py`** (v2.0, GPU not required)
Extracts features from GiantSteps+, Ballroom, and Jamendo from Google Drive. FMA-Small features are already on Google Drive from script 01 and are not re-extracted. Handles archive extraction, macOS ghost file filtering, and fallback paths automatically.

**`08_exp3_generalization.py`** (v2.0, GPU required)
Applies pitch transpositions (±2, ±4 semitones) and tempo perturbations (±10%) to all four datasets and computes HAC for each transformation. Pitch transposition uses semitone-accurate CQT bin shifting. Tempo perturbation uses feature-level bilinear interpolation (see Limitations in the manuscript). Only track-transformation pairs where the model's predicted class is unchanged are included (prediction invariance filter). Checkpoint-and-resume every 10 tracks.

---

## Script Input/Output Table

| Script | GPU | Inputs | Outputs |
|--------|-----|--------|---------|
| `harmonicshap_core.py` | No | — | Upload to Google Drive once |
| `01_fma_data_prep_and_features.py` | No | FMA-Small audio + metadata (Google Drive) | `fma_features_train_chunk_N.pkl`, `fma_features_validation.pkl`, `fma_features_test.pkl`, `fma_training_baseline.pkl`, `label_encoder.pkl` |
| `02_fma_train_models.py` | Yes | `fma_features_train_chunk_N.pkl`, `fma_features_validation.pkl`, `fma_features_test.pkl`, `label_encoder.pkl` | `crnn_backbone_weights.pth`, `xgboost_baseline.json`, `fma_training_baseline.pkl` (log-transformed), `training_log.json` |
| `03_exp1_attributions_and_visualization.py` | Yes | `fma_features_train_chunk_1.pkl`, `crnn_backbone_weights.pth`, `xgboost_baseline.json`, `fma_training_baseline.pkl`, `label_encoder.pkl` | `fig_03_01_attribution_comparison.png` |
| `04_exp1_quantitative_metrics.py` | Yes | `fma_features_train_chunk_1.pkl`, `crnn_backbone_weights.pth`, `xgboost_baseline.json`, `fma_training_baseline.pkl`, `label_encoder.pkl` | `exp1_quantitative_results.json`, `exp1_metrics_state.pkl` |
| `05_exp2_ablation.py` | Yes | `fma_features_train_chunk_1.pkl`, `crnn_backbone_weights.pth`, `fma_training_baseline.pkl`, `label_encoder.pkl` | `exp2_ablation_results.json`, `exp2_ablation_state.pkl` |
| `09_genre_attribution_profiles.py` | Yes | `fma_features_train_chunk_N.pkl` (all), `crnn_backbone_weights.pth`, `fma_training_baseline.pkl`, `label_encoder.pkl` | `fig_09_genre_attribution_heatmap.png`, `exp_genre_attribution_profiles.json` |
| `07_exp3_data_prep.py` | No | GiantSteps+ `audio.zip`, Ballroom `data1/2.tar.gz`, Jamendo `wav_24/` (all Google Drive) | `exp3_features_giantsteps.pkl`, `exp3_features_ballroom.pkl`, `exp3_features_jamendo.pkl` |
| `08_exp3_generalization.py` | Yes | `exp3_features_giantsteps.pkl`, `exp3_features_ballroom.pkl`, `exp3_features_jamendo.pkl`, `fma_features_validation.pkl`, `crnn_backbone_weights.pth`, `fma_training_baseline.pkl`, `label_encoder.pkl` | `exp3_generalization_results.json`, `exp3_state.pkl` |

---

## Hardware and Dependencies

**GPU:** Required for scripts 02, 03, 04, 05, 08, 09. Google Colab T4 used throughout.

**Storage:** Google Drive at `/content/drive/MyDrive/paper/HarmonicSHAP/` for feature files, checkpoints, and outputs. Google Drive at `/content/drive/MyDrive/datasets/` for raw audio archives.

**Core libraries:** `torch`, `librosa`, `shap`, `xgboost`, `scikit-learn`, `scipy`, `numpy`, `matplotlib`, `cv2`

**Feature extraction constants** (defined in `harmonicshap_core.py` and shared across all scripts):

| Parameter | Value |
|-----------|-------|
| Sample rate | 22,050 Hz |
| Hop length | 512 |
| Mel bins | 128 |
| CQT bins | 168 (7 octaves × 24 bins/octave) |
| Chroma bins | 12 |
| Target frames | 1,290 (≈ 30 s) |
| GRU input size | 2,464 |
| Timbral Mel bins suppressed | 48 |
| Top pitch classes per frame | 3 |

---

## Semantic Coalition Design

HarmonicSHAP defines four coalition players extracted from a multi-resolution signal decomposition:

| Player | Symbol | Extraction Method | Masking Operation |
|--------|--------|-------------------|-------------------|
| Harmonic | $H$ | CQT-to-chroma projection; top-3 pitch classes per frame mapped to CQT bin indices | Suppress chord-active CQT and chroma bins at each time frame |
| Rhythmic | $R$ | Beat positions via `librosa.beat.beat_track` | Suppress all frequency bins at beat-aligned frames |
| Timbral | $T$ | $k$-means ($k=4$) on MFCC frames; dominant cluster centroid projected via IDCT | Suppress 48 characteristic Mel frequency bands |
| Structural | $S$ | Foote-style novelty segmentation on CQT self-similarity matrix | Suppress all frequency bins in the most genre-diagnostic section |

All masking operations replace suppressed regions with the training-set per-bin mean rather than silence.

---

## Key Results

| Experiment | Key Finding |
|------------|-------------|
| 1 — Baseline Comparison | Complementary faithfulness profile: HarmonicSHAP achieves best Insertion AUC (0.5510); Standard Acoustic SHAP achieves best Deletion AUC (0.2085). Both differences statistically significant ($W=4194$, $p<0.0001$). |
| 2 — Ablation Study | Every ablation degrades Insertion AUC from full HarmonicSHAP (0.5895). CQT removal causes the largest single degradation (0.4176), independently validating the CQT-centric design. |
| 3 — Genre Profiles | Genre-differentiated attribution patterns consistent with music-theoretic expectations: Instrumental identified via Harmonic; Electronic via Rhythmic; Hip-Hop and International via Structural. |
| 4 — HAC Robustness | Pitch HAC 0.853–0.930 across four datasets with graceful degradation as transposition magnitude increases. Generalises to Jamendo, which shares no genre label correspondence with the training taxonomy. |

---

## Reproducibility

All random seeds are fixed at `SEED=42` across numpy, torch, and Python's random module. Deterministic seeded subsampling from sorted track key lists ensures identical sample sets across reruns. Checkpoint files on Google Drive enable resume from any interruption point without data loss.

The tempo perturbation in Experiment 4 is approximated via feature-level bilinear interpolation rather than audio-level time stretching, since raw audio is not stored in the feature pickle files. This approximation and its boundary conditions are discussed in the manuscript's Limitations section.