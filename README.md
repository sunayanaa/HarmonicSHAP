# HarmonicSHAP: Semantic Coalition Attribution for Explainable Music Genre Classification
## Sridharan Sankaran (sridharan.sankaran@ieee.org)

This repository contains the complete execution pipeline for the experiments detailed in the manuscript. The codebase is structured sequentially, covering raw data ingestion, feature extraction, model training, baseline comparisons, ablation studies, and generalization robustness testing.

## Execution Pipeline

### Experiment 1: Baselines and Quantitative Metrics

* **`01_exp1_data_prep_and_features.py` (v1.0)**
* **Description:** Prepares the GTZAN dataset. Extracts the multi-resolution features (Mel Spectrogram, MFCC, CQT, Chroma) required for all baseline models and HarmonicSHAP. Saves processed features as chunked pickle checkpoints to the Google Drive project folder.

* **`02_exp1_train_models.py` (v1.0)**
* **Description:** Trains the baseline classifiers. Includes an XGBoost model trained on aggregated MFCCs and a lightweight Multi-Branch CRNN trained on Mel, CQT, and Chroma features. Uploads model weights and training histories to the Google Drive project folder.

* **`03_exp1_attributions_and_visualization.py` (v2.0)**
* **Description:** Generates baseline explanations (Vanilla SHAP, Grad-CAM) and the temporal semantic attribution profile for a sample track. Creates the side-by-side visual comparisons for manuscript figures using an exact 4-player Shapley game for temporal sections. Saves figures to the Google Drive project folder.

* **`04_exp1_quantitative_metrics.py` (v1.0)**
* **Description:** Computes the full quantitative metrics (Deletion and Insertion AUC) for all baselines (including SHAP-LM and Standard Acoustic SHAP) across 200 samples. Implements state-saving checkpoints to the Google Drive project folder to ensure resumability.

* **`04_exp1_Wilcoxon_signed-rank_test.py`**
* **Description:** Conducts the statistical significance testing. Performs a Wilcoxon signed-rank test on the per-sample Deletion AUC values comparing HarmonicSHAP against the Standard Acoustic SHAP baseline.

### Experiment 2: Ablation Study

* **`05_exp2_ablation.py` (v3.0)**
* **Description:** Isolates the impact of the semantic and structural groupings. Computes Harmonic Attribution Consistency (HAC), Deletion AUC, and Insertion AUC across five architectural ablations. Implements deterministic sampling and calculates standard errors for rigorous evaluation. Results saved to Google Drive.

* **`06_exp2_diagnostics.py` (v1.0)**
* **Description:** Diagnoses boundary conditions identified during the ablation study. Prints per-sample HAC comparisons and plots raw Deletion Confidence curves to empirically demonstrate the baseline CRNN confidence collapse (model dependency finding) when the Harmonic or CQT inputs are ablated, justifying the metric exclusions in the manuscript.

### Experiment 3: Generalization and HAC Robustness

* **`07_exp3_data_prep.py` (v3.1)**
* **Description:** Ingestion engine for unseen datasets (GiantSteps+, FMA-Small, Ballroom). Hardened against remote session disconnects via local caching and incremental saving. Automatically handles macOS ghost files and `.tar` extractions, outputting standardized feature checkpoints saved to Google Drive.

* **`08_exp3_generalization.py` (v1.0)**
* **Description:** Conducts the core robustness study. Subjects the unseen datasets to musically meaning-preserving transformations, specifically pitch transpositions ($\pm2$, $\pm4$ semitones) and tempo perturbations ($\pm10\%$), calculating the resulting HAC to evaluate the stability of the semantic hierarchy at scale. Results saved to Google Drive.

## Program Outputs Cross-Reference

| Program | Output Files | Storage Location |
|---------|--------------|------------------|
| **`01_exp1_data_prep_and_features.py`** | `gtzan_features_chunk_N.pkl` (multiple chunks) | `PROJECT_DIR/` |
| | `fig_01_01_sample_multires_features.png` | `PROJECT_DIR/` |
| **`02_exp1_train_models.py`** | `xgboost_baseline.json` | `PROJECT_DIR/` |
| | `crnn_backbone_weights.pth` | `PROJECT_DIR/` |
| | `label_encoder.pkl` | `PROJECT_DIR/` |
| | `fig_02_01_crnn_training_history.png` | `PROJECT_DIR/` |
| **`03_exp1_attributions_and_visualization.py`** | `fig_03_01_attribution_comparison.png` | `PROJECT_DIR/` |
| **`04_exp1_quantitative_metrics.py`** | `exp1_quantitative_results.json` | `PROJECT_DIR/` |
| | `exp1_metrics_state.pkl` (checkpoint) | `PROJECT_DIR/` |
| **`04_exp1_Wilcoxon_signed-rank_test.py`** | `wilcoxon_results.txt` (or similar) | `PROJECT_DIR/` |
| **`05_exp2_ablation.py`** | `exp2_ablation_results.json` | `PROJECT_DIR/` |
| **`06_exp2_diagnostics.py`** | `diag_curves.png` | `PROJECT_DIR/` |
| **`07_exp3_data_prep.py`** | `exp3_features_giantsteps.pkl` | `PROJECT_DIR/` |
| | `exp3_features_fma.pkl` | `PROJECT_DIR/` |
| | `exp3_features_ballroom.pkl` | `PROJECT_DIR/` |
| **`08_exp3_generalization.py`** | `exp3_generalization_results.json` | `PROJECT_DIR/` |

### Input Dependencies (Files Required by Downstream Scripts)

| Required File | Generated By | Used By |
|---------------|--------------|---------|
| `gtzan_features_chunk_*.pkl` | `01_exp1_data_prep_and_features.py` | `02_exp1_train_models.py`, `03_exp1_attributions_and_visualization.py`, `04_exp1_quantitative_metrics.py`, `05_exp2_ablation.py`, `06_exp2_diagnostics.py` |
| `xgboost_baseline.json` | `02_exp1_train_models.py` | `03_exp1_attributions_and_visualization.py`, `04_exp1_quantitative_metrics.py` |
| `crnn_backbone_weights.pth` | `02_exp1_train_models.py` | `03_exp1_attributions_and_visualization.py`, `04_exp1_quantitative_metrics.py`, `05_exp2_ablation.py`, `06_exp2_diagnostics.py`, `08_exp3_generalization.py` |
| `label_encoder.pkl` | `02_exp1_train_models.py` | `03_exp1_attributions_and_visualization.py`, `04_exp1_quantitative_metrics.py`, `05_exp2_ablation.py`, `06_exp2_diagnostics.py`, `08_exp3_generalization.py` |
| `exp3_features_*.pkl` | `07_exp3_data_prep.py` | `08_exp3_generalization.py` |

### Execution Order

For a complete run of all experiments, execute the scripts in the following order:

1. `01_exp1_data_prep_and_features.py` — Extract GTZAN features
2. `02_exp1_train_models.py` — Train baseline models
3. `03_exp1_attributions_and_visualization.py` — Generate centerpiece figure
4. `04_exp1_quantitative_metrics.py` — Compute Deletion/Insertion AUC
5. `04_exp1_Wilcoxon_signed-rank_test.py` — Statistical significance
6. `05_exp2_ablation.py` — Run ablation study
7. `06_exp2_diagnostics.py` — Diagnostic plots (optional)
8. `07_exp3_data_prep.py` — Prepare external datasets
9. `08_exp3_generalization.py` — Run generalization robustness study

```

**Summary of the table content:**

| Section | Description |
|---------|-------------|
| **Program Outputs Cross-Reference** | Lists every output file produced by each script and where it is stored (`PROJECT_DIR/` = Google Drive project folder) |
| **Input Dependencies** | Shows which files are required by downstream scripts and which scripts generate them |
| **Execution Order** | Provides the recommended sequence for running all experiments from start to finish |

## Hardware and Dependencies

* **GPU Requirement:** A GPU environment (e.g., Google Colab T4/V100) is highly recommended for PyTorch CRNN training and deep Shapley value computations.
* **Core Libraries:** `torch`, `shap`, `librosa`, `xgboost`, `scikit-learn`, `scipy`, `numpy`, `matplotlib`, `cv2`.

## Storage Configuration

All scripts use Google Drive as the persistent storage layer. The project expects the following configuration:

```python
PROJECT_DIR = "/content/drive/MyDrive/paper/harmonic-SHAP/"
```

The following files are stored in `PROJECT_DIR`:
- Model weights (`crnn_backbone_weights.pth`, `xgboost_baseline.json`)
- Label encoder (`label_encoder.pkl`)
- Feature checkpoints (`gtzan_features_chunk_*.pkl`, `exp3_features_*.pkl`)
- Results JSON files (`exp1_quantitative_results.json`, `exp2_ablation_results.json`, `exp3_generalization_results.json`)
- Visualization figures (`fig_*.png`)

The script also expects the following datasets in `/content/drive/MyDrive/datasets/`:
- `GTZAN.zip`
- `GiantSteps+.zip`
- `GiantSteps+.xlsx`
- `FMA-small.zip`
- `Ballroom/data1.tar.gz`
```
