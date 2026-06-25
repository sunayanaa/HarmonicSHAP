# ==============================================================================
# Program Name: 03_exp1_attributions_and_visualization.py
# Version: 2.0 (FMA-Small, harmonicshap_core)
# Description: Generates the three-panel centerpiece figure for Experiment 1:
#              Panel 1 — Vanilla SHAP on XGBoost MFCC features
#              Panel 2 — Grad-CAM acoustic saliency on Mel spectrogram
#              Panel 3 — HarmonicSHAP temporal semantic attribution profile
#
# Change Log:
#   1.0: GTZAN, full-branch zeroing, silence baseline, uniform quartering.
#   2.0: FMA-Small. Imports MultiBranchCRNN and all attribution logic from
#        harmonicshap_core. Correct semantic time-frequency masking, training-
#        mean baseline, Foote novelty segmentation, librosa beat tracking.
#        Section-restricted Shapley games for the temporal profile.
#        Log1p applied to Mel/CQT consistent with training in script 02.
#
# GPU Required: Yes
# Dependencies: torch, shap, cv2, matplotlib, harmonicshap_core
# Inputs (copied from Google Drive):
#   fma_features_train_chunk_1.pkl
#   xgboost_baseline.json
#   crnn_backbone_weights.pth
#   fma_training_baseline.pkl
#   label_encoder.pkl
# Outputs:
#   fig_03_01_attribution_comparison.png  (saved to Google Drive)
# ==============================================================================

import os, sys, pickle, warnings
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import shap
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from google.colab import drive
warnings.filterwarnings('ignore')

# --- Mount Google Drive ---
drive.mount('/content/drive')

# --- GPU check ---
if not torch.cuda.is_available():
    print("[ERROR] GPU not detected.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")
device = torch.device("cuda")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/HarmonicSHAP/"
CHECKPOINT_DIR  = "/content/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Google Drive file helpers ---
def copy_from_project(remote_name, local_path):
    """Copy a file from PROJECT_DIR to local path."""
    try:
        remote_path = os.path.join(PROJECT_DIR, remote_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Copy file from Google Drive to local
        import shutil
        shutil.copy2(remote_path, local_path)
        print(f"[Google Drive] Copied {remote_name} to {local_path}")
        return True
    except Exception as e:
        print(f"[Google Drive ERROR] {remote_name}: {e}")
        return False

def copy_to_project(local_path, remote_name):
    """Copy a file from local path to PROJECT_DIR."""
    try:
        remote_path = os.path.join(PROJECT_DIR, remote_name)
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        
        # Copy file from local to Google Drive
        import shutil
        shutil.copy2(local_path, remote_path)
        print(f"[Google Drive] Copied {local_path} to {remote_name}")
        return True
    except Exception as e:
        print(f"[Google Drive ERROR] {local_path}: {e}")
        return False

# --- Load harmonicshap_core ---
core_path = os.path.join(CHECKPOINT_DIR, "harmonicshap_core.py")
if not os.path.exists(core_path):
    copy_from_project("harmonicshap_core.py", core_path)
sys.path.insert(0, CHECKPOINT_DIR)
import harmonicshap_core as hsc
from harmonicshap_core import MultiBranchCRNN

# =============================================================================
# Step 1: Load models, baseline, and chunk data
# =============================================================================
print("\n--- Step 1: Loading Models & Data ---")

for fname in ["xgboost_baseline.json", "crnn_backbone_weights.pth",
              "label_encoder.pkl",       "fma_training_baseline.pkl",
              "fma_features_train_chunk_1.pkl"]:
    local = os.path.join(CHECKPOINT_DIR, fname)
    if not os.path.exists(local):
        copy_from_project(fname, local)

with open(os.path.join(CHECKPOINT_DIR, "label_encoder.pkl"), 'rb') as f:
    le = pickle.load(f)
num_classes = len(le.classes_)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(CHECKPOINT_DIR, "xgboost_baseline.json"))

crnn_model = MultiBranchCRNN(num_classes).to(device)
crnn_model.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "crnn_backbone_weights.pth"),
               map_location=device))
crnn_model.eval()

mel_mean, cqt_mean, chroma_mean = hsc.load_baseline(
    os.path.join(CHECKPOINT_DIR, "fma_training_baseline.pkl"), device)

with open(os.path.join(CHECKPOINT_DIR,
                        "fma_features_train_chunk_1.pkl"), 'rb') as f:
    chunk_data = pickle.load(f)
print(f"Chunk loaded: {len(chunk_data)} tracks")

# =============================================================================
# Step 2: Select a correctly-classified representative track
# =============================================================================
print("\n--- Step 2: Selecting representative track ---")

sample_data  = None
sample_id    = None
sample_genre = None

for track_id, track_info in chunk_data.items():
    # XGBoost prediction
    mfcc    = track_info['features']['mfcc']
    X_xgb   = np.concatenate([mfcc.mean(axis=1),
                               mfcc.var(axis=1)]).reshape(1, -1)
    xgb_pred_idx = xgb_model.predict(X_xgb)[0]

    # CRNN prediction
    mel_t = hsc.format_tensor(track_info['features']['mel'],    device,
                               apply_log=True)
    cqt_t = hsc.format_tensor(track_info['features']['cqt'],    device,
                               apply_log=True)
    chr_t = hsc.format_tensor(track_info['features']['chroma'], device)

    with torch.no_grad():
        crnn_pred_idx = crnn_model(mel_t, cqt_t, chr_t).argmax(dim=1).item()

    # Select first track where CRNN predicts correctly
    true_idx = int(le.transform([track_info['genre']])[0])
    if crnn_pred_idx == true_idx:
        sample_data  = track_info
        sample_id    = track_id
        sample_genre = track_info['genre']
        print(f"Selected  : {track_id}")
        print(f"True genre: {sample_genre}")
        print(f"CRNN pred : {le.classes_[crnn_pred_idx]}")
        print(f"XGB pred  : {le.classes_[xgb_pred_idx]}")
        break

if sample_data is None:
    sample_id    = list(chunk_data.keys())[0]
    sample_data  = chunk_data[sample_id]
    sample_genre = sample_data['genre']
    print(f"[WARN] No correctly classified track found in chunk 1. "
          f"Using {sample_id} ({sample_genre}) as fallback.")

# Prepare final tensors
mel_t = hsc.format_tensor(sample_data['features']['mel'],    device,
                           apply_log=True)
cqt_t = hsc.format_tensor(sample_data['features']['cqt'],    device,
                           apply_log=True)
chr_t = hsc.format_tensor(sample_data['features']['chroma'], device)

with torch.no_grad():
    out_full   = crnn_model(mel_t, cqt_t, chr_t)
    pred_class = out_full.argmax(dim=1).item()
    pred_label = le.classes_[pred_class]

mfcc    = sample_data['features']['mfcc']
X_xgb   = np.concatenate([mfcc.mean(axis=1),
                           mfcc.var(axis=1)]).reshape(1, -1)
xgb_pred_idx = xgb_model.predict(X_xgb)[0]

# =============================================================================
# Step 3: Panel 1 — Vanilla SHAP (XGBoost MFCC)
# =============================================================================
print("\n--- Step 3: Computing Vanilla SHAP (XGBoost) ---")

explainer = shap.TreeExplainer(xgb_model)
shap_vals = explainer.shap_values(X_xgb)

if isinstance(shap_vals, list):
    shap_for_class = shap_vals[xgb_pred_idx][0]
else:
    shap_for_class = shap_vals[0, :, xgb_pred_idx]

feat_names = ([f"MFCC_Mean_{i}" for i in range(20)] +
              [f"MFCC_Var_{i}"  for i in range(20)])

# =============================================================================
# Step 4: Panel 2 — Grad-CAM on Mel spectrogram
# =============================================================================
print("\n--- Step 4: Computing Grad-CAM ---")

gradients_store  = [None]
activations_store = [None]

def bwd_hook(module, grad_in, grad_out):
    gradients_store[0] = grad_out[0]

def fwd_hook(module, inp, out_):
    activations_store[0] = out_

target_layer = crnn_model.mel_cnn[-1]
h_fwd = target_layer.register_forward_hook(fwd_hook)
h_bwd = target_layer.register_full_backward_hook(bwd_hook)

# Train mode for gradient flow; keep BN and Dropout in eval for stability
crnn_model.train()
for m in crnn_model.modules():
    if isinstance(m, (nn.BatchNorm2d, nn.Dropout)):
        m.eval()

crnn_model.zero_grad()
out = crnn_model(mel_t, cqt_t, chr_t)
out[0, pred_class].backward()

h_fwd.remove()
h_bwd.remove()
crnn_model.eval()

grads = gradients_store[0]
acts  = activations_store[0]

pooled = torch.mean(grads, dim=[0, 2, 3])
for i in range(acts.size(1)):
    acts[:, i, :, :] *= pooled[i]

heatmap = torch.mean(acts, dim=1).squeeze().cpu().detach().numpy()
heatmap = np.maximum(heatmap, 0)
heatmap /= (np.max(heatmap) + 1e-10)
heatmap  = np.power(heatmap, 0.5)
heatmap  = cv2.resize(heatmap, (mel_t.shape[3], mel_t.shape[2]))

# =============================================================================
# Step 5: Panel 3 — HarmonicSHAP Temporal Attribution Profile
# =============================================================================
print("\n--- Step 5: Computing HarmonicSHAP Temporal Attribution Profile ---")

# Extract global semantic entities for this track
(beat_frames, _section_frames_global,
 timbral_mel_bins, mask_H_cqt, mask_H_chr,
 mask_R, sections, most_diag_idx) = hsc.extract_track_entities(
    sample_data, device, hsc.TARGET_FRAMES
)

print(f"Structural sections : {sections}")
print(f"Most diagnostic     : Section {most_diag_idx + 1}")

temporal_attributions = {p: [] for p in hsc.PLAYERS}

for sec_idx, (s_start, s_end) in enumerate(sections):
    sec_frames = np.arange(s_start, s_end, dtype=int)

    # Restrict beat mask to this section
    sec_beats  = beat_frames[(beat_frames >= s_start) &
                              (beat_frames <  s_end)]
    sec_mask_R = hsc.build_beat_mask(sec_beats, hsc.TARGET_FRAMES, device)

    # Restrict harmonic masks to this section only
    sec_mask_H_cqt              = mask_H_cqt.clone()
    sec_mask_H_cqt[:, :s_start] = 0.0
    sec_mask_H_cqt[:, s_end:]   = 0.0

    sec_mask_H_chr              = mask_H_chr.clone()
    sec_mask_H_chr[:, :s_start] = 0.0
    sec_mask_H_chr[:, s_end:]   = 0.0

    shap_sec, _ = hsc.compute_shapley_game(
        mel_t, cqt_t, chr_t, crnn_model, pred_class,
        timbral_mel_bins,
        sec_mask_H_cqt, sec_mask_H_chr,
        sec_mask_R, sec_frames,
        mel_mean, cqt_mean, chroma_mean
    )

    for p in hsc.PLAYERS:
        temporal_attributions[p].append(shap_sec[p])

    t_start = s_start * hsc.HOP_LENGTH / hsc.SR
    t_end   = s_end   * hsc.HOP_LENGTH / hsc.SR
    print(f"  Section {sec_idx + 1} ({t_start:.1f}-{t_end:.1f}s) : "
          f"H={shap_sec['H']:.4f}  R={shap_sec['R']:.4f}  "
          f"T={shap_sec['T']:.4f}  S={shap_sec['S']:.4f}")

# =============================================================================
# Step 6: Generate and save figure
# =============================================================================
print("\n--- Step 6: Generating Figure ---")

fig = plt.figure(figsize=(14, 12))

# --- Panel 1: Vanilla SHAP ---
ax1 = plt.subplot(3, 1, 1)
shap.summary_plot(shap_for_class.reshape(1, -1), X_xgb,
                  feature_names=feat_names, plot_type="bar", show=False)
plt.title(f"Baseline 1: Vanilla SHAP on MFCCs  "
          f"(Predicted: {le.classes_[xgb_pred_idx]})",
          fontsize=11, fontweight='bold')
plt.xlabel("Mean |SHAP Value|")
ax1.tick_params(axis='y', labelsize=5)

# --- Panel 2: Grad-CAM ---
ax2 = plt.subplot(3, 1, 2)
mel_display = mel_t.squeeze().cpu().numpy()
plt.imshow(mel_display, aspect='auto', origin='lower', cmap='magma')
plt.imshow(heatmap,     aspect='auto', origin='lower', cmap='jet', alpha=0.5)
plt.title(f"Baseline 2: Grad-CAM Acoustic Saliency  "
          f"(Predicted: {pred_label})",
          fontsize=11, fontweight='bold')
plt.ylabel("Mel Bins")
plt.xlabel("Time Frames")

# --- Panel 3: HarmonicSHAP temporal profile ---
ax3 = plt.subplot(3, 1, 3)
n_sec   = len(sections)
x       = np.arange(n_sec)
width   = 0.18
offsets = [-1.5, -0.5, 0.5, 1.5]
colors  = {'T': '#1f77b4', 'H': '#ff7f0e', 'R': '#2ca02c', 'S': '#d62728'}
labels  = {'T': 'Timbral (T)', 'H': 'Harmonic (H)',
           'R': 'Rhythmic (R)', 'S': 'Structural (S)'}

for j, p in enumerate(hsc.PLAYERS):
    ax3.bar(x + offsets[j] * width,
            temporal_attributions[p],
            width, label=labels[p], color=colors[p])

sec_labels = [
    f"Section {i + 1}\n"
    f"({s * hsc.HOP_LENGTH / hsc.SR:.1f}-"
    f"{e * hsc.HOP_LENGTH / hsc.SR:.1f}s)"
    for i, (s, e) in enumerate(sections)
]
ax3.set_xticks(x)
ax3.set_xticklabels(sec_labels, fontsize=9)
ax3.set_ylabel("Genre Confidence Drop (Exact Shapley)")
ax3.set_title("HarmonicSHAP: Temporal Semantic Attribution Profile (NC-3)",
              fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', fontsize=8)
ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout(pad=2.0)
fig_path = os.path.join(CHECKPOINT_DIR, "fig_03_01_attribution_comparison.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
copy_to_project(fig_path, "fig_03_01_attribution_comparison.png")

print(f"\n[SUCCESS] Script 03 complete.")
print(f"Track         : {sample_id}")
print(f"True genre    : {sample_genre}")
print(f"CRNN predicted: {pred_label}")
print(f"Figure saved  : fig_03_01_attribution_comparison.png")
print("Next step     : run 04_exp1_quantitative_metrics.py")