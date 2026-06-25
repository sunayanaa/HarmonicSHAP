# ==============================================================================
# Program Name: 04_exp1_quantitative_metrics.py
# Version: 2.0 (FMA-Small, harmonicshap_core)
# Description: Computes Deletion AUC and Insertion AUC for all four methods
#              in Experiment 1 on 200 FMA-Small training samples.
#              Runs Wilcoxon signed-rank test between HarmonicSHAP and
#              Standard Acoustic SHAP on per-sample Deletion AUC.
#
# Change Log:
#   1.0: GTZAN, full-branch zeroing, silence baseline, scalar SHAP broadcast.
#   2.0: FMA-Small. All HarmonicSHAP masking via harmonicshap_core (FIX 1-5).
#        Player-level deletion/insertion (5 points) for HarmonicSHAP.
#        Frame-level deletion/insertion (11 steps) for acoustic baselines.
#        Training-mean baseline for all masking. Log1p on Mel/CQT.
#        Checkpoint/resume every 5 samples via Google Drive.
#
# GPU Required: Yes
# EVAL_SAMPLES: 200 (deterministic, seeded selection from chunk 1)
# Dependencies: torch, shap, xgboost, scipy, sklearn, harmonicshap_core
# Inputs (Google Drive): fma_features_train_chunk_1.pkl, xgboost_baseline.json,
#               crnn_backbone_weights.pth, fma_training_baseline.pkl,
#               label_encoder.pkl
# Outputs (Google Drive): exp1_quantitative_results.json
#                exp1_metrics_state.pkl  (checkpoint)
# ==============================================================================

import os, sys, pickle, json, random
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import shap
from sklearn.metrics import auc
from scipy.stats import wilcoxon, sem
from google.colab import drive

# --- Mount Google Drive ---
drive.mount('/content/drive')

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/HarmonicSHAP/"
CHECKPOINT_DIR  = "/content/checkpoints"
EVAL_SAMPLES    = 200
SEED            = 42
DEL_INS_STEPS   = 10   # steps for acoustic baseline curves (11 points)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if not torch.cuda.is_available():
    print("[ERROR] GPU not detected.")
    sys.exit(1)
device = torch.device("cuda")
print("CUDA available: True. Proceeding...")

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
# Step 1: Load models, baseline, and data
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

# Deterministic sample selection
all_keys      = sorted(chunk_data.keys())
np.random.seed(SEED)
selected_keys = np.random.choice(
    all_keys, size=min(EVAL_SAMPLES, len(all_keys)), replace=False
).tolist()
print(f"Selected {len(selected_keys)} samples for evaluation")

# Pre-compute MFCC training mean for acoustic baseline masking
mfcc_mean_train = np.zeros(40, dtype=np.float32)
n_mfcc_tracks   = 0
for key in all_keys:
    mfcc = chunk_data[key]['features']['mfcc']
    mfcc_mean_train += np.concatenate([mfcc.mean(axis=1), mfcc.var(axis=1)])
    n_mfcc_tracks   += 1
mfcc_mean_train /= n_mfcc_tracks

# XGBoost TreeExplainer (built once)
xgb_explainer = shap.TreeExplainer(xgb_model)

# =============================================================================
# Step 2: Load or initialise checkpoint
# =============================================================================
state_local = os.path.join(CHECKPOINT_DIR, "exp1_metrics_state.pkl")
if copy_from_project("exp1_metrics_state.pkl", state_local):
    with open(state_local, 'rb') as f:
        state = pickle.load(f)
    print(f"[INFO] Resuming from checkpoint: "
          f"{state['n_processed']}/{EVAL_SAMPLES} samples processed")
else:
    state = {
        'n_processed': 0,
        'results': {
            'Vanilla SHAP (XGBoost)'      : {'del': [], 'ins': []},
            'SHAP-LM Proxy (XGBoost)'     : {'del': [], 'ins': []},
            'Standard Acoustic SHAP (CRNN)': {'del': [], 'ins': []},
            'HarmonicSHAP (CRNN)'          : {'del': [], 'ins': []},
        }
    }
    print("[INFO] Starting fresh evaluation")

# =============================================================================
# AUC helper functions
# =============================================================================

def mfcc_del_ins_auc(X_sample, shap_vals_abs, model_fn, n_features=40):
    """
    Deletion/Insertion AUC for XGBoost MFCC-level methods.
    Masks features by replacing with training-set mean in order of importance.
    Uses DEL_INS_STEPS evenly-spaced steps.
    """
    sorted_feats = np.argsort(-shap_vals_abs)   # highest importance first
    step         = max(1, n_features // DEL_INS_STEPS)
    del_confs, ins_confs = [], []

    X_del = X_sample.copy()
    X_ins = mfcc_mean_train.reshape(1, -1).copy()

    del_confs.append(float(np.max(model_fn(X_del),  axis=1)[0]))
    ins_confs.append(float(np.max(model_fn(X_ins),  axis=1)[0]))

    for start in range(0, n_features, step):
        idxs = sorted_feats[start:start + step]
        X_del[:, idxs] = mfcc_mean_train[idxs]
        X_ins[:, idxs] = X_sample[:, idxs]
        del_confs.append(float(np.max(model_fn(X_del), axis=1)[0]))
        ins_confs.append(float(np.max(model_fn(X_ins), axis=1)[0]))

    x = np.linspace(0, 1, len(del_confs))
    return float(auc(x, del_confs)), float(auc(x, ins_confs))


def acoustic_del_ins_auc(mel_t, cqt_t, chr_t, crnn, pred_class,
                           mel_mean_t, cqt_mean_t, chroma_mean_t):
    """
    Deletion/Insertion AUC for Standard Acoustic SHAP (CRNN).
    Attribution: gradient × input on Mel branch, aggregated over frequency.
    Masking: replace time frames with training mean in order of attribution.
    DEL_INS_STEPS evenly-spaced steps over 1290 time frames.
    """
    # Gradient × input attribution
    mel_g = mel_t.clone().requires_grad_(True)
    crnn.train()
    for m in crnn.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.Dropout)):
            m.eval()
    crnn.zero_grad()
    out = crnn(mel_g, cqt_t, chr_t)
    out[0, pred_class].backward()
    crnn.eval()

    attribution = (mel_g.grad * mel_g).squeeze().abs().mean(dim=0) \
                                                 .detach().cpu().numpy()  # (T,)
    sorted_frames = np.argsort(-attribution)
    T             = mel_t.shape[3]
    step          = max(1, T // DEL_INS_STEPS)

    fill_mel    = mel_mean_t.view(1, 1, -1, 1).expand(1, 1, hsc.N_MEL_BINS, T)
    fill_cqt    = cqt_mean_t.view(1, 1, -1, 1).expand(1, 1, hsc.N_CQT_BINS, T)
    fill_chroma = chroma_mean_t.view(1, 1, -1, 1).expand(
        1, 1, hsc.N_CHROMA_BINS, T)

    m_del_mel = mel_t.clone()
    m_del_cqt = cqt_t.clone()
    m_del_chr = chr_t.clone()
    m_ins_mel = fill_mel.clone()
    m_ins_cqt = fill_cqt.clone()
    m_ins_chr = fill_chroma.clone()

    del_confs, ins_confs = [], []

    with torch.no_grad():
        del_confs.append(
            torch.softmax(crnn(m_del_mel, m_del_cqt, m_del_chr),
                          dim=1)[0, pred_class].item())
        ins_confs.append(
            torch.softmax(crnn(m_ins_mel, m_ins_cqt, m_ins_chr),
                          dim=1)[0, pred_class].item())

        for start in range(0, T, step):
            idxs = sorted_frames[start:start + step]
            m_del_mel[:, :, :, idxs] = fill_mel[:, :, :, idxs]
            m_del_cqt[:, :, :, idxs] = fill_cqt[:, :, :, idxs]
            m_del_chr[:, :, :, idxs] = fill_chroma[:, :, :, idxs]
            m_ins_mel[:, :, :, idxs] = mel_t[:, :, :, idxs]
            m_ins_cqt[:, :, :, idxs] = cqt_t[:, :, :, idxs]
            m_ins_chr[:, :, :, idxs] = chr_t[:, :, :, idxs]
            del_confs.append(
                torch.softmax(crnn(m_del_mel, m_del_cqt, m_del_chr),
                              dim=1)[0, pred_class].item())
            ins_confs.append(
                torch.softmax(crnn(m_ins_mel, m_ins_cqt, m_ins_chr),
                              dim=1)[0, pred_class].item())

    x = np.linspace(0, 1, len(del_confs))
    return float(auc(x, del_confs)), float(auc(x, ins_confs))


def harmonic_del_ins_auc(mel_t, cqt_t, chr_t, crnn, pred_class,
                           shapley_vals,
                           timbral_mel_bins, mask_H_cqt, mask_H_chr,
                           mask_R, section_frames,
                           mel_mean_t, cqt_mean_t, chroma_mean_t):
    """
    Deletion/Insertion AUC for HarmonicSHAP.
    Players ordered by descending Shapley value.
    5-point curve (including all-present and all-absent endpoints).
    """
    players        = hsc.PLAYERS
    sorted_players = sorted(players, key=lambda p: shapley_vals[p], reverse=True)

    del_coal = set(sorted_players)
    ins_coal = set()
    del_confs, ins_confs = [], []

    with torch.no_grad():
        def conf(coal):
            mm, mc, mh = hsc.apply_semantic_mask(
                mel_t, cqt_t, chr_t, frozenset(coal),
                timbral_mel_bins, mask_H_cqt, mask_H_chr,
                mask_R, section_frames,
                mel_mean_t, cqt_mean_t, chroma_mean_t
            )
            return torch.softmax(crnn(mm, mc, mh),
                                  dim=1)[0, pred_class].item()

        del_confs.append(conf(del_coal))
        ins_confs.append(conf(ins_coal))

        for p in sorted_players:
            del_coal.discard(p)
            ins_coal.add(p)
            del_confs.append(conf(del_coal))
            ins_confs.append(conf(ins_coal))

    x = np.linspace(0, 1, len(del_confs))
    return float(auc(x, del_confs)), float(auc(x, ins_confs))


# =============================================================================
# Step 3: Main evaluation loop
# =============================================================================
print(f"\n--- Step 3: Evaluating {EVAL_SAMPLES} samples ---")
print(f"Already processed: {state['n_processed']}")

keys_to_process = selected_keys[state['n_processed']:]

for loop_idx, key in enumerate(keys_to_process):
    global_idx = state['n_processed'] + loop_idx + 1
    sys.stdout.write(f"\rProcessing {global_idx}/{EVAL_SAMPLES}...")
    sys.stdout.flush()

    track = chunk_data[key]
    true_idx = int(le.transform([track['genre']])[0])

    # --- Prepare tensors ---
    mel_t = hsc.format_tensor(track['features']['mel'],    device, apply_log=True)
    cqt_t = hsc.format_tensor(track['features']['cqt'],    device, apply_log=True)
    chr_t = hsc.format_tensor(track['features']['chroma'], device)

    with torch.no_grad():
        pred_class = crnn_model(mel_t, cqt_t, chr_t).argmax(dim=1).item()

    # --- MFCC features for XGBoost baselines ---
    mfcc    = track['features']['mfcc']
    X_xgb   = np.concatenate([mfcc.mean(axis=1),
                               mfcc.var(axis=1)]).reshape(1, -1)
    xgb_pred_idx  = xgb_model.predict(X_xgb)[0]
    shap_xgb_vals = xgb_explainer.shap_values(X_xgb)
    if isinstance(shap_xgb_vals, list):
        shap_abs = np.abs(shap_xgb_vals[xgb_pred_idx][0])
    else:
        shap_abs = np.abs(shap_xgb_vals[0, :, xgb_pred_idx])

    def xgb_predict_proba(X):
        return xgb_model.predict_proba(X)

    # 1. Vanilla SHAP (XGBoost) — all 40 features
    d, i = mfcc_del_ins_auc(X_xgb, shap_abs, xgb_predict_proba)
    state['results']['Vanilla SHAP (XGBoost)']['del'].append(d)
    state['results']['Vanilla SHAP (XGBoost)']['ins'].append(i)

    # 2. SHAP-LM Proxy (XGBoost) — top-20 features only
    top20_mask        = np.zeros(40, dtype=np.float32)
    top20_mask[np.argsort(-shap_abs)[:20]] = shap_abs[np.argsort(-shap_abs)[:20]]
    d, i = mfcc_del_ins_auc(X_xgb, top20_mask, xgb_predict_proba)
    state['results']['SHAP-LM Proxy (XGBoost)']['del'].append(d)
    state['results']['SHAP-LM Proxy (XGBoost)']['ins'].append(i)

    # 3. Standard Acoustic SHAP (CRNN) — gradient × input, frame-level
    d, i = acoustic_del_ins_auc(
        mel_t, cqt_t, chr_t, crnn_model, pred_class,
        mel_mean, cqt_mean, chroma_mean)
    state['results']['Standard Acoustic SHAP (CRNN)']['del'].append(d)
    state['results']['Standard Acoustic SHAP (CRNN)']['ins'].append(i)

    # 4. HarmonicSHAP (CRNN) — player-level exact Shapley
    (beat_frames, section_frames, timbral_mel_bins,
     mask_H_cqt, mask_H_chr, mask_R, sections, _) = hsc.extract_track_entities(
        track, device, hsc.TARGET_FRAMES)

    shap_vals, _ = hsc.compute_shapley_game(
        mel_t, cqt_t, chr_t, crnn_model, pred_class,
        timbral_mel_bins, mask_H_cqt, mask_H_chr,
        mask_R, section_frames,
        mel_mean, cqt_mean, chroma_mean
    )

    d, i = harmonic_del_ins_auc(
        mel_t, cqt_t, chr_t, crnn_model, pred_class,
        shap_vals, timbral_mel_bins, mask_H_cqt, mask_H_chr,
        mask_R, section_frames,
        mel_mean, cqt_mean, chroma_mean)
    state['results']['HarmonicSHAP (CRNN)']['del'].append(d)
    state['results']['HarmonicSHAP (CRNN)']['ins'].append(i)

    state['n_processed'] += 1

    # Checkpoint every 5 samples
    if state['n_processed'] % 5 == 0:
        with open(state_local, 'wb') as f:
            pickle.dump(state, f)
        copy_to_project(state_local, "exp1_metrics_state.pkl")

# Final checkpoint
with open(state_local, 'wb') as f:
    pickle.dump(state, f)
copy_to_project(state_local, "exp1_metrics_state.pkl")

# =============================================================================
# Step 4: Report results and Wilcoxon test
# =============================================================================
print("\n\n=== Experiment 1: Quantitative Results (AUC) ===")
print("Lower Deletion AUC is better. Higher Insertion AUC is better.\n")

final_metrics = {}
for method, data in state['results'].items():
    del_arr = np.array(data['del'])
    ins_arr = np.array(data['ins'])
    final_metrics[method] = {
        'del_mean': float(np.mean(del_arr)),
        'del_se'  : float(sem(del_arr)),
        'ins_mean': float(np.mean(ins_arr)),
        'ins_se'  : float(sem(ins_arr)),
    }
    print(f"{method}:")
    print(f"  Deletion AUC  : {np.mean(del_arr):.4f} ± {sem(del_arr):.4f}")
    print(f"  Insertion AUC : {np.mean(ins_arr):.4f} ± {sem(ins_arr):.4f}\n")

# Wilcoxon signed-rank test: HarmonicSHAP vs Standard Acoustic SHAP
del_hs  = np.array(state['results']['HarmonicSHAP (CRNN)']['del'])
del_sas = np.array(state['results']['Standard Acoustic SHAP (CRNN)']['del'])

stat, p_val = wilcoxon(del_hs, del_sas)
print(f"Wilcoxon signed-rank test (Deletion AUC):")
print(f"  HarmonicSHAP vs Standard Acoustic SHAP")
print(f"  W = {stat:.2f},  p = {p_val:.4f}")
if p_val < 0.05:
    print("  Result: statistically significant difference (p < 0.05)")
else:
    print("  Result: no statistically significant difference — "
          "competitive performance confirmed (p = {:.3f})".format(p_val))

final_metrics['wilcoxon'] = {'statistic': float(stat), 'p_value': float(p_val)}

# =============================================================================
# Step 5: Save and upload results
# =============================================================================
results_path = os.path.join(CHECKPOINT_DIR, "exp1_quantitative_results.json")
with open(results_path, 'w') as f:
    json.dump(final_metrics, f, indent=2)
copy_to_project(results_path, "exp1_quantitative_results.json")

print(f"\n[SUCCESS] Script 04 complete. Results uploaded.")
print("Next step: run 05_exp2_ablation.py")