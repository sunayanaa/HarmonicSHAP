# ==============================================================================
# Program Name: 06_exp2_diagnostics.py
# Version: 1.0
# Description: Diagnoses the anomalies in Experiment 2. 
#              1. Prints per-sample HAC comparison for HarmonicSHAP vs Ablation-H.
#              2. Plots raw Deletion Confidence curves for 3 samples to diagnose 
#                 the near-zero AUC collapse in Ablation-H and Ablation-CQT.
# ==============================================================================

import sys
import os
import torch
import glob
import pickle
import shutil
import numpy as np
import math
import itertools
import random
from scipy.stats import pearsonr
import torch.nn as nn
from sklearn.metrics import auc
import matplotlib.pyplot as plt

# --- Rigorous Seed Locking ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/harmonic-SHAP/"  # Persistent storage
CHECKPOINT_DIR = "/content/checkpoints"
DIAGNOSTIC_SAMPLES = 10 

# --- Google Drive Helper Functions ---
def ensure_project_dir():
    """Create project directory in Google Drive if it doesn't exist."""
    os.makedirs(PROJECT_DIR, exist_ok=True)

def load_from_drive(remote_filename, local_filepath):
    """Copy a file from Google Drive project folder to local path."""
    ensure_project_dir()
    src_path = os.path.join(PROJECT_DIR, remote_filename)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, local_filepath)
            print(f"  [DRIVE OK] {src_path}  →  {local_filepath}")
            return True
        except Exception as e:
            print(f"  [DRIVE FAIL] copy from {src_path}: {e}")
            return False
    else:
        print(f"  [DRIVE MISSING] {src_path} not found")
        return False

# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Step 1: Load Model and Data ---
class MultiBranchCRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d((2, 4))
            )
        self.mel_cnn = nn.Sequential(conv_block(1, 16), conv_block(16, 32))
        self.cqt_cnn = nn.Sequential(conv_block(1, 16), conv_block(16, 32))
        self.chroma_cnn = nn.Sequential(conv_block(1, 16), conv_block(16, 32))
        self.gru = nn.GRU(input_size=1792, hidden_size=128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, mel, cqt, chroma):
        x_mel = self.mel_cnn(mel)
        x_cqt = self.cqt_cnn(cqt)
        x_chroma = self.chroma_cnn(chroma)
        
        def prep_for_rnn(x):
            b, c, f, t = x.size()
            return x.permute(0, 3, 1, 2).reshape(b, t, -1)
            
        x_mel, x_cqt, x_chroma = prep_for_rnn(x_mel), prep_for_rnn(x_cqt), prep_for_rnn(x_chroma)
        x_fused = torch.cat([x_mel, x_cqt, x_chroma], dim=-1)
        
        out, _ = self.gru(x_fused)
        out = out[:, -1, :] 
        return self.fc(out)

# Download required files from Drive if missing locally
required_files = ["label_encoder.pkl", "crnn_backbone_weights.pth", "gtzan_features_chunk_1.pkl"]
for file in required_files:
    local_path = os.path.join(CHECKPOINT_DIR, file)
    if not os.path.exists(local_path):
        print(f"[DRIVE] Downloading {file}...")
        if not load_from_drive(file, local_path):
            print(f"[ERROR] Failed to download {file} from Drive.")
            sys.exit(1)

with open(os.path.join(CHECKPOINT_DIR, "label_encoder.pkl"), 'rb') as f:
    le = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crnn = MultiBranchCRNN(len(le.classes_)).to(device)
crnn.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "crnn_backbone_weights.pth")))
crnn.eval()

chunk_path = os.path.join(CHECKPOINT_DIR, "gtzan_features_chunk_1.pkl")
with open(chunk_path, 'rb') as f:
    chunk_data = pickle.load(f)

sample_keys = list(chunk_data.keys())
sample_keys.sort() 
sample_keys = random.sample(sample_keys, min(DIAGNOSTIC_SAMPLES, len(sample_keys)))

# --- Step 2: Define Masking & AUC Logic ---
TARGET_FRAMES = 1290

def format_tensor(x):
    if x.shape[1] < TARGET_FRAMES:
        x = np.pad(x, ((0, 0), (0, TARGET_FRAMES - x.shape[1])))
    return torch.FloatTensor(x[:, :TARGET_FRAMES]).unsqueeze(0).unsqueeze(0).to(device)

def get_onsets_and_structure(mel_t):
    mel_np = mel_t.squeeze().cpu().numpy()
    flux = np.mean(np.diff(mel_np, axis=1) > 0, axis=0)
    onset_frames = np.where(flux > np.percentile(flux, 85))[0]
    struct_frames = np.arange(TARGET_FRAMES // 2, TARGET_FRAMES)
    return onset_frames, struct_frames

def apply_ablation_mask(mel, cqt, chroma, coal, onset_frames, struct_frames, mode):
    m_mel, m_cqt, m_chr = mel.clone(), cqt.clone(), chroma.clone()
    
    if mode in ["HarmonicSHAP", "Ablation-H"]:
        if 'T' not in coal: 
            m_mel *= 0
        if 'H' not in coal: 
            m_cqt *= 0
            m_chr *= 0
        if 'R' not in coal: 
            m_mel[:,:,:,onset_frames] *= 0.5
            m_cqt[:,:,:,onset_frames] *= 0.5
            m_chr[:,:,:,onset_frames] *= 0.5
        if 'S' not in coal and struct_frames is not None:
            m_mel[:,:,:,struct_frames] *= 0
            m_cqt[:,:,:,struct_frames] *= 0
            m_chr[:,:,:,struct_frames] *= 0

    elif mode == "Ablation-CQT":
        m_cqt *= 0 
        if 'T' not in coal: 
            m_mel *= 0
        if 'H' not in coal: 
            m_chr *= 0
        if 'R' not in coal: 
            m_mel[:,:,:,onset_frames] *= 0.5
            m_chr[:,:,:,onset_frames] *= 0.5
        if 'S' not in coal:
            m_mel[:,:,:,struct_frames] *= 0
            m_chr[:,:,:,struct_frames] *= 0
            
    return m_mel, m_cqt, m_chr

def compute_exact_shapley(mel, cqt, chroma, players, mode, true_label_idx, onset_frames, struct_frames=None):
    num_players = len(players)
    coalitions = [frozenset(c) for r in range(num_players + 1) for c in itertools.combinations(players, r)]
    val_dict = {}
    
    with torch.no_grad():
        for coal in coalitions:
            m_mel, m_cqt, m_chr = apply_ablation_mask(mel, cqt, chroma, coal, onset_frames, struct_frames, mode)
            out = crnn(m_mel, m_cqt, m_chr)
            val_dict[coal] = torch.softmax(out, dim=1)[0, true_label_idx].item()
            
    shapley_vals = {p: 0 for p in players}
    for player in players:
        for coal in coalitions:
            if player not in coal:
                coal_with = coal.union([player])
                weight = (math.factorial(len(coal)) * math.factorial(num_players - len(coal) - 1)) / float(math.factorial(num_players))
                shapley_vals[player] += weight * (val_dict[coal_with] - val_dict[coal])
    return shapley_vals

def get_raw_deletion_curve(mel, cqt, chroma, players, shap_vals, mode, true_label_idx, onset_frames, struct_frames=None):
    sorted_players = sorted(players, key=lambda p: shap_vals[p], reverse=True)
    del_confs = []
    current_del_coal = set(sorted_players)
    
    with torch.no_grad():
        m_mel, m_cqt, m_chr = apply_ablation_mask(mel, cqt, chroma, current_del_coal, onset_frames, struct_frames, mode)
        del_confs.append(torch.softmax(crnn(m_mel, m_cqt, m_chr), dim=1)[0, true_label_idx].item())
        
        for p in sorted_players:
            current_del_coal.remove(p)
            m_mel, m_cqt, m_chr = apply_ablation_mask(mel, cqt, chroma, current_del_coal, onset_frames, struct_frames, mode)
            del_confs.append(torch.softmax(crnn(m_mel, m_cqt, m_chr), dim=1)[0, true_label_idx].item())
            
    return del_confs

# --- Step 3: Diagnostic Loop ---
print("\n=== DIAGNOSTIC 1: Per-Sample HAC (HarmonicSHAP vs Ablation-H) ===")
print(f"{'Sample ID':<15} | {'True Genre':<10} | {'HarmonicSHAP HAC':<20} | {'Ablation-H HAC':<20}")
print("-" * 75)

curves = {"HarmonicSHAP": [], "Ablation-H": [], "Ablation-CQT": []}

for idx, key in enumerate(sample_keys):
    track = chunk_data[key]
    true_label_idx = le.transform([track['genre']])[0]
    
    mel_t = format_tensor(track['features']['mel'])
    cqt_t = format_tensor(track['features']['cqt'])
    chr_t = format_tensor(track['features']['chroma'])
    onset_frames, struct_frames = get_onsets_and_structure(mel_t)
    
    mel_shift = torch.roll(mel_t, shifts=1, dims=2)
    cqt_shift = torch.roll(cqt_t, shifts=1, dims=2) 
    chr_shift = torch.roll(chr_t, shifts=1, dims=2)
    
    # HarmonicSHAP
    shap_base_full = compute_exact_shapley(mel_t, cqt_t, chr_t, ['T', 'H', 'R', 'S'], "HarmonicSHAP", true_label_idx, onset_frames, struct_frames)
    shap_shft_full = compute_exact_shapley(mel_shift, cqt_shift, chr_shift, ['T', 'H', 'R', 'S'], "HarmonicSHAP", true_label_idx, onset_frames, struct_frames)
    vec_base_f, vec_shft_f = [shap_base_full[p] for p in ['T', 'H', 'R', 'S']], [shap_shft_full[p] for p in ['T', 'H', 'R', 'S']]
    hac_full, _ = pearsonr(vec_base_f, vec_shft_f) if np.std(vec_base_f)>0 and np.std(vec_shft_f)>0 else (0,0)
    
    # Ablation-H
    shap_base_abH = compute_exact_shapley(mel_t, cqt_t, chr_t, ['T', 'R', 'S'], "Ablation-H", true_label_idx, onset_frames, struct_frames)
    shap_shft_abH = compute_exact_shapley(mel_shift, cqt_shift, chr_shift, ['T', 'R', 'S'], "Ablation-H", true_label_idx, onset_frames, struct_frames)
    vec_base_h, vec_shft_h = [shap_base_abH[p] for p in ['T', 'R', 'S']], [shap_shft_abH[p] for p in ['T', 'R', 'S']]
    hac_abH, _ = pearsonr(vec_base_h, vec_shft_h) if np.std(vec_base_h)>0 and np.std(vec_shft_h)>0 else (0,0)

    # Ablation-CQT Shapley (needed for curves)
    shap_base_abCQT = compute_exact_shapley(mel_t, cqt_t, chr_t, ['T', 'H', 'R', 'S'], "Ablation-CQT", true_label_idx, onset_frames, struct_frames)
    
    print(f"{key[:13]:<15} | {track['genre']:<10} | {hac_full:.4f}               | {hac_abH:.4f}")

    # Collect curves for the first 3 samples
    if idx < 3:
        curves["HarmonicSHAP"].append(get_raw_deletion_curve(mel_t, cqt_t, chr_t, ['T', 'H', 'R', 'S'], shap_base_full, "HarmonicSHAP", true_label_idx, onset_frames, struct_frames))
        curves["Ablation-H"].append(get_raw_deletion_curve(mel_t, cqt_t, chr_t, ['T', 'R', 'S'], shap_base_abH, "Ablation-H", true_label_idx, onset_frames, struct_frames))
        curves["Ablation-CQT"].append(get_raw_deletion_curve(mel_t, cqt_t, chr_t, ['T', 'H', 'R', 'S'], shap_base_abCQT, "Ablation-CQT", true_label_idx, onset_frames, struct_frames))

print("\n=== DIAGNOSTIC 2: Generating Deletion Curve Plots ===")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i in range(3):
    axes[i].plot(np.linspace(0, 1, len(curves["HarmonicSHAP"][i])), curves["HarmonicSHAP"][i], marker='o', label="HarmonicSHAP (4 players)")
    axes[i].plot(np.linspace(0, 1, len(curves["Ablation-H"][i])), curves["Ablation-H"][i], marker='x', label="Ablation-H (3 players)")
    axes[i].plot(np.linspace(0, 1, len(curves["Ablation-CQT"][i])), curves["Ablation-CQT"][i], marker='s', label="Ablation-CQT (4 players)")
    axes[i].set_title(f"Sample {i+1} Deletion Curve")
    axes[i].set_xlabel("Proportion of Features Deleted")
    axes[i].set_ylabel("Model Confidence")
    axes[i].set_ylim(0, 1.05)
    if i == 0: 
        axes[i].legend()

plt.tight_layout()
plot_path = os.path.join(CHECKPOINT_DIR, "diag_curves.png")
plt.savefig(plot_path)
plt.close()
print(f"Plot saved to {plot_path}. Please review the starting confidence at Step 0.")

# --- Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")