# ==============================================================================
# Program Name: 08_exp3_generalization.py
# Version: 1.0
# Description: Conducts Experiment 3 (HAC Robustness Study).
#              1. Downloads Exp 3 datasets from the Google Drive project folder.
#              2. Applies pitch transpositions (±2, ±4 semitones) and tempo 
#                 perturbations (±10%).
#              3. Computes Harmonic Attribution Consistency (HAC) to evaluate 
#                 the robustness of the semantic hierarchy across datasets.
# Dependencies: torch, scipy, numpy, itertools, math
# Outputs: 
#   - exp3_generalization_results.json (saved to Google Drive)
# ==============================================================================

import os
import sys
import torch
import torch.nn.functional as F
import pickle
import shutil
import numpy as np
import math
import itertools
import random
from scipy.stats import pearsonr, sem
import torch.nn as nn
import json

# --- Rigorous Seed Locking ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Configuration & Google Drive Helpers ---
PROJECT_DIR = "/content/drive/MyDrive/paper/harmonic-SHAP/"  # Persistent storage
CHECKPOINT_DIR = "/content/checkpoints"

def ensure_project_dir():
    """Create project directory in Google Drive if it doesn't exist."""
    os.makedirs(PROJECT_DIR, exist_ok=True)

def save_to_drive(local_filepath, remote_filename):
    """Copy a local file to Google Drive project folder."""
    ensure_project_dir()
    dest_path = os.path.join(PROJECT_DIR, remote_filename)
    try:
        shutil.copy2(local_filepath, dest_path)
        print(f"  [DRIVE OK] {local_filepath}  →  {dest_path}")
    except Exception as e:
        print(f"  [DRIVE FAIL] {local_filepath}: {e}")

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

def list_drive_files():
    """List files in the Google Drive project directory."""
    ensure_project_dir()
    try:
        return [f for f in os.listdir(PROJECT_DIR) if os.path.isfile(os.path.join(PROJECT_DIR, f))]
    except Exception as e:
        print(f"  [DRIVE] Could not list files: {e}")
        return []

# --- Step 1: Load Model and Data ---
print("\n--- Step 1: Loading Checkpoints & Datasets from Google Drive ---")

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Get list of files already on Drive
drive_files = list_drive_files()

required_files = [
    "crnn_backbone_weights.pth", 
    "label_encoder.pkl",
    "exp3_features_giantsteps.pkl",
    "exp3_features_fma.pkl",
    "exp3_features_ballroom.pkl"
]

for file in required_files:
    local_path = os.path.join(CHECKPOINT_DIR, file)
    if not os.path.exists(local_path):
        if file in drive_files:
            print(f"Downloading {file} from Google Drive...")
            load_from_drive(file, local_path)
        else:
            print(f"[ERROR] {file} not found in Drive or locally.")
            sys.exit(1)
    else:
        print(f"{file} already exists locally.")

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

with open(os.path.join(CHECKPOINT_DIR, "label_encoder.pkl"), 'rb') as f:
    le = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crnn = MultiBranchCRNN(len(le.classes_)).to(device)
crnn.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "crnn_backbone_weights.pth")))
crnn.eval()

# --- Step 2: Define Transformation & Shapley Logic ---
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

def shift_pitch(tensor, semitones):
    """Simulates pitch transposition by rolling the frequency axis."""
    return torch.roll(tensor, shifts=semitones, dims=2)

def shift_tempo(tensor, factor):
    """Simulates tempo change by interpolating along the time axis."""
    b, c, f, t = tensor.shape
    tensor_3d = tensor.squeeze(0)  # (1, f, t)
    # Linear interpolation along time axis
    shifted = F.interpolate(tensor_3d, scale_factor=factor, mode='linear', align_corners=False)
    shifted = shifted.unsqueeze(0)  # Back to (1, 1, f, t_new)
    
    # Repad or truncate to TARGET_FRAMES
    if shifted.size(3) < TARGET_FRAMES:
        pad_amount = TARGET_FRAMES - shifted.size(3)
        shifted = F.pad(shifted, (0, pad_amount, 0, 0))
    else:
        shifted = shifted[:, :, :, :TARGET_FRAMES]
    return shifted

def apply_semantic_mask(mel, cqt, chroma, coal, onset_frames, struct_frames):
    m_mel, m_cqt, m_chr = mel.clone(), cqt.clone(), chroma.clone()
    if 'T' not in coal: 
        m_mel *= 0
    if 'H' not in coal: 
        m_cqt *= 0
        m_chr *= 0
    if 'R' not in coal: 
        m_mel[:,:,:,onset_frames] *= 0.5
        m_cqt[:,:,:,onset_frames] *= 0.5
        m_chr[:,:,:,onset_frames] *= 0.5
    if 'S' not in coal:
        m_mel[:,:,:,struct_frames] *= 0
        m_cqt[:,:,:,struct_frames] *= 0
        m_chr[:,:,:,struct_frames] *= 0
    return m_mel, m_cqt, m_chr

def compute_exact_shapley(mel, cqt, chroma, true_label_idx, onset_frames, struct_frames):
    players = ['T', 'H', 'R', 'S']
    coalitions = [frozenset(c) for r in range(5) for c in itertools.combinations(players, r)]
    val_dict = {}
    
    with torch.no_grad():
        for coal in coalitions:
            m_mel, m_cqt, m_chr = apply_semantic_mask(mel, cqt, chroma, coal, onset_frames, struct_frames)
            out = crnn(m_mel, m_cqt, m_chr)
            val_dict[coal] = torch.softmax(out, dim=1)[0, true_label_idx].item()
            
    shapley_vals = {p: 0 for p in players}
    for player in players:
        for coal in coalitions:
            if player not in coal:
                coal_with = coal.union([player])
                weight = (math.factorial(len(coal)) * math.factorial(4 - len(coal) - 1)) / 24.0
                shapley_vals[player] += weight * (val_dict[coal_with] - val_dict[coal])
    return shapley_vals

# --- Step 3: Robustness Evaluation Loop ---
print("\n--- Step 3: Running Robustness Transformations ---")

datasets = {
    "GiantSteps+": "exp3_features_giantsteps.pkl",
    "FMA-Small": "exp3_features_fma.pkl",
    "Ballroom": "exp3_features_ballroom.pkl"
}

transformations = {
    "Pitch_Plus_2": lambda m, c, ch: (shift_pitch(m, 2), shift_pitch(c, 2), shift_pitch(ch, 2)),
    "Pitch_Minus_2": lambda m, c, ch: (shift_pitch(m, -2), shift_pitch(c, -2), shift_pitch(ch, -2)),
    "Pitch_Plus_4": lambda m, c, ch: (shift_pitch(m, 4), shift_pitch(c, 4), shift_pitch(ch, 4)),
    "Pitch_Minus_4": lambda m, c, ch: (shift_pitch(m, -4), shift_pitch(c, -4), shift_pitch(ch, -4)),
    "Tempo_Plus_10": lambda m, c, ch: (shift_tempo(m, 0.9), shift_tempo(c, 0.9), shift_tempo(ch, 0.9)),  # 10% faster = scale time by 0.9
    "Tempo_Minus_10": lambda m, c, ch: (shift_tempo(m, 1.1), shift_tempo(c, 1.1), shift_tempo(ch, 1.1))   # 10% slower = scale time by 1.1
}

results = {ds: {trans: [] for trans in transformations.keys()} for ds in datasets.keys()}

for ds_name, pkl_file in datasets.items():
    print(f"\nProcessing {ds_name}...")
    pkl_path = os.path.join(CHECKPOINT_DIR, pkl_file)
    
    with open(pkl_path, 'rb') as f:
        ds_data = pickle.load(f)
        
    track_keys = list(ds_data.keys())
    
    for idx, key in enumerate(track_keys):
        sys.stdout.write(f"\r  Evaluating track {idx+1}/{len(track_keys)}...")
        sys.stdout.flush()
        
        track = ds_data[key]
        
        # We need a proxy label for entirely unseen datasets to calculate baseline confidence drop. 
        # For a rigorous XAI stability test, we use the model's own pseudo-label as the target.
        mel_t = format_tensor(track['features']['mel'])
        cqt_t = format_tensor(track['features']['cqt'])
        chr_t = format_tensor(track['features']['chroma'])
        onset_frames, struct_frames = get_onsets_and_structure(mel_t)
        
        with torch.no_grad():
            base_out = crnn(mel_t, cqt_t, chr_t)
            pseudo_label_idx = base_out.argmax(dim=1).item()
            
        shap_base = compute_exact_shapley(mel_t, cqt_t, chr_t, pseudo_label_idx, onset_frames, struct_frames)
        vec_base = [shap_base[p] for p in ['T', 'H', 'R', 'S']]
        
        # Standard deviation check to prevent undefined Pearson correlation
        if np.std(vec_base) == 0: 
            continue
            
        for trans_name, trans_func in transformations.items():
            m_trans, c_trans, ch_trans = trans_func(mel_t, cqt_t, chr_t)
            
            # Recalculate onsets/structure for tempo shifts as time mapping changes
            if "Tempo" in trans_name:
                trans_onsets, trans_structs = get_onsets_and_structure(m_trans)
            else:
                trans_onsets, trans_structs = onset_frames, struct_frames
                
            shap_trans = compute_exact_shapley(m_trans, c_trans, ch_trans, pseudo_label_idx, trans_onsets, trans_structs)
            vec_trans = [shap_trans[p] for p in ['T', 'H', 'R', 'S']]
            
            if np.std(vec_trans) > 0:
                hac, _ = pearsonr(vec_base, vec_trans)
                results[ds_name][trans_name].append(hac)

print("\n\n=== Experiment 3: HAC Robustness Results (Mean ± SE) ===")

final_metrics = {}
for ds_name in datasets.keys():
    print(f"\n[{ds_name}]")
    final_metrics[ds_name] = {}
    
    for trans_name in transformations.keys():
        data = results[ds_name][trans_name]
        if len(data) > 0:
            mean_val, se_val = np.mean(data), sem(data)
            print(f"  {trans_name:<15}: {mean_val:.4f} ± {se_val:.4f}")
            final_metrics[ds_name][trans_name] = {"Mean": float(mean_val), "SE": float(se_val)}
        else:
            print(f"  {trans_name:<15}: N/A (Failed to correlate)")

json_path = os.path.join(CHECKPOINT_DIR, "exp3_generalization_results.json")
with open(json_path, 'w') as f:
    json.dump(final_metrics, f, indent=4)
save_to_drive(json_path, "exp3_generalization_results.json")

# --- Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")

print("\n[SUCCESS] Script 08 completed. Robustness metrics saved to Google Drive.")