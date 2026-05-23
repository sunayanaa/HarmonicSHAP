# ==============================================================================
# Program Name: 04_exp1_quantitative_metrics.py
# Version: 1.0
# Description: Computes the full quantitative metrics (Deletion/Insertion AUC) 
#              for all baselines outlined in Experiment 1. Implements the missing
#              SHAP-LM and Standard Acoustic SHAP baselines. Generates a JSON
#              table of the final metrics and saves it to the Google Drive project folder.
#				EVAL_SAMPLES = 200 # Number of samples to evaluate (20% of total)
#				.pkl state file that saves eval_data queue and current results to the Google Drive folder every 5 iterations,
# Change Log: 
#   - 1.0: Full implementation of Deletion/Insertion AUC pipeline for all models.
# GPU Required: Yes (Highly recommended for Gradient/Deep SHAP and CRNN passes)
# Dependencies: torch, shap, sklearn
# Inputs: 
#   - gtzan_features_chunk_*.pkl
#   - xgboost_baseline.json
#   - crnn_backbone_weights.pth
# Outputs: 
#   - exp1_quantitative_results.json (saved to Google Drive)
# ==============================================================================
!pip install shap scikit-learn numpy

import sys
import os
import torch

# --- GPU Check ---
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU to compute attribution metrics in a reasonable time.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

import os
import glob
import pickle
import shutil
import numpy as np
import xgboost as xgb
import shap
import torch.nn as nn
from sklearn.metrics import auc
from tqdm import tqdm
import json
import warnings
import itertools
import math
warnings.filterwarnings('ignore')  # Suppress SHAP/PyTorch warnings for clean output

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/harmonic-SHAP/"  # Persistent storage

CHECKPOINT_DIR = "/content/checkpoints"
EVAL_SAMPLES = 200  # Number of samples to evaluate (20% of total)
MASK_STEPS = 10     # Number of steps for Deletion/Insertion curves (10%, 20%, etc.)
TARGET_FRAMES = 1290

# --- Google Drive Helper Functions ---
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

# --- CRNN Architecture Definition ---
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
            
        x_fused = torch.cat([prep_for_rnn(x_mel), prep_for_rnn(x_cqt), prep_for_rnn(x_chroma)], dim=-1)
        out, _ = self.gru(x_fused)
        return self.fc(out[:, -1, :])

# --- Helper: Tensor Formatting ---
def format_tensor(x, device):
    if x.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad_width)), mode='constant')
    else:
        x = x[:, :TARGET_FRAMES]
    return torch.FloatTensor(x).unsqueeze(0).unsqueeze(0).to(device)

# --- Metric: Deletion / Insertion AUC ---
def calc_del_ins_auc(model, inputs, attribution_map, true_class, is_xgb=False):
    """
    Masks features from most important to least (Deletion) and least to most (Insertion).
    Returns Area Under the Curve (AUC). 
    Lower Deletion AUC = Better (model drops confidence fast).
    Higher Insertion AUC = Better (model gains confidence fast).
    """
    if is_xgb:
        original_input = inputs.copy()
        flat_attr = attribution_map.flatten()
        sorted_indices = np.argsort(-np.abs(flat_attr))  # Descending importance
    else:
        mel, cqt, chroma = inputs
        original_mel, original_cqt, original_chroma = mel.clone(), cqt.clone(), chroma.clone()
        # Flatten and sort all acoustic pixels
        flat_attr = attribution_map.flatten()
        sorted_indices = np.argsort(-np.abs(flat_attr))

    total_features = len(flat_attr)
    step_size = total_features // MASK_STEPS
    
    del_probs, ins_probs = [], []
    
    for step in range(MASK_STEPS + 1):
        num_mask = step * step_size
        if step == MASK_STEPS: 
            num_mask = total_features
        
        mask_indices = sorted_indices[:num_mask]
        
        if is_xgb:
            # Deletion: Start full, zero out top features
            del_input = original_input.copy()
            del_input[0, mask_indices] = 0
            del_probs.append(model.predict_proba(del_input)[0][true_class])
            
            # Insertion: Start empty, add top features
            ins_input = np.zeros_like(original_input)
            ins_input[0, mask_indices] = original_input[0, mask_indices]
            ins_probs.append(model.predict_proba(ins_input)[0][true_class])
        else:
            # For CRNN, mapping flattened indices back to 3D tensors is complex for this script's scope,
            # so we approximate by masking whole time-frames based on aggregated temporal attribution.
            
            # Simplified proxy for CRNN pixel-masking for fast evaluation
            # FIX: axis=0 averages over the frequency dimension (128) to give temporal importance (1290)
            time_attr = np.mean(attribution_map, axis=0)  # Aggregate over freq
            sorted_time_idx = np.argsort(-np.abs(time_attr))
            mask_t_idx = sorted_time_idx[:(step * TARGET_FRAMES // MASK_STEPS)]
            
            del_mel, del_cqt, del_chroma = original_mel.clone(), original_cqt.clone(), original_chroma.clone()
            ins_mel, ins_cqt, ins_chroma = torch.zeros_like(original_mel), torch.zeros_like(original_cqt), torch.zeros_like(original_chroma)
            
            for t in mask_t_idx:
                del_mel[:,:,:,t] = 0
                del_cqt[:,:,:,t] = 0
                del_chroma[:,:,:,t] = 0
                ins_mel[:,:,:,t] = original_mel[:,:,:,t]
                ins_cqt[:,:,:,t] = original_cqt[:,:,:,t]
                ins_chroma[:,:,:,t] = original_chroma[:,:,:,t]
                
            with torch.no_grad():
                del_probs.append(torch.softmax(model(del_mel, del_cqt, del_chroma), dim=1)[0, true_class].item())
                ins_probs.append(torch.softmax(model(ins_mel, ins_cqt, ins_chroma), dim=1)[0, true_class].item())

    x_axis = np.linspace(0, 1, MASK_STEPS + 1)
    return auc(x_axis, del_probs), auc(x_axis, ins_probs)

# --- Execution ---
print("\n--- Loading Models & Data ---")

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

device = torch.device("cuda")

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Get list of files already on Drive
drive_files = list_drive_files()

# Download required models and encoders from Drive if missing locally
required_files = [
    "label_encoder.pkl", 
    "xgboost_baseline.json", 
    "crnn_backbone_weights.pth"
]

for file in required_files:
    local_path = os.path.join(CHECKPOINT_DIR, file)
    if not os.path.exists(local_path):
        if file in drive_files:
            print(f"[DRIVE] Downloading {file}...")
            load_from_drive(file, local_path)
        else:
            print(f"[ERROR] {file} not found in Drive or locally.")
            sys.exit(1)

# Now safely load the models
with open(os.path.join(CHECKPOINT_DIR, "label_encoder.pkl"), 'rb') as f:
    le = pickle.load(f)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(CHECKPOINT_DIR, "xgboost_baseline.json"))
xgb_explainer = shap.TreeExplainer(xgb_model)

crnn = MultiBranchCRNN(len(le.classes_)).to(device)
crnn.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "crnn_backbone_weights.pth")))
crnn.eval()

# Ensure feature chunks exist locally. If not, download them from Drive.
local_chunks = glob.glob(os.path.join(CHECKPOINT_DIR, "gtzan_features_chunk_*.pkl"))
if not local_chunks:
    print("[DRIVE] No feature chunks found locally. Checking Google Drive...")
    chunk_files = [f for f in drive_files if f.startswith("gtzan_features_chunk_")]
    if chunk_files:
        for chunk_file in chunk_files:
            print(f"[DRIVE] Downloading {chunk_file}...")
            load_from_drive(chunk_file, os.path.join(CHECKPOINT_DIR, chunk_file))
        local_chunks = glob.glob(os.path.join(CHECKPOINT_DIR, "gtzan_features_chunk_*.pkl"))
    else:
        print("[ERROR] No feature chunks found in Drive or locally.")
        sys.exit(1)

# --- State Management & Resumption ---
CHECKPOINT_NAME = "exp1_metrics_state.pkl"
local_state_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

if load_from_drive(CHECKPOINT_NAME, local_state_path):
    print("\n[INFO] Found existing state file on Drive. Resuming evaluation...")
    with open(local_state_path, 'rb') as f:
        state = pickle.load(f)
        eval_data = state['eval_data']
        results = state['results']
        processed_count = state['processed_count']
else:
    print("\n[INFO] Starting fresh evaluation...")
    # Load all chunks to sample from
    all_files = glob.glob(os.path.join(CHECKPOINT_DIR, "gtzan_features_chunk_*.pkl"))
    eval_data = []
    for file in all_files:
        with open(file, 'rb') as f:
            chunk = pickle.load(f)
            eval_data.extend([(k, v) for k, v in chunk.items()])

    # Create a deterministic, shuffled evaluation queue
    np.random.seed(42)
    np.random.shuffle(eval_data)
    eval_data = eval_data[:EVAL_SAMPLES]
    
    results = {
        "Vanilla SHAP (XGBoost)": {"del": [], "ins": []},
        "SHAP-LM Proxy (XGBoost)": {"del": [], "ins": []},
        "Standard Acoustic SHAP (CRNN)": {"del": [], "ins": []},
        "HarmonicSHAP (CRNN)": {"del": [], "ins": []}
    }
    processed_count = 0

print(f"Total samples to evaluate: {EVAL_SAMPLES}. Already processed: {processed_count}")

# Only process what remains in the queue
remaining_data = eval_data[processed_count:]

for i, (name, data) in enumerate(tqdm(remaining_data, desc="Calculating Explanations & Metrics")):
    true_label_idx = le.transform([data['genre']])[0]
    
    # 1. XGBoost Data Prep
    mfcc = data['features']['mfcc']
    X_sample = np.concatenate([np.mean(mfcc, axis=1), np.var(mfcc, axis=1)]).reshape(1, -1)
    
    # 2. CRNN Data Prep
    mel_t = format_tensor(data['features']['mel'], device)
    cqt_t = format_tensor(data['features']['cqt'], device)
    chroma_t = format_tensor(data['features']['chroma'], device)
    
    # --- Baseline 1: Vanilla SHAP (XGB) ---
    shap_vals_xgb = xgb_explainer.shap_values(X_sample)
    if isinstance(shap_vals_xgb, list): 
        shap_vals_xgb = shap_vals_xgb[true_label_idx][0]
    else: 
        shap_vals_xgb = shap_vals_xgb[0, :, true_label_idx]
    
    del_auc, ins_auc = calc_del_ins_auc(xgb_model, X_sample, shap_vals_xgb, true_label_idx, is_xgb=True)
    results["Vanilla SHAP (XGBoost)"]["del"].append(del_auc)
    results["Vanilla SHAP (XGBoost)"]["ins"].append(ins_auc)
    
    # --- Baseline 4: SHAP-LM Proxy (Guided Feature Selection) ---
    top_50_idx = np.argsort(-np.abs(shap_vals_xgb))[:20]
    shap_lm_attr = np.zeros_like(shap_vals_xgb)
    shap_lm_attr[top_50_idx] = shap_vals_xgb[top_50_idx]
    
    del_auc, ins_auc = calc_del_ins_auc(xgb_model, X_sample, shap_lm_attr, true_label_idx, is_xgb=True)
    results["SHAP-LM Proxy (XGBoost)"]["del"].append(del_auc)
    results["SHAP-LM Proxy (XGBoost)"]["ins"].append(ins_auc)
    
    # --- Baseline 3: Standard Acoustic DeepSHAP (CRNN) ---
    mel_t.requires_grad = True
    
    # FIX: Temporarily set model to train mode for cuDNN backward pass, 
    # but keep BatchNorm layers in eval mode to preserve statistics.
    crnn.train()
    for m in crnn.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            
    crnn.zero_grad()
    out = crnn(mel_t, cqt_t, chroma_t)
    out[0, true_label_idx].backward()
    acoustic_shap_approx = (mel_t.grad * mel_t).squeeze().cpu().detach().numpy()
    
    # Return model to eval mode and clean up gradients
    crnn.eval()
    mel_t.requires_grad = False
    crnn.zero_grad()
    
    del_auc, ins_auc = calc_del_ins_auc(crnn, (mel_t, cqt_t, chroma_t), acoustic_shap_approx, true_label_idx, is_xgb=False)
    results["Standard Acoustic SHAP (CRNN)"]["del"].append(del_auc)
    results["Standard Acoustic SHAP (CRNN)"]["ins"].append(ins_auc)
    
    # --- HarmonicSHAP (Exact Semantic Coalition Players) ---
    # 1. Define the semantic masks (The 4 Players)
    # T: Timbral (Mel), H: Harmonic (CQT/Chroma), R: Rhythmic (Onsets), S: Structural (Time segment)
    def apply_coalition_mask(mel, cqt, chroma, coalition, onset_frames, structural_frames):
        m_mel, m_cqt, m_chroma = mel.clone(), cqt.clone(), chroma.clone()
        
        if 'T' not in coalition: 
            m_mel *= 0
        if 'H' not in coalition: 
            m_cqt *= 0
            m_chroma *= 0
        if 'R' not in coalition:  # Mask rhythmic onsets
            m_mel[:,:,:,onset_frames] *= 0.5  # Suppress transients rather than hard zero
            m_cqt[:,:,:,onset_frames] *= 0.5
            m_chroma[:,:,:,onset_frames] *= 0.5
        if 'S' not in coalition:  # Mask structural segment (e.g., 2nd half of track)
            m_mel[:,:,:,structural_frames] *= 0
            m_cqt[:,:,:,structural_frames] *= 0
            m_chroma[:,:,:,structural_frames] *= 0
            
        return m_mel, m_cqt, m_chroma

    # Detect onset frames (rough proxy for Rhythmic player locations)
    mel_np = mel_t.squeeze().cpu().numpy()
    spectral_flux = np.mean(np.diff(mel_np, axis=1) > 0, axis=0)  # positive energy changes
    onset_threshold = np.percentile(spectral_flux, 85)
    onset_frames = np.where(spectral_flux > onset_threshold)[0]
    
    # Define structural frames (e.g., frames after midpoint)
    structural_frames = np.arange(TARGET_FRAMES // 2, TARGET_FRAMES)
    
    # 2. Compute exact Shapley Values for the 4 players (2^4 = 16 inferences)
    players = ['T', 'H', 'R', 'S']
    coalitions = []
    for r in range(len(players) + 1):
        coalitions.extend(itertools.combinations(players, r))
        
    value_dict = {}
    crnn.eval()
    with torch.no_grad():
        for coal in coalitions:
            m_mel, m_cqt, m_chroma = apply_coalition_mask(mel_t, cqt_t, chroma_t, coal, onset_frames, structural_frames)
            out = crnn(m_mel, m_cqt, m_chroma)
            # Use frozenset as the dictionary key to completely ignore order
            value_dict[frozenset(coal)] = torch.softmax(out, dim=1)[0, true_label_idx].item()
            
    shapley_vals = {'T': 0, 'H': 0, 'R': 0, 'S': 0}
    for player in players:
        for coal in coalitions:
            if player not in coal:
                # Use frozensets for the lookup
                coal_with = frozenset(list(coal) + [player])
                coal_without = frozenset(coal)
                
                weight = (math.factorial(len(coal)) * math.factorial(4 - len(coal) - 1)) / math.factorial(4)
                marginal_contribution = value_dict[coal_with] - value_dict.get(coal_without, 0)
                shapley_vals[player] += weight * marginal_contribution

    # 3. Construct the True Semantic Attribution Map
    # Broadcast the computed Shapley values to the time-frequency bins they govern
    semantic_map = np.zeros((128, TARGET_FRAMES))
    
    # Timbral importance applies to Mel representation
    semantic_map += shapley_vals['T'] 
    # Harmonic importance applies heavily where CQT/Chroma energy exists
    cqt_energy = cqt_t.squeeze().cpu().numpy().mean(axis=0)
    semantic_map += (cqt_energy / (np.max(cqt_energy) + 1e-10)) * shapley_vals['H']
    # Rhythmic importance applies to onset frames
    semantic_map[:, onset_frames] += shapley_vals['R']
    # Structural importance applies to the targeted segment
    semantic_map[:, structural_frames] += shapley_vals['S']
    
    del_auc, ins_auc = calc_del_ins_auc(crnn, (mel_t, cqt_t, chroma_t), semantic_map, true_label_idx, is_xgb=False)
    results["HarmonicSHAP (CRNN)"]["del"].append(del_auc)
    results["HarmonicSHAP (CRNN)"]["ins"].append(ins_auc)    
    processed_count += 1
    
    # --- Periodic Checkpointing (Every 5 samples) ---
    if processed_count % 5 == 0 or processed_count == EVAL_SAMPLES:
        print(f"\n[INFO] Saving checkpoint at {processed_count}/{EVAL_SAMPLES} samples...")
        state = {
            'eval_data': eval_data,
            'results': results,
            'processed_count': processed_count
        }
        with open(local_state_path, 'wb') as f:
            pickle.dump(state, f)
        save_to_drive(local_state_path, CHECKPOINT_NAME)

# --- Aggregate and Save Final JSON Results ---
print("\n=== Experiment 1: Quantitative Results (AUC) ===")
print("Lower Deletion AUC is better. Higher Insertion AUC is better.\n")

final_metrics = {}
for model_name, metrics in results.items():
    avg_del = float(np.mean(metrics["del"]))
    avg_ins = float(np.mean(metrics["ins"]))
    final_metrics[model_name] = {"Avg_Deletion_AUC": avg_del, "Avg_Insertion_AUC": avg_ins}
    
    print(f"{model_name}:")
    print(f"  - Deletion AUC:  {avg_del:.4f}")
    print(f"  - Insertion AUC: {avg_ins:.4f}\n")

json_path = os.path.join(CHECKPOINT_DIR, "exp1_quantitative_results.json")
with open(json_path, 'w') as f:
    json.dump(final_metrics, f, indent=4)

save_to_drive(json_path, "exp1_quantitative_results.json")

# --- Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")

print("[SUCCESS] Script 04 completed. Metrics table saved to Google Drive.")