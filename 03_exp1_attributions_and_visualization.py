# ==============================================================================
# Program Name: 03_exp1_attributions_and_visualization.py
# Version: 2.0
# Description: Generates the baseline explanations (Vanilla SHAP, Grad-CAM) and 
#              the temporal semantic attribution profile for a sample track. 
#              Creates the side-by-side visual comparison for Experiment 1 
#              as specified in the Blueprint.
# Change Log: 
#   - 1.0: Implementation of XGBoost SHAP, CRNN Mel Grad-CAM, and Temporal SHAP.
#   - 2.0: Implemented exact 4-player Shapley game for temporal sections, 
#          enhanced Grad-CAM normalization, fixed missing hooks/data vars, 
#          and updated plot labels for manuscript alignment.
# GPU Required: Yes
# Dependencies: torch, shap, cv2, matplotlib, itertools, math
# Inputs: 
#   - gtzan_features_chunk_1.pkl (for sample extraction)
#   - xgboost_baseline.json
#   - crnn_backbone_weights.pth
# Outputs: 
#   - fig_03_01_attribution_comparison.png (saved to Google Drive)
# ==============================================================================

import sys
import os
import torch
import glob
import pickle
import shutil
import numpy as np
import xgboost as xgb
import shap
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import itertools
import math

# --- GPU Check ---
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/harmonic-SHAP/"  # Persistent storage

CHECKPOINT_DIR = "/content/checkpoints"

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

# --- End Google Drive Helper Functions ---

# --- Step 1: Load Models and Sample Data ---
print("\n--- Step 1: Loading Models & Sample Data ---")

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Check which files are needed and download from Drive if not present
required_files = ["xgboost_baseline.json", "crnn_backbone_weights.pth", "label_encoder.pkl"]
drive_files = list_drive_files()

for file in required_files:
    local_path = os.path.join(CHECKPOINT_DIR, file)
    if not os.path.exists(local_path):
        if file in drive_files:
            print(f"Downloading {file} from Google Drive...")
            load_from_drive(file, local_path)
        else:
            print(f"Error: {file} not found in Drive or locally.")
            sys.exit(1)

# Also check for feature chunk file
chunk_file = "gtzan_features_chunk_1.pkl"
chunk_path = os.path.join(CHECKPOINT_DIR, chunk_file)
if not os.path.exists(chunk_path):
    if chunk_file in drive_files:
        print(f"Downloading {chunk_file} from Google Drive...")
        load_from_drive(chunk_file, chunk_path)
    else:
        print(f"Error: {chunk_file} not found in Drive or locally.")
        sys.exit(1)

with open(os.path.join(CHECKPOINT_DIR, "label_encoder.pkl"), 'rb') as f:
    le = pickle.load(f)
num_classes = len(le.classes_)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(CHECKPOINT_DIR, "xgboost_baseline.json"))

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

device = torch.device("cuda")
crnn_model = MultiBranchCRNN(num_classes).to(device)
crnn_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "crnn_backbone_weights.pth")))
crnn_model.eval()

# Load chunk data and find a Reggae track for the visualization
with open(chunk_path, 'rb') as f:
    chunk_data = pickle.load(f)

sample_data = None
for track_id, track_info in chunk_data.items():
    if 'reggae' in str(track_info.get('genre', '')).lower():
        sample_data = track_info
        print(f"Selected Sample: {track_id} (True Genre: {track_info['genre']})")
        break

if sample_data is None:
    sample_key = list(chunk_data.keys())[0]
    sample_data = chunk_data[sample_key]
    print(f"Reggae track not found. Falling back to: {sample_key} (True Genre: {sample_data['genre']})")

# --- Step 2: Compute Vanilla SHAP (XGBoost) ---
print("\n--- Step 2: Computing Vanilla SHAP (XGBoost) ---")
mfcc = sample_data['features']['mfcc']
mfcc_mean = np.mean(mfcc, axis=1)
mfcc_var = np.var(mfcc, axis=1)
X_sample = np.concatenate([mfcc_mean, mfcc_var]).reshape(1, -1)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)
pred_class_xgb = xgb_model.predict(X_sample)[0]

if isinstance(shap_values, list):
    shap_vals_sample = shap_values[pred_class_xgb][0]
else:
    shap_vals_sample = shap_values[0, :, pred_class_xgb]

# --- Step 3: Compute Grad-CAM (CRNN Mel Saliency) ---
print("\n--- Step 3: Computing Grad-CAM (CRNN Mel Saliency) ---")
target_frames = 1290
def format_tensor(x):
    if x.shape[1] < target_frames:
        pad_width = target_frames - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad_width)), mode='constant')
    else:
        x = x[:, :target_frames]
    return torch.FloatTensor(x).unsqueeze(0).unsqueeze(0).to(device)

mel_t = format_tensor(sample_data['features']['mel'])
cqt_t = format_tensor(sample_data['features']['cqt'])
chroma_t = format_tensor(sample_data['features']['chroma'])

gradients = None
activations = None

def bwd_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def fwd_hook(module, input, output):
    global activations
    activations = output

target_layer = crnn_model.mel_cnn[-1]
handle_fwd = target_layer.register_forward_hook(fwd_hook)
handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

crnn_model.train()
for m in crnn_model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

crnn_model.zero_grad()
out = crnn_model(mel_t, cqt_t, chroma_t)
pred_class_crnn = out.argmax(dim=1).item()
out[0, pred_class_crnn].backward()

handle_fwd.remove()
handle_bwd.remove()

pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
for i in range(activations.size(1)):
    activations[:, i, :, :] *= pooled_gradients[i]
    
heatmap = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()
heatmap = np.maximum(heatmap, 0)
heatmap /= (np.max(heatmap) + 1e-10)
heatmap = np.power(heatmap, 0.5) 
heatmap = cv2.resize(heatmap, (mel_t.shape[3], mel_t.shape[2]))

crnn_model.eval()

# --- Step 4: Temporal Coalition Attribution (Exact HarmonicSHAP) ---
print("\n--- Step 4: Computing Exact Temporal Attribution Profile ---")
n_sections = 4
section_length = target_frames // n_sections
players = ['T', 'H', 'R', 'S']
coalitions = [frozenset(c) for r in range(5) for c in itertools.combinations(players, r)]

temporal_attributions = {p: [] for p in players}

for i in range(n_sections):
    start = i * section_length
    end = (i + 1) * section_length
    
    val_dict = {}
    crnn_model.eval()
    
    for coal in coalitions:
        m_mel, m_cqt, m_chroma = mel_t.clone(), cqt_t.clone(), chroma_t.clone()
        
        if 'T' not in coal: m_mel[:, :, :, start:end] *= 0
        if 'H' not in coal: m_cqt[:, :, :, start:end] *= 0; m_chroma[:, :, :, start:end] *= 0
        if 'R' not in coal:
            m_mel[:, :, :, start:end] *= 0.5 
            m_cqt[:, :, :, start:end] *= 0.5
            m_chroma[:, :, :, start:end] *= 0.5
        if 'S' not in coal:
            m_mel[:, :, :, start:end] *= 0
            m_cqt[:, :, :, start:end] *= 0
            m_chroma[:, :, :, start:end] *= 0
            
        with torch.no_grad():
            out = crnn_model(m_mel, m_cqt, m_chroma)
            val_dict[coal] = torch.softmax(out, dim=1)[0, pred_class_crnn].item()
            
    for player in players:
        shap_val = 0
        for coal in coalitions:
            if player not in coal:
                coal_with = coal.union([player])
                weight = (math.factorial(len(coal)) * math.factorial(4 - len(coal) - 1)) / 24.0
                shap_val += weight * (val_dict[coal_with] - val_dict[coal])
        temporal_attributions[player].append(shap_val)

# --- Step 5: Plotting the Centerpiece Figure ---
print("\n--- Step 5: Generating Visualization ---")
fig = plt.figure(figsize=(15, 12))

# Subplot 1: Vanilla SHAP (MFCC Features)
ax1 = plt.subplot(3, 1, 1)
features = [f"MFCC_Mean_{i}" for i in range(20)] + [f"MFCC_Var_{i}" for i in range(20)]
shap.summary_plot(shap_vals_sample.reshape(1, -1), X_sample, feature_names=features, plot_type="bar", show=False)
plt.title(f"Baseline 1: Vanilla SHAP on MFCCs (Predicted: {le.classes_[pred_class_xgb]})", fontsize=12, fontweight='bold')
plt.xlabel("Mean |SHAP Value|")
ax1.tick_params(axis='y', labelsize=5) 
fig.axes[-1].set_aspect('auto') 

# Subplot 2: Grad-CAM (Mel Spectrogram)
ax2 = plt.subplot(3, 1, 2)
mel_db = mel_t.squeeze().cpu().numpy()
plt.imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
plt.imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
plt.title(f"Baseline 2: Grad-CAM Acoustic Saliency (Predicted: {le.classes_[pred_class_crnn]})", fontsize=12, fontweight='bold')
plt.ylabel("Mel Bins")
plt.xlabel("Time Frames")

# Subplot 3: HarmonicSHAP Temporal Profile
ax3 = plt.subplot(3, 1, 3)
x_labels = ['Section 1\n(0-7.5s)', 'Section 2\n(7.5-15s)', 'Section 3\n(15-22.5s)', 'Section 4\n(22.5-30s)']
x = np.arange(len(x_labels))
width = 0.2

ax3.bar(x - 1.5*width, temporal_attributions['T'], width, label='Timbral (T)', color='#1f77b4')
ax3.bar(x - 0.5*width, temporal_attributions['H'], width, label='Harmonic (H)', color='#ff7f0e')
ax3.bar(x + 0.5*width, temporal_attributions['R'], width, label='Rhythmic (R)', color='#2ca02c')
ax3.bar(x + 1.5*width, temporal_attributions['S'], width, label='Structural (S)', color='#d62728')

ax3.set_xticks(x)
ax3.set_xticklabels(x_labels)
ax3.set_ylabel("Genre Confidence Drop (Exact Shapley)")
ax3.set_title("HarmonicSHAP: Temporal Semantic Attribution Profile (NC-3)", fontsize=12, fontweight='bold')
ax3.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
fig_path = os.path.join(CHECKPOINT_DIR, "fig_03_01_attribution_comparison.png")
plt.savefig(fig_path, dpi=300)
plt.close()
save_to_drive(fig_path, "fig_03_01_attribution_comparison.png")

# --- Step 6: Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")

print("\n[SUCCESS] Script 03 completed. Centerpiece figure saved to Google Drive.")