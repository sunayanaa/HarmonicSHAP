# ==============================================================================
# Program Name: 02_exp1_train_models.py
# Version: 1.0
# Description: Trains the baseline classifiers for Experiment 1 using the 
#              extracted multi-resolution features. Trains an XGBoost model on 
#              aggregated MFCCs (baseline 1) and a lightweight Multi-Branch CRNN on Mel, 
#              CQT, and Chroma features (baseline 3). Saves model weights and metrics to Google Drive.
# Change Log: 
#   - 1.0: Implementation of XGBoost and Multi-Branch CRNN training loops.
# GPU Required: Yes (For PyTorch CRNN training)
# Dependencies: torch, xgboost, scikit-learn
# Inputs: gtzan_features_chunk_*.pkl (from local or Google Drive)
# Outputs: 
#   - crnn_backbone_weights.pth (saved to Google Drive)
#   - xgboost_baseline.json (saved to Google Drive)
#   - label_encoder.pkl (saved to Google Drive)
#   - fig_02_01_crnn_training_history.png (saved to Google Drive)
# ==============================================================================

import sys
import torch

# --- GPU Check ---
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU")
    print("Please switch your Colab runtime to a T4 GPU and restart.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

import os
import glob
import pickle
import shutil
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/harmonic-SHAP/"  # Persistent storage

CHECKPOINT_DIR = "/content/checkpoints"
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

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

# --- End Google Drive Helper Functions ---

# --- Step 1: Load Data ---
print("\n--- Step 1: Loading Feature Checkpoints ---")
chunk_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "gtzan_features_chunk_*.pkl")))

if not chunk_files:
    # Try to download from Drive if not found locally
    print("[INFO] No feature chunks found locally. Checking Google Drive...")
    ensure_project_dir()
    drive_files = [f for f in os.listdir(PROJECT_DIR) if f.startswith("gtzan_features_chunk_") and f.endswith(".pkl")]
    
    if not drive_files:
        print("[ERROR] No feature chunks found in Drive or locally. Please ensure Script 01 ran successfully.")
        sys.exit(1)
    
    print(f"Found {len(drive_files)} chunks on Drive. Downloading...")
    for fname in drive_files:
        local_path = os.path.join(CHECKPOINT_DIR, fname)
        if load_from_drive(fname, local_path):
            print(f"  Downloaded {fname}")
        else:
            print(f"  Failed to download {fname}")
    
    chunk_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "gtzan_features_chunk_*.pkl")))

all_features = []
all_labels = []

for chunk_file in chunk_files:
    with open(chunk_file, 'rb') as f:
        chunk_data = pickle.load(f)
        for filename, data in chunk_data.items():
            all_features.append(data['features'])
            all_labels.append(data['genre'])

print(f"Loaded {len(all_labels)} total tracks.")

# Encode Labels
le = LabelEncoder()
encoded_labels = le.fit_transform(all_labels)
num_classes = len(le.classes_)

# Save and upload label encoder to Drive
le_path = os.path.join(CHECKPOINT_DIR, "label_encoder.pkl")
with open(le_path, 'wb') as f:
    pickle.dump(le, f)
save_to_drive(le_path, "label_encoder.pkl")

# Train/Test Split (80/20)
indices = np.arange(len(all_labels))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=encoded_labels)

# --- Step 2: Train XGBoost Baseline (MFCCs) ---
print("\n--- Step 2: Training XGBoost Baseline (MFCCs) ---")
# Aggregate MFCCs over time (mean and variance) to create 1D feature vector per track
def aggregate_mfcc(features_list, indices):
    X = []
    y = []
    for idx in indices:
        mfcc = features_list[idx]['mfcc']
        # Flatten by taking mean and var across the time axis
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        X.append(np.concatenate([mfcc_mean, mfcc_var]))
        y.append(encoded_labels[idx])
    return np.array(X), np.array(y)

X_train_xgb, y_train_xgb = aggregate_mfcc(all_features, train_idx)
X_test_xgb, y_test_xgb = aggregate_mfcc(all_features, test_idx)

xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_xgb, y_train_xgb)

xgb_preds = xgb_model.predict(X_test_xgb)
xgb_acc = accuracy_score(y_test_xgb, xgb_preds)
print(f"XGBoost Baseline Accuracy: {xgb_acc:.4f}")

xgb_path = os.path.join(CHECKPOINT_DIR, "xgboost_baseline.json")
xgb_model.save_model(xgb_path)
save_to_drive(xgb_path, "xgboost_baseline.json")

# --- Step 3: Train Multi-Branch CRNN ---
print("\n--- Step 3: Training Multi-Branch CRNN Backbone ---")

class GTZANDataset(Dataset):
    def __init__(self, features_list, labels, indices):
        self.features = features_list
        self.labels = labels
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        feats = self.features[real_idx]
        
        # Ensure consistent time frames (truncate/pad to 1290 frames for 30s at 22050Hz/512hop)
        target_frames = 1290
        
        def process_tensor(x):
            if x.shape[1] < target_frames:
                pad_width = target_frames - x.shape[1]
                x = np.pad(x, ((0, 0), (0, pad_width)), mode='constant')
            else:
                x = x[:, :target_frames]
            return torch.FloatTensor(x).unsqueeze(0)  # Add channel dim
            
        mel = process_tensor(feats['mel'])
        cqt = process_tensor(feats['cqt'])
        chroma = process_tensor(feats['chroma'])
        label = self.labels[real_idx]
        
        return mel, cqt, chroma, label

train_dataset = GTZANDataset(all_features, encoded_labels, train_idx)
test_dataset = GTZANDataset(all_features, encoded_labels, test_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class MultiBranchCRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Simplified CNN blocks for SPL (Lightweight)
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
        
        # RNN for temporal modeling
        # Input size to GRU depends on the flattened feature maps after CNNs
        self.gru = nn.GRU(input_size=1792, hidden_size=128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, num_classes)  # 128 * 2 for bidirectional

    def forward(self, mel, cqt, chroma):
        x_mel = self.mel_cnn(mel)
        x_cqt = self.cqt_cnn(cqt)
        x_chroma = self.chroma_cnn(chroma)
        
        # Reshape for RNN: (Batch, Channels, Freq, Time) -> (Batch, Time, Features)
        def prep_for_rnn(x):
            b, c, f, t = x.size()
            return x.permute(0, 3, 1, 2).reshape(b, t, -1)
            
        x_mel = prep_for_rnn(x_mel)
        x_cqt = prep_for_rnn(x_cqt)
        x_chroma = prep_for_rnn(x_chroma)
        
        # Early Fusion (Concatenate along feature dimension)
        x_fused = torch.cat([x_mel, x_cqt, x_chroma], dim=-1)
        
        out, _ = self.gru(x_fused)
        out = out[:, -1, :]  # Take last hidden state
        out = self.fc(out)
        return out

device = torch.device("cuda")
crnn_model = MultiBranchCRNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(crnn_model.parameters(), lr=LEARNING_RATE)

train_losses, test_accs = [], []

for epoch in range(EPOCHS):
    crnn_model.train()
    running_loss = 0.0
    for mel, cqt, chroma, labels in train_loader:
        mel, cqt, chroma, labels = mel.to(device), cqt.to(device), chroma.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = crnn_model(mel, cqt, chroma)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    crnn_model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for mel, cqt, chroma, labels in test_loader:
            mel, cqt, chroma, labels = mel.to(device), cqt.to(device), chroma.to(device), labels.to(device)
            outputs = crnn_model(mel, cqt, chroma)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    val_acc = accuracy_score(all_targets, all_preds)
    test_accs.append(val_acc)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.4f}")

# Save CRNN Weights
crnn_path = os.path.join(CHECKPOINT_DIR, "crnn_backbone_weights.pth")
torch.save(crnn_model.state_dict(), crnn_path)
save_to_drive(crnn_path, "crnn_backbone_weights.pth")

# --- Step 4: Plot Training History ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.title('CRNN Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accs, label='Val Accuracy', color='orange')
plt.title('CRNN Validation Accuracy')
plt.legend()

plt.tight_layout()
fig_path = os.path.join(CHECKPOINT_DIR, "fig_02_01_crnn_training_history.png")
plt.savefig(fig_path)
plt.close()
save_to_drive(fig_path, "fig_02_01_crnn_training_history.png")

# --- Step 5: Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")

print("\n[SUCCESS] Script 02 completed. All outputs saved to Google Drive.")