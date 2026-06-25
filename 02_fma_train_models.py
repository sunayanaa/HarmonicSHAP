# ==============================================================================
# Program Name: 02_fma_train_models.py
# Version: 2.2 (log transform + dropout)
# Description: Trains the XGBoost MFCC baseline and the MultiBranchCRNN
#              on FMA-Small features produced by 01_fma_data_prep_and_features.py.
#
# Change Log:
#   2.0: Initial FMA-Small version (bulk loading — caused OOM crash).
#   2.1: FMALazyDataset with chunk caching — fixed OOM.
#   2.2: Applied log1p transform to Mel and CQT at load time (FIX-LOG).
#        Added Dropout(0.3) to MultiBranchCRNN via harmonicshap_core v2.1.
#        Recomputes training-mean baseline with log-transformed features
#        so masking baseline matches inference representation (FIX-BASE).
#        Expected test accuracy: 55-65% (up from 40% in v2.1).
#
# IMPORTANT: harmonicshap_core.py must be v2.1 or later (with dropout
#            and apply_log parameter in format_tensor). Upload the latest
#            version to Google Drive before running this script.
#
# GPU Required: Yes
# Dependencies: torch, xgboost, sklearn, numpy
# ==============================================================================

import os, sys, pickle, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import xgboost as xgb
from sklearn.metrics import accuracy_score
from google.colab import drive

# --- Mount Google Drive ---
drive.mount('/content/drive')

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/HarmonicSHAP/"
CHECKPOINT_DIR = "/content/checkpoints"

BATCH_SIZE      = 32
MAX_EPOCHS      = 60
PATIENCE        = 7
LR              = 1e-3
SEED            = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if not torch.cuda.is_available():
    print("[ERROR] GPU not detected. CRNN training requires GPU.")
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

# --- Load harmonicshap_core (copy from Google Drive to get v2.1) ---
core_path = os.path.join(CHECKPOINT_DIR, "harmonicshap_core.py")
print("Copying latest harmonicshap_core.py from Google Drive...")
copy_from_project("harmonicshap_core.py", core_path)
sys.path.insert(0, CHECKPOINT_DIR)
import harmonicshap_core as hsc
from harmonicshap_core import MultiBranchCRNN

# =============================================================================
# Step 1: Copy feature chunks and label encoder from Google Drive
# =============================================================================
print("\n--- Step 1: Copying feature chunks from Google Drive ---")

le_path = os.path.join(CHECKPOINT_DIR, "label_encoder.pkl")
if not os.path.exists(le_path):
    copy_from_project("label_encoder.pkl", le_path)
with open(le_path, "rb") as f:
    le = pickle.load(f)
num_classes = len(le.classes_)
print(f"Classes ({num_classes}): {list(le.classes_)}")

# Copy training chunks
train_chunks = []
chunk_n = 1
while True:
    fname = f"fma_features_train_chunk_{chunk_n}.pkl"
    local = os.path.join(CHECKPOINT_DIR, fname)
    if not os.path.exists(local):
        if not copy_from_project(fname, local):
            break
    train_chunks.append(local)
    chunk_n += 1
print(f"Training chunks found: {len(train_chunks)}")

val_path  = os.path.join(CHECKPOINT_DIR, "fma_features_validation.pkl")
test_path = os.path.join(CHECKPOINT_DIR, "fma_features_test.pkl")
for fname, local in [("fma_features_validation.pkl", val_path),
                      ("fma_features_test.pkl",       test_path)]:
    if not os.path.exists(local):
        copy_from_project(fname, local)

# =============================================================================
# Step 2: Recompute training-mean baseline with log1p transform (FIX-BASE)
#
# The baseline saved by script 01 was computed on raw linear features.
# Since training and inference now use log1p(Mel) and log1p(CQT), the
# masking baseline must match. This step rewrites fma_training_baseline.pkl.
# =============================================================================
print("\n--- Step 2: Recomputing log-transformed training baseline ---")

mel_acc  = np.zeros(hsc.N_MEL_BINS,    dtype=np.float64)
cqt_acc  = np.zeros(hsc.N_CQT_BINS,    dtype=np.float64)
chr_acc  = np.zeros(hsc.N_CHROMA_BINS, dtype=np.float64)
n_tracks = 0

for path in train_chunks:
    with open(path, 'rb') as f:
        chunk = pickle.load(f)
    for _, track in chunk.items():
        mel_acc  += np.log1p(track['features']['mel']).mean(axis=1)
        cqt_acc  += np.log1p(track['features']['cqt']).mean(axis=1)
        chr_acc  += track['features']['chroma'].mean(axis=1)  # chroma unchanged
        n_tracks += 1
    del chunk

baseline = {
    'mel_mean'    : (mel_acc  / n_tracks).astype(np.float32),
    'cqt_mean'    : (cqt_acc  / n_tracks).astype(np.float32),
    'chroma_mean' : (chr_acc  / n_tracks).astype(np.float32),
}
baseline_path = os.path.join(CHECKPOINT_DIR, "fma_training_baseline.pkl")
with open(baseline_path, 'wb') as f:
    pickle.dump(baseline, f)
copy_to_project(baseline_path, "fma_training_baseline.pkl")
print(f"Baseline recomputed on {n_tracks} training tracks (log1p applied to Mel/CQT).")

# =============================================================================
# Step 3: Count tracks without bulk loading
# =============================================================================
print("\n--- Step 3: Indexing feature chunks (lazy loading) ---")

def count_tracks(paths):
    n = 0
    for p in (paths if isinstance(paths, list) else [paths]):
        with open(p, 'rb') as f:
            n += len(pickle.load(f))
    return n

print(f"Train : {count_tracks(train_chunks)} | "
      f"Val   : {count_tracks(val_path)}     | "
      f"Test  : {count_tracks(test_path)}")

# =============================================================================
# Step 4: Train XGBoost baseline (MFCC features; no log transform for MFCC)
# =============================================================================
print("\n--- Step 4: Training XGBoost baseline ---")

def build_mfcc_features(paths):
    X, y = [], []
    for p in (paths if isinstance(paths, list) else [paths]):
        with open(p, 'rb') as f:
            chunk = pickle.load(f)
        for _, track in chunk.items():
            mfcc = track['features']['mfcc']
            X.append(np.concatenate([mfcc.mean(axis=1),
                                     mfcc.var(axis=1)]))
            y.append(int(le.transform([track['genre']])[0]))
        del chunk
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)

X_train, y_train = build_mfcc_features(train_chunks)
X_val,   y_val   = build_mfcc_features(val_path)
X_test,  y_test  = build_mfcc_features(test_path)

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=SEED
)
xgb_model.fit(X_train, y_train,
               eval_set=[(X_val, y_val)],
               verbose=False)

xgb_val_acc  = accuracy_score(y_val,  xgb_model.predict(X_val))
xgb_test_acc = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"XGBoost — Val Acc: {xgb_val_acc:.4f}  Test Acc: {xgb_test_acc:.4f}")

xgb_path = os.path.join(CHECKPOINT_DIR, "xgboost_baseline.json")
xgb_model.save_model(xgb_path)
copy_to_project(xgb_path, "xgboost_baseline.json")

# =============================================================================
# Step 5: Train MultiBranchCRNN with log1p transform on Mel and CQT (FIX-LOG)
# =============================================================================
print("\n--- Step 5: Training MultiBranchCRNN ---")

class FMALazyDataset(Dataset):
    """
    Memory-efficient Dataset. Caches one chunk at a time (~800 MB peak).
    Applies log1p to Mel and CQT at load time (FIX-LOG).
    Must use num_workers=0 in DataLoader.
    Call reshuffle() before each epoch.
    """
    def __init__(self, paths, label_encoder, target_frames):
        if isinstance(paths, str):
            paths = [paths]
        self.label_encoder = label_encoder
        self.target_frames  = target_frames
        self._cached_path   = None
        self._cached_chunk  = None

        self.index = []
        for path in sorted(paths):
            with open(path, 'rb') as f:
                chunk = pickle.load(f)
            for key, track in chunk.items():
                label = int(label_encoder.transform([track['genre']])[0])
                self.index.append((path, key, label))
            del chunk
        self.reshuffle()

    def reshuffle(self):
        groups = defaultdict(list)
        for item in self.index:
            groups[item[0]].append(item)
        order = list(groups.keys())
        random.shuffle(order)
        self.index = []
        for path in order:
            items = groups[path]
            random.shuffle(items)
            self.index.extend(items)

    def _load_chunk(self, path):
        if self._cached_path != path:
            with open(path, 'rb') as f:
                self._cached_chunk = pickle.load(f)
            self._cached_path = path

    def _fix(self, x, apply_log=False):
        T = self.target_frames
        if apply_log:
            x = np.log1p(x)
        if x.shape[1] < T:
            x = np.pad(x, ((0, 0), (0, T - x.shape[1])))
        return torch.FloatTensor(x[:, :T]).unsqueeze(0)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, key, label = self.index[idx]
        self._load_chunk(path)
        feats = self._cached_chunk[key]['features']
        return (
            self._fix(feats['mel'],    apply_log=True),   # log1p scaling
            self._fix(feats['cqt'],    apply_log=True),   # log1p scaling
            self._fix(feats['chroma'], apply_log=False),  # already [0,1]
            torch.tensor(label, dtype=torch.long)
        )


train_ds = FMALazyDataset(train_chunks, le, hsc.TARGET_FRAMES)
val_ds   = FMALazyDataset(val_path,     le, hsc.TARGET_FRAMES)
test_ds  = FMALazyDataset(test_path,    le, hsc.TARGET_FRAMES)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

model     = MultiBranchCRNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3)

best_val_acc   = 0.0
patience_count = 0
history        = []
best_path      = os.path.join(CHECKPOINT_DIR, "crnn_backbone_weights.pth")

for epoch in range(1, MAX_EPOCHS + 1):
    train_ds.reshuffle()

    # Training
    model.train()
    train_loss = 0.0
    for mel, cqt, chroma, labels in train_loader:
        mel, cqt, chroma, labels = (mel.to(device), cqt.to(device),
                                     chroma.to(device), labels.to(device))
        optimizer.zero_grad()
        loss = criterion(model(mel, cqt, chroma), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, cqt, chroma, labels in val_loader:
            mel, cqt, chroma = (mel.to(device), cqt.to(device),
                                 chroma.to(device))
            preds = model(mel, cqt, chroma).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    val_acc = accuracy_score(all_labels, all_preds)
    scheduler.step(val_acc)

    is_best = val_acc > best_val_acc
    history.append({'epoch'     : epoch,
                    'train_loss': round(train_loss, 4),
                    'val_acc'   : round(val_acc,    4)})
    print(f"Epoch {epoch:03d}/{MAX_EPOCHS} — "
          f"Loss: {train_loss:.4f}  Val Acc: {val_acc:.4f}"
          + ("  [BEST]" if is_best else ""))

    if is_best:
        best_val_acc   = val_acc
        patience_count = 0
        torch.save(model.state_dict(), best_path)
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"[INFO] Early stopping triggered at epoch {epoch}.")
            break

# Test accuracy
model.load_state_dict(torch.load(best_path))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for mel, cqt, chroma, labels in test_loader:
        mel, cqt, chroma = mel.to(device), cqt.to(device), chroma.to(device)
        preds = model(mel, cqt, chroma).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
test_acc = accuracy_score(all_labels, all_preds)

print(f"\n=== Final Results ===")
print(f"Best Val Acc   : {best_val_acc:.4f}")
print(f"Test Acc       : {test_acc:.4f}")
print(f"Num classes    : {num_classes}")
print(f"GRU input size : {hsc.GRU_INPUT_SIZE}")

copy_to_project(best_path, "crnn_backbone_weights.pth")

log_path = os.path.join(CHECKPOINT_DIR, "training_log.json")
with open(log_path, "w") as f:
    json.dump({
        'history'       : history,
        'best_val_acc'  : best_val_acc,
        'test_acc'      : test_acc,
        'num_classes'   : num_classes,
        'gru_input_size': hsc.GRU_INPUT_SIZE,
        'classes'       : list(le.classes_),
        'log_transform' : 'log1p applied to Mel and CQT'
    }, f, indent=2)
copy_to_project(log_path, "training_log.json")

print("\n[SUCCESS] Script 02 v2.2 complete.")
print("Outputs: crnn_backbone_weights.pth, xgboost_baseline.json, "
      "fma_training_baseline.pkl, training_log.json")
print("Next step: run 03_exp1_attributions_and_visualization.py")