# ==============================================================================
# Program Name: 07_exp3_data_prep.py
# Version: 3.1 (Anti-Disconnect, macOS Ghost Filter, Tar Filter Fix, Drive-based)
# Description: Ingestion and feature extraction engine for Experiment 3.
#              Hardened against Colab "Errno 107" Drive disconnects by copying 
#              archives to local disk prior to extraction.
#              GiantSteps+ dataset is loaded from Google Drive (not downloaded from Zenodo).
# Dependencies: librosa, numpy, zipfile, tarfile, shutil
# Outputs:
# `/content/checkpoints/exp3_features_giantsteps.pkl`
# `/content/checkpoints/exp3_features_fma.pkl`
# `/content/checkpoints/exp3_features_ballroom.pkl`
# All 3 outputs saved to Google Drive project folder
# ==============================================================================

import os
import sys
import zipfile
import tarfile
import shutil
import librosa
import numpy as np
import pickle
import random

from google.colab import drive
drive.mount('/content/drive')

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

# --- Seed Locking ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- Configuration ---
DRIVE_DIR = "/content/drive/MyDrive/datasets"
EXP3_DIR = "/content/exp3_data"
CHECKPOINT_DIR = "/content/checkpoints"
os.makedirs(EXP3_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SAMPLES_PER_DATASET = 100 
SAVE_INTERVAL = 5 

# Standardized Feature Extraction Parameters
SR = 22050
HOP_LENGTH = 512
N_MELS = 128
CQT_BINS = 84

# --- Helper Functions ---
def is_dir_populated(dir_path):
    return os.path.exists(dir_path) and len(os.listdir(dir_path)) > 0

def check_for_actual_audio(dir_path, ext):
    """Verifies that the directory actually contains audio files, not just empty subfolders."""
    if not os.path.exists(dir_path): return False
    for root, _, files in os.walk(dir_path):
        if any(f.lower().endswith(ext) for f in files):
            return True
    return False

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=SR, duration=30.0)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, n_bins=CQT_BINS))
        chroma = librosa.feature.chroma_cqt(C=cqt, sr=sr, hop_length=HOP_LENGTH)
        return {"mel": mel_db, "cqt": cqt, "chroma": chroma}
    except Exception as e:
        print(f"\n[ERROR] Processing {audio_path}: {e}")
        return None

# ==============================================================================
# Phase 1: GiantSteps+ (Load from Google Drive, not Zenodo)
# ==============================================================================
print("\n=== Phase 1: Preparing GiantSteps+ ===")
gs_dir = os.path.join(EXP3_DIR, "giantsteps")
os.makedirs(gs_dir, exist_ok=True)

gs_audio_zip = os.path.join(gs_dir, "audio.zip")
gs_meta_xlsx = os.path.join(gs_dir, "GiantSteps+.xlsx")

# Expected files in Drive datasets folder
GS_ZIP_DRIVE = os.path.join(DRIVE_DIR, "GiantSteps+.zip")
GS_META_DRIVE = os.path.join(DRIVE_DIR, "GiantSteps+.xlsx")

if os.path.exists(GS_ZIP_DRIVE) and os.path.exists(GS_META_DRIVE):
    # Copy from Drive to local if not already present
    if not os.path.exists(gs_audio_zip):
        print("Copying GiantSteps+ audio from Drive to local disk...")
        shutil.copy2(GS_ZIP_DRIVE, gs_audio_zip)
    else:
        print("GiantSteps+ audio zip already exists locally.")
    
    if not os.path.exists(gs_meta_xlsx):
        print("Copying GiantSteps+ metadata from Drive to local disk...")
        shutil.copy2(GS_META_DRIVE, gs_meta_xlsx)
    else:
        print("GiantSteps+ metadata already exists locally.")
else:
    print(f"[WARNING] GiantSteps+ files not found in {DRIVE_DIR}")
    print("Expected files: GiantSteps+.zip and GiantSteps+.xlsx")
    print("Skipping GiantSteps+ dataset.")

if not check_for_actual_audio(os.path.join(gs_dir, "audio"), ".mp3"): 
    if os.path.exists(gs_audio_zip):
        print("Extracting GiantSteps+ audio from local copy...")
        with zipfile.ZipFile(gs_audio_zip, 'r') as zip_ref:
            zip_ref.extractall(gs_dir)
    else:
        print("GiantSteps+ audio zip not found. Skipping extraction.")
else:
    print("GiantSteps+ audio already extracted. Skipping.")

# ==============================================================================
# Phase 2: FMA-Small & Ballroom (Hardened Local Extraction)
# ==============================================================================
print("\n=== Phase 2: Preparing FMA-Small & Ballroom ===")

fma_zip_drive = os.path.join(DRIVE_DIR, "FMA-small.zip")
fma_zip_local = os.path.join(EXP3_DIR, "FMA-small.zip")
fma_dir = os.path.join(EXP3_DIR, "fma_small")

if os.path.exists(fma_zip_drive):
    # Fix: Actually check for .mp3 files. If a previous extraction crashed, nuke the corrupted folder.
    if not check_for_actual_audio(fma_dir, ".mp3"):
        if os.path.exists(fma_dir): 
            shutil.rmtree(fma_dir) 
        
        if not os.path.exists(fma_zip_local):
            print("Copying FMA-Small to local disk to bypass Drive I/O limits...")
            shutil.copy2(fma_zip_drive, fma_zip_local)
        
        print("Extracting FMA-Small from local disk...")
        with zipfile.ZipFile(fma_zip_local, 'r') as zip_ref:
            zip_ref.extractall(fma_dir)
            
        os.remove(fma_zip_local) 
        print("FMA-Small extraction complete. Local zip removed.")
    else:
        print("FMA-Small already extracted and validated. Skipping.")
else:
    print(f"[WARNING] {fma_zip_drive} not found.")

ballroom_tar_drive = os.path.join(DRIVE_DIR, "Ballroom/data1.tar.gz")
ballroom_tar_local = os.path.join(EXP3_DIR, "data1.tar.gz")
ballroom_dir = os.path.join(EXP3_DIR, "ballroom")

if os.path.exists(ballroom_tar_drive):
    if not check_for_actual_audio(ballroom_dir, ".wav"):
        if os.path.exists(ballroom_dir): 
            shutil.rmtree(ballroom_dir)

        if not os.path.exists(ballroom_tar_local):
            print("Copying Ballroom to local disk to bypass Drive I/O limits...")
            shutil.copy2(ballroom_tar_drive, ballroom_tar_local)
            
        print("Extracting Ballroom from local disk...")
        with tarfile.open(ballroom_tar_local, "r:gz") as tar:
            # Fix: Added filter='data' to suppress the Python 3.12+ security warning
            tar.extractall(path=ballroom_dir, filter='data')
            
        os.remove(ballroom_tar_local)
        print("Ballroom extraction complete. Local tar removed.")
    else:
        print("Ballroom already extracted and validated. Skipping.")
else:
    print(f"[WARNING] {ballroom_tar_drive} not found.")

# ==============================================================================
# Phase 3: Resumable Feature Extraction Pipeline
# ==============================================================================
print("\n=== Phase 3: Extracting Multi-Resolution Features (Resumable) ===")

datasets = {
    "giantsteps": {"path": gs_dir, "ext": ".mp3", "out": "exp3_features_giantsteps.pkl"},
    "fma": {"path": fma_dir, "ext": ".mp3", "out": "exp3_features_fma.pkl"},
    "ballroom": {"path": ballroom_dir, "ext": ".wav", "out": "exp3_features_ballroom.pkl"}
}

for ds_name, info in datasets.items():
    print(f"\nProcessing {ds_name.upper()}...")
    out_path = os.path.join(CHECKPOINT_DIR, info["out"])
    
    if os.path.exists(out_path):
        with open(out_path, 'rb') as f:
            dataset_features = pickle.load(f)
        print(f"Loaded checkpoint: {len(dataset_features)} tracks already processed.")
    else:
        dataset_features = {}
    
    all_files = []
    for root, dirs, files in os.walk(info["path"]):
        # Fix: Ignore hidden macOS directories
        if "__MACOSX" in root:
            continue
        for file in files:
            # Fix: Ignore hidden macOS ghost files
            if file.startswith("._"):
                continue
            if file.lower().endswith(info["ext"]):
                all_files.append(os.path.join(root, file))
                
    if not all_files:
        print(f"No valid audio files found for {ds_name}. Check directory paths. Skipping.")
        continue
        
    all_files.sort()
    sample_files = random.sample(all_files, min(SAMPLES_PER_DATASET, len(all_files)))
    pending_files = [f for f in sample_files if os.path.basename(f) not in dataset_features]
    
    if not pending_files:
        print(f"All {len(sample_files)} requested samples are already processed. Moving on.")
        continue
        
    print(f"{len(pending_files)} tracks remaining to process.")
    
    for idx, filepath in enumerate(pending_files):
        sys.stdout.write(f"\r  Extracting {idx+1}/{len(pending_files)}...")
        sys.stdout.flush()
        
        feats = extract_features(filepath)
        if feats is not None:
            track_id = os.path.basename(filepath)
            dataset_features[track_id] = {"features": feats, "path": filepath}
            
        if (idx + 1) % SAVE_INTERVAL == 0 or (idx + 1) == len(pending_files):
            with open(out_path, 'wb') as f:
                pickle.dump(dataset_features, f)
            
    print(f"\nCompleted {ds_name.upper()}. Total saved: {len(dataset_features)} tracks.")

print("\n[SUCCESS] Script 07 completed.")

# --- Upload results to Google Drive ---
print("\n--- Uploading feature files to Google Drive ---")
save_to_drive("/content/checkpoints/exp3_features_giantsteps.pkl", "exp3_features_giantsteps.pkl")
save_to_drive("/content/checkpoints/exp3_features_fma.pkl", "exp3_features_fma.pkl")
save_to_drive("/content/checkpoints/exp3_features_ballroom.pkl", "exp3_features_ballroom.pkl")

# --- Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")

print("\n[SUCCESS] Script 07 completed. All feature files saved to Google Drive.")