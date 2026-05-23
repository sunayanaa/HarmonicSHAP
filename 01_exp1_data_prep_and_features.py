# ==============================================================================
# Program Name: 01_exp1_data_prep_and_features.py
# Version: 1.0
# Description: Prepares the GTZAN dataset for Experiment 1. Mounts Google Drive, 
#              copies the zip to local Colab storage, extracts it, and processes 
#              the multi-resolution features (Mel Spectrogram, MFCC, CQT, Chroma) 
#              required for all baseline models and HarmonicSHAP. Saves processed 
#              features as chunked pickle checkpoints to the Google Drive project folder.
# Change Log: 
#   - 1.0: Initial implementation of GTZAN multi-resolution extraction.
# GPU Required: Yes (Included to ensure the runtime is prepared for subsequent model training)
# Inputs: /content/drive/MyDrive/datasets/GTZAN.zip
# Outputs: 
#   - gtzan_features_chunk_N.pkl (saved to Google Drive)
#   - fig_01_01_sample_multires_features.png (saved to Google Drive)
# ==============================================================================

!pip install librosa tqdm matplotlib numpy


import sys
import os
import torch

# --- GPU Check ---
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU")
    print("Please switch your Colab runtime to a T4 GPU and restart.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

import os
import shutil
import zipfile
import pickle
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from google.colab import drive

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/harmonic-SHAP/"  # Persistent storage

DRIVE_ZIP_PATH = "/content/drive/MyDrive/datasets/GTZAN.zip"
LOCAL_DATA_DIR = "/content/dataset"
LOCAL_GTZAN_DIR = os.path.join(LOCAL_DATA_DIR, "genres")  # Typical GTZAN internal structure
CHECKPOINT_DIR = "/content/checkpoints"
SR = 22050  # Target sample rate for MGC
CHUNK_SIZE = 100  # Number of audio files to process before saving and uploading a checkpoint

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

# --- Step 1: Mount Drive and Prepare Local Storage ---
print("\n--- Step 1: Mounting Drive & Preparing Data ---")
drive.mount('/content/drive')

os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

local_zip_path = os.path.join(LOCAL_DATA_DIR, "GTZAN.zip")

if not os.path.exists(LOCAL_GTZAN_DIR):
    print("Copying GTZAN.zip from Drive to local disk...")
    shutil.copy2(DRIVE_ZIP_PATH, local_zip_path)
    print("Extracting dataset...")
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(LOCAL_DATA_DIR)
    print("Extraction complete.")
else:
    print("Dataset already exists locally.")

# --- Step 2: Define Feature Extraction Pipeline ---
def extract_features(file_path):
    """Extracts Mel, MFCC, CQT, and Chroma representations as per Blueprint Section 5."""
    try:
        y, sr = librosa.load(file_path, sr=SR, duration=30.0)  # Ensure uniform 30s length
        
        # 1. Mel Spectrogram (For CNN Baseline)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 2. MFCC (For XGBoost/Vanilla SHAP Baseline)
        mfcc = librosa.feature.mfcc(S=mel_spec_db, n_mfcc=20)
        
        # 3. CQT (For HarmonicSHAP Substrate)
        cqt = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C1'), n_bins=84, bins_per_octave=12))
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        
        # 4. Chroma (For Harmonic Coalition Extraction)
        chroma = librosa.feature.chroma_cqt(C=cqt, sr=sr)
        
        return {
            "mel": mel_spec_db,
            "mfcc": mfcc,
            "cqt": cqt_db,
            "chroma": chroma
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- Step 3: Process and Chunk Data with Google Drive Checkpointing ---
print("\n--- Step 3: Processing Features & Checkpointing ---")
audio_files = []
for root, dirs, files in os.walk(LOCAL_DATA_DIR):
    for file in files:
        if file.endswith('.wav'):
            audio_files.append(os.path.join(root, file))

print(f"Found {len(audio_files)} audio files. Processing in chunks of {CHUNK_SIZE}...")

# Get list of existing files on Drive to check for completed chunks
drive_files = list_drive_files()

features_dict = {}
chunk_idx = 1
processed_count = 0

for i, filepath in enumerate(tqdm(audio_files)):
    checkpoint_name = f"gtzan_features_chunk_{chunk_idx}.pkl"
    local_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    
    # Skip processing if we just started a chunk and it already exists on Drive
    if len(features_dict) == 0 and checkpoint_name in drive_files:
        print(f"\nSkipping chunk {chunk_idx}, already exists on Drive.")
        chunk_idx += 1
        processed_count += CHUNK_SIZE
        continue

    genre = os.path.basename(os.path.dirname(filepath))
    filename = os.path.basename(filepath)
    
    feats = extract_features(filepath)
    if feats:
        features_dict[filename] = {"genre": genre, "features": feats}
    
    processed_count += 1
    
    # Save and upload chunk
    if processed_count % CHUNK_SIZE == 0 or processed_count == len(audio_files):
        print(f"\nSaving and uploading {checkpoint_name} to Google Drive...")
        with open(local_checkpoint, 'wb') as f:
            pickle.dump(features_dict, f)
        save_to_drive(local_checkpoint, checkpoint_name)
        
        # Clear dictionary for next chunk and increment index
        features_dict = {}
        chunk_idx += 1

# --- Step 4: Generate Sample Figure ---
print("\n--- Step 4: Generating Sample Figure ---")
if len(audio_files) > 0:
    sample_file = audio_files[0]
    sample_feats = extract_features(sample_file)
    
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    librosa.display.specshow(sample_feats['mel'], x_axis='time', y_axis='mel', sr=SR, ax=ax[0])
    ax[0].set_title('Mel Spectrogram')
    
    librosa.display.specshow(sample_feats['mfcc'], x_axis='time', ax=ax[1])
    ax[1].set_title('MFCCs')
    
    librosa.display.specshow(sample_feats['cqt'], x_axis='time', y_axis='cqt_note', sr=SR, ax=ax[2])
    ax[2].set_title('Constant-Q Transform (CQT)')
    
    librosa.display.specshow(sample_feats['chroma'], x_axis='time', y_axis='chroma', sr=SR, ax=ax[3])
    ax[3].set_title('Chroma (Pitch Class Profiles)')
    
    plt.tight_layout()
    fig_name = "fig_01_01_sample_multires_features.png"
    fig_path = os.path.join(CHECKPOINT_DIR, fig_name)
    plt.savefig(fig_path)
    plt.close()
    
    save_to_drive(fig_path, fig_name)
    print(f"Sample visualization saved to Drive as {fig_name}")

# --- Step 5: Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")

print("\n[SUCCESS] Script 01 completed. All outputs saved to Google Drive.")