# ==============================================================================
# Program Name: 01_fma_data_prep_and_features.py
# Version: 2.0 (FMA-Small)
# Description: Downloads FMA-Small audio and metadata, extracts features
#              (Mel, CQT, Chroma, MFCC) for all 8000 tracks, saves split-
#              labelled chunk .pkl files, uploads to Google Drive, and computes the
#              training-set per-bin mean baseline required by
#              harmonicshap_core.py (FIX-2).
#
# Feature extraction parameters are imported from harmonicshap_core.py:
#   N_MEL_BINS=128, N_CQT_BINS=168 (7 oct × 24 bins/oct), N_CHROMA_BINS=12,
#   SR=22050, HOP_LENGTH=512, TARGET_FRAMES=1290
#
# Drive outputs:
#   harmonicshap_core.py              (uploaded once for all scripts)
#   label_encoder.pkl
#   fma_features_train_chunk_N.pkl    (N=1..~13, 500 tracks each)
#   fma_features_val.pkl              (800 tracks)
#   fma_features_test.pkl             (800 tracks)
#   fma_training_baseline.pkl         (per-bin mean for masking baseline)
#
# Prerequisites:
#   - harmonicshap_core.py already uploaded to Drive
#   - GPU not required for this script
# ==============================================================================

import os, sys, pickle, shutil, warnings
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/HarmonicSHAP/"  # Persistent storage
CHECKPOINT_DIR = "/content/checkpoints"
CHUNK_SIZE = 500    # tracks per training chunk
FMA_AUDIO_DIR = "/content/fma_small/fma_small"
FMA_META_DIR = "/content/fma_metadata/fma_metadata"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PROJECT_DIR, exist_ok=True)

# --- Google Drive helpers ---
def save_to_drive(local_path, remote_name):
    """Copy a local file to Google Drive project folder."""
    dest_path = os.path.join(PROJECT_DIR, remote_name)
    try:
        shutil.copy2(local_path, dest_path)
        print(f"  [DRIVE OK] {local_path}  →  {dest_path}")
    except Exception as e:
        print(f"  [DRIVE FAIL] {local_path}: {e}")

def load_from_drive(remote_name, local_path):
    """Copy a file from Google Drive project folder to local path."""
    src_path = os.path.join(PROJECT_DIR, remote_name)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, local_path)
            print(f"  [DRIVE OK] {src_path}  →  {local_path}")
            return True
        except Exception as e:
            print(f"  [DRIVE FAIL] copy from {src_path}: {e}")
            return False
    else:
        print(f"  [DRIVE MISSING] {src_path} not found")
        return False

def list_drive_files():
    """List files in the Google Drive project directory."""
    try:
        return [f for f in os.listdir(PROJECT_DIR) if os.path.isfile(os.path.join(PROJECT_DIR, f))]
    except Exception as e:
        print(f"  [DRIVE] Could not list files: {e}")
        return []

# --- Load harmonicshap_core from Drive ---
core_path = os.path.join(CHECKPOINT_DIR, "harmonicshap_core.py")
drive_files = list_drive_files()

if not os.path.exists(core_path):
    if "harmonicshap_core.py" in drive_files:
        print("Downloading harmonicshap_core.py from Google Drive...")
        load_from_drive("harmonicshap_core.py", core_path)
    else:
        print("[ERROR] harmonicshap_core.py not found on Drive.")
        sys.exit(1)

sys.path.insert(0, CHECKPOINT_DIR)
import harmonicshap_core as hsc

print(f"[INFO] Using constants: SR={hsc.SR}, HOP={hsc.HOP_LENGTH}, "
      f"MEL={hsc.N_MEL_BINS}, CQT={hsc.N_CQT_BINS}, "
      f"CHROMA={hsc.N_CHROMA_BINS}, T={hsc.TARGET_FRAMES}")

# =============================================================================
# Step 1: Copy FMA-Small audio and metadata from Google Drive
# =============================================================================
print("\n--- Step 1: Copying FMA-Small from Google Drive ---")

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

DRIVE_DIR = "/content/drive/MyDrive/datasets"

if not os.path.exists(FMA_AUDIO_DIR):
    print("Copying fma_small.zip from Drive (7.2 GB)...")
    os.system(f"cp '{DRIVE_DIR}/fma_small.zip' /content/fma_small.zip")
    print("Extracting audio archive...")
    os.system("unzip -q /content/fma_small.zip -d /content/")
    print("Audio extracted.")
else:
    print("FMA-Small audio already present — skipping copy.")

if not os.path.exists(FMA_META_DIR):
    print("Copying fma_metadata.zip from Drive...")
    os.system(f"cp '{DRIVE_DIR}/fma_metadata.zip' /content/fma_metadata.zip")
    print("Extracting metadata...")
    os.system("unzip -q /content/fma_metadata.zip -d /content/")
    print("Metadata extracted.")
else:
    print("FMA metadata already present — skipping copy.")

# =============================================================================
# Step 2: Parse metadata — genres, splits, track IDs
# =============================================================================
print("\n--- Step 2: Parsing FMA-Small metadata ---")

tracks = pd.read_csv(
    os.path.join(FMA_META_DIR, "tracks.csv"),
    index_col=0, header=[0, 1]
)

# Filter to small subset only
small = tracks[tracks[('set', 'subset')] == 'small'].copy()
genre_col = ('track', 'genre_top')
split_col = ('set',   'split')

# Drop rows with missing genre
valid = small[genre_col].notna()
small = small[valid]
t_ids = small.index.tolist()
genres = small[genre_col].tolist()
splits = small[split_col].tolist()

print(f"Total tracks with genre labels : {len(t_ids)}")
print(f"Genres : {sorted(set(genres))}")
print(f"Splits : { {s: splits.count(s) for s in set(splits)} }")

# Label encoder
le = LabelEncoder()
le.fit(sorted(set(genres)))
print(f"Classes ({len(le.classes_)}) : {list(le.classes_)}")

le_path = os.path.join(CHECKPOINT_DIR, "label_encoder.pkl")
with open(le_path, "wb") as f:
    pickle.dump(le, f)
save_to_drive(le_path, "label_encoder.pkl")

# Bucket by split
split_buckets = {"training": [], "validation": [], "test": []}
for tid, genre, split in zip(t_ids, genres, splits):
    if split in split_buckets:
        split_buckets[split].append((tid, genre))

# =============================================================================
# Step 3: Feature extraction
# =============================================================================
print("\n--- Step 3: Extracting features ---")

# --- Diagnostic: verify audio path construction ---
sample_tid = split_buckets['training'][0][0]
sample_path = os.path.join(FMA_AUDIO_DIR, f"{sample_tid:06d}"[:3],
                            f"{sample_tid:06d}.mp3")
print(f"Sample track ID : {sample_tid}")
print(f"Constructed path: {sample_path}")
print(f"Path exists     : {os.path.exists(sample_path)}")
print(f"FMA_AUDIO_DIR contents (top level):")
os.system(f"ls {FMA_AUDIO_DIR} | head -5")
print(f"First subdir contents:")
os.system(f"ls {FMA_AUDIO_DIR}/{f'{sample_tid:06d}'[:3]} | head -5")
# --- End diagnostic ---

def audio_path(track_id):
    """FMA path convention: 000/000002.mp3, 001/001000.mp3, etc."""
    s = f"{track_id:06d}"
    return os.path.join(FMA_AUDIO_DIR, s[:3], f"{s}.mp3")

def extract_features(tid):
    """Load audio and extract Mel, CQT, Chroma, MFCC."""
    y, _ = librosa.load(audio_path(tid), sr=hsc.SR, mono=True, duration=30.0)

    mel = librosa.feature.melspectrogram(
        y=y, sr=hsc.SR, n_mels=hsc.N_MEL_BINS,
        hop_length=hsc.HOP_LENGTH, n_fft=2048
    ).astype(np.float32)

    cqt = np.abs(librosa.cqt(
        y, sr=hsc.SR, hop_length=hsc.HOP_LENGTH,
        n_bins=hsc.N_CQT_BINS,
        bins_per_octave=hsc.N_CQT_BINS_PER_OCTAVE,
        fmin=librosa.note_to_hz('C1')
    )).astype(np.float32)

    chroma = librosa.feature.chroma_cqt(
        C=cqt, sr=hsc.SR,
        bins_per_octave=hsc.N_CQT_BINS_PER_OCTAVE,
        n_chroma=hsc.N_CHROMA_BINS
    ).astype(np.float32)

    mfcc = librosa.feature.mfcc(
        y=y, sr=hsc.SR, n_mfcc=20,
        hop_length=hsc.HOP_LENGTH
    ).astype(np.float32)

    def fix(x):
        T = hsc.TARGET_FRAMES
        if x.shape[1] < T:
            return np.pad(x, ((0, 0), (0, T - x.shape[1])))
        return x[:, :T]

    return {'mel': fix(mel), 'cqt': fix(cqt),
            'chroma': fix(chroma), 'mfcc': fix(mfcc)}


def process_split(split_name, track_list, chunk_size):
    """
    Extract features for all tracks in a split, save and upload chunks.
    Training split is chunked at chunk_size; val and test are single files.

    Returns list of local chunk paths (training only — used for baseline).
    """
    chunk_paths = []
    current_chunk = {}
    chunk_num = 1
    n_ok = 0
    n_fail = 0

    for i, (tid, genre) in enumerate(track_list):
        try:
            feats = extract_features(tid)
            current_chunk[f"fma_{tid:06d}"] = {
                'features': feats,
                'genre': genre,
                'split': split_name,
                'track_id': tid
            }
            n_ok += 1
        except Exception as e:
            n_fail += 1
            continue

        end_of_list = (i == len(track_list) - 1)
        chunk_full = (len(current_chunk) >= chunk_size)

        if (chunk_full or end_of_list) and current_chunk:
            if split_name == 'training':
                fname = f"fma_features_train_chunk_{chunk_num}.pkl"
            else:
                fname = f"fma_features_{split_name}.pkl"

            local = os.path.join(CHECKPOINT_DIR, fname)
            with open(local, "wb") as f:
                pickle.dump(current_chunk, f)
            save_to_drive(local, fname)

            if split_name == 'training':
                chunk_paths.append(local)

            current_chunk = {}
            chunk_num += 1

        sys.stdout.write(
            f"\r[{split_name}] {i+1}/{len(track_list)} "
            f"(ok={n_ok}, fail={n_fail})"
        )
        sys.stdout.flush()

    print(f"\n[{split_name}] Complete — {n_ok} extracted, {n_fail} failed.")
    return chunk_paths


train_chunk_paths = process_split(
    'training', split_buckets['training'], CHUNK_SIZE)
process_split('validation', split_buckets['validation'],
              len(split_buckets['validation']))
process_split('test', split_buckets['test'],
              len(split_buckets['test']))

# =============================================================================
# Step 4: Compute training-set per-bin mean baseline (FIX-2)
# =============================================================================
print("\n--- Step 4: Computing training-set per-bin mean baseline ---")

baseline_path = os.path.join(CHECKPOINT_DIR, "fma_training_baseline.pkl")
hsc.precompute_training_mean(
    chunk_paths=train_chunk_paths,
    output_path=baseline_path
)
save_to_drive(baseline_path, "fma_training_baseline.pkl")

# --- Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")

print("\n[SUCCESS] Script 01 complete. All chunks and baseline saved to Google Drive.")
print(f"Training chunks : {len(train_chunk_paths)}")
print("Next step       : run 02_fma_train_models.py")