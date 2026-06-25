# ==============================================================================
# Program Name: 07_exp3_data_prep.py
# Version: 2.0
# Description: Feature extraction for Experiment 3 datasets.
#              Extracts Mel, CQT, Chroma, MFCC features from:
#                - GiantSteps+ (600 tracks, first 30s of each 2-min clip)
#                - Ballroom     (data1.tar.gz + data2.tar.gz, all 30s WAVs)
#                - Jamendo      (wav_24/ folder, capped at 200 tracks)
#              FMA-Small features are already on Google Drive from script 01 and
#              are NOT re-extracted here.
#
#              Feature extraction parameters match harmonicshap_core.py:
#                N_MEL_BINS=128, N_CQT_BINS=168, N_CHROMA_BINS=12
#                SR=22050, HOP_LENGTH=512, TARGET_FRAMES=1290
#              Raw features stored (no log1p) — log1p applied at load
#              time in script 08, consistent with scripts 02-05.
#
# Drive inputs:
#   /content/drive/MyDrive/datasets/GiantStepsPlus/audio.zip
#   /content/drive/MyDrive/datasets/Ballroom/data1.tar.gz
#   /content/drive/MyDrive/datasets/Ballroom/data2.tar.gz
#   /content/drive/MyDrive/datasets/Jamendo/wav_24/
#
# Google Drive outputs:
#   exp3_features_giantsteps.pkl
#   exp3_features_ballroom.pkl
#   exp3_features_jamendo.pkl
# ==============================================================================

import os, sys, pickle, tarfile, zipfile, glob, warnings
import numpy as np
import librosa
from google.colab import drive
warnings.filterwarnings('ignore')

# --- Mount Google Drive ---
drive.mount('/content/drive')

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/HarmonicSHAP/"
CHECKPOINT_DIR  = "/content/checkpoints"
DRIVE_BASE      = "/content/drive/MyDrive/datasets"
JAMENDO_MAX     = 200    # cap Jamendo at 200 tracks
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

# --- Load harmonicshap_core for constants ---
core_path = os.path.join(CHECKPOINT_DIR, "harmonicshap_core.py")
if not os.path.exists(core_path):
    copy_from_project("harmonicshap_core.py", core_path)
sys.path.insert(0, CHECKPOINT_DIR)
import harmonicshap_core as hsc

print(f"[INFO] Constants: SR={hsc.SR}, HOP={hsc.HOP_LENGTH}, "
      f"MEL={hsc.N_MEL_BINS}, CQT={hsc.N_CQT_BINS}, T={hsc.TARGET_FRAMES}")

# =============================================================================
# Feature extraction function (shared across all datasets)
# =============================================================================

def extract_features(audio_path, offset=0.0, duration=30.0):
    """
    Load audio and extract Mel, CQT, Chroma, MFCC.
    offset   : start time in seconds (default 0 — use first 30s)
    duration : clip length in seconds (default 30s)
    Returns dict with keys mel, cqt, chroma, mfcc, each (n_bins, T).
    """
    y, _ = librosa.load(audio_path, sr=hsc.SR, mono=True,
                         offset=offset, duration=duration)

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

    return {'mel'   : fix(mel),
            'cqt'   : fix(cqt),
            'chroma': fix(chroma),
            'mfcc'  : fix(mfcc)}


def process_dataset(audio_paths, dataset_name, offset=0.0, duration=30.0):
    """
    Extract features for a list of audio files.
    Returns dict {track_id: {'features': {...}, 'dataset': dataset_name}}.
    """
    data    = {}
    n_ok    = 0
    n_fail  = 0
    n_total = len(audio_paths)

    for i, path in enumerate(audio_paths):
        track_id = os.path.splitext(os.path.basename(path))[0]
        # Skip macOS ghost files
        if track_id.startswith('._') or track_id.startswith('.'):
            continue
        try:
            feats = extract_features(path, offset=offset, duration=duration)
            data[track_id] = {'features': feats, 'dataset': dataset_name}
            n_ok += 1
        except Exception:
            n_fail += 1
        sys.stdout.write(f"\r[{dataset_name}] {i+1}/{n_total} "
                         f"(ok={n_ok}, fail={n_fail})")
        sys.stdout.flush()

    print(f"\n[{dataset_name}] Complete — {n_ok} extracted, {n_fail} failed.")
    return data


# =============================================================================
# Step 1: GiantSteps+
# =============================================================================
print("\n--- Step 1: GiantSteps+ ---")

GIANTSTEPS_AUDIO_DIR = "/content/giantsteps_audio"
GIANTSTEPS_ZIP       = os.path.join(DRIVE_BASE,
                                     "GiantStepsPlus", "audio.zip")

if not os.path.exists(GIANTSTEPS_AUDIO_DIR):
    print("Extracting audio.zip...")
    with zipfile.ZipFile(GIANTSTEPS_ZIP, 'r') as z:
        z.extractall("/content/giantsteps_raw")
    # Find actual audio files regardless of zip internal structure
    mp3_files = glob.glob("/content/giantsteps_raw/**/*.mp3",
                           recursive=True)
    os.makedirs(GIANTSTEPS_AUDIO_DIR, exist_ok=True)
    for f in mp3_files:
        os.rename(f, os.path.join(GIANTSTEPS_AUDIO_DIR,
                                   os.path.basename(f)))
    print(f"Extracted {len(mp3_files)} MP3 files.")
else:
    print("GiantSteps+ audio already extracted.")

gs_paths = sorted(glob.glob(os.path.join(GIANTSTEPS_AUDIO_DIR, "*.mp3")))
print(f"Found {len(gs_paths)} GiantSteps+ tracks. "
      f"Using first 30s of each 2-min clip.")

gs_data = process_dataset(gs_paths, "GiantStepsPlus",
                            offset=0.0, duration=30.0)

gs_path_local = os.path.join(CHECKPOINT_DIR, "exp3_features_giantsteps.pkl")
with open(gs_path_local, 'wb') as f:
    pickle.dump(gs_data, f)
copy_to_project(gs_path_local, "exp3_features_giantsteps.pkl")

# =============================================================================
# Step 2: Ballroom (data1.tar.gz + data2.tar.gz)
# =============================================================================
print("\n--- Step 2: Ballroom ---")

BALLROOM_AUDIO_DIR = "/content/ballroom_audio"

if not os.path.exists(BALLROOM_AUDIO_DIR):
    os.makedirs(BALLROOM_AUDIO_DIR, exist_ok=True)
    for tar_name in ["data1.tar.gz", "data2.tar.gz"]:
        tar_path = os.path.join(DRIVE_BASE, "Ballroom", tar_name)
        if not os.path.exists(tar_path):
            print(f"[WARN] {tar_path} not found — skipping.")
            continue
        print(f"Extracting {tar_name}...")
        with tarfile.open(tar_path, 'r:gz') as t:
            t.extractall("/content/ballroom_raw")
    # Collect all WAV files
    wav_files = glob.glob("/content/ballroom_raw/**/*.wav",
                           recursive=True)
    for f in wav_files:
        dest = os.path.join(BALLROOM_AUDIO_DIR, os.path.basename(f))
        if not os.path.exists(dest):
            os.rename(f, dest)
    print(f"Collected {len(wav_files)} WAV files.")
else:
    print("Ballroom audio already extracted.")

ball_paths = sorted(glob.glob(os.path.join(BALLROOM_AUDIO_DIR, "*.wav")))
print(f"Found {len(ball_paths)} Ballroom tracks.")

ball_data = process_dataset(ball_paths, "Ballroom",
                             offset=0.0, duration=30.0)

ball_path_local = os.path.join(CHECKPOINT_DIR,
                                "exp3_features_ballroom.pkl")
with open(ball_path_local, 'wb') as f:
    pickle.dump(ball_data, f)
copy_to_project(ball_path_local, "exp3_features_ballroom.pkl")

# =============================================================================
# Step 3: Jamendo (wav_24/ folder, capped at JAMENDO_MAX tracks)
# =============================================================================
print(f"\n--- Step 3: Jamendo (cap={JAMENDO_MAX} tracks) ---")

JAMENDO_WAV_DIR = os.path.join(DRIVE_BASE, "Jamendo", "wav_24")

# Recursively find WAV files
jamendo_all = sorted(glob.glob(
    os.path.join(JAMENDO_WAV_DIR, "**", "*.wav"), recursive=True
))
if not jamendo_all:
    # Fallback to audio_data if wav_24 is empty
    print("[INFO] wav_24 empty — trying audio_data/ for MP3s...")
    JAMENDO_WAV_DIR = os.path.join(DRIVE_BASE, "Jamendo", "audio_data")
    jamendo_all = sorted(glob.glob(
        os.path.join(JAMENDO_WAV_DIR, "**", "*.mp3"), recursive=True
    ))

# Fixed seed subsample for reproducibility
np.random.seed(42)
if len(jamendo_all) > JAMENDO_MAX:
    jamendo_paths = list(
        np.random.choice(jamendo_all, size=JAMENDO_MAX, replace=False)
    )
    jamendo_paths = sorted(jamendo_paths)
else:
    jamendo_paths = jamendo_all
print(f"Found {len(jamendo_all)} Jamendo tracks. "
      f"Using {len(jamendo_paths)} (seed=42 subsample).")

# Extract 30s from middle of track to avoid intros/outros
jamendo_data = process_dataset(jamendo_paths, "Jamendo",
                                offset=30.0, duration=30.0)

jam_path_local = os.path.join(CHECKPOINT_DIR,
                               "exp3_features_jamendo.pkl")
with open(jam_path_local, 'wb') as f:
    pickle.dump(jamendo_data, f)
copy_to_project(jam_path_local, "exp3_features_jamendo.pkl")

# =============================================================================
# Summary
# =============================================================================
print("\n=== Step 7 Summary ===")
print(f"GiantSteps+ : {len(gs_data)} tracks → exp3_features_giantsteps.pkl")
print(f"Ballroom    : {len(ball_data)} tracks → exp3_features_ballroom.pkl")
print(f"Jamendo     : {len(jamendo_data)} tracks → exp3_features_jamendo.pkl")
print("\n[SUCCESS] Script 07 complete.")
print("Next step: run 08_exp3_generalization.py")