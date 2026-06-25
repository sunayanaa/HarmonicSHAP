# ==============================================================================
# Program Name: 08_exp3_generalization.py
# Version: 2.0
# Description: Experiment 3 — HAC Robustness Study.
#              Applies pitch transpositions (±2, ±4 semitones) and tempo
#              perturbations (±10%) to four datasets and computes HAC
#              to evaluate semantic attribution stability.
#
#              Datasets (four rows in Table 3):
#                - GiantSteps+  (exp3_features_giantsteps.pkl)
#                - FMA-Small    (fma_features_validation.pkl, 100 tracks)
#                - Ballroom     (exp3_features_ballroom.pkl)
#                - Jamendo      (exp3_features_jamendo.pkl)
#
# Change Log:
#   1.0: GTZAN-era, Pearson HAC, 1-bin circular roll, no prediction
#        invariance filter, silence baseline.
#   2.0: All HAC via harmonicshap_core.compute_hac_for_track:
#        cosine distance, semitone-accurate CQT bin shift, prediction
#        invariance filter (FIX-5). Training-mean baseline (FIX-2).
#        Log1p on Mel/CQT. Tempo perturbation via feature-level
#        interpolation (documented limitation). Checkpoint/resume.
#        Jamendo added as fourth dataset.
#
# GPU Required: Yes
# Dependencies: torch, scipy, numpy, harmonicshap_core
# Inputs (Google Drive):
#   exp3_features_giantsteps.pkl
#   exp3_features_ballroom.pkl
#   exp3_features_jamendo.pkl
#   fma_features_validation.pkl
#   crnn_backbone_weights.pth
#   fma_training_baseline.pkl
#   label_encoder.pkl
# Outputs (Google Drive):
#   exp3_generalization_results.json
#   exp3_state.pkl  (checkpoint)
# ==============================================================================

import os, sys, pickle, json
import numpy as np
import torch
import scipy.ndimage
from scipy.stats import sem
from google.colab import drive

# --- Mount Google Drive ---
drive.mount('/content/drive')

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/HarmonicSHAP/"
CHECKPOINT_DIR  = "/content/checkpoints"
FMA_EVAL_TRACKS = 100      # number of FMA-Small validation tracks to use
SEED            = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if not torch.cuda.is_available():
    print("[ERROR] GPU not detected.")
    sys.exit(1)
device = torch.device("cuda")
print("CUDA available: True. Proceeding...")

# Transformations to apply
PITCH_SHIFTS    = (-4, -2, 2, 4)    # semitones
TEMPO_FACTORS   = (0.9, 1.1)        # ±10%

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

# --- Load harmonicshap_core ---
core_path = os.path.join(CHECKPOINT_DIR, "harmonicshap_core.py")
if not os.path.exists(core_path):
    copy_from_project("harmonicshap_core.py", core_path)
sys.path.insert(0, CHECKPOINT_DIR)
import harmonicshap_core as hsc
from harmonicshap_core import MultiBranchCRNN

# =============================================================================
# Step 1: Load model, baseline, and label encoder
# =============================================================================
print("\n--- Step 1: Loading Model & Baseline ---")

for fname in ["crnn_backbone_weights.pth", "label_encoder.pkl",
              "fma_training_baseline.pkl"]:
    local = os.path.join(CHECKPOINT_DIR, fname)
    if not os.path.exists(local):
        copy_from_project(fname, local)

with open(os.path.join(CHECKPOINT_DIR, "label_encoder.pkl"), 'rb') as f:
    le = pickle.load(f)
num_classes = len(le.classes_)

crnn = MultiBranchCRNN(num_classes).to(device)
crnn.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "crnn_backbone_weights.pth"),
               map_location=device))
crnn.eval()

mel_mean, cqt_mean, chroma_mean = hsc.load_baseline(
    os.path.join(CHECKPOINT_DIR, "fma_training_baseline.pkl"), device)

print(f"Model loaded: {num_classes} classes, GRU input {hsc.GRU_INPUT_SIZE}")

# =============================================================================
# Step 2: Load all datasets
# =============================================================================
print("\n--- Step 2: Loading Datasets from Google Drive ---")

datasets = {
    'GiantSteps+' : 'exp3_features_giantsteps.pkl',
    'FMA-Small'   : 'fma_features_validation.pkl',
    'Ballroom'    : 'exp3_features_ballroom.pkl',
    'Jamendo'     : 'exp3_features_jamendo.pkl',
}

dataset_data = {}
for name, fname in datasets.items():
    local = os.path.join(CHECKPOINT_DIR, fname)
    if not os.path.exists(local):
        copy_from_project(fname, local)
    with open(local, 'rb') as f:
        raw = pickle.load(f)

    # For FMA-Small validation, subsample to FMA_EVAL_TRACKS
    if name == 'FMA-Small':
        keys = sorted(raw.keys())
        np.random.seed(SEED)
        keys = list(np.random.choice(keys,
                                      size=min(FMA_EVAL_TRACKS, len(keys)),
                                      replace=False))
        raw = {k: raw[k] for k in sorted(keys)}

    dataset_data[name] = raw
    print(f"  {name}: {len(raw)} tracks")

# =============================================================================
# Tempo perturbation helper
#
# Since raw audio is not stored, tempo perturbation is approximated by
# resampling the feature matrices along the time axis using bilinear
# interpolation. Speed-up (factor>1) compresses frames; slow-down
# (factor<1) stretches frames. The result is padded/trimmed to
# TARGET_FRAMES. This is a known approximation documented as a boundary
# condition in the manuscript's Limitations section.
# =============================================================================

def tempo_perturb_features(track_data, factor):
    """
    Apply tempo perturbation by resampling feature matrices along time axis.
    factor > 1.0 = speed up (fewer frames preserved)
    factor < 1.0 = slow down (frames stretched)

    Returns a new track_data dict with perturbed feature arrays.
    """
    T    = hsc.TARGET_FRAMES
    T_new = int(round(T * factor))

    perturbed = {}
    for key in ['mel', 'cqt', 'chroma', 'mfcc']:
        feat = track_data['features'][key]          # (n_bins, T)
        n_bins = feat.shape[0]
        # Resample along time axis using zoom
        zoom_factor = T_new / T
        resampled   = scipy.ndimage.zoom(feat, (1.0, zoom_factor), order=1)
        # Pad or trim to TARGET_FRAMES
        if resampled.shape[1] < T:
            resampled = np.pad(resampled,
                                ((0, 0), (0, T - resampled.shape[1])))
        else:
            resampled = resampled[:, :T]
        perturbed[key] = resampled.astype(np.float32)

    return {'features': perturbed,
            'dataset' : track_data.get('dataset', 'unknown')}

# =============================================================================
# Step 3: Load or initialise checkpoint
# =============================================================================
state_local = os.path.join(CHECKPOINT_DIR, "exp3_state.pkl")
if copy_from_project("exp3_state.pkl", state_local):
    with open(state_local, 'rb') as f:
        state = pickle.load(f)
    print(f"\n[INFO] Resuming from checkpoint")
else:
    state = {ds: {} for ds in datasets.keys()}
    print("\n[INFO] Starting fresh evaluation")

# =============================================================================
# Step 4: HAC robustness evaluation loop
# =============================================================================
print("\n--- Step 4: Running Robustness Transformations ---")

initial_n = {ds: len(state[ds]) for ds in datasets.keys()}
n_processed_total = 0

for ds_name, ds_tracks in dataset_data.items():
    print(f"\nProcessing {ds_name}...")
    already_done = set(state[ds_name].keys())
    tracks_todo  = [(k, v) for k, v in ds_tracks.items()
                    if k not in already_done]

    for idx, (track_id, track_data) in enumerate(tracks_todo):
        sys.stdout.write(
            f"\r  Track {len(state[ds_name]) + 1}/{len(ds_tracks)}...")
        sys.stdout.flush()

        result = {}

        # Prepare original tensors
        mel_t = hsc.format_tensor(track_data['features']['mel'],
                                   device, apply_log=True)
        cqt_t = hsc.format_tensor(track_data['features']['cqt'],
                                   device, apply_log=True)
        chr_t = hsc.format_tensor(track_data['features']['chroma'], device)

        # Extract semantic entities for this track
        try:
            (beat_frames, section_frames, timbral_mel_bins,
             mask_H_cqt, mask_H_chr, mask_R,
             sections, _) = hsc.extract_track_entities(
                track_data, device, hsc.TARGET_FRAMES)
        except Exception:
            continue   # skip tracks where entity extraction fails

        # ── Pitch transpositions ────────────────────────────────────────────
        for n_semi in PITCH_SHIFTS:
            label = (f"Pitch_Plus_{abs(n_semi)}"
                     if n_semi > 0 else
                     f"Pitch_Minus_{abs(n_semi)}")
            try:
                hac_val, n_valid = hsc.compute_hac_for_track(
                    mel_t, cqt_t, chr_t, crnn, device,
                    timbral_mel_bins, mask_H_cqt, mask_H_chr,
                    mask_R, section_frames,
                    mel_mean, cqt_mean, chroma_mean,
                    semitone_shifts=(n_semi,)
                )
                if hac_val is not None:
                    result[label] = float(hac_val)
            except Exception:
                pass

        # ── Tempo perturbations ─────────────────────────────────────────────
        # Documented approximation: tempo perturbation applied at feature
        # level via bilinear interpolation (see function above).
        # True audio-level time stretching not possible since raw audio
        # is not stored. This is reported as a boundary condition in the
        # manuscript Limitations section.
        for factor in TEMPO_FACTORS:
            label = ("Tempo_Plus_10"  if factor > 1.0 else
                     "Tempo_Minus_10")
            try:
                perturbed = tempo_perturb_features(track_data, factor)
                mel_p = hsc.format_tensor(perturbed['features']['mel'],
                                           device, apply_log=True)
                cqt_p = hsc.format_tensor(perturbed['features']['cqt'],
                                           device, apply_log=True)
                chr_p = hsc.format_tensor(perturbed['features']['chroma'],
                                           device)

                # For tempo perturbation, compare attribution vectors
                # before and after, with prediction invariance filter
                with torch.no_grad():
                    pred_orig  = crnn(mel_t, cqt_t, chr_t).argmax(1).item()
                    pred_pert  = crnn(mel_p, cqt_p, chr_p).argmax(1).item()

                if pred_pert != pred_orig:
                    continue   # prediction invariance filter

                # Extract entities for perturbed track
                (bf_p, sf_p, tmb_p,
                 mHcqt_p, mHchr_p, mR_p,
                 _, _) = hsc.extract_track_entities(
                    perturbed, device, hsc.TARGET_FRAMES)

                phi_orig, _ = hsc.compute_shapley_game(
                    mel_t, cqt_t, chr_t, crnn, pred_orig,
                    timbral_mel_bins, mask_H_cqt, mask_H_chr,
                    mask_R, section_frames,
                    mel_mean, cqt_mean, chroma_mean)

                phi_pert, _ = hsc.compute_shapley_game(
                    mel_p, cqt_p, chr_p, crnn, pred_orig,
                    tmb_p, mHcqt_p, mHchr_p,
                    mR_p, sf_p,
                    mel_mean, cqt_mean, chroma_mean)

                vec_o = np.array([phi_orig[p] for p in hsc.PLAYERS])
                vec_p = np.array([phi_pert[p] for p in hsc.PLAYERS])

                if (np.linalg.norm(vec_o) > 1e-9 and
                        np.linalg.norm(vec_p) > 1e-9):
                    from scipy.spatial.distance import cosine as cdist
                    result[label] = float(1.0 - cdist(vec_o, vec_p))

            except Exception:
                pass

        if result:
            state[ds_name][track_id] = result
        n_processed_total += 1

        # Checkpoint every 10 tracks
        if n_processed_total % 10 == 0:
            with open(state_local, 'wb') as f:
                pickle.dump(state, f)
            copy_to_project(state_local, "exp3_state.pkl")

    print(f"\n  {ds_name}: {len(state[ds_name])} tracks evaluated")

# Final checkpoint
with open(state_local, 'wb') as f:
    pickle.dump(state, f)
copy_to_project(state_local, "exp3_state.pkl")

# =============================================================================
# Step 5: Aggregate and report results
# =============================================================================
print("\n\n=== Experiment 3: HAC Robustness Results (Mean ± SE) ===\n")

TRANSFORM_LABELS = [
    'Pitch_Plus_2', 'Pitch_Minus_2',
    'Pitch_Plus_4', 'Pitch_Minus_4',
    'Tempo_Plus_10', 'Tempo_Minus_10',
]

final_metrics = {}

for ds_name in datasets.keys():
    print(f"[{ds_name}]")
    final_metrics[ds_name] = {}

    for label in TRANSFORM_LABELS:
        values = [v[label] for v in state[ds_name].values()
                  if label in v]
        if values:
            m  = float(np.mean(values))
            se = float(sem(values))
            final_metrics[ds_name][label] = {'mean': m, 'se': se,
                                              'n': len(values)}
            print(f"  {label:<20}: {m:.4f} ± {se:.4f}  (n={len(values)})")
        else:
            final_metrics[ds_name][label] = None
            print(f"  {label:<20}: N/A")
    print()

# =============================================================================
# Step 6: Save and upload results
# =============================================================================
results_path = os.path.join(CHECKPOINT_DIR,
                             "exp3_generalization_results.json")
with open(results_path, 'w') as f:
    json.dump(final_metrics, f, indent=2)
copy_to_project(results_path, "exp3_generalization_results.json")

print("[SUCCESS] Script 08 complete. Results uploaded.")
print("All three experiments complete. Ready to draft LaTeX.")