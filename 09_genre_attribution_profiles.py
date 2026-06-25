# ==============================================================================
# Program Name: 09_genre_attribution_profiles.py
# Version: 1.0
# Description: Computes genre-level mean Shapley attribution profiles for all
#              8 FMA-Small genres. For each correctly-classified track, runs
#              the full 4-player Shapley game over the entire 30-second clip
#              and records [phi_H, phi_R, phi_T, phi_S]. Averages across all
#              correctly-classified tracks per genre to produce an 8x4 mean
#              attribution matrix. Generates a compact heatmap figure for the
#              manuscript (Fig. 2).
#
# Design: Full-track Shapley (one game per track, not per section).
#         Only correctly-classified tracks contribute to each genre's profile,
#         ensuring the attribution reflects genuine genre-discriminative
#         reasoning rather than misclassification artifacts.
#
# GPU Required: Yes
# Inputs (Google Drive): fma_features_train_chunk_*.pkl (all training chunks)
#               crnn_backbone_weights.pth
#               fma_training_baseline.pkl
#               label_encoder.pkl
# Outputs (Google Drive): fig_09_genre_attribution_heatmap.png
#                exp_genre_attribution_profiles.json
# ==============================================================================

import os, sys, pickle, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import sem
from google.colab import drive

# --- Mount Google Drive ---
drive.mount('/content/drive')

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/HarmonicSHAP/"
CHECKPOINT_DIR  = "/content/checkpoints"
MAX_TRACKS      = 50   # max correctly-classified tracks per genre to average
SEED            = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if not torch.cuda.is_available():
    print("[ERROR] GPU not detected.")
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

# --- Load harmonicshap_core ---
core_path = os.path.join(CHECKPOINT_DIR, "harmonicshap_core.py")
if not os.path.exists(core_path):
    copy_from_project("harmonicshap_core.py", core_path)
sys.path.insert(0, CHECKPOINT_DIR)
import harmonicshap_core as hsc
from harmonicshap_core import MultiBranchCRNN

# =============================================================================
# Step 1: Load model, baseline, label encoder
# =============================================================================
print("\n--- Step 1: Loading Model & Baseline ---")

for fname in ["crnn_backbone_weights.pth", "label_encoder.pkl",
              "fma_training_baseline.pkl"]:
    local = os.path.join(CHECKPOINT_DIR, fname)
    if not os.path.exists(local):
        copy_from_project(fname, local)

with open(os.path.join(CHECKPOINT_DIR, "label_encoder.pkl"), 'rb') as f:
    le = pickle.load(f)
num_classes  = len(le.classes_)
genre_names  = list(le.classes_)
print(f"Genres: {genre_names}")

crnn = MultiBranchCRNN(num_classes).to(device)
crnn.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "crnn_backbone_weights.pth"),
               map_location=device))
crnn.eval()

mel_mean, cqt_mean, chroma_mean = hsc.load_baseline(
    os.path.join(CHECKPOINT_DIR, "fma_training_baseline.pkl"), device)

# =============================================================================
# Step 2: Load all training chunks
# =============================================================================
print("\n--- Step 2: Loading Training Chunks ---")

all_tracks = {}
chunk_n = 1
while True:
    fname = f"fma_features_train_chunk_{chunk_n}.pkl"
    local = os.path.join(CHECKPOINT_DIR, fname)
    if not os.path.exists(local):
        if not copy_from_project(fname, local):
            break
    with open(local, 'rb') as f:
        chunk = pickle.load(f)
    all_tracks.update(chunk)
    del chunk
    chunk_n += 1

print(f"Total training tracks loaded: {len(all_tracks)}")

# =============================================================================
# Step 3: Compute per-genre attribution profiles
# =============================================================================
print("\n--- Step 3: Computing Genre-Level Attribution Profiles ---")
print(f"Using up to {MAX_TRACKS} correctly-classified tracks per genre.\n")

# Accumulate Shapley vectors per genre
genre_profiles = {g: [] for g in genre_names}
genre_counts   = {g: 0  for g in genre_names}

# Shuffle tracks for random sampling
track_keys = list(all_tracks.keys())
np.random.seed(SEED)
np.random.shuffle(track_keys)

for track_id in track_keys:
    # Check if all genres have enough tracks
    if all(genre_counts[g] >= MAX_TRACKS for g in genre_names):
        break

    track = all_tracks[track_id]
    genre = track['genre']

    # Skip if this genre already has enough tracks
    if genre_counts[genre] >= MAX_TRACKS:
        continue

    true_idx = int(le.transform([genre])[0])

    # Prepare tensors
    mel_t = hsc.format_tensor(track['features']['mel'],    device,
                               apply_log=True)
    cqt_t = hsc.format_tensor(track['features']['cqt'],    device,
                               apply_log=True)
    chr_t = hsc.format_tensor(track['features']['chroma'], device)

    # Check correct classification
    with torch.no_grad():
        pred_class = crnn(mel_t, cqt_t, chr_t).argmax(dim=1).item()

    if pred_class != true_idx:
        continue   # only use correctly-classified tracks

    # Extract semantic entities
    try:
        (beat_frames, section_frames, timbral_mel_bins,
         mask_H_cqt, mask_H_chr, mask_R,
         sections, _) = hsc.extract_track_entities(
            track, device, hsc.TARGET_FRAMES)
    except Exception:
        continue

    # Full-track Shapley game (section_frames from most diagnostic section)
    try:
        shap_vals, _ = hsc.compute_shapley_game(
            mel_t, cqt_t, chr_t, crnn, pred_class,
            timbral_mel_bins, mask_H_cqt, mask_H_chr,
            mask_R, section_frames,
            mel_mean, cqt_mean, chroma_mean
        )
    except Exception:
        continue

    vec = [shap_vals[p] for p in hsc.PLAYERS]   # [T, H, R, S]
    genre_profiles[genre].append(vec)
    genre_counts[genre] += 1

    # Progress
    done = sum(genre_counts.values())
    needed = num_classes * MAX_TRACKS
    sys.stdout.write(f"\r  Progress: {done}/{needed}  "
                     + "  ".join(f"{g[:4]}:{genre_counts[g]}"
                                 for g in genre_names))
    sys.stdout.flush()

print("\n")
for g in genre_names:
    print(f"  {g:<15}: {genre_counts[g]} correctly-classified tracks")

# =============================================================================
# Step 4: Compute mean attribution matrix (8 genres x 4 players)
# =============================================================================
print("\n--- Step 4: Computing Mean Attribution Matrix ---")

# PLAYERS order in harmonicshap_core: ['T', 'H', 'R', 'S']
# Reorder for display: H, R, T, S (harmonic first — more intuitive)
DISPLAY_PLAYERS = ['H', 'R', 'T', 'S']
PLAYER_IDX      = {p: hsc.PLAYERS.index(p) for p in DISPLAY_PLAYERS}
PLAYER_LABELS   = {
    'H': 'Harmonic ($H$)',
    'R': 'Rhythmic ($R$)',
    'T': 'Timbral ($T$)',
    'S': 'Structural ($S$)'
}

mean_matrix = np.zeros((num_classes, len(DISPLAY_PLAYERS)),
                        dtype=np.float32)
se_matrix   = np.zeros_like(mean_matrix)

for gi, genre in enumerate(genre_names):
    vecs = np.array(genre_profiles[genre])   # (n_tracks, 4) in T,H,R,S order
    for pi, player in enumerate(DISPLAY_PLAYERS):
        col = hsc.PLAYERS.index(player)
        if len(vecs) > 0:
            mean_matrix[gi, pi] = float(np.mean(vecs[:, col]))
            se_matrix[gi,   pi] = float(sem(vecs[:, col]))
        else:
            mean_matrix[gi, pi] = 0.0
            se_matrix[gi,   pi] = 0.0

print("\nRaw mean attribution matrix:")
header = f"{'Genre':<16}" + "".join(f"{p:>12}" for p in DISPLAY_PLAYERS)
print(header)
print("-" * len(header))
for gi, genre in enumerate(genre_names):
    row = f"{genre:<16}" + "".join(
        f"{mean_matrix[gi, pi]:>12.4f}" for pi in range(len(DISPLAY_PLAYERS)))
    print(row)

# =============================================================================
# Step 5: Generate heatmap figure
# =============================================================================
print("\n--- Step 5: Generating Heatmap Figure ---")

fig, ax = plt.subplots(figsize=(4.5, 3.2))

# Diverging colormap centred at zero, capped at ±0.3
# Green = player increases genre confidence
# Orange/red = player suppresses genre confidence
vmax = 0.3
im = ax.imshow(mean_matrix, cmap='RdYlGn', aspect='auto',
                vmin=-vmax, vmax=vmax)

# Axis labels
ax.set_xticks(range(len(DISPLAY_PLAYERS)))
ax.set_xticklabels([PLAYER_LABELS[p] for p in DISPLAY_PLAYERS],
                    fontsize=7.5)
ax.set_yticks(range(num_classes))
ax.set_yticklabels(genre_names, fontsize=7.5)

# Annotate cells with raw mean Shapley value
for gi in range(num_classes):
    for pi in range(len(DISPLAY_PLAYERS)):
        val = mean_matrix[gi, pi]
        text_color = 'white' if abs(val) > 0.20 else 'black'
        ax.text(pi, gi, f"{val:.3f}", ha='center', va='center',
                fontsize=6.5, color=text_color, fontweight='bold')

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Mean Shapley Value', fontsize=7)
cbar.ax.tick_params(labelsize=6)

ax.set_title(
    'Genre-Level Semantic Attribution Profiles',
    fontsize=8.5, fontweight='bold', pad=6)
ax.set_xlabel('Semantic Player', fontsize=7.5)
ax.set_ylabel('Genre', fontsize=7.5)

plt.tight_layout(pad=0.8)

fig_path = os.path.join(CHECKPOINT_DIR,
                         "fig_09_genre_attribution_heatmap.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
copy_to_project(fig_path, "fig_09_genre_attribution_heatmap.png")

# =============================================================================
# Step 6: Save results JSON
# =============================================================================
results = {}
for gi, genre in enumerate(genre_names):
    results[genre] = {
        'n_tracks': genre_counts[genre],
        'mean_attribution': {
            p: float(mean_matrix[gi, hsc.PLAYERS.index(p)])
            for p in hsc.PLAYERS
        },
        'se_attribution': {
            p: float(se_matrix[gi, hsc.PLAYERS.index(p)])
            for p in hsc.PLAYERS
        }
    }

json_path = os.path.join(CHECKPOINT_DIR,
                          "exp_genre_attribution_profiles.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
copy_to_project(json_path, "exp_genre_attribution_profiles.json")

print(f"\n[SUCCESS] Script 09 complete.")
print(f"Heatmap : fig_09_genre_attribution_heatmap.png")
print(f"Data    : exp_genre_attribution_profiles.json")
print("Next step: add figure and analysis paragraph to LaTeX.")