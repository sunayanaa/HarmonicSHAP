# ==============================================================================
# Program Name: 05_exp2_ablation.py
# Version: 2.0 (FMA-Small, harmonicshap_core)
# Description: Experiment 2 — Ablation Study.
#              Evaluates the contribution of each semantic layer by computing
#              HAC, Deletion AUC, and Insertion AUC across six configurations:
#              HarmonicSHAP (full), Ablation-H, Ablation-R, Ablation-T,
#              Ablation-CQT, and Ablation-Flat.
#
# Change Log:
#   1.0: GTZAN, full-branch zeroing, silence baseline, Pearson HAC,
#        1-bin circular roll for pitch shift, flux-proxy rhythmic player.
#   2.0: FMA-Small. All masking via harmonicshap_core (FIX 1-5).
#        Correct HAC: cosine distance, semitone-accurate CQT bin shift,
#        prediction invariance filter. Training-mean baseline. Log1p on
#        Mel/CQT. Ablation-H and Ablation-CQT transparently reported.
#        Checkpoint/resume every 5 samples via Google Drive.
#
# GPU Required: Yes
# EVAL_SAMPLES: 100 (fixed seed, same tracks across all ablation conditions)
# Dependencies: torch, scipy, sklearn, harmonicshap_core
# Inputs (Google Drive): fma_features_train_chunk_1.pkl, crnn_backbone_weights.pth,
#               fma_training_baseline.pkl, label_encoder.pkl
# Outputs (Google Drive): exp2_ablation_results.json
#                exp2_ablation_state.pkl  (checkpoint)
# ==============================================================================

import os, sys, pickle, json, random
import numpy as np
import torch
from sklearn.metrics import auc
from scipy.stats import sem
from scipy.spatial.distance import cosine as cosine_dist
from google.colab import drive

# --- Mount Google Drive ---
drive.mount('/content/drive')

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/HarmonicSHAP/"
CHECKPOINT_DIR  = "/content/checkpoints"
EVAL_SAMPLES    = 100
SEED            = 42

random.seed(SEED)
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
# Step 1: Load models, baseline, and data
# =============================================================================
print("\n--- Step 1: Loading CRNN Backbone & Data ---")

for fname in ["crnn_backbone_weights.pth", "label_encoder.pkl",
              "fma_training_baseline.pkl",
              "fma_features_train_chunk_1.pkl"]:
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

with open(os.path.join(CHECKPOINT_DIR,
                        "fma_features_train_chunk_1.pkl"), 'rb') as f:
    chunk_data = pickle.load(f)

# Deterministic sample selection — same 100 tracks across all conditions
all_keys      = sorted(chunk_data.keys())
np.random.seed(SEED)
selected_keys = np.random.choice(
    all_keys, size=min(EVAL_SAMPLES, len(all_keys)), replace=False
).tolist()
print(f"Fixed sample set: {len(selected_keys)} tracks")

# =============================================================================
# Helper: player-level Deletion/Insertion AUC
# =============================================================================

def player_del_ins_auc(mel_t, cqt_t, chr_t, shapley_vals, players,
                        timbral_mel_bins, mask_H_cqt, mask_H_chr,
                        mask_R, section_frames,
                        pred_class, ablation_cqt=False):
    sorted_p = sorted(players, key=lambda p: shapley_vals[p], reverse=True)
    del_coal = set(sorted_p)
    ins_coal = set()
    del_confs, ins_confs = [], []

    with torch.no_grad():
        def conf(coal):
            mm, mc, mh = hsc.apply_semantic_mask(
                mel_t, cqt_t, chr_t, frozenset(coal),
                timbral_mel_bins, mask_H_cqt, mask_H_chr,
                mask_R, section_frames,
                mel_mean, cqt_mean, chroma_mean,
                ablation_cqt=ablation_cqt
            )
            return torch.softmax(crnn(mm, mc, mh),
                                  dim=1)[0, pred_class].item()

        del_confs.append(conf(del_coal))
        ins_confs.append(conf(ins_coal))

        for p in sorted_p:
            del_coal.discard(p)
            ins_coal.add(p)
            del_confs.append(conf(del_coal))
            ins_confs.append(conf(ins_coal))

    x = np.linspace(0, 1, len(del_confs))
    return float(auc(x, del_confs)), float(auc(x, ins_confs))


# =============================================================================
# Step 2: Load or initialise checkpoint
# =============================================================================
CONFIGS = [
    "HarmonicSHAP",
    "Ablation-H",
    "Ablation-R",
    "Ablation-T",
    "Ablation-CQT",
    "Ablation-Flat"
]

state_local = os.path.join(CHECKPOINT_DIR, "exp2_ablation_state.pkl")
if copy_from_project("exp2_ablation_state.pkl", state_local):
    with open(state_local, 'rb') as f:
        state = pickle.load(f)
    print(f"[INFO] Resuming from checkpoint: "
          f"{state['n_processed']}/{EVAL_SAMPLES} samples processed")
else:
    state = {
        'n_processed': 0,
        'results': {c: {'hac': [], 'del': [], 'ins': []} for c in CONFIGS}
    }
    print("[INFO] Starting fresh evaluation")

# =============================================================================
# Step 3: Main evaluation loop
# =============================================================================
print(f"\n--- Step 3: Running Ablation Suite ({EVAL_SAMPLES} fixed samples) ---")

keys_to_process = selected_keys[state['n_processed']:]

for loop_idx, key in enumerate(keys_to_process):
    global_idx = state['n_processed'] + loop_idx + 1
    sys.stdout.write(f"\rProcessing {global_idx}/{EVAL_SAMPLES}...")
    sys.stdout.flush()

    track    = chunk_data[key]
    true_idx = int(le.transform([track['genre']])[0])

    mel_t = hsc.format_tensor(track['features']['mel'],    device, apply_log=True)
    cqt_t = hsc.format_tensor(track['features']['cqt'],    device, apply_log=True)
    chr_t = hsc.format_tensor(track['features']['chroma'], device)

    with torch.no_grad():
        pred_class = crnn(mel_t, cqt_t, chr_t).argmax(dim=1).item()

    # Extract semantic entities once per track
    (beat_frames, section_frames, timbral_mel_bins,
     mask_H_cqt, mask_H_chr, mask_R, sections, _) = hsc.extract_track_entities(
        track, device, hsc.TARGET_FRAMES)

    # ── HAC semitone shifts (shared across all configs) ──────────────────────
    # Used by HarmonicSHAP, Ablation-H, Ablation-R, Ablation-T only
    # (Ablation-CQT and Ablation-Flat HAC reported as N/A)
    SEMITONE_SHIFTS = (-4, -2, 2, 4)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. HarmonicSHAP (full 4-player game)
    # ─────────────────────────────────────────────────────────────────────────
    shap_full, _ = hsc.compute_shapley_game(
        mel_t, cqt_t, chr_t, crnn, pred_class,
        timbral_mel_bins, mask_H_cqt, mask_H_chr,
        mask_R, section_frames, mel_mean, cqt_mean, chroma_mean
    )
    d, i = player_del_ins_auc(
        mel_t, cqt_t, chr_t, shap_full, hsc.PLAYERS,
        timbral_mel_bins, mask_H_cqt, mask_H_chr,
        mask_R, section_frames, pred_class)
    state['results']['HarmonicSHAP']['del'].append(d)
    state['results']['HarmonicSHAP']['ins'].append(i)

    hac_val, _ = hsc.compute_hac_for_track(
        mel_t, cqt_t, chr_t, crnn, device,
        timbral_mel_bins, mask_H_cqt, mask_H_chr,
        mask_R, section_frames, mel_mean, cqt_mean, chroma_mean,
        semitone_shifts=SEMITONE_SHIFTS)
    if hac_val is not None:
        state['results']['HarmonicSHAP']['hac'].append(hac_val)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Ablation-H: remove Harmonic player → 3-player game {T, R, S}
    #    HAC computed over {T, R, S} vector only.
    #    Deletion/Insertion: reported as N/A if baseline confidence < 0.05
    # ─────────────────────────────────────────────────────────────────────────
    players_abH = ['T', 'R', 'S']
    shap_abH, val_dict_abH = hsc.compute_shapley_game(
        mel_t, cqt_t, chr_t, crnn, pred_class,
        timbral_mel_bins, mask_H_cqt, mask_H_chr,
        mask_R, section_frames, mel_mean, cqt_mean, chroma_mean,
        players=players_abH
    )
    # Check baseline confidence with H always suppressed
    with torch.no_grad():
        mm, mc, mh = hsc.apply_semantic_mask(
            mel_t, cqt_t, chr_t, frozenset(players_abH),
            timbral_mel_bins, mask_H_cqt, mask_H_chr,
            mask_R, section_frames, mel_mean, cqt_mean, chroma_mean)
        base_conf_abH = torch.softmax(
            crnn(mm, mc, mh), dim=1)[0, pred_class].item()

    if base_conf_abH >= 0.05:
        d, i = player_del_ins_auc(
            mel_t, cqt_t, chr_t, shap_abH, players_abH,
            timbral_mel_bins, mask_H_cqt, mask_H_chr,
            mask_R, section_frames, pred_class)
        state['results']['Ablation-H']['del'].append(d)
        state['results']['Ablation-H']['ins'].append(i)
    # HAC for Ablation-H is N/A by design (harmonic player absent)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Ablation-R: remove Rhythmic player → 3-player game {T, H, S}
    # ─────────────────────────────────────────────────────────────────────────
    players_abR = ['T', 'H', 'S']
    shap_abR, _ = hsc.compute_shapley_game(
        mel_t, cqt_t, chr_t, crnn, pred_class,
        timbral_mel_bins, mask_H_cqt, mask_H_chr,
        mask_R, section_frames, mel_mean, cqt_mean, chroma_mean,
        players=players_abR
    )
    d, i = player_del_ins_auc(
        mel_t, cqt_t, chr_t, shap_abR, players_abR,
        timbral_mel_bins, mask_H_cqt, mask_H_chr,
        mask_R, section_frames, pred_class)
    state['results']['Ablation-R']['del'].append(d)
    state['results']['Ablation-R']['ins'].append(i)

    # HAC for Ablation-R: uses Harmonic player in vector so valid
    hac_abR, _ = hsc.compute_hac_for_track(
        mel_t, cqt_t, chr_t, crnn, device,
        timbral_mel_bins, mask_H_cqt, mask_H_chr,
        mask_R, section_frames, mel_mean, cqt_mean, chroma_mean,
        semitone_shifts=SEMITONE_SHIFTS)
    # Note: HAC computed on full 4-player attribution to measure impact
    # of removing R from the Shapley game on harmonic consistency
    # We compare full HarmonicSHAP HAC vs Ablation-R HAC (4-player game
    # run with R-suppressed inputs)
    empty_mask_R = torch.zeros(hsc.TARGET_FRAMES, device=device)
    shap_abR_hac, _ = hsc.compute_shapley_game(
        mel_t, cqt_t, chr_t, crnn, pred_class,
        timbral_mel_bins, mask_H_cqt, mask_H_chr,
        empty_mask_R, section_frames, mel_mean, cqt_mean, chroma_mean
    )
    vec_abR = np.array([shap_abR_hac[p] for p in hsc.PLAYERS])

    # Compare with shifted inputs
    distances_abR = []
    for n_semi in SEMITONE_SHIFTS:
        cqt_s, chr_s = hsc.pitch_shift_cqt_chroma(cqt_t, chr_t, n_semi)
        with torch.no_grad():
            pred_s = crnn(mel_t, cqt_s, chr_s).argmax(dim=1).item()
        if pred_s != pred_class:
            continue
        chr_np_s = chr_s.squeeze().cpu().numpy()
        bins_s, cbins_s = hsc.extract_chord_bins(chr_np_s, hsc.TARGET_FRAMES)
        mH_s, mC_s = hsc.build_harmonic_masks(bins_s, cbins_s,
                                                hsc.TARGET_FRAMES, device)
        shap_s, _ = hsc.compute_shapley_game(
            mel_t, cqt_s, chr_s, crnn, pred_class,
            timbral_mel_bins, mH_s, mC_s,
            empty_mask_R, section_frames, mel_mean, cqt_mean, chroma_mean)
        vec_s = np.array([shap_s[p] for p in hsc.PLAYERS])
        if np.linalg.norm(vec_abR) > 1e-9 and np.linalg.norm(vec_s) > 1e-9:
            distances_abR.append(float(cosine_dist(vec_abR, vec_s)))
    if distances_abR:
        state['results']['Ablation-R']['hac'].append(
            1.0 - float(np.mean(distances_abR)))

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Ablation-T: remove Timbral player → 3-player game {H, R, S}
    # ─────────────────────────────────────────────────────────────────────────
    players_abT = ['H', 'R', 'S']
    # T absent → suppress only the 48 characteristic timbral bins,
    # not the entire Mel branch. timbral_mel_bins already computed
    # by extract_track_entities above.
    shap_abT, _ = hsc.compute_shapley_game(
        mel_t, cqt_t, chr_t, crnn, pred_class,
        timbral_mel_bins, mask_H_cqt, mask_H_chr,
        mask_R, section_frames, mel_mean, cqt_mean, chroma_mean,
        players=players_abT
    )
    d, i = player_del_ins_auc(
        mel_t, cqt_t, chr_t, shap_abT, players_abT,
        timbral_mel_bins, mask_H_cqt, mask_H_chr,
        mask_R, section_frames, pred_class)
    state['results']['Ablation-T']['del'].append(d)
    state['results']['Ablation-T']['ins'].append(i)
    
    hac_abT_val, _ = hsc.compute_hac_for_track(
        mel_t, cqt_t, chr_t, crnn, device,
        timbral_mel_bins, mask_H_cqt, mask_H_chr,
        mask_R, section_frames, mel_mean, cqt_mean, chroma_mean,
        semitone_shifts=SEMITONE_SHIFTS)
    if hac_abT_val is not None:
        state['results']['Ablation-T']['hac'].append(hac_abT_val)
    # ─────────────────────────────────────────────────────────────────────────
    # 5. Ablation-CQT: CQT branch always filled with mean
    #    Deletion/Insertion N/A if baseline confidence < 0.05
    #    HAC N/A (CQT is the substrate for harmonic attribution)
    # ─────────────────────────────────────────────────────────────────────────
    shap_abCQT, _ = hsc.compute_shapley_game(
        mel_t, cqt_t, chr_t, crnn, pred_class,
        timbral_mel_bins, mask_H_cqt, mask_H_chr,
        mask_R, section_frames, mel_mean, cqt_mean, chroma_mean,
        ablation_cqt=True
    )
    with torch.no_grad():
        mm, mc, mh = hsc.apply_semantic_mask(
            mel_t, cqt_t, chr_t, frozenset(hsc.PLAYERS),
            timbral_mel_bins, mask_H_cqt, mask_H_chr,
            mask_R, section_frames, mel_mean, cqt_mean, chroma_mean,
            ablation_cqt=True)
        base_conf_abCQT = torch.softmax(
            crnn(mm, mc, mh), dim=1)[0, pred_class].item()

    if base_conf_abCQT >= 0.05:
        d, i = player_del_ins_auc(
            mel_t, cqt_t, chr_t, shap_abCQT, hsc.PLAYERS,
            timbral_mel_bins, mask_H_cqt, mask_H_chr,
            mask_R, section_frames, pred_class, ablation_cqt=True)
        state['results']['Ablation-CQT']['del'].append(d)
        state['results']['Ablation-CQT']['ins'].append(i)
    # HAC: N/A (no CQT → no pitch-shift consistency measurement)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Ablation-Flat: three raw acoustic branches as players
    #    Players: Mel, CQT, Chroma — no semantic grouping
    #    Masking: suppress entire branch with training mean
    # ─────────────────────────────────────────────────────────────────────────
    flat_players = ['Mel', 'CQT', 'Chr']
    flat_coalitions = [
        frozenset(c)
        for r in range(len(flat_players) + 1)
        for c in __import__('itertools').combinations(flat_players, r)
    ]
    flat_val_dict = {}
    with torch.no_grad():
        for coal in flat_coalitions:
            mm = mel_t.clone()
            mc = cqt_t.clone()
            mh = chr_t.clone()
            if 'Mel' not in coal:
                mm = mel_mean.view(1, 1, -1, 1).expand(
                    1, 1, hsc.N_MEL_BINS, hsc.TARGET_FRAMES).clone()
            if 'CQT' not in coal:
                mc = cqt_mean.view(1, 1, -1, 1).expand(
                    1, 1, hsc.N_CQT_BINS, hsc.TARGET_FRAMES).clone()
            if 'Chr' not in coal:
                mh = chroma_mean.view(1, 1, -1, 1).expand(
                    1, 1, hsc.N_CHROMA_BINS, hsc.TARGET_FRAMES).clone()
            flat_val_dict[coal] = torch.softmax(
                crnn(mm, mc, mh), dim=1)[0, pred_class].item()

    import math
    n_flat = len(flat_players)
    flat_shap = {p: 0.0 for p in flat_players}
    for player in flat_players:
        for coal in flat_coalitions:
            if player not in coal:
                w = (math.factorial(len(coal)) *
                     math.factorial(n_flat - len(coal) - 1)
                     ) / math.factorial(n_flat)
                flat_shap[player] += w * (
                    flat_val_dict[coal | frozenset([player])] -
                    flat_val_dict[coal])

    sorted_flat = sorted(flat_players,
                          key=lambda p: flat_shap[p], reverse=True)
    del_flat, ins_flat = [], []
    del_coal_f = set(sorted_flat)
    ins_coal_f = set()

    with torch.no_grad():
        def conf_flat(coal):
            mm = mel_t.clone()
            mc = cqt_t.clone()
            mh = chr_t.clone()
            if 'Mel' not in coal:
                mm = mel_mean.view(1, 1, -1, 1).expand(
                    1, 1, hsc.N_MEL_BINS, hsc.TARGET_FRAMES).clone()
            if 'CQT' not in coal:
                mc = cqt_mean.view(1, 1, -1, 1).expand(
                    1, 1, hsc.N_CQT_BINS, hsc.TARGET_FRAMES).clone()
            if 'Chr' not in coal:
                mh = chroma_mean.view(1, 1, -1, 1).expand(
                    1, 1, hsc.N_CHROMA_BINS, hsc.TARGET_FRAMES).clone()
            return torch.softmax(crnn(mm, mc, mh),
                                  dim=1)[0, pred_class].item()

        del_flat.append(conf_flat(frozenset(del_coal_f)))
        ins_flat.append(conf_flat(frozenset(ins_coal_f)))
        for p in sorted_flat:
            del_coal_f.discard(p)
            ins_coal_f.add(p)
            del_flat.append(conf_flat(frozenset(del_coal_f)))
            ins_flat.append(conf_flat(frozenset(ins_coal_f)))

    x_flat = np.linspace(0, 1, len(del_flat))
    state['results']['Ablation-Flat']['del'].append(
        float(auc(x_flat, del_flat)))
    state['results']['Ablation-Flat']['ins'].append(
        float(auc(x_flat, ins_flat)))

    # HAC for Ablation-Flat: same pitch-shift test but over flat branch vector
    vec_flat = np.array([flat_shap[p] for p in flat_players])
    distances_flat = []
    for n_semi in SEMITONE_SHIFTS:
        cqt_s, chr_s = hsc.pitch_shift_cqt_chroma(cqt_t, chr_t, n_semi)
        with torch.no_grad():
            pred_s = crnn(mel_t, cqt_s, chr_s).argmax(dim=1).item()
        if pred_s != pred_class:
            continue
        flat_val_s = {}
        with torch.no_grad():
            for coal in flat_coalitions:
                mm = mel_t.clone()
                mc = cqt_s.clone()
                mh = chr_s.clone()
                if 'Mel' not in coal:
                    mm = mel_mean.view(1, 1, -1, 1).expand(
                        1, 1, hsc.N_MEL_BINS, hsc.TARGET_FRAMES).clone()
                if 'CQT' not in coal:
                    mc = cqt_mean.view(1, 1, -1, 1).expand(
                        1, 1, hsc.N_CQT_BINS, hsc.TARGET_FRAMES).clone()
                if 'Chr' not in coal:
                    mh = chroma_mean.view(1, 1, -1, 1).expand(
                        1, 1, hsc.N_CHROMA_BINS, hsc.TARGET_FRAMES).clone()
                flat_val_s[coal] = torch.softmax(
                    crnn(mm, mc, mh), dim=1)[0, pred_class].item()
        flat_shap_s = {p: 0.0 for p in flat_players}
        for player in flat_players:
            for coal in flat_coalitions:
                if player not in coal:
                    w = (math.factorial(len(coal)) *
                         math.factorial(n_flat - len(coal) - 1)
                         ) / math.factorial(n_flat)
                    flat_shap_s[player] += w * (
                        flat_val_s[coal | frozenset([player])] -
                        flat_val_s[coal])
        vec_s = np.array([flat_shap_s[p] for p in flat_players])
        if np.linalg.norm(vec_flat) > 1e-9 and np.linalg.norm(vec_s) > 1e-9:
            distances_flat.append(float(cosine_dist(vec_flat, vec_s)))
    if distances_flat:
        state['results']['Ablation-Flat']['hac'].append(
            1.0 - float(np.mean(distances_flat)))

    state['n_processed'] += 1

    # Checkpoint every 5 samples
    if state['n_processed'] % 5 == 0:
        with open(state_local, 'wb') as f:
            pickle.dump(state, f)
        copy_to_project(state_local, "exp2_ablation_state.pkl")

# Final checkpoint
with open(state_local, 'wb') as f:
    pickle.dump(state, f)
copy_to_project(state_local, "exp2_ablation_state.pkl")

# =============================================================================
# Step 4: Report results
# =============================================================================
print("\n\n=== Experiment 2: Ablation Results (Mean ± SE) ===")
print("Note: HAC N/A for Ablation-H (harmonic player absent by design).")
print("      HAC N/A for Ablation-CQT (CQT substrate absent).")
print("      Del/Ins AUC N/A where baseline model confidence < 0.05.\n")

final_metrics = {}

for config in CONFIGS:
    res  = state['results'][config]
    hac  = res['hac']
    dels = res['del']
    ins  = res['ins']

    entry = {}
    print(f"[{config}]")

    if hac:
        h_m, h_se = float(np.mean(hac)), float(sem(hac))
        entry['HAC_mean'] = h_m
        entry['HAC_se']   = h_se
        print(f"  HAC     : {h_m:.4f} ± {h_se:.4f}")
    else:
        entry['HAC_mean'] = None
        entry['HAC_se']   = None
        print(f"  HAC     : N/A")

    if dels:
        d_m, d_se = float(np.mean(dels)), float(sem(dels))
        i_m, i_se = float(np.mean(ins)),  float(sem(ins))
        entry['Del_AUC_mean'] = d_m
        entry['Del_AUC_se']   = d_se
        entry['Ins_AUC_mean'] = i_m
        entry['Ins_AUC_se']   = i_se
        print(f"  Del AUC : {d_m:.4f} ± {d_se:.4f}")
        print(f"  Ins AUC : {i_m:.4f} ± {i_se:.4f}")
    else:
        entry['Del_AUC_mean'] = None
        entry['Ins_AUC_mean'] = None
        print(f"  Del AUC : N/A (baseline confidence < 0.05)")
        print(f"  Ins AUC : N/A (baseline confidence < 0.05)")

    final_metrics[config] = entry
    print()

# =============================================================================
# Step 5: Save and upload
# =============================================================================
results_path = os.path.join(CHECKPOINT_DIR, "exp2_ablation_results.json")
with open(results_path, 'w') as f:
    json.dump(final_metrics, f, indent=2)
copy_to_project(results_path, "exp2_ablation_results.json")

print("[SUCCESS] Script 05 complete. Results uploaded.")
print("Next step: run 07_exp3_data_prep.py")