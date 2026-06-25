# ==============================================================================
# Module Name: harmonicshap_core.py
# Version: 2.0
# Description: Shared core module for HarmonicSHAP.
#              Import this module in every experiment script.
#              Contains: CRNN architecture, semantic masking, Shapley
#              attribution, and HAC evaluation.
#
# Five fixes over the original per-script implementations:
#   FIX-1  Masking is time-frequency specific, not full-branch zeroing.
#          T  → suppress MFCC-cluster-characteristic Mel frequency bands.
#          H  → suppress chord-active CQT/chroma bins per time frame.
#          R  → suppress beat-aligned frames (librosa beat tracker).
#          S  → suppress the most genre-diagnostic structural section
#               (Foote-style novelty detection).
#   FIX-2  Suppressed regions are filled with the training-set per-bin
#          mean, not silence.
#   FIX-3  Structural player uses Foote novelty detection, not a
#          hardcoded second-half split.
#   FIX-4  Rhythmic player uses librosa.beat.beat_track, not a
#          spectral flux percentile proxy.
#   FIX-5  HAC uses cosine distance (not Pearson), semitone-accurate
#          CQT bin shifting, and a prediction invariance filter.
#
# Dependencies: torch, numpy, librosa, scipy, sklearn
# Usage:  import harmonicshap_core as hsc
# ==============================================================================

import os
import pickle
import math
import itertools

import numpy as np
import torch
import torch.nn as nn
import librosa
from scipy.fft import idct
from scipy.signal import find_peaks, fftconvolve
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine as cosine_dist

# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS  (must match feature extraction settings in 01_data_prep)
# ─────────────────────────────────────────────────────────────────────────────

PLAYERS               = ['T', 'H', 'R', 'S']
N_PLAYERS             = len(PLAYERS)           # 4  → 2^4 = 16 coalitions
N_MEL_BINS            = 128
N_CQT_BINS_PER_OCTAVE = 24                     # B  (2 bins per semitone)
N_OCTAVES             = 7
N_CQT_BINS            = N_CQT_BINS_PER_OCTAVE * N_OCTAVES   # 168
N_CHROMA_BINS         = 12
BINS_PER_SEMITONE     = N_CQT_BINS_PER_OCTAVE // 12          # 2
N_TIMBRAL_CLUSTERS    = 4
N_TOP_TIMBRAL_BINS    = 48         # suppress ~37% of Mel bins for player T
N_TOP_PITCH_CLASSES   = 3          # top-3 pitch classes per frame for player H
SR                    = 22050
HOP_LENGTH            = 512
TARGET_FRAMES         = 1290       # 30 s × 22050 / 512 ≈ 1290 frames

# GRU input size (derived from architecture; do not change independently):
#   Mel branch:    32 filters × (128 // 4) = 32 × 32  = 1024
#   CQT branch:    32 filters × (168 // 4) = 32 × 42  = 1344
#   Chroma branch: 32 filters × (12  // 4) = 32 ×  3  =   96
GRU_INPUT_SIZE = (
    32 * (N_MEL_BINS    // 4) +   # 1024
    32 * (N_CQT_BINS    // 4) +   # 1344
    32 * (N_CHROMA_BINS // 4)     #   96
)                                  # = 2464


# =============================================================================
# CRNN ARCHITECTURE  (shared across all experiment scripts)
# =============================================================================


class MultiBranchCRNN(nn.Module):
    """
    Three-branch CRNN for music genre classification.
    Parallel CNN branches for Mel, CQT, and Chroma, fused before a BiGRU.

    Input  : mel    (B, 1, 128, T)
             cqt    (B, 1, 168, T)
             chroma (B, 1,  12, T)
    Output : logits (B, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d((2, 4))
            )

        self.mel_cnn    = nn.Sequential(conv_block(1, 16), conv_block(16, 32))
        self.cqt_cnn    = nn.Sequential(conv_block(1, 16), conv_block(16, 32))
        self.chroma_cnn = nn.Sequential(conv_block(1, 16), conv_block(16, 32))

        self.gru = nn.GRU(
            input_size=GRU_INPUT_SIZE, hidden_size=128,
            bidirectional=True, batch_first=True
        )
        self.dropout = nn.Dropout(p=0.3)
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, mel, cqt, chroma):
        def flatten_branch(x):
            b, c, f, t = x.size()
            return x.permute(0, 3, 1, 2).reshape(b, t, c * f)

        x = torch.cat([
            flatten_branch(self.mel_cnn(mel)),
            flatten_branch(self.cqt_cnn(cqt)),
            flatten_branch(self.chroma_cnn(chroma))
        ], dim=-1)
        x = self.dropout(x)
        out, _ = self.gru(x)
        return self.fc(self.dropout(out[:, -1, :]))
        

# =============================================================================
# UTILITY: format feature array as CRNN input tensor
# =============================================================================

def format_tensor(x, device, target_frames=TARGET_FRAMES, apply_log=False):
    """
    apply_log=True for Mel and CQT (log1p scaling).
    apply_log=False for Chroma and MFCC (already normalised).
    Must match the transform applied during training in FMALazyDataset.
    """
    if x.shape[1] < target_frames:
        x = np.pad(x, ((0, 0), (0, target_frames - x.shape[1])))
    else:
        x = x[:, :target_frames]
    if apply_log:
        x = np.log1p(x)
    return torch.FloatTensor(x).unsqueeze(0).unsqueeze(0).to(device)
    


# =============================================================================
# 1.  TRAINING-SET PER-BIN MEAN BASELINE  (FIX-2)
# =============================================================================

def precompute_training_mean(chunk_paths, output_path):
    """
    Compute per-frequency-bin mean over all frames in training tracks.
    Call once at the end of data prep (script 01) on training chunks only.

    Parameters
    ----------
    chunk_paths : list[str]
        Paths to training .pkl chunk files.  Each file is a dict:
        {track_id: {'features': {'mel', 'cqt', 'chroma', 'mfcc'},
                    'genre': str, 'split': str}}.
    output_path : str
        Where to save the resulting baseline .pkl.

    Saves
    -----
    Dict with keys 'mel_mean' (128,), 'cqt_mean' (168,), 'chroma_mean' (12,).
    """
    mel_acc  = np.zeros(N_MEL_BINS,    dtype=np.float64)
    cqt_acc  = np.zeros(N_CQT_BINS,    dtype=np.float64)
    chr_acc  = np.zeros(N_CHROMA_BINS, dtype=np.float64)
    n_tracks = 0

    for path in chunk_paths:
        with open(path, 'rb') as f:
            chunk = pickle.load(f)
        for _, track in chunk.items():
            mel_acc  += track['features']['mel'].mean(axis=1)
            cqt_acc  += track['features']['cqt'].mean(axis=1)
            chr_acc  += track['features']['chroma'].mean(axis=1)
            n_tracks += 1

    if n_tracks == 0:
        raise ValueError("No tracks found in chunk_paths.")

    baseline = {
        'mel_mean'    : (mel_acc  / n_tracks).astype(np.float32),
        'cqt_mean'    : (cqt_acc  / n_tracks).astype(np.float32),
        'chroma_mean' : (chr_acc  / n_tracks).astype(np.float32),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(baseline, f)
    print(f"[core] Baseline saved: {output_path}  ({n_tracks} training tracks)")
    return baseline


def load_baseline(baseline_path, device):
    """
    Load precomputed training-mean tensors onto `device`.

    Returns mel_mean (128,), cqt_mean (168,), chroma_mean (12,).
    """
    with open(baseline_path, 'rb') as f:
        bl = pickle.load(f)
    return (
        torch.from_numpy(bl['mel_mean']).to(device),
        torch.from_numpy(bl['cqt_mean']).to(device),
        torch.from_numpy(bl['chroma_mean']).to(device),
    )


# =============================================================================
# 2.  SEMANTIC ENTITY EXTRACTION
# =============================================================================

def extract_beat_frames(mel_np, target_frames=TARGET_FRAMES,
                         sr=SR, hop_length=HOP_LENGTH):
    """
    Estimate beat positions with librosa's dynamic-programming beat tracker.
    FIX-4: replaces spectral flux percentile proxy.

    Returns ndarray[int] of beat frame indices < target_frames.
    """
    onset_env = librosa.onset.onset_strength(
        S=librosa.power_to_db(mel_np[:, :target_frames], ref=np.max),
        sr=sr, hop_length=hop_length
    )
    _, beats = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    return beats[beats < target_frames].astype(int)


def extract_structural_sections(cqt_np, target_frames=TARGET_FRAMES,
                                  n_sections=4):
    """
    Foote-style novelty segmentation (checkerboard kernel on CQT SSM).
    FIX-3: replaces hardcoded second-half split.

    Returns
    -------
    sections            : list[(start, end)] of length n_sections
    most_diagnostic_idx : int — section with highest CQT energy variance
    """
    X = cqt_np[:, :target_frames].astype(np.float32)
    norms  = np.linalg.norm(X, axis=0, keepdims=True) + 1e-8
    S      = (X / norms).T @ (X / norms)

    k = max(4, target_frames // (n_sections * 6))
    blk    = np.ones((k, k), dtype=np.float32)
    kernel = np.block([[-blk, blk], [blk, -blk]])

    novelty = np.maximum(np.diag(fftconvolve(S, kernel, mode='same')), 0)
    peaks, _ = find_peaks(novelty, distance=max(1, target_frames // (n_sections + 1)))

    if len(peaks) >= n_sections - 1:
        boundaries = np.sort(peaks[np.argsort(novelty[peaks])[::-1][:n_sections - 1]])
    else:
        boundaries = np.array(
            [target_frames * i // n_sections for i in range(1, n_sections)]
        )

    boundaries = np.concatenate([[0], boundaries, [target_frames]])
    sections   = [(int(boundaries[i]), int(boundaries[i + 1]))
                  for i in range(n_sections)]
    most_diag  = int(np.argmax([np.var(X[:, s:e]) for s, e in sections]))
    return sections, most_diag


def extract_chord_bins(chroma_np, target_frames=TARGET_FRAMES,
                        n_top=N_TOP_PITCH_CLASSES):
    """
    For each time frame, return CQT and chroma bin indices of the top-N
    most energetic pitch classes.

    With B=24, pitch class p → CQT bins {p*2 + o*24 | o in 0..6}.

    Returns
    -------
    cqt_bins_per_frame    : list[ndarray[int]], length target_frames
    chroma_bins_per_frame : list[ndarray[int]], length target_frames
    """
    T = min(chroma_np.shape[1], target_frames)
    cqt_list, chr_list = [], []
    for t in range(T):
        top = np.argsort(-chroma_np[:, t])[:n_top].astype(int)
        cqt_bins = np.array([
            p * BINS_PER_SEMITONE + o * N_CQT_BINS_PER_OCTAVE
            for p in top for o in range(N_OCTAVES)
        ], dtype=int)
        cqt_list.append(cqt_bins[cqt_bins < N_CQT_BINS])
        chr_list.append(top)
    return cqt_list, chr_list


def extract_timbral_mel_bins(mfcc_np, n_clusters=N_TIMBRAL_CLUSTERS,
                              n_top_bins=N_TOP_TIMBRAL_BINS):
    """
    K-means on MFCC frames → dominant cluster centroid → zero-padded IDCT
    → top-N characteristic Mel bins.
    FIX-1 (timbral variant): replaces full-branch Mel zeroing.

    Returns ndarray[int] of Mel bin indices to suppress.
    """
    n_mfcc, T = mfcc_np.shape
    if T < n_clusters:
        return np.arange(30, 30 + n_top_bins, dtype=int)

    labels   = KMeans(n_clusters=n_clusters, random_state=42,
                      n_init=10).fit_predict(mfcc_np.T)
    centroid = KMeans(n_clusters=n_clusters, random_state=42,
                      n_init=10).fit(mfcc_np.T).cluster_centers_[
                          np.bincount(labels).argmax()]

    padded          = np.zeros(N_MEL_BINS)
    padded[:n_mfcc] = centroid
    return np.sort(
        np.argsort(-np.abs(idct(padded, norm='ortho')))[:n_top_bins]
    ).astype(int)


# =============================================================================
# 3.  BINARY MASK PRECOMPUTATION  (called once per track)
# =============================================================================

def build_harmonic_masks(cqt_bins_per_frame, chroma_bins_per_frame,
                          target_frames, device):
    """
    Build binary time-frequency masks for the Harmonic player.

    Returns
    -------
    mask_H_cqt : FloatTensor (N_CQT_BINS,    target_frames)
    mask_H_chr : FloatTensor (N_CHROMA_BINS, target_frames)
    """
    mc = torch.zeros(N_CQT_BINS,    target_frames, device=device)
    mh = torch.zeros(N_CHROMA_BINS, target_frames, device=device)
    for t in range(min(target_frames, len(cqt_bins_per_frame))):
        cb, pb = cqt_bins_per_frame[t], chroma_bins_per_frame[t]
        if len(cb): mc[torch.from_numpy(cb).long(), t] = 1.0
        if len(pb): mh[torch.from_numpy(pb).long(), t] = 1.0
    return mc, mh


def build_beat_mask(beat_frames, target_frames, device):
    """
    Build binary time mask for the Rhythmic player.

    Returns FloatTensor (target_frames,) with 1 at beat positions.
    """
    mask = torch.zeros(target_frames, device=device)
    bf   = beat_frames[beat_frames < target_frames].astype(int)
    if len(bf):
        mask[torch.from_numpy(bf).long()] = 1.0
    return mask


def extract_track_entities(track_data, device,
                            target_frames=TARGET_FRAMES):
    """
    One-stop wrapper: extract all semantic entities and precompute masks
    for a single track.  Pass results to apply_semantic_mask / compute_shapley_game.

    Returns
    -------
    beat_frames      : ndarray[int]
    section_frames   : ndarray[int]  (indices of most diagnostic section)
    timbral_mel_bins : ndarray[int]
    mask_H_cqt       : FloatTensor (168, T)
    mask_H_chr       : FloatTensor (12,  T)
    mask_R           : FloatTensor (T,)
    sections         : list[(start, end)]
    most_diag_idx    : int
    """
    mel_np    = track_data['features']['mel']
    cqt_np    = track_data['features']['cqt']
    chroma_np = track_data['features']['chroma']
    mfcc_np   = track_data['features']['mfcc']

    beat_frames              = extract_beat_frames(mel_np, target_frames)
    sections, most_diag_idx  = extract_structural_sections(cqt_np, target_frames)
    s0, s1                   = sections[most_diag_idx]
    section_frames           = np.arange(s0, s1, dtype=int)
    cqt_bins_pf, chr_bins_pf = extract_chord_bins(chroma_np, target_frames)
    timbral_mel_bins         = extract_timbral_mel_bins(mfcc_np)
    mask_H_cqt, mask_H_chr   = build_harmonic_masks(
        cqt_bins_pf, chr_bins_pf, target_frames, device
    )
    mask_R = build_beat_mask(beat_frames, target_frames, device)

    return (beat_frames, section_frames, timbral_mel_bins,
            mask_H_cqt, mask_H_chr, mask_R,
            sections, most_diag_idx)


# =============================================================================
# 4.  SEMANTIC MASKING WITH TRAINING-MEAN BASELINE  (FIX-1 + FIX-2)
# =============================================================================

def apply_semantic_mask(mel, cqt, chroma, coalition,
                         timbral_mel_bins,
                         mask_H_cqt, mask_H_chr,
                         mask_R, section_frames,
                         mel_mean, cqt_mean, chroma_mean,
                         ablation_cqt=False):
    """
    Apply coalition-aware semantic mask; suppressed regions → training mean.

    Parameters
    ----------
    mel, cqt, chroma  : FloatTensor (1, 1, n_bins, T)
    coalition         : frozenset of player labels in the coalition
    timbral_mel_bins  : ndarray[int]
    mask_H_cqt        : FloatTensor (168, T)
    mask_H_chr        : FloatTensor (12,  T)
    mask_R            : FloatTensor (T,)
    section_frames    : ndarray[int]
    mel_mean, cqt_mean, chroma_mean : FloatTensor (n_bins,)
    ablation_cqt      : bool  — True for Ablation-CQT condition

    Returns
    -------
    m_mel, m_cqt, m_chr : masked copies of inputs
    """
    T     = mel.shape[3]
    m_mel = mel.clone()
    m_cqt = cqt.clone()
    m_chr = chroma.clone()

    # Ablation-CQT: fill entire CQT branch with training mean
    if ablation_cqt:
        m_cqt = cqt_mean.view(1, 1, -1, 1).expand(1, 1, N_CQT_BINS, T).clone()

    # T absent: suppress timbral Mel frequency bands
    if 'T' not in coalition:
        fill = mel_mean[timbral_mel_bins].view(1, 1, -1, 1).expand(
            1, 1, len(timbral_mel_bins), T)
        m_mel[:, :, timbral_mel_bins, :] = fill

    # H absent: suppress chord-active CQT / chroma bins
    if 'H' not in coalition:
        h4 = mask_H_cqt.unsqueeze(0).unsqueeze(0).bool()
        m_cqt = torch.where(
            h4, cqt_mean.view(1, 1, N_CQT_BINS, 1).expand(1, 1, N_CQT_BINS, T),
            m_cqt)
        hc4 = mask_H_chr.unsqueeze(0).unsqueeze(0).bool()
        m_chr = torch.where(
            hc4, chroma_mean.view(1, 1, N_CHROMA_BINS, 1).expand(
                1, 1, N_CHROMA_BINS, T), m_chr)

    # R absent: suppress beat-aligned frames in all branches
    if 'R' not in coalition:
        r4 = mask_R.view(1, 1, 1, T).bool()
        m_mel = torch.where(r4.expand(1, 1, N_MEL_BINS, T),
                            mel_mean.view(1, 1, -1, 1).expand(1, 1, N_MEL_BINS, T),
                            m_mel)
        m_cqt = torch.where(r4.expand(1, 1, N_CQT_BINS, T),
                            cqt_mean.view(1, 1, -1, 1).expand(1, 1, N_CQT_BINS, T),
                            m_cqt)
        m_chr = torch.where(r4.expand(1, 1, N_CHROMA_BINS, T),
                            chroma_mean.view(1, 1, -1, 1).expand(
                                1, 1, N_CHROMA_BINS, T), m_chr)

    # S absent: suppress most genre-diagnostic structural section
    if 'S' not in coalition and len(section_frames) > 0:
        sf = torch.from_numpy(
            section_frames[section_frames < T].astype(int)).long().to(mel.device)
        if len(sf):
            n = len(sf)
            m_mel[:, :, :, sf] = mel_mean.view(1, 1, -1, 1).expand(
                1, 1, N_MEL_BINS, n)
            m_cqt[:, :, :, sf] = cqt_mean.view(1, 1, -1, 1).expand(
                1, 1, N_CQT_BINS, n)
            m_chr[:, :, :, sf] = chroma_mean.view(1, 1, -1, 1).expand(
                1, 1, N_CHROMA_BINS, n)

    return m_mel, m_cqt, m_chr


# =============================================================================
# 5.  EXACT SHAPLEY GAME  (2^|players| coalitions)
# =============================================================================

def compute_shapley_game(mel, cqt, chroma, model, pred_class,
                          timbral_mel_bins,
                          mask_H_cqt, mask_H_chr,
                          mask_R, section_frames,
                          mel_mean, cqt_mean, chroma_mean,
                          players=None, ablation_cqt=False):
    """
    Compute exact Shapley values.

    v(C) = f(x_C) − f(x_∅) where x_∅ fills all branches with training mean.
    With |players|=4, exactly 16 model evaluations are performed.

    Parameters
    ----------
    model      : MultiBranchCRNN in eval mode
    pred_class : int  — predicted class for x (full signal)
    players    : list[str] or None (defaults to ['T','H','R','S'])
    ablation_cqt : bool  — True for Ablation-CQT condition

    Returns
    -------
    shapley_vals : dict {player: float}
    val_dict     : dict {frozenset: float}  — raw confidence values
    """
    if players is None:
        players = PLAYERS
    n          = len(players)
    coalitions = [frozenset(c)
                  for r in range(n + 1)
                  for c in itertools.combinations(players, r)]
    val_dict   = {}

    model.eval()
    with torch.no_grad():
        for coal in coalitions:
            mm, mc, mh = apply_semantic_mask(
                mel, cqt, chroma, coal,
                timbral_mel_bins,
                mask_H_cqt, mask_H_chr,
                mask_R, section_frames,
                mel_mean, cqt_mean, chroma_mean,
                ablation_cqt=ablation_cqt
            )
            val_dict[coal] = torch.softmax(
                model(mm, mc, mh), dim=1)[0, pred_class].item()

    shapley_vals = {p: 0.0 for p in players}
    for player in players:
        for coal in coalitions:
            if player not in coal:
                w = (math.factorial(len(coal)) *
                     math.factorial(n - len(coal) - 1)) / math.factorial(n)
                shapley_vals[player] += w * (
                    val_dict[coal | frozenset([player])] - val_dict[coal])

    return shapley_vals, val_dict


# =============================================================================
# 6.  HAC METRIC  (FIX-5)
# =============================================================================

def pitch_shift_cqt_chroma(cqt, chroma, n_semitones):
    """
    Semitone-accurate pitch shift.
    CQT  : shift along freq axis by n_semitones × BINS_PER_SEMITONE bins.
           ±2 semitones → ±4 bins;  ±4 semitones → ±8 bins.
    Chroma: circular rotation by n_semitones pitch classes.
    Mel  : unchanged (timbral content is pitch-invariant).
    """
    k = int(round(n_semitones * BINS_PER_SEMITONE))
    return (torch.roll(cqt,    shifts=k,             dims=2),
            torch.roll(chroma, shifts=n_semitones,   dims=2))


def compute_hac_for_track(mel, cqt, chroma, model, device,
                           timbral_mel_bins,
                           mask_H_cqt, mask_H_chr,
                           mask_R, section_frames,
                           mel_mean, cqt_mean, chroma_mean,
                           semitone_shifts=(-4, -2, 2, 4)):
    """
    Compute HAC for one track.

    HAC = 1 − (1/K) Σ cosine_distance(φ(x), φ(T_k(x)))

    Only shifts where model prediction is unchanged are used
    (prediction invariance filter, FIX-5).

    Returns
    -------
    hac_value : float or None (if no valid shifts after filtering)
    n_valid   : int
    """
    model.eval()
    with torch.no_grad():
        pred_orig = torch.argmax(model(mel, cqt, chroma), dim=1).item()

    phi_orig, _ = compute_shapley_game(
        mel, cqt, chroma, model, pred_orig,
        timbral_mel_bins, mask_H_cqt, mask_H_chr,
        mask_R, section_frames,
        mel_mean, cqt_mean, chroma_mean
    )
    vec_orig = np.array([phi_orig[p] for p in PLAYERS])

    distances = []
    for n_semi in semitone_shifts:
        cqt_s, chr_s = pitch_shift_cqt_chroma(cqt, chroma, n_semi)

        with torch.no_grad():
            pred_s = torch.argmax(model(mel, cqt_s, chr_s), dim=1).item()

        if pred_s != pred_orig:     # prediction invariance filter
            continue

        chr_np_s = chr_s.squeeze().cpu().numpy()
        bins_s, cbins_s = extract_chord_bins(chr_np_s, mel.shape[3])
        mH_s, mC_s      = build_harmonic_masks(bins_s, cbins_s,
                                                mel.shape[3], device)

        phi_s, _ = compute_shapley_game(
            mel, cqt_s, chr_s, model, pred_orig,
            timbral_mel_bins, mH_s, mC_s,
            mask_R, section_frames,
            mel_mean, cqt_mean, chroma_mean
        )
        vec_s = np.array([phi_s[p] for p in PLAYERS])

        if np.linalg.norm(vec_orig) < 1e-9 or np.linalg.norm(vec_s) < 1e-9:
            continue

        distances.append(float(cosine_dist(vec_orig, vec_s)))

    if not distances:
        return None, 0
    return 1.0 - float(np.mean(distances)), len(distances)
