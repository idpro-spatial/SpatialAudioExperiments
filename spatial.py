#!/usr/bin/env python3
"""
spatialize.py — Simple robust single-file SOFA spatializer

Usage examples:
  python spatialize.py -i input.wav -s HRIR.sofa -o out.wav -a 45 -e 0
  python spatialize.py --list-sofa-dir ./hrtf_dir
"""

from pathlib import Path
import argparse
import logging
import sys
import tempfile

from typing import Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import netCDF4 as nc
from scipy.signal import fftconvolve

# Optional reverb (soft dependency)
try:
    from pedalboard import Pedalboard, Reverb
    _HAS_PEDALBOARD = True
except Exception:
    _HAS_PEDALBOARD = False

# Constants
DEFAULT_SR = 44100
LOG = logging.getLogger("spatialize")

# ======================= NEW: MOVING SOURCE SUPPORT =======================
class SourcePath:
    """Represents a (linear) movement path for a sound source in (az, el) space.

    Currently implements straight-line movement with constant *angular* speed
    (degrees per second) between start and end positions.
    """
    def __init__(self, start_az: float, start_el: float,
                 end_az: float, end_el: float, speed_deg_per_sec: float,
                 start_dist: float = 1.0, end_dist: float = 1.0):
        self.start = np.array([float(start_az), float(start_el)], dtype=np.float64)
        self.end = np.array([float(end_az), float(end_el)], dtype=np.float64)
        self.speed = float(speed_deg_per_sec)
        self.distance = float(np.linalg.norm(self.end - self.start))
        # Distance (radial) path parameters
        self.start_dist = float(start_dist)
        self.end_dist = float(end_dist)
        if self.speed <= 0.0 and self.start_dist != self.end_dist:
            # If only distance changes, create synthetic duration based on 1 deg/sec equivalent
            self.speed = 1.0
            self.distance = abs(self.end_dist - self.start_dist)
        if self.speed <= 0.0:
            self.duration = float('inf')
        else:
            # Share one unified duration over which BOTH angle and distance interpolate linearly
            self.duration = max(self.distance / self.speed, 1e-12)
        LOG.info("SourcePath: angles %.2f deg travel, distance %.2f -> %.2f (dur=%.3f s)",
                 self.distance, self.start_dist, self.end_dist, self.duration)

    def position_at(self, t_sec: float) -> Tuple[float, float]:
        """Return (az, el) at time t (seconds). Clamps to end after duration."""
        if t_sec >= self.duration:
            return float(self.end[0]), float(self.end[1])
        if self.distance == 0.0 or self.duration == float('inf'):
            return float(self.start[0]), float(self.start[1])
        frac = t_sec / self.duration
        pos = self.start + frac * (self.end - self.start)
        return float(pos[0]), float(pos[1])

    def distance_at(self, t_sec: float) -> float:
        if t_sec >= self.duration:
            return self.end_dist
        if self.duration == float('inf'):
            return self.start_dist
        frac = t_sec / self.duration
        return self.start_dist + frac * (self.end_dist - self.start_dist)
# ==========================================================================

# ================= Distance / Air Absorption Helpers =====================

def compute_distance_gain(dist: float, ref_dist: float = 1.0, rolloff: float = 1.0, min_dist: float = 0.1) -> float:
    """Inverse style distance attenuation.
    gain = (ref/dist)^rolloff with clamping to avoid infinities."""
    d = max(dist, min_dist, 1e-6)
    ref = max(ref_dist, min_dist)
    return float((ref / d) ** rolloff)


def apply_air_absorption(block: np.ndarray, sr: int, dist: float,
                          prev_state: float,  # NEW: persistent one-pole state
                          ref_dist: float = 1.0,
                          air_strength: float = 0.35) -> Tuple[np.ndarray, float]:
    """Stateful lightweight HF damping approximating air absorption.
    Returns (processed_block, new_state). If bypassed, state passes through unchanged."""
    if air_strength <= 0 or dist <= ref_dist:
        return block, prev_state
    fc_near = 0.45 * sr
    fc = fc_near * np.exp(-air_strength * (dist - ref_dist))
    fc = float(np.clip(fc, 200.0, fc_near))
    a = (2 * np.pi * fc) / (2 * np.pi * fc + sr)
    y = np.empty_like(block)
    prev = prev_state
    for i, x in enumerate(block):
        prev += a * (x - prev)
        y[i] = prev
    return y, prev
# =========================================================================


def angular_distance_rad(az1_deg: float, el1_deg: float,
                         az2_deg: np.ndarray, el2_deg: np.ndarray) -> np.ndarray:
    """
    Compute angular distance (radians) between (az1, el1) and arrays az2, el2.
    Uses unit-vector dot product on sphere after converting degrees to radians.

    Returns array of angular distances in radians.
    """
    az1 = np.deg2rad(az1_deg)
    el1 = np.deg2rad(el1_deg)
    az2 = np.deg2rad(az2_deg)
    el2 = np.deg2rad(el2_deg)

    v1 = np.array([np.cos(el1) * np.cos(az1),
                   np.cos(el1) * np.sin(az1),
                   np.sin(el1)])  # shape (3,)

    v2 = np.stack([np.cos(el2) * np.cos(az2),
                   np.cos(el2) * np.sin(az2),
                   np.sin(el2)], axis=-1)  # shape (..., 3)

    dot = np.clip(np.sum(v2 * v1, axis=-1), -1.0, 1.0)
    return np.arccos(dot)


def read_sofa(hrtf_path: Path):
    """
    Read SOFA file and return (hrir_array, positions, sr_sofa)
    Expected Data.IR shape: (M, R, N) with R usually 2 (L/R).
    SourcePosition shape: (M, 3) or (M, 2) — expects az, el in degrees in first two cols.
    """
    if not hrtf_path.exists():
        raise FileNotFoundError(f"SOFA file not found: {hrtf_path}")

    with nc.Dataset(str(hrtf_path), "r") as ds:
        # Basic checks
        if "Data.IR" not in ds.variables:
            raise RuntimeError("SOFA missing variable 'Data.IR'")
        if "SourcePosition" not in ds.variables:
            raise RuntimeError("SOFA missing variable 'SourcePosition'")

        hrir = np.array(ds.variables["Data.IR"][:])  # copy out as numpy array
        pos = np.array(ds.variables["SourcePosition"][:])

        # Sampling rate: sometimes scalar or array; extract safely
        if "Data.SamplingRate" in ds.variables:
            sr_val = ds.variables["Data.SamplingRate"][:]
            try:
                sr_sofa = int(np.asarray(sr_val).item())
            except Exception:
                # fallback: convert first element
                sr_sofa = int(np.asarray(sr_val)[0])
        else:
            LOG.debug("SOFA has no Data.SamplingRate; assuming %d", DEFAULT_SR)
            sr_sofa = DEFAULT_SR

    LOG.debug("Loaded SOFA: HRIR shape=%s, SourcePosition shape=%s, sr=%s",
              hrir.shape, pos.shape, sr_sofa)
    return hrir, pos, sr_sofa


def pick_closest_hrir(hrir: np.ndarray, positions: np.ndarray,
                      az_deg: float, el_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pick the HRIR measurement closest to requested az/el.
    Returns (left_hrir, right_hrir) as 1D numpy arrays (float32).
    """
    if positions.ndim < 2 or positions.shape[1] < 2:
        raise RuntimeError("SourcePosition must contain at least azimuth and elevation columns")

    azs = positions[:, 0]
    els = positions[:, 1]

    dists = angular_distance_rad(az_deg, el_deg, azs, els)
    idx = int(np.argmin(dists))
    LOG.debug("Chosen SOFA index=%d (angular distance=%.3f deg)",
              idx, np.rad2deg(dists[idx]))

    # Validate HRIR dims
    if hrir.ndim != 3:
        raise RuntimeError(f"Unexpected Data.IR shape {hrir.shape}; expected 3D (M,R,N)")

    # Typical convention: Data.IR[measurement, receiver, sample]
    # Assume receiver 0 = left, 1 = right. Validate length
    if hrir.shape[1] < 2:
        raise RuntimeError("HRIR receiver axis has fewer than 2 channels")

    left = np.asarray(hrir[idx, 0, :], dtype=np.float32)
    right = np.asarray(hrir[idx, 1, :], dtype=np.float32)
    return left, right


def interpolate_hrir(hrir: np.ndarray, positions: np.ndarray,
                      az_deg: float, el_deg: float,
                      k: int = 4, power: float = 2.0,
                      exact_thresh_deg: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate HRIR at arbitrary (az, el) by blending k nearest measurements
    using inverse-distance weighting on the sphere.

    - k: number of neighbors to blend (default 4)
    - power: exponent for inverse-distance weights (2.0 = IDW^2)
    - exact_thresh_deg: if the closest sample is within this angular threshold,
      return it directly (avoids unnecessary blending).
    """
    if positions.ndim < 2 or positions.shape[1] < 2:
        raise RuntimeError("SourcePosition must contain at least azimuth and elevation columns")

    azs = positions[:, 0]
    els = positions[:, 1]

    dists_rad = angular_distance_rad(az_deg, el_deg, azs, els)
    min_idx = int(np.argmin(dists_rad))
    min_deg = float(np.rad2deg(dists_rad[min_idx]))

    # If we're effectively at a measured position, just return that HRIR
    if min_deg <= exact_thresh_deg:
        LOG.debug("Interpolation skipped; exact/nearby sample at idx=%d (%.6f deg)", min_idx, min_deg)
        left = np.asarray(hrir[min_idx, 0, :], dtype=np.float32)
        right = np.asarray(hrir[min_idx, 1, :], dtype=np.float32)
        return left, right

    # Blend k nearest neighbors
    m = dists_rad.shape[0]
    k_eff = int(max(1, min(k, m)))
    # Get indices of k smallest distances
    nn_idx = np.argpartition(dists_rad, k_eff - 1)[:k_eff]
    nn_d = dists_rad[nn_idx]

    # Inverse-distance weights on spherical distance (avoid div-by-zero)
    eps = 1e-12
    w = 1.0 / np.maximum(nn_d, eps) ** power
    w /= np.sum(w)

    LOG.debug("Interpolating HRIR using %d-NN (min=%.3f deg)", k_eff, min_deg)

    # Weighted sum of HRIRs
    # Assume Data.IR shape (M, R, N); R>=2
    if hrir.ndim != 3 or hrir.shape[1] < 2:
        raise RuntimeError(f"Unexpected Data.IR shape {hrir.shape}; expected (M,>=2,N)")

    lefts = hrir[nn_idx, 0, :].astype(np.float64)
    rights = hrir[nn_idx, 1, :].astype(np.float64)

    left = np.tensordot(w, lefts, axes=(0, 0)).astype(np.float32)
    right = np.tensordot(w, rights, axes=(0, 0)).astype(np.float32)
    return left, right


def resample_if_needed(sig: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample 1D signal using librosa if sample rates differ.
    For short HRIRs this is efficient.
    """
    if orig_sr == target_sr:
        return sig
    LOG.debug("Resampling from %d -> %d", orig_sr, target_sr)
    return librosa.resample(sig.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def convolve_hrir(audio: np.ndarray, left_hrir: np.ndarray,
                  right_hrir: np.ndarray) -> np.ndarray:
    """
    Convolve mono audio with left/right HRIRs using FFT convolution.
    Returns the FULL convolution result (len(audio)+len(hrir)-1, 2).
    Caller decides whether to trim (static case) or overlap-add (moving case).
    """
    left_full = fftconvolve(audio, left_hrir, mode="full")
    right_full = fftconvolve(audio, right_hrir, mode="full")
    stereo = np.stack([left_full, right_full], axis=-1).astype(np.float32)
    return stereo


def apply_optional_reverb(stereo: np.ndarray, sr: int,
                          room_size: float = 0.25, wet_level: float = 0.12) -> np.ndarray:
    """
    If pedalboard is available, apply a simple stereo reverb and return processed stereo.
    Otherwise returns the input unchanged.
    """
    if not _HAS_PEDALBOARD:
        LOG.debug("Pedalboard not available; skipping reverb")
        return stereo

    LOG.debug("Applying pedalboard Reverb (room_size=%.2f, wet=%.2f)",
              room_size, wet_level)
    board = Pedalboard([
        Reverb(room_size=room_size,
               damping=0.5,
               wet_level=wet_level,
               dry_level=1.0 - wet_level,
               width=1.0,
               freeze_mode=0.0)
    ])
    # pedalboard expects shape (n_channels, n_samples)
    processed = board(stereo.T, sample_rate=sr)
    return np.asarray(processed.T, dtype=np.float32)


def normalize_audio(stereo: np.ndarray, peak_target: float = 0.95) -> np.ndarray:
    """
    Peak-normalize stereo audio to peak_target (0..1). Returns float32.
    """
    peak = float(np.max(np.abs(stereo)))
    if peak <= 0:
        return stereo
    gain = peak_target / peak
    return (stereo * gain).astype(np.float32)


def write_atomic(path: Path, data: np.ndarray, sr: int, subtype: str = "PCM_24"):
    """
    Write audio to disk atomically: write to a temp file in same dir then replace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False,
                                   dir=str(path.parent),
                                   prefix=path.name + ".tmp",
                                   suffix=".wav") as tf:
        tmp_path = Path(tf.name)
    # soundfile writes to given path
    sf.write(str(tmp_path), data, sr, subtype=subtype)
    tmp_path.replace(path)
    LOG.info("Wrote output: %s", path)


def spatialize_file(input_path: Path, sofa_path: Path, output_path: Path,
                    azimuth: float, elevation: float, target_sr: int,
                    apply_reverb: bool):
    """
    Full pipeline: load audio, read SOFA, pick HRIR, (resample HRIR), convolve,
    optionally reverb, normalize and write to disk.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    LOG.info("Loading audio: %s", input_path)
    audio, sr_audio = librosa.load(str(input_path), sr=target_sr, mono=True)
    audio = audio.astype(np.float32)
    LOG.debug("Audio length=%d samples, sr=%d", len(audio), target_sr)

    LOG.info("Reading SOFA: %s", sofa_path)
    hrir, pos, sr_sofa = read_sofa(sofa_path)

    # Use interpolated HRIR instead of nearest-only
    left_hrir, right_hrir = interpolate_hrir(hrir, pos, azimuth, elevation, k=4)

    # Resample HRIRs to target sample rate if needed
    left_hrir = resample_if_needed(left_hrir, sr_sofa, target_sr)
    right_hrir = resample_if_needed(right_hrir, sr_sofa, target_sr)

    # Convolve
    LOG.info("Convolving audio with HRIRs...")
    stereo_full = convolve_hrir(audio, left_hrir, right_hrir)
    # Trim full convolution to original program length (classic zero-latency rendering)
    stereo = stereo_full[:len(audio)]
    if apply_reverb:
        LOG.info("Applying optional reverb (pedalboard)...")
        stereo = apply_optional_reverb(stereo, target_sr)
    stereo = normalize_audio(stereo)
    write_atomic(output_path, stereo, target_sr)
    LOG.info("Spatialization complete")


def spatialize_moving_source(input_path: Path, sofa_path: Path, output_path: Path,
                              movement: SourcePath, target_sr: int, apply_reverb: bool,
                              block_size: int = 2048, interp_k: int = 4,
                              interp_power: float = 2.0, filter_morph: bool = True,
                              morph_tol_deg: float = 1e-4,
                              # Distance params
                              ref_dist: float = 1.0, rolloff: float = 1.0,
                              air_strength: float = 0.35,
                              wet_max: float = 0.3, max_dist: float = 10.0,
                              boundary_xfade: int = 256):
    """Artifact-minimized moving source renderer.

    Improvements vs previous version:
      * Removes per-block reverb (tail discontinuities) -> single global reverb pass.
      * Distance-based direct/reverb ratio applied via continuous per-sample envelope.
      * Stores only dry field during block convolution (with optional HRIR morphing).
      * Wet envelope derived analytically from SourcePath distance (vectorized, no block seams).
      * NEW: Optional short boundary crossfade (boundary_xfade) between successive block contributions
        to further suppress subtle clicks from rapid HRIR / distance changes. Set 0 to disable.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")
    LOG.info("Loading audio (moving source): %s", input_path)
    audio, _ = librosa.load(str(input_path), sr=target_sr, mono=True)
    audio = audio.astype(np.float32)
    n_samples = len(audio)
    LOG.debug("Audio length=%d samples (%.2f s)", n_samples, n_samples / target_sr)

    LOG.info("Reading SOFA: %s", sofa_path)
    hrir_all, positions, sr_sofa = read_sofa(sofa_path)
    hrir_len = hrir_all.shape[2]
    if sr_sofa != target_sr:
        LOG.debug("HRIR sample rate (%d) will be resampled per block to %d", sr_sofa, target_sr)

    # Output dry buffer (with tail) & build only once
    dry_out = np.zeros((n_samples + hrir_len - 1, 2), dtype=np.float32)

    num_blocks = int(np.ceil(n_samples / block_size))
    LOG.info("Processing %d blocks (block_size=%d, morph=%s, global reverb=%s)",
             num_blocks, block_size, filter_morph, apply_reverb)

    # Shared air absorption state
    air_state = 0.0

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, n_samples)
        block = audio[start:end]
        if block.size == 0:
            continue
        t_start = start / target_sr
        t_end = end / target_sr
        az_start, el_start = movement.position_at(t_start)
        az_end, el_end = movement.position_at(t_end)
        dist_start = movement.distance_at(t_start)
        dist_end = movement.distance_at(t_end)
        if b % 25 == 0:
            LOG.debug("Block %d/%d t=[%.2f,%.2f] az=(%.2f->%.2f) el=(%.2f->%.2f) dist=(%.2f->%.2f)",
                      b, num_blocks, t_start, t_end, az_start, az_end, el_start, el_end, dist_start, dist_end)

        gain_start = compute_distance_gain(dist_start, ref_dist, rolloff)
        need_morph = filter_morph and (abs(az_end - az_start) >= morph_tol_deg or abs(dist_end - dist_start) >= 1e-6)

        if not need_morph:
            proc, air_state = apply_air_absorption(block * gain_start, target_sr, dist_start,
                                                   air_state, ref_dist, air_strength)
            l_hrir, r_hrir = interpolate_hrir(hrir_all, positions, az_start, el_start,
                                              k=interp_k, power=interp_power)
            l_hrir = resample_if_needed(l_hrir, sr_sofa, target_sr)
            r_hrir = resample_if_needed(r_hrir, sr_sofa, target_sr)
            stereo = convolve_hrir(proc, l_hrir, r_hrir)
            # NEW: boundary crossfade
            if boundary_xfade > 0 and start > 0:
                xfade = min(boundary_xfade, stereo.shape[0])
                existing = dry_out[start:start + xfade]
                ramp = np.linspace(0.0, 1.0, xfade, dtype=np.float32)[:, None]
                dry_out[start:start + xfade] = existing * (1.0 - ramp) + stereo[:xfade] * ramp
                if xfade < stereo.shape[0]:
                    dry_out[start + xfade:start + stereo.shape[0]] += stereo[xfade:]
            else:
                dry_out[start:start + stereo.shape[0]] += stereo
            continue

        # Morph path (dual dry render -> crossfade)
        air_state_initial = air_state
        # Start variant
        dry_start, air_state_after = apply_air_absorption(block * gain_start, target_sr, dist_start,
                                                          air_state_initial, ref_dist, air_strength)
        l_s, r_s = interpolate_hrir(hrir_all, positions, az_start, el_start,
                                    k=interp_k, power=interp_power)
        l_s = resample_if_needed(l_s, sr_sofa, target_sr)
        r_s = resample_if_needed(r_s, sr_sofa, target_sr)
        stereo_start = convolve_hrir(dry_start, l_s, r_s)
        # End variant (simulate filter at end; do NOT advance global state yet)
        gain_end = compute_distance_gain(dist_end, ref_dist, rolloff)
        dry_end, air_state_end = apply_air_absorption(block * gain_end, target_sr, dist_end,
                                                       air_state_initial, ref_dist, air_strength)
        l_e, r_e = interpolate_hrir(hrir_all, positions, az_end, el_end,
                                    k=interp_k, power=interp_power)
        l_e = resample_if_needed(l_e, sr_sofa, target_sr)
        r_e = resample_if_needed(r_e, sr_sofa, target_sr)
        stereo_end = convolve_hrir(dry_end, l_e, r_e)
        L = stereo_start.shape[0]
        ramp = np.linspace(0.0, 1.0, L, dtype=np.float32)
        stereo_morph = (stereo_start * (1.0 - ramp)[:, None] + stereo_end * ramp[:, None]).astype(np.float32)
        # NEW: boundary crossfade for morph case
        if boundary_xfade > 0 and start > 0:
            xfade = min(boundary_xfade, stereo_morph.shape[0])
            existing = dry_out[start:start + xfade]
            ramp2 = np.linspace(0.0, 1.0, xfade, dtype=np.float32)[:, None]
            dry_out[start:start + xfade] = existing * (1.0 - ramp2) + stereo_morph[:xfade] * ramp2
            if xfade < stereo_morph.shape[0]:
                dry_out[start + xfade:start + stereo_morph.shape[0]] += stereo_morph[xfade:]
        else:
            dry_out[start:start + L] += stereo_morph
        air_state = air_state_end  # advance

    # Trim dry with tail kept
    final_len = n_samples + hrir_len - 1
    dry_trim = dry_out[:final_len]

    # Build per-sample distance-based wet envelope (0..wet_max)
    times = np.arange(final_len, dtype=np.float32) / float(target_sr)
    # Vectorized distance curve (linear path clamped after duration)
    dur = movement.duration
    if not np.isfinite(dur) or dur <= 0:
        distances = np.full(final_len, movement.end_dist, dtype=np.float32)
    else:
        frac = np.clip(times / dur, 0.0, 1.0)
        distances = (movement.start_dist + frac * (movement.end_dist - movement.start_dist)).astype(np.float32)
    wet_env = np.clip((distances - ref_dist) / max(1e-6, (max_dist - ref_dist)), 0.0, 1.0) * float(wet_max)

    if apply_reverb and _HAS_PEDALBOARD and wet_max > 0:
        LOG.info("Applying single global reverb pass + distance envelope mix (max wet=%.2f)", wet_max)
        wet_full = apply_optional_reverb(dry_trim, target_sr)  # uses default modest settings
        # Mix with per-sample envelope
        dry_gain = (1.0 - wet_env)[:, None]
        wet_gain = wet_env[:, None]
        mixed = (dry_trim * dry_gain + wet_full * wet_gain).astype(np.float32)
    else:
        mixed = dry_trim

    final = normalize_audio(mixed)
    write_atomic(output_path, final, target_sr)
    LOG.info("Moving source spatialization complete (morph=%s, global reverb=%s, crackle-minimized)",
             filter_morph, apply_reverb)


def list_sofa_files(directory: Path):
    """
    List .sofa files in directory
    """
    if not directory.exists():
        raise FileNotFoundError(directory)
    files = sorted(directory.glob("*.sofa"))
    for f in files:
        print(f)


def build_cli():
    p = argparse.ArgumentParser(prog="spatialize",
                                description="Simple robust SOFA spatializer (static or moving source)")
    p.add_argument("-i", "--input", type=Path, required=False,
                   help="Input mono/stereo audio file (loaded as mono)")
    p.add_argument("-s", "--sofa", type=Path, required=False,
                   help="SOFA HRTF file to use")
    p.add_argument("-o", "--output", type=Path, default=Path("spatial_out.wav"),
                   help="Output WAV file")
    p.add_argument("-a", "--azimuth", type=float, default=0.0,
                   help="Static azimuth in degrees")
    p.add_argument("-e", "--elevation", type=float, default=0.0,
                   help="Static elevation in degrees")
    p.add_argument("-r", "--samplerate", type=int, default=DEFAULT_SR,
                   help="Target sample rate (default 44100)")
    p.add_argument("--reverb", action="store_true",
                   help="Apply a small stereo reverb (requires pedalboard)")
    p.add_argument("--list-sofa-dir", type=Path, default=None,
                   help="List .sofa files in given directory and exit")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    # Moving source arguments
    p.add_argument("--start-az", type=float, help="Start azimuth (deg) for moving source")
    p.add_argument("--start-el", type=float, default=0.0, help="Start elevation (deg)")
    p.add_argument("--end-az", type=float, help="End azimuth (deg) for moving source")
    p.add_argument("--end-el", type=float, default=0.0, help="End elevation (deg)")
    p.add_argument("--speed", type=float, help="Angular speed (deg/sec) for movement")
    p.add_argument("--block-size", type=int, default=2048, help="Block size for moving source processing")
    p.add_argument("--interp-k", type=int, default=4, help="k-NN for HRIR interpolation")
    p.add_argument("--interp-power", type=float, default=2.0, help="Inverse distance weight power")
    p.add_argument("--no-morph", action="store_true", help="Disable intra-block HRIR morph (crossfade)")
    p.add_argument("--start-dist", type=float, default=1.0, help="Start distance (meters, ref ~1m)")
    p.add_argument("--end-dist", type=float, default=1.0, help="End distance (meters)")
    p.add_argument("--ref-dist", type=float, default=1.0, help="Reference distance for unity gain")
    p.add_argument("--rolloff", type=float, default=1.0, help="Distance rolloff exponent (1=inverse, 2=inverse-square approx dB)")
    p.add_argument("--air-strength", type=float, default=0.35, help="Air absorption strength (0 disables)")
    p.add_argument("--wet-max", type=float, default=0.3, help="Max reverb wet level at max distance")
    p.add_argument("--max-dist", type=float, default=10.0, help="Distance for max reverb wet level")
    p.add_argument("--boundary-xfade", type=int, default=256, help="Samples of crossfade at block boundaries (reduce clicks)")
    return p


def main(argv=None):
    parser = build_cli()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if args.list_sofa_dir:
        try:
            list_sofa_files(args.list_sofa_dir)
        except Exception as exc:
            LOG.error("Error listing SOFA files: %s", exc)
        return

    if not args.input or not args.sofa:
        parser.print_help()
        LOG.error("Input audio and SOFA file are required (use -i and -s).")
        return

    # Detect moving source request
    moving = (args.start_az is not None and args.end_az is not None and args.speed is not None)

    try:
        if moving:
            LOG.info("=== Moving source mode ===")
            movement = SourcePath(args.start_az, args.start_el, args.end_az, args.end_el, args.speed,
                                   start_dist=args.start_dist, end_dist=args.end_dist)
            spatialize_moving_source(args.input, args.sofa, args.output,
                                      movement, args.samplerate, args.reverb,
                                      block_size=args.block_size,
                                      interp_k=args.interp_k,
                                      interp_power=args.interp_power,
                                      filter_morph=not args.no_morph,
                                      ref_dist=args.ref_dist,
                                      rolloff=args.rolloff,
                                      air_strength=args.air_strength,
                                      wet_max=args.wet_max,
                                      max_dist=args.max_dist,
                                      boundary_xfade=args.boundary_xfade)
        else:
            LOG.info("=== Static source mode ===")
            spatialize_file(args.input, args.sofa, args.output,
                            args.azimuth, args.elevation, args.samplerate,
                            args.reverb)
    except Exception as exc:
        LOG.exception("Processing failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()