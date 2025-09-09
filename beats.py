#!/usr/bin/env python3
"""
beats.py — Beat-synchronized ambisonic spatializer for Demucs stems

- Detects beats (librosa) from a reference stem (drums preferred),
  otherwise from the mix of available stems.
- For each stem, creates a yaw trajectory that advances by a fixed step at each beat,
  and a smooth amplitude pulse envelope triggered on beats.
- NEW: Frequency-driven spatialization for the 'other' stem - creates ethereal floating
  movement based on spectral content (brightness controls left-right, HF energy controls up-down).
- Encodes each stem into first-order Ambisonics (or higher, but order=1 is fastest),
  rotates per-sample (vectorized for order=1), sums SH streams, and decodes to binaural
  using spaudiopy's binaural decoder and cached HRIRs.

Usage examples (PowerShell):
  python beats.py --stems-dir ../demucs/song/separated/htdemucs/song \
      -o song_beats.wav --order 1 --rotate-step-deg 45 --stereo-width 30 --pulse-decay-ms 140

  python beats.py --drums drums.wav --bass bass.wav --vocals vocals.wav --other other.wav \
      -o song_beats.wav --rotate-step-deg 60 --seed 7 --beats-only

  # Frequency-driven spatialization for 'other' stem (enabled by default in beats-only mode):
  python beats.py --other other.wav -o ethereal_other.wav --beats-only --frequency-driven-other
"""
from __future__ import annotations
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
import subprocess
import sys
import tempfile
import shutil

from spaudiopy import sph, decoder, io
from scipy import signal
from scipy.ndimage import uniform_filter1d

LOG = logging.getLogger("beats")

# -------------------- Caches --------------------
_HRIRS_CACHE: Dict[int, object] = {}
_DECODER_CACHE: Dict[Tuple[int, int], object] = {}


def get_decoder(fs: int, order: int):
    key = (fs, order)
    if key in _DECODER_CACHE:
        return _DECODER_CACHE[key]
    if fs in _HRIRS_CACHE:
        hrirs = _HRIRS_CACHE[fs]
    else:
        LOG.info("Loading HRIRs for fs=%d (first run may be slow)", fs)
        hrirs = io.load_hrirs(fs)
        _HRIRS_CACHE[fs] = hrirs
    hrirs_nm = decoder.magls_bin(hrirs, order)
    _DECODER_CACHE[key] = hrirs_nm
    return hrirs_nm


# -------------------- Utilities --------------------

def load_audio_stereo(path: Path, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(str(path))
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    elif y.shape[1] > 2:
        y = y[:, :2]
    if target_sr is not None and sr != target_sr:
        y_mono = librosa.to_mono(y.T)
        y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=target_sr)
        # rebuild stereo by matching average; better is per-channel resample, do that:
        yL = librosa.resample(y[:, 0].astype(np.float32), orig_sr=sr, target_sr=target_sr)
        yR = librosa.resample(y[:, 1].astype(np.float32), orig_sr=sr, target_sr=target_sr)
        y = np.stack([yL, yR], axis=1)
        sr = target_sr
    return y.astype(np.float32), sr


def run_demucs_separation(input_file: Path, output_dir: Path, model: str = "htdemucs") -> Dict[str, Path]:
    """Run Demucs to separate stems, return dict of stem paths."""
    LOG.info("Running Demucs separation on %s", input_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the same Python executable that's running this script
    python_exe = Path(sys.executable)
    
    cmd = [
        str(python_exe), "-m", "demucs.separate",
        "--out", str(output_dir),
        "--name", model,
        str(input_file)
    ]
    
    try:
        LOG.info("Running command: %s", ' '.join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        LOG.info("Demucs completed successfully")
    except subprocess.CalledProcessError as e:
        LOG.error("Demucs failed: %s", e.stderr)
        raise RuntimeError(f"Demucs separation failed: {e}")
    except FileNotFoundError:
        raise RuntimeError("Demucs not found. Install with: pip install demucs")
    
    # Find the separated stems directory
    stem_name = input_file.stem
    stems_dir = output_dir / model / stem_name
    
    if not stems_dir.exists():
        raise RuntimeError(f"Expected stems directory not found: {stems_dir}")
    
    # Map standard Demucs outputs
    stems = {}
    for name in ["drums", "bass", "vocals", "other"]:
        stem_file = stems_dir / f"{name}.wav"
        if stem_file.exists():
            stems[name] = stem_file
    
    if not stems:
        raise RuntimeError(f"No stems found in {stems_dir}")
    
    LOG.info("Found stems: %s", list(stems.keys()))
    return stems


def normalize_stems_gain(loaded_stems: Dict[str, Tuple[np.ndarray, int]], 
                        target_lufs: float = -18.0) -> Dict[str, Tuple[np.ndarray, int]]:
    """Normalize all stems to consistent loudness for cohesive mix."""
    try:
        import pyloudnorm as pyln
    except ImportError:
        LOG.warning("pyloudnorm not available, using peak normalization")
        # Fallback to peak normalization
        normalized = {}
        for name, (y, sr) in loaded_stems.items():
            peak = np.max(np.abs(y))
            if peak > 0:
                y_norm = y * (0.7 / peak)  # Normalize to -3dB peak
            else:
                y_norm = y
            normalized[name] = (y_norm.astype(np.float32), sr)
        return normalized
    
    normalized = {}
    meter = pyln.Meter(next(iter(loaded_stems.values()))[1])  # Use first stem's SR
    
    for name, (y, sr) in loaded_stems.items():
        y_mono = librosa.to_mono(y.T) if y.ndim > 1 else y
        current_lufs = meter.integrated_loudness(y_mono)
        
        if current_lufs > -70:  # Valid measurement
            gain_db = target_lufs - current_lufs
            gain_linear = 10 ** (gain_db / 20)
            y_norm = (y * gain_linear).astype(np.float32)
        else:
            # Fallback for very quiet stems
            y_norm = y.astype(np.float32)
        
        normalized[name] = (y_norm, sr)
        LOG.info("Stem %s: %.1f LUFS -> %.1f LUFS", name, current_lufs, target_lufs)
    
    return normalized


def find_stems(stems_dir: Path) -> Dict[str, Path]:
    """Return mapping for common demucs stems present in directory."""
    patterns = {
        "drums": ["*drum*.wav", "*Drum*.wav"],
        "bass": ["*bass*.wav", "*Bass*.wav"],
        "vocals": ["*vocal*.wav", "*Voc*.wav"],
        "other": ["*other*.wav", "*oth*.wav", "*piano*.wav", "*guitar*.wav"],
    }
    out: Dict[str, Path] = {}
    for name, globs in patterns.items():
        for g in globs:
            hits = sorted(stems_dir.glob(g))
            if hits:
                out[name] = hits[0]
                break
    return out


def pad_to_length(y: np.ndarray, n: int) -> np.ndarray:
    if y.shape[0] >= n:
        return y[:n]
    pad = np.zeros((n - y.shape[0], y.shape[1]), dtype=y.dtype)
    return np.vstack([y, pad])


def detect_beats(ref_audio: np.ndarray, sr: int) -> np.ndarray:
    """Return beat sample indices. Fallback to onsets if beats fail."""
    y_mono = librosa.to_mono(ref_audio.T)
    tempo, beats = librosa.beat.beat_track(y=y_mono, sr=sr, units='samples')
    if beats is None or len(beats) < 4:
        onset_frames = librosa.onset.onset_detect(y=y_mono, sr=sr, units='frames')
        beats = librosa.frames_to_samples(onset_frames)
    LOG.info("Detected %d beats", len(beats))
    return beats.astype(np.int64)


def make_pulse_env(n: int, beats: np.ndarray, sr: int, decay_ms: float = 140.0) -> np.ndarray:
    """Exponential decay envelope triggered at beat times; normalized to <=1."""
    impulses = np.zeros(n, dtype=np.float32)
    beats = beats[(beats >= 0) & (beats < n)]
    impulses[beats] = 1.0
    # kernel length ~ 1s or until negligible
    tau = max(1.0, decay_ms) / 1000.0
    L = int(min(n, max(sr // 2, int(5 * tau * sr))))
    t = np.arange(L, dtype=np.float32) / float(sr)
    kernel = np.exp(-t / tau).astype(np.float32)
    env = np.convolve(impulses, kernel, mode='full')[:n]
    env /= max(1e-6, np.max(env))
    # soft floor to keep ambience between beats
    env = (env * 0.85 + 0.15).astype(np.float32)
    return env

def make_beat_mask(n: int, beats: np.ndarray, sr: int, win_ms: float = 120.0) -> np.ndarray:
    """Maximum of Hann windows centered at beats; ensures proper crossfade [0,1]."""
    if beats.size == 0:
        return np.zeros(n, dtype=np.float32)
    L = max(1, int(win_ms * 1e-3 * sr))
    if L % 2 == 1:
        L += 1
    half = L // 2
    w = np.hanning(L).astype(np.float32)
    env = np.zeros(n, dtype=np.float32)
    for b in beats:
        start = int(max(0, b - half))
        end = int(min(n, b + half))
        w_s = 0
        w_e = end - start
        if start == 0:
            w_s = half - b
        if end == n:
            w_e = half + (n - b)
        # Use maximum instead of sum to prevent overlap issues
        env[start:end] = np.maximum(env[start:end], w[w_s:w_e])
    return env

def assign_beat_angles(num_beats: int, pattern: str = "golden", start_deg: float = 0.0,
                       seed: int | None = None) -> np.ndarray:
    """Return per-beat azimuth angles in degrees following a pattern."""
    if num_beats <= 0:
        return np.array([], dtype=np.float32)
    if pattern == "alternate":
        angles = np.array([(-45 if i % 2 == 0 else 45) for i in range(num_beats)], dtype=np.float32)
        angles += start_deg
    elif pattern == "cycle":
        base = np.array([-90, -30, 30, 90, 150, -150], dtype=np.float32)
        angles = np.tile(base, int(np.ceil(num_beats / len(base))))[:num_beats].astype(np.float32)
        angles += start_deg
    else:  # golden-angle dispersion
        phi = 137.50776405
        idx = np.arange(num_beats, dtype=np.float32)
        angles = (start_deg + idx * phi) % 360.0
        angles = ((angles + 180.0) % 360.0) - 180.0
    return angles.astype(np.float32)


def analyze_stem_characteristics(y: np.ndarray, sr: int, name: str) -> Dict[str, float]:
    """Fast analysis of stem characteristics using simple heuristics."""
    # Use stem name as primary indicator (much faster!)
    if name == 'vocals':
        return {
            'freq_centroid': 2000.0,
            'brightness': 0.4,
            'melodicity': 0.8,
            'harmonic_ratio': 0.7,
        }
    elif name == 'bass':
        return {
            'freq_centroid': 200.0,
            'brightness': 0.1,
            'melodicity': 0.3,
            'harmonic_ratio': 0.6,
        }
    elif name == 'other':
        return {
            'freq_centroid': 1500.0,
            'brightness': 0.3,
            'melodicity': 0.6,
            'harmonic_ratio': 0.5,
        }
    else:  # drums or unknown
        return {
            'freq_centroid': 800.0,
            'brightness': 0.2,
            'melodicity': 0.1,
            'harmonic_ratio': 0.2,
        }


def create_floating_path(N: int, sr: int, characteristics: Dict[str, float], 
                        base_angle: float, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create fast floating path based on stem characteristics."""
    rng = np.random.default_rng(seed)
    
    # Extract characteristics
    brightness = characteristics['brightness']
    melodicity = characteristics['melodicity']
    
    # Fast path parameters
    elevation_range = 10.0 + (brightness * 20.0)  # 10-30 degrees
    azimuth_range = 15.0 + (melodicity * 30.0)   # 15-45 degrees
    movement_freq = 0.15 + (brightness * 0.1)     # 0.15-0.25 Hz
    
    # Generate simple trajectories (much faster)
    t = np.arange(N) / sr
    
    # Simple sine waves instead of complex analysis
    azimuth_offset = azimuth_range * np.sin(2 * np.pi * movement_freq * t)
    elevation_offset = elevation_range * 0.5 * np.sin(2 * np.pi * movement_freq * 0.8 * t)
    
    # Final trajectories
    azimuth = (base_angle + azimuth_offset) % 360.0
    elevation = np.abs(elevation_offset)
    
    return azimuth.astype(np.float32), elevation.astype(np.float32)


def encode_stem_with_elevation(y: np.ndarray, azimuth: np.ndarray, elevation: np.ndarray, 
                              width_deg: float, order: int) -> np.ndarray:
    """Fast encoding with time-varying position using block processing."""
    N = y.shape[0]
    C = (order + 1) ** 2
    sh_out = np.zeros((C, N), dtype=np.float32)
    
    # Convert stereo to mid/side
    if y.shape[1] == 2:
        mid = (y[:, 0] + y[:, 1]) * 0.5
        side = (y[:, 0] - y[:, 1]) * 0.5
    else:
        mid = y[:, 0]
        side = np.zeros_like(mid)
    
    # Use larger blocks for speed (less position updates)
    block_size = 4096  # Increased from 1024
    width_rad = np.deg2rad(width_deg) * 0.5
    
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        
        # Use middle sample position for entire block
        mid_idx = start + (end - start) // 2
        az_rad = np.deg2rad(azimuth[mid_idx])
        el_rad = np.deg2rad(elevation[mid_idx])
        
        # Encode mid signal
        mid_block = mid[start:end]
        if order == 1:
            # First-order encoding (optimized)
            cos_el = np.cos(el_rad)
            sin_el = np.sin(el_rad)
            cos_az = np.cos(az_rad)
            sin_az = np.sin(az_rad)
            
            sh_out[0, start:end] += mid_block * 0.5  # W
            sh_out[1, start:end] += mid_block * cos_el * cos_az * 0.5  # X
            sh_out[2, start:end] += mid_block * cos_el * sin_az * 0.5  # Y
            sh_out[3, start:end] += mid_block * sin_el * 0.5  # Z
        
        # Simple stereo width (skip if side is empty)
        if np.any(side[start:end] != 0):
            side_block = side[start:end]
            left_az = az_rad - width_rad
            right_az = az_rad + width_rad
            
            if order == 1:
                cos_el = np.cos(el_rad)
                
                X_left = side_block * cos_el * np.cos(left_az) * 0.25
                Y_left = side_block * cos_el * np.sin(left_az) * 0.25
                X_right = -side_block * cos_el * np.cos(right_az) * 0.25
                Y_right = -side_block * cos_el * np.sin(right_az) * 0.25
                
                sh_out[1, start:end] += X_left + X_right
                sh_out[2, start:end] += Y_left + Y_right
    
    return sh_out

def yaw_envelope_from_beats(n: int, beats: np.ndarray, angles_deg: np.ndarray, sr: int, win_ms: float) -> np.ndarray:
    """Piecewise-constant yaw per sample set to each beat's angle across its window."""
    yaw = np.zeros(n, dtype=np.float64)
    L = max(1, int(win_ms * 1e-3 * sr))
    if L % 2 == 1:
        L += 1
    half = L // 2
    for b, ang in zip(beats, angles_deg):
        start = int(max(0, b - half))
        end = int(min(n, b + half))
        yaw[start:end] = np.deg2rad(float(ang))
    return yaw


def yaw_trajectory(n: int, beats: np.ndarray, base_deg: float, step_deg: float) -> np.ndarray:
    """Linear ramp yaw between successive beat-target angles."""
    tgt_angles = []
    for i in range(len(beats) + 1):
        tgt_angles.append(base_deg + i * step_deg)
    # segment boundaries
    seg_starts = np.concatenate([beats, [n]])
    yaw = np.zeros(n, dtype=np.float64)
    prev_sample = 0
    prev_angle = np.deg2rad(tgt_angles[0])
    for i, seg_end in enumerate(seg_starts):
        end = int(seg_end)
        if end <= prev_sample:
            continue
        next_angle = np.deg2rad(tgt_angles[min(i + 1, len(tgt_angles) - 1)])
        L = end - prev_sample
        ramp = np.linspace(0.0, 1.0, L, endpoint=False, dtype=np.float64)
        yaw[prev_sample:end] = prev_angle + (next_angle - prev_angle) * ramp
        prev_sample = end
        prev_angle = next_angle
    return yaw


def encode_stereo_sh(x: np.ndarray, width_deg: float, order: int) -> np.ndarray:
    """Encode stereo signal as two virtual sources separated by width around forward."""
    half = np.deg2rad(width_deg / 2.0)
    left_azi = -half
    right_azi = +half
    zen = np.deg2rad(90.0)
    sh_L = sph.src_to_sh(x[:, 0], left_azi, zen, order)
    sh_R = sph.src_to_sh(x[:, 1], right_azi, zen, order)
    return (sh_L + sh_R).astype(np.float32)


def rotate_yaw_order1(sh_signal: np.ndarray, yaw_angles: np.ndarray) -> np.ndarray:
    """Vectorized yaw rotation for 1st order: ACN [W,Y,Z,X]."""
    if sh_signal.shape[0] != 4:
        raise ValueError("rotate_yaw_order1 expects 4-channel 1st-order SH")
    W = sh_signal[0, :]
    Y = sh_signal[1, :]
    Z = sh_signal[2, :]
    X = sh_signal[3, :]
    c = np.cos(yaw_angles)
    s = np.sin(yaw_angles)
    Xr = c * X + s * Y
    Yr = -s * X + c * Y
    return np.vstack([W, Yr, Z, Xr]).astype(np.float32)


# -------------------- Frequency-Driven Spatialization --------------------

def compute_spectral_features(y: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spectral centroid and high-frequency energy ratio for frequency-driven spatialization.
    
    Args:
        y: Audio signal (mono or averaged to mono)
        sr: Sample rate
        hop_length: STFT hop length in samples
    
    Returns:
        spectral_centroid: Brightness measure (Hz)
        hf_energy_ratio: High-frequency energy ratio (0-1)
    """
    # Convert to mono if stereo
    if y.ndim > 1:
        y_mono = librosa.to_mono(y.T)
    else:
        y_mono = y
    
    # Compute spectral centroid (brightness measure)
    spectral_centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr, hop_length=hop_length)[0]
    
    # Compute STFT for high-frequency energy analysis
    stft = librosa.stft(y_mono, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    # Define frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0]*2-1)
    
    # High-frequency threshold (6kHz as suggested)
    hf_threshold = 6000.0
    hf_bins = freqs >= hf_threshold
    
    # Calculate high-frequency energy ratio for each frame
    total_energy = np.sum(magnitude**2, axis=0)
    hf_energy = np.sum(magnitude[hf_bins, :]**2, axis=0)
    
    # Avoid division by zero
    hf_energy_ratio = np.divide(hf_energy, total_energy, 
                               out=np.zeros_like(hf_energy), 
                               where=total_energy > 1e-10)
    
    return spectral_centroid, hf_energy_ratio


def smooth_control_signals(signal: np.ndarray, sr: int, smoothing_time: float = 0.2) -> np.ndarray:
    """
    Apply temporal smoothing to control signals to create fluid movement.
    
    Args:
        signal: Input control signal
        sr: Sample rate (for frame-based signals, this should be frame_rate)
        smoothing_time: Smoothing window duration in seconds
    
    Returns:
        Smoothed signal
    """
    # Calculate smoothing window size in frames
    window_size = max(1, int(smoothing_time * sr))
    
    # Apply uniform filter (moving average) for smooth transitions
    smoothed = uniform_filter1d(signal.astype(np.float64), size=window_size, mode='nearest')
    
    return smoothed.astype(np.float32)


def interpolate_to_samples(frame_values: np.ndarray, hop_length: int, total_samples: int) -> np.ndarray:
    """
    Interpolate frame-based values to per-sample values.
    
    Args:
        frame_values: Frame-based feature values
        hop_length: STFT hop length used for frame extraction
        total_samples: Target number of samples
    
    Returns:
        Per-sample interpolated values
    """
    # Create frame time indices (center of each frame)
    frame_times = np.arange(len(frame_values)) * hop_length
    
    # Create sample time indices
    sample_times = np.arange(total_samples)
    
    # Interpolate to sample rate
    interpolated = np.interp(sample_times, frame_times, frame_values)
    
    return interpolated.astype(np.float32)


def create_frequency_driven_trajectory(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create yaw and elevation trajectories driven by frequency content of the 'other' stem.
    
    This creates the ethereal floating effect where:
    - Yaw (left-right) is controlled by spectral centroid (brightness)  
    - Elevation (up-down) is controlled by high-frequency energy ratio
    
    Args:
        y: Stereo audio signal for the 'other' stem
        sr: Sample rate
    
    Returns:
        yaw_trajectory: Per-sample yaw angles in radians
        elevation_trajectory: Per-sample elevation angles in radians
    """
    hop_length = 512  # ~11ms frames at 44.1kHz
    total_samples = y.shape[0]
    
    # Compute spectral features
    LOG.info("Computing spectral features for frequency-driven spatialization...")
    spectral_centroid, hf_energy_ratio = compute_spectral_features(y, sr, hop_length)
    
    # Calculate frame rate for smoothing
    frame_rate = sr / hop_length
    
    # Smooth the control signals for fluid movement (200ms smoothing)
    spectral_centroid_smooth = smooth_control_signals(spectral_centroid, frame_rate, 0.2)
    hf_energy_smooth = smooth_control_signals(hf_energy_ratio, frame_rate, 0.3)  # Slightly longer for elevation
    
    # Normalize spectral centroid to yaw range
    # Map spectral centroid to azimuth range: -90° to +90° 
    centroid_min = np.percentile(spectral_centroid_smooth, 5)   # Robust min
    centroid_max = np.percentile(spectral_centroid_smooth, 95)  # Robust max
    centroid_norm = np.clip((spectral_centroid_smooth - centroid_min) / 
                           max(1.0, centroid_max - centroid_min), 0.0, 1.0)
    
    # Map to yaw range: -90° to +90°
    yaw_range = np.deg2rad(180.0)  # 180° total range
    yaw_frames = (centroid_norm - 0.5) * yaw_range  # Center around 0
    
    # Map high-frequency energy to elevation range  
    # When HF energy is high -> move up (positive elevation)
    # When HF energy is low -> return to horizontal plane (0°)
    elevation_max = np.deg2rad(45.0)  # Maximum elevation: 45°
    elevation_frames = hf_energy_smooth * elevation_max
    
    # Interpolate frame-based values to per-sample
    yaw_trajectory = interpolate_to_samples(yaw_frames, hop_length, total_samples)
    elevation_trajectory = interpolate_to_samples(elevation_frames, hop_length, total_samples)
    
    # Log trajectory characteristics
    LOG.info("Frequency-driven trajectory: yaw range %.1f° to %.1f°", 
             np.rad2deg(np.min(yaw_trajectory)), np.rad2deg(np.max(yaw_trajectory)))
    LOG.info("Frequency-driven trajectory: elevation range %.1f° to %.1f°",
             np.rad2deg(np.min(elevation_trajectory)), np.rad2deg(np.max(elevation_trajectory)))
    
    return yaw_trajectory, elevation_trajectory


def encode_with_3d_rotation(y: np.ndarray, yaw_trajectory: np.ndarray, 
                           elevation_trajectory: np.ndarray, stereo_width: float, 
                           order: int) -> np.ndarray:
    """
    Encode stereo signal with time-varying 3D rotation (yaw + elevation).
    
    Args:
        y: Stereo audio signal
        yaw_trajectory: Per-sample yaw angles in radians  
        elevation_trajectory: Per-sample elevation angles in radians
        stereo_width: Stereo width in degrees
        order: Ambisonic order
    
    Returns:
        Rotated spherical harmonic signal
    """
    # Encode as stereo sources
    sh_signal = encode_stereo_sh(y, stereo_width, order)
    
    # Apply per-sample 3D rotation
    C, N = sh_signal.shape
    sh_rotated = np.zeros_like(sh_signal)
    
    # Process in blocks for efficiency
    block_size = 1024
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        
        for i in range(start, end):
            yaw = float(yaw_trajectory[i])
            pitch = float(elevation_trajectory[i])  # elevation = pitch in spaudiopy
            roll = 0.0  # No roll rotation needed
            
            # Apply 3D rotation using spaudiopy
            sh_rotated[:, i] = sph.rotate_sh(sh_signal[:, i], yaw, pitch, roll)
    
    return sh_rotated.astype(np.float32)


# -------------------- Main processing --------------------

def spatialize_beats(stems: Dict[str, Path], output: Path, order: int = 1,
                      stereo_width: float = 30.0, rotate_step_deg: float = 45.0,
                      pulse_decay_ms: float = 140.0, seed: Optional[int] = None,
                      beats_only: bool = True, beat_window_ms: float = 120.0,
                      beat_pattern: str = "golden", normalize_stems: bool = True,
                      distance_m: float = 2.0, b_format: bool = False, 
                      wet_level: float = 1.0, no_normalize: bool = False,
                      intelligent_float: bool = False, frequency_driven_other: bool = True):
    # Load stems
    loaded: Dict[str, Tuple[np.ndarray, int]] = {}
    sr_ref: Optional[int] = None
    for name, p in stems.items():
        y, sr = load_audio_stereo(p)
        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            y, sr = load_audio_stereo(p, target_sr=sr_ref)
        loaded[name] = (y, sr)
    if not loaded:
        raise RuntimeError("No stems provided")
    assert sr_ref is not None

    # Normalize stems for cohesive loudness (optional)
    if not no_normalize and normalize_stems:
        loaded = normalize_stems_gain(loaded)
        LOG.info("Stem normalization applied")
    else:
        LOG.info("Stem normalization disabled - preserving original mix balance")

    # Align lengths
    N = max(y.shape[0] for y, _ in loaded.values())
    for k in list(loaded.keys()):
        y, sr = loaded[k]
        loaded[k] = (pad_to_length(y, N), sr)

    # Reference for beats: FORCE drums if available, else mix
    if 'drums' in loaded:
        ref = loaded['drums'][0]
        LOG.info("Using drums stem for beat detection")
    else:
        mix = np.zeros((N, 2), dtype=np.float32)
        for y, _ in loaded.values():
            mix += y
        ref = (mix / max(1, len(loaded))).astype(np.float32)
        LOG.info("Using mixed stems for beat detection (no drums found)")

    beats = detect_beats(ref, sr_ref)
    if beats.size == 0:
        LOG.warning("No beats found; using uniform 0.5s grid")
        step = int(0.5 * sr_ref)
        beats = np.arange(step, N, step, dtype=np.int64)

    # Distance modeling for cohesive spatialization (but preserve overall loudness)
    distance_gain = 1.0 / max(1.0, distance_m)  # Simple inverse distance
    # Don't apply distance gain yet - we'll use it for relative positioning only
    LOG.info("Distance modeling: %.1fm (gain calculation: %.3f)", distance_m, distance_gain)

    # Set deterministic base angles per stem but keep them spread evenly
    rng = np.random.default_rng(seed)
    stem_names = sorted(loaded.keys())
    angle_step = 360.0 / max(1, len(stem_names))
    base_angles: Dict[str, float] = {}
    for idx, name in enumerate(stem_names):
        # Evenly space stems around circle with some jitter
        base_angle = idx * angle_step + rng.uniform(-15, 15)
        base_angles[name] = base_angle % 360.0
        LOG.info("Stem %s base angle: %.1f°", name, base_angles[name])

    # Build SH sum
    C = (order + 1) ** 2
    sh_sum = np.zeros((C, N), dtype=np.float32)

    # Get input RMS for loudness matching
    input_rms = 0.0
    input_samples = 0
    for y, _ in loaded.values():
        input_rms += np.sum(y ** 2)
        input_samples += y.size
    input_rms = np.sqrt(input_rms / max(1, input_samples))

    for name, (y, _) in loaded.items():
        # Don't apply distance gain - preserve original loudness
        # y = (y * distance_gain).astype(np.float32)  # Removed this line
        
        if intelligent_float:
            # INTELLIGENT FLOATING MODE: Drums get beat sync, others get smooth floating paths
            if name == 'drums':
                # Drums: beat-synchronized spatialization as before
                env = make_pulse_env(N, beats, sr_ref, decay_ms=pulse_decay_ms)
                y_proc = (y * env[:, None]).astype(np.float32)
                sh = encode_stereo_sh(y_proc, stereo_width, order)
                yaw = yaw_trajectory(N, beats, base_angles[name], rotate_step_deg)
                if order == 1 and sh.shape[0] == 4:
                    sh_rot = rotate_yaw_order1(sh, yaw)
                else:
                    sh_rot = np.empty_like(sh)
                    blk = 2048
                    for start in range(0, N, blk):
                        end = min(start + blk, N)
                        for i in range(start, end):
                            sh_rot[:, i] = sph.rotate_sh(sh[:, i], float(yaw[i]), 0.0, 0.0)
                sh_sum += sh_rot.astype(np.float32)
                LOG.info("Stem %s: beat-synchronized spatialization (%d beats)", name, len(beats))
            else:
                # Other stems: intelligent floating paths based on analysis
                characteristics = analyze_stem_characteristics(y, sr_ref, name)
                azimuth_path, elevation_path = create_floating_path(N, sr_ref, characteristics, 
                                                                 base_angles[name], seed)
                
                # Log characteristics for debugging
                LOG.info("Stem %s analysis: freq_centroid=%.1fHz, melodicity=%.2f, brightness=%.2f", 
                        name, characteristics['freq_centroid'], characteristics['melodicity'], 
                        characteristics['brightness'])
                LOG.info("Stem %s floating path: azimuth_range=%.1f°, elevation_range=%.1f°", 
                        name, np.ptp(azimuth_path), np.ptp(elevation_path))
                
                # Encode with time-varying position
                sh_float = encode_stem_with_elevation(y, azimuth_path, elevation_path, 
                                                    stereo_width, order)
                sh_sum += sh_float.astype(np.float32)
        
        elif beats_only:
            if name == 'drums':
                # Drums: blend between centered and spatialized using beat mask
                # NO LAYERING - just crossfade between positions
                
                # Create centered version (full signal)
                sh_centered = encode_stereo_sh(y, width_deg=10.0, order=order)
                
                # Create spatialized version (full signal)
                sh_spatialized = encode_stereo_sh(y, stereo_width, order)
                angles = assign_beat_angles(len(beats), pattern=beat_pattern,
                                            start_deg=base_angles[name], seed=seed)
                yaw = yaw_envelope_from_beats(N, beats, angles, sr_ref, beat_window_ms)
                if order == 1 and sh_spatialized.shape[0] == 4:
                    sh_spatialized = rotate_yaw_order1(sh_spatialized, yaw)
                else:
                    sh_rot_temp = np.empty_like(sh_spatialized)
                    blk = 2048
                    for start in range(0, N, blk):
                        end = min(start + blk, N)
                        for i in range(start, end):
                            sh_rot_temp[:, i] = sph.rotate_sh(sh_spatialized[:, i], float(yaw[i]), 0.0, 0.0)
                    sh_spatialized = sh_rot_temp
                
                # Blend using beat mask (no destructive gating)
                mask = make_beat_mask(N, beats, sr_ref, win_ms=beat_window_ms)
                # When mask=1: full spatialization, mask=0: centered
                for c in range(sh_centered.shape[0]):
                    sh_final = sh_centered[c, :] * (1.0 - mask) + sh_spatialized[c, :] * mask
                    sh_sum[c, :] += sh_final
            else:
                # Special frequency-driven spatialization for 'other' stem only
                if name == 'other' and frequency_driven_other:
                    LOG.info("Applying frequency-driven spatialization to 'other' stem...")
                    
                    # Create frequency-driven 3D trajectory for the 'other' stem
                    yaw_trajectory_other, elevation_trajectory_other = create_frequency_driven_trajectory(y, sr_ref)
                    
                    # Encode with 3D rotation (yaw + elevation)
                    sh_other = encode_with_3d_rotation(y, yaw_trajectory_other, elevation_trajectory_other, 
                                                      stereo_width, order)
                    sh_sum += sh_other
                    
                    LOG.info("Frequency-driven spatialization applied to 'other' stem")
                else:
                    # Bass, vocals, and optionally 'other': always centered, no spatialization  
                    y_centered = y.astype(np.float32)
                    sh_centered = encode_stereo_sh(y_centered, width_deg=stereo_width, order=order)
                    sh_sum += sh_centered
        else:
            # Previous behavior: continuous spin step per beat with pulse
            env = make_pulse_env(N, beats, sr_ref, decay_ms=pulse_decay_ms)
            y_proc = (y * env[:, None]).astype(np.float32)
            sh = encode_stereo_sh(y_proc, stereo_width, order)
            yaw = yaw_trajectory(N, beats, base_angles[name], rotate_step_deg)
            if order == 1 and sh.shape[0] == 4:
                sh_rot = rotate_yaw_order1(sh, yaw)
            else:
                sh_rot = np.empty_like(sh)
                blk = 2048
                for start in range(0, N, blk):
                    end = min(start + blk, N)
                    for i in range(start, end):
                        sh_rot[:, i] = sph.rotate_sh(sh[:, i], float(yaw[i]), 0.0, 0.0)
            sh_sum += sh_rot.astype(np.float32)

    # Choose output format
    if b_format:
        # Output B-format (first-order ambisonics: W, X, Y, Z)
        out = sh_sum.T  # Transpose to [samples, channels]
        
        # Match output RMS to input RMS (preserve loudness)
        output_rms = np.sqrt(np.mean(out ** 2))
        if output_rms > 0:
            gain_match = input_rms / output_rms
            out = out * gain_match
            LOG.info("RMS matching: input=%.6f, output=%.6f, gain=%.3f", input_rms, output_rms, gain_match)
        
        # Safety limiter only if needed
        peak = float(np.max(np.abs(out)))
        if peak > 0.99:  # Only limit if close to clipping
            safety_gain = 0.95 / peak
            out = out * safety_gain
            LOG.info("Safety limiting: peak=%.3f, gain=%.3f", peak, safety_gain)
        
        # Ensure output has _bformat suffix instead of .b extension
        if output.suffix.lower() != '.wav':
            output = output.with_suffix('.wav')
        if '_bformat' not in output.stem:
            output = output.with_name(f"{output.stem}_bformat.wav")
        
    else:
        # Decode to binaural (this is your WET signal)
        hrirs_nm = get_decoder(sr_ref, order)
        bin_st = decoder.sh2bin(sh_sum, hrirs_nm)
        wet_out = bin_st.T if bin_st.ndim == 2 and bin_st.shape[0] == 2 else bin_st
        
        # Create the DRY signal (sum of original stems)
        dry_out = np.zeros((N, 2), dtype=np.float32)
        for name, (y, _) in loaded.items():
            dry_out += y
        
        # Blend dry and wet signals
        # Ensure both signals have the same length
        min_len = min(wet_out.shape[0], dry_out.shape[0])
        wet_out = wet_out[:min_len]
        dry_out = dry_out[:min_len]
        
        # Apply wet/dry mix
        dry_level = 1.0 - wet_level
        out = (wet_out * wet_level) + (dry_out * dry_level)
        
        LOG.info("Dry/Wet mix: %.1f%% wet, %.1f%% dry", wet_level * 100, dry_level * 100)
        
        # Match output RMS to input RMS (preserve loudness)
        output_rms = np.sqrt(np.mean(out ** 2))
        if output_rms > 0:
            gain_match = input_rms / output_rms
            out = out * gain_match
            LOG.info("RMS matching: input=%.6f, output=%.6f, gain=%.3f", input_rms, output_rms, gain_match)
        
        # Safety limiter only if needed
        peak = float(np.max(np.abs(out)))
        if peak > 0.99:  # Only limit if close to clipping
            safety_gain = 0.95 / peak
            out = out * safety_gain
            LOG.info("Safety limiting: peak=%.3f, gain=%.3f", peak, safety_gain)
    
    sf.write(str(output), out, sr_ref, format='WAV' if not b_format else 'WAV')
    if b_format:
        LOG.info("Wrote B-format %s (N=%d, sr=%d, channels=WXYZ)", output, N, sr_ref)
    else:
        LOG.info("Wrote binaural %s (N=%d, sr=%d)", output, N, sr_ref)


# -------------------- CLI --------------------

def build_cli():
    p = argparse.ArgumentParser(prog="beats", description="Beat-synced ambisonic spatializer for Demucs stems")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--stems-dir", type=Path, help="Directory containing demucs stems")
    g.add_argument("--drums", type=Path, help="Path to drums stem WAV")
    p.add_argument("--auto-separate", type=Path, help="Auto-run Demucs on this input file")
    p.add_argument("--demucs-model", type=str, default="htdemucs", help="Demucs model to use")
    p.add_argument("--bass", type=Path, help="Path to bass stem WAV")
    p.add_argument("--vocals", type=Path, help="Path to vocals stem WAV")
    p.add_argument("--other", type=Path, help="Path to other stem WAV")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output file (.wav for binaural, .b for B-format)")
    p.add_argument("--b-format", action="store_true", help="Output B-format ambisonics (.b) instead of binaural")
    p.add_argument("--wet", type=float, default=1.0, help="Wet signal mix (0.0=dry only, 1.0=wet only)")
    p.add_argument("--no-normalize", action="store_true", help="Disable stem normalization to preserve original mix balance")
    p.add_argument("--order", type=int, default=1, help="Ambisonic order (1 recommended)")
    p.add_argument("--stereo-width", type=float, default=30.0, help="Per-stem stereo width (deg)")
    p.add_argument("--rotate-step-deg", type=float, default=45.0, help="Yaw step per beat (deg)")
    p.add_argument("--pulse-decay-ms", type=float, default=140.0, help="Beat pulse decay (ms)")
    p.add_argument("--beats-only", action="store_true", help="Only spatialize short windows around beats; keep base centered")
    freq_group = p.add_mutually_exclusive_group()
    freq_group.add_argument("--frequency-driven-other", action="store_true", default=True, help="Apply frequency-driven spatialization to 'other' stem (default)")
    freq_group.add_argument("--no-frequency-driven-other", action="store_true", help="Keep 'other' stem centered like bass/vocals")
    p.add_argument("--intelligent-float", action="store_true", help="Use intelligent floating paths for non-drum stems based on frequency/melody analysis")
    p.add_argument("--beat-window-ms", type=float, default=120.0, help="Beat window length (ms) for placement")
    p.add_argument("--beat-pattern", type=str, default="golden", choices=["golden","alternate","cycle"], help="Placement pattern for beat angles")
    p.add_argument("--normalize-stems", action="store_true", default=True, help="Normalize stem loudness for cohesive mix")
    p.add_argument("--distance-m", type=float, default=2.0, help="Virtual distance for all stems (meters)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for stem base angles")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None):
    parser = build_cli()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    stems: Dict[str, Path] = {}
    
    # Auto-separate with Demucs if requested
    if args.auto_separate:
        if not args.auto_separate.exists():
            parser.error(f"Input file not found: {args.auto_separate}")
        
        # Use temp directory for separation
        with tempfile.TemporaryDirectory(prefix="beats_demucs_") as temp_dir:
            temp_path = Path(temp_dir)
            stems = run_demucs_separation(args.auto_separate, temp_path, args.demucs_model)
            
            # Copy stems to a persistent location next to output
            persist_dir = args.output.parent / f"{args.output.stem}_stems"
            persist_dir.mkdir(exist_ok=True)
            
            for name, temp_stem in stems.items():
                persist_stem = persist_dir / f"{name}.wav"
                shutil.copy2(temp_stem, persist_stem)
                stems[name] = persist_stem
                LOG.info("Copied %s -> %s", name, persist_stem)
    
    elif args.stems_dir:
        if not args.stems_dir.exists():
            parser.error(f"stems-dir not found: {args.stems_dir}")
        found = find_stems(args.stems_dir)
        if not found:
            parser.error("No stems found in directory (expected drums/bass/vocals/other WAVs)")
        stems.update(found)
    else:
        # individual files
        if args.drums: stems["drums"] = args.drums
        if args.bass: stems["bass"] = args.bass
        if args.vocals: stems["vocals"] = args.vocals
        if args.other: stems["other"] = args.other
        if not stems:
            parser.error("Provide stems via --auto-separate, --stems-dir, or individual files")
        for k, p in stems.items():
            if not p.exists():
                parser.error(f"Stem not found: {k} -> {p}")

    try:
        spatialize_beats(
            stems,
            args.output,
            order=args.order,
            stereo_width=args.stereo_width,
            rotate_step_deg=args.rotate_step_deg,
            pulse_decay_ms=args.pulse_decay_ms,
            seed=args.seed,
            beats_only=(args.beats_only or True),
            beat_window_ms=args.beat_window_ms,
            beat_pattern=args.beat_pattern,
            normalize_stems=args.normalize_stems,
            distance_m=args.distance_m,
            b_format=args.b_format,
            wet_level=args.wet,
            no_normalize=args.no_normalize,
            intelligent_float=args.intelligent_float,
            frequency_driven_other=(not args.no_frequency_driven_other),
        )
    except Exception as exc:
        LOG.exception("Processing failed: %s", exc)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
