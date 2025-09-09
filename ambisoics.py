import numpy as np
import soundfile as sf
from spaudiopy import sph, decoder, sig, io
import time
import argparse

# Simple in-process caches
_HRIRS_CACHE = {}
_DECODER_CACHE = {}

def get_decoder(fs: int, order: int):
    """Cache HRIRs and binaural decoder per (fs, order)."""
    key = (fs, order)
    if key in _DECODER_CACHE:
        return _DECODER_CACHE[key]
    if fs in _HRIRS_CACHE:
        hrirs = _HRIRS_CACHE[fs]
    else:
        hrirs = io.load_hrirs(fs)  # can be slow on first call only
        _HRIRS_CACHE[fs] = hrirs
    hrirs_nm = decoder.magls_bin(hrirs, order)
    _DECODER_CACHE[key] = hrirs_nm
    return hrirs_nm

def rotate_yaw_order1(sh_signal: np.ndarray, yaw_angles: np.ndarray) -> np.ndarray:
    """
    Fast, vectorized yaw rotation for 1st-order Ambisonics (ACN/SN3D).
    ACN channel order: [0=W(Y00), 1=Y(Y1-1), 2=Z(Y10), 3=X(Y11)].
    Rotation about z mixes X<->Y; W,Z unchanged.
    sh_signal: (4, N), yaw_angles: (N,)
    """
    if sh_signal.shape[0] != 4:
        raise ValueError("rotate_yaw_order1 expects 1st-order (4 channels).")
    W = sh_signal[0, :]
    Y = sh_signal[1, :]
    Z = sh_signal[2, :]
    X = sh_signal[3, :]
    c = np.cos(yaw_angles)
    s = np.sin(yaw_angles)
    Xr = c * X + s * Y
    Yr = -s * X + c * Y
    return np.vstack([W, Yr, Z, Xr])

# -------- Motion envelope helpers (vectorized, click-free) --------
def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(max(1, win))
    k = np.ones(win, dtype=np.float32) / float(win)
    y = np.convolve(x.astype(np.float32), k, mode="same")
    return y.astype(np.float32)

def smooth_random_envelope(n: int, fs: int, vmin: float, vmax: float,
                           change_every_s: float = 2.0, smooth_ms: float = 30.0,
                           seed: int | None = None) -> np.ndarray:
    """
    Piecewise-linear random targets, then smoothed by moving average.
    Returns length-n float32 array in [vmin, vmax].
    """
    rng = np.random.default_rng(seed)
    step = max(1, int(change_every_s * fs))
    knots = np.arange(0, n + step, step)
    vals = rng.uniform(vmin, vmax, size=len(knots)).astype(np.float32)
    t = np.arange(n, dtype=np.int64)
    env = np.interp(t, knots[:len(vals)], vals).astype(np.float32)
    smooth_win = max(1, int(smooth_ms * 1e-3 * fs))
    return _moving_average(env, smooth_win)

def yaw_with_jitter(n: int, fs: int, base_rps: float = 0.5,
                    jitter_depth: float = 0.4, change_every_s: float = 0.5,
                    seed: int | None = None) -> np.ndarray:
    """
    Integrate time-varying angular speed: omega(t) = 2*pi*(base_rps + jitter(t)).
    jitter(t) is a smooth random in [-jitter_depth, +jitter_depth].
    Returns yaw angle (radians) per sample, length n.
    """
    if jitter_depth <= 0:
        yaw = 2.0 * np.pi * base_rps * (np.arange(n, dtype=np.float64) / fs)
        return yaw.astype(np.float64)
    jitter = smooth_random_envelope(n, fs, -jitter_depth, +jitter_depth,
                                    change_every_s=change_every_s, smooth_ms=25.0, seed=seed).astype(np.float64)
    omega = 2.0 * np.pi * (base_rps + jitter)  # rad/sec
    yaw = np.cumsum(omega) / float(fs)
    return yaw.astype(np.float64)

def distance_gain_env(n: int, fs: int, dmin: float, dmax: float,
                      change_every_s: float = 2.5, smooth_ms: float = 30.0,
                      ref_dist: float = 1.0, rolloff: float = 1.0,
                      seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (distance(t), gain(t)) arrays. gain = (ref/d)^rolloff, both float32.
    """
    d = smooth_random_envelope(n, fs, dmin, dmax, change_every_s, smooth_ms, seed=seed).astype(np.float32)
    d_safe = np.maximum(d, 1e-6)
    ref = max(ref_dist, 1e-6)
    g = np.power(ref / d_safe, rolloff, dtype=np.float32)
    return d, g

# ------------------- Random spin with near/far motion -------------------
def spin_audio_random(input_file: str, output_file: str,
                      base_rps: float = 0.5,
                      stereo_width: float = 30.0,
                      ambisonic_order: int = 1,
                      dmin: float = 1.0, dmax: float = 8.0,
                      dist_change_s: float = 2.5,
                      yaw_jitter_depth: float = 0.4,
                      yaw_change_s: float = 0.5,
                      rolloff: float = 1.0,
                      seed: int | None = None):
    """
    Random yaw (with jitter) + smooth near/far distance loop (vectorized).
    """
    t0 = time.perf_counter()
    audio_in, fs = sf.read(input_file)
    if audio_in.ndim == 1:
        audio_in = np.stack([audio_in, audio_in], axis=1)
    elif audio_in.shape[1] > 2:
        audio_in = audio_in[:, :2]
    N = audio_in.shape[0]

    # Distance gain envelope (applied pre-encoding for linearity)
    _, gain = distance_gain_env(N, fs, dmin, dmax,
                                change_every_s=dist_change_s, smooth_ms=30.0,
                                ref_dist=1.0, rolloff=rolloff, seed=seed)
    x = (audio_in.astype(np.float32) * gain[:, None]).astype(np.float32)

    # Ambisonics encode two virtual sources for stereo width
    half_width = np.deg2rad(stereo_width / 2.0)
    left_azi = -half_width
    right_azi = half_width
    source_zen = np.deg2rad(90.0)
    sh_L = sph.src_to_sh(x[:, 0], left_azi, source_zen, ambisonic_order)
    sh_R = sph.src_to_sh(x[:, 1], right_azi, source_zen, ambisonic_order)
    sh = sh_L + sh_R  # (C, N)

    # Randomized yaw trajectory
    yaw = yaw_with_jitter(N, fs, base_rps=base_rps,
                          jitter_depth=yaw_jitter_depth,
                          change_every_s=yaw_change_s, seed=seed)

    if ambisonic_order == 1:
        sh_rot = rotate_yaw_order1(sh, yaw)
    else:
        # Fallback: per-block rotation (slower for higher orders)
        sh_rot = np.empty_like(sh)
        blk = 2048
        for start in range(0, N, blk):
            end = min(start + blk, N)
            for i in range(start, end):
                sh_rot[:, i] = sph.rotate_sh(sh[:, i], yaw[i], 0.0, 0.0)

    # Binaural decode (cached HRIR/decoder)
    hrirs_nm = get_decoder(fs, ambisonic_order)
    bin_st = decoder.sh2bin(sh_rot, hrirs_nm)  # (2, N) or (N, 2)
    out = bin_st.T if bin_st.ndim == 2 and bin_st.shape[0] == 2 else bin_st
    sf.write(output_file, out, fs)
    print(f"Wrote {output_file} in {time.perf_counter()-t0:.2f}s (length {N/fs:.1f}s)")

def spin_audio(input_file, output_file, rotation_speed=1.0, ambisonic_order=1, stereo_width=30):
    """
    Takes a stereo audio file and makes it spin around the listener's head
    using ambisonics.

    Args:
        input_file (str): Path to the input stereo audio file.
        output_file (str): Path to save the output binaural audio file.
        rotation_speed (float, optional): Speed of rotation in revolutions
                                           per second. Defaults to 1.0.
        ambisonic_order (int, optional): The order of ambisonics to use.
                                         Defaults to 1.
        stereo_width (float, optional): The width of the stereo image in
                                        degrees. Defaults to 30.
    """
    t0 = time.perf_counter()
    # Load the audio file
    audio_in, fs = sf.read(input_file)
    if audio_in.ndim == 1:
        # If the audio is mono, duplicate it to create a stereo signal
        audio_in = np.stack([audio_in, audio_in], axis=1)
    elif audio_in.shape[1] > 2:
        # If more than 2 channels, take the first two
        audio_in = audio_in[:, :2]


    num_samples = audio_in.shape[0]
    left_channel = audio_in[:, 0]
    right_channel = audio_in[:, 1]


    # --- Ambisonics Setup ---
    # Place the left and right channels to create a stereo image
    half_width = np.deg2rad(stereo_width / 2)
    left_azi = -half_width
    right_azi = half_width
    source_zen = np.deg2rad(90)

    # Encode the left and right channels into separate ambisonic sources
    sh_signal_left = sph.src_to_sh(left_channel, left_azi, source_zen, ambisonic_order)
    sh_signal_right = sph.src_to_sh(right_channel, right_azi, source_zen, ambisonic_order)

    # Combine the two sources
    sh_signal = sh_signal_left + sh_signal_right


    # --- Rotation ---
    # Create a time vector
    t = np.linspace(0., num_samples / fs, num_samples, endpoint=False)
    # Calculate the yaw angle for each sample
    yaw_angles = 2 * np.pi * rotation_speed * t

    # Apply rotation efficiently
    if ambisonic_order == 1:
        rotated_sh_signal = rotate_yaw_order1(sh_signal, yaw_angles)
    else:
        # Fallback: coarse blockwise rotation to reduce Python overhead
        rotated_sh_signal = np.empty_like(sh_signal)
        blk = 2048
        for start in range(0, num_samples, blk):
            end = min(start + blk, num_samples)
            # Approximate constant yaw within block (good for small blk)
            ya = float(yaw_angles[start])
            R = sph.rotate_sh(np.eye((ambisonic_order + 1) ** 2)[:, 0], ya, 0, 0)  # build once
            for i in range(start, end):
                rotated_sh_signal[:, i] = sph.rotate_sh(sh_signal[:, i], yaw_angles[i], 0, 0)


    # --- Binaural Decoding ---
    # Get cached binaural decoder (loads HRIRs once per fs)
    hrirs_nm = get_decoder(fs, ambisonic_order)

    # Decode the rotated ambisonic signal to binaural
    binaural_out = decoder.sh2bin(rotated_sh_signal, hrirs_nm)

    # --- Output ---
    # Ensure shape (N, 2) then write
    out = binaural_out.T if binaural_out.ndim == 2 and binaural_out.shape[0] == 2 else binaural_out
    sf.write(output_file, out, fs)
    print(f"Wrote {output_file} in {time.perf_counter()-t0:.2f}s (length {num_samples/fs:.1f}s)")


if __name__ == '__main__':
     p = argparse.ArgumentParser(prog="ambisoics", description="Ambisonic spin and random near/far motion")
     p.add_argument("-i", "--input", required=True, help="Input mono/stereo audio")
     p.add_argument("-o", "--output", required=True, help="Output binaural WAV")
     p.add_argument("--order", type=int, default=1, help="Ambisonic order (1 recommended)")
     p.add_argument("--stereo-width", type=float, default=30.0, help="Stereo width in degrees")
     sub = p.add_subparsers(dest="mode", required=False)
     # Simple constant-speed spin
     sp = sub.add_parser("spin", help="Constant yaw spin")
     sp.add_argument("--rps", type=float, default=0.5, help="Revolutions per second")
     # Random spin + distance loop
     rnd = sub.add_parser("random", help="Random yaw + near/far loop")
     rnd.add_argument("--base-rps", type=float, default=0.5, help="Base revolutions per second")
     rnd.add_argument("--yaw-jitter-depth", type=float, default=0.4, help="Yaw speed jitter depth (0..1, in rps)")
     rnd.add_argument("--yaw-change-s", type=float, default=0.5, help="Yaw jitter change period (s)")
     rnd.add_argument("--dist-min", type=float, default=1.0, help="Nearest distance")
     rnd.add_argument("--dist-max", type=float, default=8.0, help="Farthest distance")
     rnd.add_argument("--dist-change-s", type=float, default=2.5, help="Distance change period (s)")
     rnd.add_argument("--rolloff", type=float, default=1.0, help="Distance rolloff exponent")
     rnd.add_argument("--seed", type=int, default=None, help="Random seed")
     args = p.parse_args()

     if args.mode == "spin":
         spin_audio(args.input, args.output, rotation_speed=getattr(args, "rps", 0.5),
                    ambisonic_order=args.order, stereo_width=args.stereo_width)
     else:
         # default to random mode
         spin_audio_random(args.input, args.output,
                           base_rps=getattr(args, "base_rps", 0.5),
                           stereo_width=args.stereo_width,
                           ambisonic_order=args.order,
                           dmin=getattr(args, "dist_min", 1.0),
                           dmax=getattr(args, "dist_max", 8.0),
                           dist_change_s=getattr(args, "dist_change_s", 2.5),
                           yaw_jitter_depth=getattr(args, "yaw_jitter_depth", 0.4),
                           yaw_change_s=getattr(args, "yaw_change_s", 0.5),
                           rolloff=getattr(args, "rolloff", 1.0),
                           seed=getattr(args, "seed", None))
