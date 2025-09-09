#!/usr/bin/env python3
import argparse
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import spaudiopy as spa

def load_mono(path, max_duration=None):
    audio, fs = sf.read(path, always_2d=True)
    if max_duration is not None:
        max_samps = int(max_duration * fs)
        if audio.shape[0] > max_samps:
            audio = audio[:max_samps]
    mono = np.mean(audio, axis=1).astype(np.float32)
    return mono, fs

def resample_hrirs(hrirs, target_fs):
    if hrirs.fs == target_fs:
        return hrirs
    g = math.gcd(int(hrirs.fs), int(target_fs))
    up = target_fs // g
    down = hrirs.fs // g
    left_rs = resample_poly(hrirs.left, up, down, axis=1)
    right_rs = resample_poly(hrirs.right, up, down, axis=1)
    return spa.sig.HRIRs(left_rs, right_rs, hrirs.azi, hrirs.zen, target_fs)

def encode_spinning_source(sig, fs, order, rev_seconds, block=65536):

    n = sig.shape[0]
    C = (order + 1) ** 2
    ambi = np.zeros((C, n), dtype=np.float32)

    omega = 2 * np.pi / float(rev_seconds)  # rad/s
    t = np.arange(n, dtype=np.float64) / fs
    azi_all = (omega * t)  # allow to grow; SH matrix only cares modulo 2π internally
    zen_all = np.full_like(azi_all, np.pi / 2.0)  # horizontal plane

    for start in range(0, n, block):
        end = min(start + block, n)
        azi_block = np.mod(azi_all[start:end], 2 * np.pi)
        # Real ACN/N3D spherical harmonics
        Y = spa.sph.sh_matrix(order, azi_block, zen_all[start:end])  # (B, C)
        # Multiply each sample's SH vector by sample amplitude
        ambi[:, start:end] = (Y * sig[start:end, None]).T
    return ambi

def binaural_magls(ambi, fs, order, sofa_path):
    hrirs = spa.io.load_sofa_hrirs(sofa_path)
    if hrirs.fs != fs:
        hrirs = resample_hrirs(hrirs, fs)
    hrirs_nm = spa.decoder.magls_bin(hrirs, order)
    stereo = spa.decoder.sh2bin(ambi, hrirs_nm)  # (2, S)
    return stereo

def peak_normalize(x, peak=0.99):
    m = np.max(np.abs(x))
    if m > 0:
        x = x * (peak / m)
    return x

def parse_args():
    p = argparse.ArgumentParser(description="Spin a mono source around the listener and binauralize (magLS).")
    p.add_argument("--in", dest="inp", required=True, help="Input audio (mono/stereo)")
    p.add_argument("--sofa", required=True, help="SOFA HRIR file")
    p.add_argument("--out", required=True, help="Output binaural WAV")
    p.add_argument("--order", type=int, default=1, help="Ambisonic order (>=0). Perceptual gains >3 minimal for single source.")
    p.add_argument("--rev-seconds", type=float, default=6.0, help="Seconds per full 360° revolution")
    p.add_argument("--max-duration", type=float, default=None, help="Optional truncate input (s)")
    p.add_argument("--no-normalize", action="store_true", help="Disable peak normalization")
    p.add_argument("--gain", type=float, default=1.0, help="Linear gain after decode (applied pre-normalization)")
    p.add_argument("--subtype", default="PCM_24", help="Output subtype (e.g. PCM_16, PCM_24, FLOAT)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if args.order < 0:
        raise ValueError("Order must be >= 0")
    if args.verbose:
        print(f"Loading audio: {args.inp}")
    mono, fs = load_mono(args.inp, args.max_duration)
    if args.verbose:
        dur = mono.shape[0] / fs
        print(f"Audio fs={fs}Hz samples={mono.shape[0]} duration={dur:.2f}s")

    if args.verbose:
        print(f"Encoding spinning source: order={args.order} rev_seconds={args.rev_seconds}")
    ambi = encode_spinning_source(mono, fs, args.order, args.rev_seconds)

    if args.verbose:
        print(f"Decoding (magLS) with SOFA {args.sofa}")
    stereo = binaural_magls(ambi, fs, args.order, args.sofa)  # (2, S)

    stereo *= args.gain

    if not args.no_normalize:
        stereo = peak_normalize(stereo, 0.99)

    stereo_out = stereo.T  # (S,2)
    sf.write(args.out, stereo_out, fs, subtype=args.subtype)
    if args.verbose:
        print(f"Done -> {args.out}")

if __name__ == "__main__":
    main()