#!/usr/bin/env python3
"""
spatial_loops.py

Utility script providing looped / rotating spatial test scenarios on top of the
core functionality implemented in spatial.py.

Scenarios:
  1. Constant spin (azimuth rotates 0..360 repeatedly) at constant distance.
  2. Spin with breathing distance (near <-> far cosine LFO) while rotating.
  3. Inward or outward spiral (az rotates, distance ramps linearly once).

All scenarios use a generic path abstraction based on analytic functions:
   az(t), el(t), dist(t)
They reuse the lower-level primitives from spatial.py and implement:
   - Block processing with optional intra-block morph (az + distance)
   - Stateful air absorption filter
   - Global reverb pass mixed with distance envelope (crackle-minimized)

Usage examples:
  python spatial_loops.py -i input.wav -s HRIR.sofa -o spin.wav --mode spin --spin-speed 45
  python spatial_loops.py -i input.wav -s HRIR.sofa -o spin_pulse.wav --mode spin_pulse --spin-speed 60 \
      --dist-min 1 --dist-max 8 --dist-period 6 --wet-max 0.4
  python spatial_loops.py -i input.wav -s HRIR.sofa -o spiral_in.wav --mode spiral --spin-speed 90 \
      --dist-start 10 --dist-end 1
"""
from __future__ import annotations
from pathlib import Path
import argparse
import logging
import sys
import numpy as np
import librosa
from typing import Callable

import spatial  # Reuse existing functions

LOG = logging.getLogger("spatial_loops")

# ---------------------------------------------------------------------------
# Analytic path wrappers
# ---------------------------------------------------------------------------
class AnalyticPath:
    def __init__(self, az_fn: Callable[[float], float], el_fn: Callable[[float], float], dist_fn: Callable[[float], float]):
        self.az_fn = az_fn
        self.el_fn = el_fn
        self.dist_fn = dist_fn

    def position_at(self, t: float):  # compatible with SourcePath subset
        return float(self.az_fn(t)), float(self.el_fn(t))

    def distance_at(self, t: float):
        return float(self.dist_fn(t))

# ---------------------------------------------------------------------------
# Core loop spatializer (generic), mirroring artifact-minimized strategy
# ---------------------------------------------------------------------------

def spatialize_loop(input_path: Path, sofa_path: Path, output_path: Path,
                     path: AnalyticPath, target_sr: int, apply_reverb: bool,
                     duration: float | None,
                     block_size: int = 2048, interp_k: int = 4, interp_power: float = 2.0,
                     filter_morph: bool = True, morph_tol_deg: float = 1e-4,
                     ref_dist: float = 1.0, rolloff: float = 1.0, air_strength: float = 0.35,
                     wet_max: float = 0.3, max_dist: float = 10.0,
                     boundary_xfade: int = 0):
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    LOG.info("Loading audio: %s", input_path)
    audio, _ = librosa.load(str(input_path), sr=target_sr, mono=True)
    if duration is not None:
        total_samples = min(len(audio), int(duration * target_sr))
        audio = audio[:total_samples]
    audio = audio.astype(np.float32)
    n_samples = len(audio)
    LOG.debug("Audio samples=%d (%.2fs)", n_samples, n_samples / target_sr)

    LOG.info("Reading SOFA: %s", sofa_path)
    hrir_all, positions, sr_sofa = spatial.read_sofa(sofa_path)
    hrir_len = hrir_all.shape[2]
    if sr_sofa != target_sr:
        LOG.debug("HRIR sr %d -> resample to %d per block", sr_sofa, target_sr)

    dry = np.zeros((n_samples + hrir_len - 1, 2), dtype=np.float32)
    num_blocks = int(np.ceil(n_samples / block_size))
    LOG.info("Processing %d blocks (block=%d, morph=%s, global reverb=%s)", num_blocks, block_size, filter_morph, apply_reverb)

    air_state = 0.0

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, n_samples)
        block = audio[start:end]
        if block.size == 0:
            continue
        t0 = start / target_sr
        t1 = end / target_sr
        az0, el0 = path.position_at(t0)
        az1, el1 = path.position_at(t1)
        d0 = path.distance_at(t0)
        d1 = path.distance_at(t1)
        if b % 25 == 0:
            LOG.debug("Block %d/%d t=[%.2f,%.2f] az=(%.1f->%.1f) el=(%.1f->%.1f) dist=(%.2f->%.2f)",
                      b, num_blocks, t0, t1, az0, az1, el0, el1, d0, d1)

        gain0 = spatial.compute_distance_gain(d0, ref_dist, rolloff)
        need_morph = filter_morph and (abs(az1 - az0) >= morph_tol_deg or abs(d1 - d0) >= 1e-6)

        if not need_morph:
            proc, air_state = spatial.apply_air_absorption(block * gain0, target_sr, d0, air_state, ref_dist, air_strength)
            l_hrir, r_hrir = spatial.interpolate_hrir(hrir_all, positions, az0, el0, k=interp_k, power=interp_power)
            l_hrir = spatial.resample_if_needed(l_hrir, sr_sofa, target_sr)
            r_hrir = spatial.resample_if_needed(r_hrir, sr_sofa, target_sr)
            stereo = spatial.convolve_hrir(proc, l_hrir, r_hrir)
            if boundary_xfade > 0 and start > 0:
                xfade = min(boundary_xfade, stereo.shape[0])
                existing = dry[start:start + xfade]
                ramp = np.linspace(0.0, 1.0, xfade, dtype=np.float32)[:, None]
                dry[start:start + xfade] = existing * (1.0 - ramp) + stereo[:xfade] * ramp
                if xfade < stereo.shape[0]:
                    dry[start + xfade:start + stereo.shape[0]] += stereo[xfade:]
            else:
                dry[start:start + stereo.shape[0]] += stereo
            continue

        # Morph variant
        air_state_initial = air_state
        dry0, air_state_after = spatial.apply_air_absorption(block * gain0, target_sr, d0, air_state_initial, ref_dist, air_strength)
        l0, r0 = spatial.interpolate_hrir(hrir_all, positions, az0, el0, k=interp_k, power=interp_power)
        l0 = spatial.resample_if_needed(l0, sr_sofa, target_sr)
        r0 = spatial.resample_if_needed(r0, sr_sofa, target_sr)
        stereo0 = spatial.convolve_hrir(dry0, l0, r0)

        gain1 = spatial.compute_distance_gain(d1, ref_dist, rolloff)
        dry1, air_state1 = spatial.apply_air_absorption(block * gain1, target_sr, d1, air_state_initial, ref_dist, air_strength)
        l1, r1 = spatial.interpolate_hrir(hrir_all, positions, az1, el1, k=interp_k, power=interp_power)
        l1 = spatial.resample_if_needed(l1, sr_sofa, target_sr)
        r1 = spatial.resample_if_needed(r1, sr_sofa, target_sr)
        stereo1 = spatial.convolve_hrir(dry1, l1, r1)

        L = stereo0.shape[0]
        ramp = np.linspace(0.0, 1.0, L, dtype=np.float32)
        stereo_morph = (stereo0 * (1.0 - ramp)[:, None] + stereo1 * ramp[:, None]).astype(np.float32)
        if boundary_xfade > 0 and start > 0:
            xfade = min(boundary_xfade, stereo_morph.shape[0])
            existing = dry[start:start + xfade]
            ramp2 = np.linspace(0.0, 1.0, xfade, dtype=np.float32)[:, None]
            dry[start:start + xfade] = existing * (1.0 - ramp2) + stereo_morph[:xfade] * ramp2
            if xfade < stereo_morph.shape[0]:
                dry[start + xfade:start + stereo_morph.shape[0]] += stereo_morph[xfade:]
        else:
            dry[start:start + L] += stereo_morph
        air_state = air_state1

    final_len = n_samples + hrir_len - 1
    dry = dry[:final_len]

    # Build distance envelope for wet mix
    times = np.arange(final_len, dtype=np.float32) / float(target_sr)
    dist_vec = np.array([path.distance_at(t) for t in times], dtype=np.float32)
    wet_env = np.clip((dist_vec - ref_dist) / max(1e-6, max_dist - ref_dist), 0.0, 1.0) * wet_max

    if apply_reverb and spatial._HAS_PEDALBOARD and wet_max > 0:
        LOG.info("Global reverb + distance envelope blend (wet_max=%.2f)", wet_max)
        wet_full = spatial.apply_optional_reverb(dry, target_sr)
        mixed = (dry * (1.0 - wet_env)[:, None] + wet_full * wet_env[:, None]).astype(np.float32)
    else:
        mixed = dry

    final = spatial.normalize_audio(mixed)
    spatial.write_atomic(output_path, final, target_sr)
    LOG.info("Loop spatialization complete -> %s", output_path)

# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def build_spin(spin_speed: float, distance: float, elevation: float = 0.0) -> AnalyticPath:
    def az_fn(t: float):
        return (spin_speed * t) % 360.0
    def el_fn(t: float):
        return elevation
    def dist_fn(t: float):
        return distance
    return AnalyticPath(az_fn, el_fn, dist_fn)

def build_spin_pulse(spin_speed: float, dist_min: float, dist_max: float, period: float, elevation: float = 0.0) -> AnalyticPath:
    amp = (dist_max - dist_min) * 0.5
    mid = dist_min + amp
    def az_fn(t: float):
        return (spin_speed * t) % 360.0
    def el_fn(t: float):
        return elevation
    def dist_fn(t: float):
        return mid + amp * (0.5 * (1 - np.cos(2 * np.pi * t / max(1e-6, period))))  # cosine ease in/out
    return AnalyticPath(az_fn, el_fn, dist_fn)

def build_spiral(spin_speed: float, dist_start: float, dist_end: float, total_time: float, elevation: float = 0.0) -> AnalyticPath:
    def az_fn(t: float):
        return (spin_speed * t) % 360.0
    def el_fn(t: float):
        return elevation
    def dist_fn(t: float):
        if t >= total_time:
            return dist_end
        frac = t / total_time
        return dist_start + frac * (dist_end - dist_start)
    return AnalyticPath(az_fn, el_fn, dist_fn)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_cli():
    p = argparse.ArgumentParser(prog="spatial_loops", description="Loop / rotating spatial test generator")
    p.add_argument('-i', '--input', type=Path, required=True, help='Input mono/stereo file (loaded mono)')
    p.add_argument('-s', '--sofa', type=Path, required=True, help='SOFA HRTF file')
    p.add_argument('-o', '--output', type=Path, default=Path('loop_out.wav'), help='Output WAV')
    p.add_argument('-r', '--samplerate', type=int, default=spatial.DEFAULT_SR, help='Target sample rate')
    p.add_argument('--mode', choices=['spin', 'spin_pulse', 'spiral'], default='spin', help='Scenario mode')
    p.add_argument('--duration', type=float, default=None, help='Optional duration limit (seconds)')
    p.add_argument('--spin-speed', type=float, default=60.0, help='Spin speed deg/sec')
    # Distance params
    p.add_argument('--distance', type=float, default=2.0, help='Constant distance for spin mode')
    p.add_argument('--dist-min', type=float, default=1.0, help='Min distance for spin_pulse')
    p.add_argument('--dist-max', type=float, default=8.0, help='Max distance for spin_pulse')
    p.add_argument('--dist-period', type=float, default=6.0, help='Near->far->near period seconds (spin_pulse)')
    p.add_argument('--dist-start', type=float, default=8.0, help='Start distance (spiral)')
    p.add_argument('--dist-end', type=float, default=1.0, help='End distance (spiral)')
    p.add_argument('--spiral-time', type=float, default=10.0, help='Spiral total time (seconds)')
    p.add_argument('--elevation', type=float, default=0.0, help='Elevation (deg)')
    # Engine params
    p.add_argument('--block-size', type=int, default=2048)
    p.add_argument('--interp-k', type=int, default=4)
    p.add_argument('--interp-power', type=float, default=2.0)
    p.add_argument('--no-morph', action='store_true', help='Disable intra-block morph')
    p.add_argument('--ref-dist', type=float, default=1.0)
    p.add_argument('--rolloff', type=float, default=1.0)
    p.add_argument('--air-strength', type=float, default=0.35)
    p.add_argument('--wet-max', type=float, default=0.3)
    p.add_argument('--max-dist', type=float, default=10.0)
    p.add_argument('--reverb', action='store_true', help='Enable global reverb blend')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--boundary-xfade', type=int, default=0, help='Samples of block boundary crossfade (reduce clicks)')
    return p


def main(argv=None):
    parser = build_cli()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    if args.mode == 'spin':
        path = build_spin(args.spin_speed, args.distance, elevation=args.elevation)
    elif args.mode == 'spin_pulse':
        path = build_spin_pulse(args.spin_speed, args.dist_min, args.dist_max, args.dist_period, elevation=args.elevation)
    else:  # spiral
        path = build_spiral(args.spin_speed, args.dist_start, args.dist_end, args.spiral_time, elevation=args.elevation)

    LOG.info("Mode=%s", args.mode)

    try:
        spatialize_loop(args.input, args.sofa, args.output, path, args.samplerate, args.reverb,
                        duration=args.duration,
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
    except Exception as exc:
        LOG.exception("Loop spatialization failed: %s", exc)
        sys.exit(1)


if __name__ == '__main__':
    main()
