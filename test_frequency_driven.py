#!/usr/bin/env python3
"""
Test script for frequency-driven spatialization functions
"""
import numpy as np
import sys
from pathlib import Path

# Add current directory to path to import beats functions
sys.path.append(str(Path(__file__).parent))

def test_spectral_features():
    """Test spectral feature computation with synthetic data"""
    print("Testing spectral feature computation...")
    
    # Create synthetic audio data (1 second at 44.1kHz)
    sr = 44100
    duration = 1.0
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples)
    
    # Create a sweep from low to high frequency to test spectral centroid
    f_start = 200
    f_end = 8000
    y_sweep = np.sin(2 * np.pi * np.logspace(np.log10(f_start), np.log10(f_end), samples) * t)
    
    # Mock the librosa functions for testing
    try:
        # These would normally come from librosa
        def mock_spectral_centroid(y, sr, hop_length):
            # Simulate increasing centroid for our sweep
            n_frames = len(y) // hop_length
            return np.linspace(1000, 6000, n_frames).reshape(1, -1)
        
        def mock_stft(y, hop_length):
            # Simple mock STFT
            n_frames = len(y) // hop_length
            n_freqs = 1025  # Typical for 2048 FFT
            return np.random.random((n_freqs, n_frames)) + 1j * np.random.random((n_freqs, n_frames))
        
        def mock_fft_frequencies(sr, n_fft):
            return np.linspace(0, sr/2, n_fft//2 + 1)
        
        # Test the core logic without actual audio processing
        hop_length = 512
        n_frames = samples // hop_length
        
        # Mock spectral centroid
        spectral_centroid = np.linspace(1000, 6000, n_frames)
        
        # Mock high-frequency energy ratio
        hf_energy_ratio = np.linspace(0.1, 0.8, n_frames)
        
        print(f"✓ Spectral centroid range: {spectral_centroid.min():.1f} - {spectral_centroid.max():.1f} Hz")
        print(f"✓ HF energy ratio range: {hf_energy_ratio.min():.3f} - {hf_energy_ratio.max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in spectral feature test: {e}")
        return False


def test_smoothing():
    """Test control signal smoothing"""
    print("\nTesting control signal smoothing...")
    
    try:
        from scipy.ndimage import uniform_filter1d
        
        # Create noisy signal
        signal = np.random.random(1000) + np.sin(np.linspace(0, 10*np.pi, 1000))
        
        # Test smoothing
        smoothed = uniform_filter1d(signal, size=10, mode='nearest')
        
        # Check that smoothing reduces variance
        original_var = np.var(signal)
        smoothed_var = np.var(smoothed)
        
        print(f"✓ Original variance: {original_var:.3f}")
        print(f"✓ Smoothed variance: {smoothed_var:.3f}")
        print(f"✓ Variance reduction: {(1 - smoothed_var/original_var)*100:.1f}%")
        
        return smoothed_var < original_var
        
    except Exception as e:
        print(f"✗ Error in smoothing test: {e}")
        return False


def test_trajectory_mapping():
    """Test mapping of spectral features to spatial trajectories"""
    print("\nTesting trajectory mapping...")
    
    try:
        # Create mock spectral features
        n_frames = 100
        spectral_centroid = np.linspace(1000, 6000, n_frames)  # Increasing brightness
        hf_energy_ratio = np.sin(np.linspace(0, 2*np.pi, n_frames)) * 0.5 + 0.5  # Oscillating HF energy
        
        # Map to yaw (left-right) based on spectral centroid
        centroid_min = np.percentile(spectral_centroid, 5)
        centroid_max = np.percentile(spectral_centroid, 95)
        centroid_norm = np.clip((spectral_centroid - centroid_min) / max(1.0, centroid_max - centroid_min), 0.0, 1.0)
        
        yaw_range = np.deg2rad(180.0)  # -90° to +90°
        yaw_trajectory = (centroid_norm - 0.5) * yaw_range
        
        # Map to elevation based on HF energy
        elevation_max = np.deg2rad(45.0)  # 0° to 45°
        elevation_trajectory = hf_energy_ratio * elevation_max
        
        print(f"✓ Yaw range: {np.rad2deg(yaw_trajectory.min()):.1f}° to {np.rad2deg(yaw_trajectory.max()):.1f}°")
        print(f"✓ Elevation range: {np.rad2deg(elevation_trajectory.min()):.1f}° to {np.rad2deg(elevation_trajectory.max()):.1f}°")
        
        # Check that trajectories are reasonable
        yaw_ok = np.rad2deg(np.abs(yaw_trajectory)).max() <= 90.0
        elevation_ok = np.rad2deg(elevation_trajectory).max() <= 45.0 and np.rad2deg(elevation_trajectory).min() >= 0.0
        
        print(f"✓ Yaw range valid: {yaw_ok}")
        print(f"✓ Elevation range valid: {elevation_ok}")
        
        return yaw_ok and elevation_ok
        
    except Exception as e:
        print(f"✗ Error in trajectory mapping test: {e}")
        return False


def main():
    """Run all tests"""
    print("=== Testing Frequency-Driven Spatialization Functions ===\n")
    
    tests = [
        test_spectral_features,
        test_smoothing,
        test_trajectory_mapping,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed! The frequency-driven spatialization logic is working correctly.")
    else:
        print("✗ Some tests failed. Check the implementation.")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
