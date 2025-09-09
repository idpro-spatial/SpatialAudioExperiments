#!/usr/bin/env python3
"""
Demo script showing how to use the new frequency-driven spatialization for the 'other' stem
"""

def print_usage_examples():
    """Print usage examples for the frequency-driven spatialization feature"""
    
    print("🎵 Frequency-Driven Spatialization for 'Other' Stem")
    print("=" * 60)
    print()
    print("This enhancement adds ethereal, floating movement to the 'other' stem")
    print("(typically containing synths, guitars, pianos, and melodic elements).")
    print()
    print("How it works:")
    print("• Brightness (spectral centroid) → Left-Right movement (-90° to +90°)")
    print("• High-frequency energy → Up-Down movement (0° to +45°)")
    print("• Smooth, continuous movement driven by music content")
    print()
    print("Basic Usage Examples:")
    print("-" * 30)
    print()
    
    examples = [
        {
            "title": "🎧 Process a full song with Demucs auto-separation",
            "command": "python beats.py --auto-separate song.mp3 -o ethereal_song.wav --beats-only",
            "description": "Automatically separates stems and applies frequency-driven spatialization to 'other'"
        },
        {
            "title": "🎹 Process individual stems with frequency-driven 'other'",
            "command": "python beats.py --drums drums.wav --bass bass.wav --vocals vocals.wav --other other.wav -o spatial_mix.wav --beats-only",
            "description": "Drums get beat-sync spatialization, bass/vocals stay centered, 'other' gets frequency-driven floating"
        },
        {
            "title": "🌟 Process only the 'other' stem for maximum ethereal effect",
            "command": "python beats.py --other synths.wav -o floating_synths.wav --beats-only",
            "description": "Apply frequency-driven spatialization to just the melodic elements"
        },
        {
            "title": "🎛️ Customize the stereo width for more immersive effect",
            "command": "python beats.py --other other.wav -o wide_floating.wav --beats-only --stereo-width 60",
            "description": "Wider stereo field makes the floating effect more pronounced"
        },
        {
            "title": "🔄 Disable frequency-driven spatialization (fallback to centered)",
            "command": "python beats.py --other other.wav -o centered_other.wav --beats-only --no-frequency-driven-other",
            "description": "Keep 'other' stem centered like bass and vocals"
        },
        {
            "title": "🎚️ Output B-format for use in spatial audio workstations",
            "command": "python beats.py --other other.wav -o floating_bformat.wav --beats-only --b-format",
            "description": "Generate B-format ambisonics output instead of binaural"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['title']}")
        print(f"   {example['command']}")
        print(f"   → {example['description']}")
        print()
    
    print("Advanced Options:")
    print("-" * 20)
    print("--order 1               : Ambisonic order (1st order recommended)")
    print("--stereo-width 30       : Stereo width in degrees (try 60+ for wider effect)")
    print("--seed 42               : Random seed for reproducible results")
    print("--distance-m 2.0        : Virtual distance modeling")
    print("--no-normalize          : Preserve original mix balance")
    print()
    
    print("Technical Details:")
    print("-" * 20)
    print("• Spectral analysis uses Short-Time Fourier Transform (STFT)")
    print("• Control signals are smoothed over 200-300ms for fluid movement")  
    print("• High-frequency threshold: 6kHz for elevation control")
    print("• Yaw range: ±90° (full left-right panorama)")
    print("• Elevation range: 0° to 45° (horizontal to upward)")
    print("• Frame rate: ~86 Hz (512-sample hop length at 44.1kHz)")
    print()
    
    print("🎯 Pro Tips:")
    print("• Works best with melodic content that has varying brightness")
    print("• Combine with drums for full rhythmic + melodic spatialization")
    print("• Use headphones for the full binaural experience")
    print("• Try different stereo widths to find your preferred effect intensity")


if __name__ == "__main__":
    print_usage_examples()
