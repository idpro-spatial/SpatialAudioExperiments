#!/usr/bin/env python3
"""
Test CLI argument parsing for the frequency-driven spatialization
"""
import argparse
from pathlib import Path

def build_test_cli():
    """Test version of the CLI without audio library dependencies"""
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

def test_argument_parsing():
    """Test different argument combinations"""
    parser = build_test_cli()
    
    test_cases = [
        {
            "name": "Default behavior (frequency-driven enabled)",
            "args": ["--other", "test.wav", "-o", "output.wav", "--beats-only"],
            "expected_freq_driven": True
        },
        {
            "name": "Explicitly enable frequency-driven",
            "args": ["--other", "test.wav", "-o", "output.wav", "--beats-only", "--frequency-driven-other"],
            "expected_freq_driven": True
        },
        {
            "name": "Disable frequency-driven",
            "args": ["--other", "test.wav", "-o", "output.wav", "--beats-only", "--no-frequency-driven-other"],
            "expected_freq_driven": False
        }
    ]
    
    print("Testing CLI argument parsing for frequency-driven spatialization:")
    print("=" * 70)
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            args = parser.parse_args(test_case["args"])
            
            # Calculate frequency_driven_other based on the new logic
            frequency_driven_other = not getattr(args, 'no_frequency_driven_other', False)
            
            passed = frequency_driven_other == test_case["expected_freq_driven"]
            status = "✓ PASS" if passed else "✗ FAIL"
            
            print(f"{i}. {test_case['name']}")
            print(f"   Command: python beats.py {' '.join(test_case['args'])}")
            print(f"   Expected frequency_driven_other: {test_case['expected_freq_driven']}")
            print(f"   Actual frequency_driven_other: {frequency_driven_other}")
            print(f"   Result: {status}")
            print()
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"{i}. {test_case['name']}: ✗ ERROR - {e}")
            print()
            all_passed = False
    
    return all_passed

def show_help():
    """Show the help message"""
    parser = build_test_cli()
    print("Help message for the enhanced beats.py:")
    print("=" * 50)
    parser.print_help()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_help()
    else:
        success = test_argument_parsing()
        
        if success:
            print("🎉 All CLI tests passed!")
            print("\nTo see the full help message, run:")
            print("python test_cli.py --help")
        else:
            print("❌ Some CLI tests failed!")
            
        sys.exit(0 if success else 1)
