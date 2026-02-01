"""
Diagnostic script to investigate why chunk_sec changes from 0.1s to 0.5s.

Run this to understand the buffer drop issue.
"""

import sys


def analyze_log(log_path):
    """Analyze log file for chunk_sec changes."""
    print("=" * 70)
    print("BUFFER DROP DIAGNOSTIC TOOL")
    print("=" * 70)

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Extract key information
    audio_config = None
    stream_info = {"input": None, "output": None}
    chunk_size_changes = []
    sample_infer = None

    for i, line in enumerate(lines):
        # Audio config
        if "Audio config:" in line:
            audio_config = line.strip()

        # Stream start
        if "Input stream started" in line:
            stream_info["input"] = line.strip()
        if "Output stream started" in line:
            stream_info["output"] = line.strip()

        # Chunk size adjustments
        if "Chunk size adjusted:" in line:
            chunk_size_changes.append(line.strip())

        # Sample inference
        if "[INFER] Chunk #" in line and sample_infer is None:
            sample_infer = line.strip()

    # Print results
    print("\n1. AUDIO CONFIGURATION")
    print("-" * 70)
    if audio_config:
        print(audio_config)
        # Parse values
        import re

        if match := re.search(r"chunk=([\d.]+)s \((\d+) samples\)", audio_config):
            chunk_sec = float(match.group(1))
            chunk_samples = int(match.group(2))
            print(f"\n   → chunk_sec: {chunk_sec}s")
            print(f"   → chunk_samples: {chunk_samples}")

            if chunk_sec != 0.1:
                print(
                    f"   ⚠️  WARNING: chunk_sec is {chunk_sec}s, expected 0.1s!"
                )

        if match := re.search(r"mic_sr=(\d+)", audio_config):
            mic_sr = int(match.group(1))
            print(f"   → mic_sample_rate: {mic_sr} Hz")

            if mic_sr not in [48000, 44100]:
                print(f"   ⚠️  WARNING: Unusual sample rate {mic_sr} Hz")
    else:
        print("   ❌ Audio config not found in log")

    print("\n2. AUDIO DRIVER SETTINGS")
    print("-" * 70)
    for stream_type, info in stream_info.items():
        if info:
            print(f"{stream_type.upper()}: {info}")
            # Parse blocksize
            import re

            if match := re.search(r"blocksize=(\d+)", info):
                blocksize = int(match.group(1))
                if match2 := re.search(r"sr=(\d+)", info):
                    sr = int(match2.group(1))
                    blocksize_ms = blocksize / sr * 1000
                    print(f"   → blocksize: {blocksize} samples = {blocksize_ms:.1f}ms")

                    if blocksize > 2000:
                        print(
                            f"   ⚠️  WARNING: Large blocksize ({blocksize_ms:.1f}ms) detected!"
                        )
        else:
            print(f"{stream_type.upper()}: ❌ Not found in log")

    print("\n3. CHUNK SIZE ADJUSTMENTS")
    print("-" * 70)
    if chunk_size_changes:
        for change in chunk_size_changes:
            print(f"   {change}")
    else:
        print("   ℹ️  No chunk size adjustments found")

    print("\n4. INFERENCE SAMPLE")
    print("-" * 70)
    if sample_infer:
        print(sample_infer)
        # Parse values
        import re

        if match := re.search(r"in=(\d+), out=(\d+)", sample_infer):
            in_samples = int(match.group(1))
            out_samples = int(match.group(2))
            print(f"\n   → Input samples: {in_samples}")
            print(f"   → Output samples: {out_samples}")

            # Calculate durations
            input_sr = 16000  # Always 16kHz for processing
            model_sr = 40000  # RVC model output

            in_sec = in_samples / input_sr
            out_sec = out_samples / model_sr

            print(f"   → Input duration: {in_sec*1000:.0f}ms @ {input_sr}Hz")
            print(f"   → Output duration: {out_sec*1000:.0f}ms @ {model_sr}Hz")

            if in_sec > 0.15:  # More than 150ms
                print(
                    f"   ⚠️  WARNING: Input duration {in_sec*1000:.0f}ms exceeds expected 100ms!"
                )
    else:
        print("   ❌ No inference data found in log")

    print("\n5. DIAGNOSIS")
    print("=" * 70)

    # Check for the smoking gun
    if audio_config and "chunk=0.5s" in audio_config:
        print("🔴 ISSUE CONFIRMED: chunk_sec = 0.5s (should be 0.1s)")
        print("\n   Root cause: chunk_sec was changed from default 0.1s to 0.5s")
        print("\n   Likely causes:")
        print("   1. GUI saved config with chunk_sec=0.5")
        print("   2. HuBERT hop alignment logic modified chunk_sec")
        print("   3. _recalculate_buffers() called with wrong values")
        print(
            "\n   Impact: Buffer overflow because max_latency calculated for 0.1s chunks"
        )
        print("           but actual chunks are 0.5s → drops every ~4 chunks")
    elif audio_config and "chunk=0.1s" in audio_config:
        print("✅ chunk_sec looks correct (0.1s)")
        print("\n   Need to investigate other causes...")
    else:
        print("⚠️  Cannot determine chunk_sec from logs")

    # Check blocksize
    for stream_type, info in stream_info.items():
        if info and "blocksize=5512" in info:
            print(
                f"\n⚠️  Large {stream_type} blocksize (5512 = 125ms @ 44.1kHz) detected"
            )
            print("   This is 5x larger than requested (1200 = 25ms)")
            print("   Audio driver may not support requested blocksize")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        # Try default location
        import os
        from pathlib import Path

        log_dir = Path.home() / ".config" / "rcwx" / "logs"
        if log_dir.exists():
            # Find most recent log
            logs = sorted(log_dir.glob("rcwx_*.log"), key=os.path.getmtime, reverse=True)
            if logs:
                log_path = str(logs[0])
                print(f"Using most recent log: {log_path}\n")
            else:
                print("No log files found!")
                sys.exit(1)
        else:
            print("Usage: python diagnose_chunk_size.py [log_file]")
            sys.exit(1)

    analyze_log(log_path)
