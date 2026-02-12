"""RCWX command-line interface."""

from __future__ import annotations

import os

# Enable ASIO support in sounddevice (must be set before importing sounddevice)
# See: https://python-sounddevice.readthedocs.io/en/latest/installation.html
os.environ["SD_ENABLE_ASIO"] = "1"

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from rcwx import __version__

# Global log file path for easy access
LOG_FILE: Path | None = None


def setup_logging(verbose: bool = False, log_to_file: bool = True) -> Path | None:
    """Setup logging configuration with optional file output.

    Args:
        verbose: Enable DEBUG level logging
        log_to_file: Write logs to file in addition to console

    Returns:
        Path to log file if file logging is enabled
    """
    global LOG_FILE

    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create handlers
    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    # File handler
    log_file = None
    if log_to_file:
        log_dir = Path.home() / ".config" / "rcwx" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Use timestamp in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"rcwx_{timestamp}.log"
        LOG_FILE = log_file

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always DEBUG for file
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

        # Clean up old logs (keep last 10)
        _cleanup_old_logs(log_dir, keep=10)

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG, handlers filter
        format=log_format,
        handlers=handlers,
        force=True,
    )

    return log_file


def _cleanup_old_logs(log_dir: Path, keep: int = 10) -> None:
    """Remove old log files, keeping the most recent ones."""
    try:
        log_files = sorted(log_dir.glob("rcwx_*.log"), key=lambda p: p.stat().st_mtime)
        for old_log in log_files[:-keep]:
            old_log.unlink()
    except Exception:
        pass  # Ignore cleanup errors


def cmd_gui(args: argparse.Namespace) -> int:
    """Launch the GUI application."""
    import logging

    from rcwx.gui.app import run_gui

    logger = logging.getLogger(__name__)
    if LOG_FILE:
        logger.info(f"Log file: {LOG_FILE}")
        print(f"Log file: {LOG_FILE}")

    run_gui()
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    """List available devices."""
    from rcwx.audio.input import list_input_devices
    from rcwx.audio.output import list_output_devices
    from rcwx.device import list_devices

    print("=== Compute Devices ===")
    for dev in list_devices():
        status = "o" if dev["available"] else "x"
        print(f"  [{status}] {dev['type'].upper()}:{dev['index']} - {dev['name']}")

    print("\n=== Audio Input Devices ===")
    for dev in list_input_devices():
        print(f"  [{dev['index']}] {dev['name']} ({int(dev['sample_rate'])}Hz, {dev['channels']}ch)")

    print("\n=== Audio Output Devices ===")
    for dev in list_output_devices():
        print(f"  [{dev['index']}] {dev['name']} ({int(dev['sample_rate'])}Hz, {dev['channels']}ch)")

    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download required models."""
    from rcwx.config import RCWXConfig
    from rcwx.downloader import check_models, download_all

    config = RCWXConfig.load()
    models_dir = Path(args.models_dir) if args.models_dir else config.get_models_dir()

    print(f"Models directory: {models_dir}")

    # Check existing models
    status = check_models(models_dir)
    print("\nCurrent status:")
    for name, exists in status.items():
        symbol = "o" if exists else "x"
        print(f"  [{symbol}] {name}")

    if all(status.values()) and not args.force:
        print("\nAll models already downloaded. Use --force to re-download.")
        return 0

    print("\nDownloading models...")

    def callback(name: str, status: str) -> None:
        if status == "downloading":
            print(f"  Downloading {name}...")
        else:
            print(f"  {name} done.")

    try:
        download_all(models_dir, force=args.force, callback=callback)
        print("\nAll models downloaded successfully!")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Run voice conversion on a file."""
    import numpy as np
    import torchaudio

    from rcwx.pipeline.inference import RVCPipeline

    # Load input audio
    print(f"Loading: {args.input}")
    audio, sr = torchaudio.load(args.input)
    audio = audio.numpy()

    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=0)

    # Create pipeline
    print(f"Loading model: {args.model}")
    pipeline = RVCPipeline(
        args.model,
        device=args.device,
        dtype=args.dtype,
        use_f0=not args.no_f0,
        use_compile=not args.no_compile,
    )
    pipeline.load()

    # Run inference
    print(
        "Processing "
        f"(pitch shift: {args.pitch}, index_rate: {args.index_rate}, "
        f"pre_hubert: {args.pre_hubert_pitch}, moe_boost: {args.moe_boost})..."
    )
    output = pipeline.infer(
        audio,
        input_sr=sr,
        pitch_shift=args.pitch,
        f0_method=args.f0_method if not args.no_f0 else "none",
        index_rate=args.index_rate,
        voice_gate_mode="off",
        pre_hubert_pitch_ratio=args.pre_hubert_pitch,
        moe_boost=args.moe_boost,
        noise_scale=args.noise_scale,
        pad_mode="batch",
        use_feature_cache=False,
    )

    # Save output
    output_path = args.output or args.input.replace(".wav", "_converted.wav")
    print(f"Saving: {output_path}")

    import torch

    output_tensor = torch.from_numpy(output).unsqueeze(0)
    torchaudio.save(output_path, output_tensor, pipeline.sample_rate)

    print("Done!")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show model information."""
    import torch

    from rcwx.models.synthesizer import detect_model_type

    print(f"Loading: {args.model}")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    version, has_f0, spk_dim = detect_model_type(checkpoint)

    print(f"\nModel Information:")
    print(f"  Version: RVC v{version} ({'768-dim' if version == 2 else '256-dim'} features)")
    print(f"  F0 Support: {'Yes (NSF decoder)' if has_f0 else 'No (standard decoder)'}")
    print(f"  Speaker Embedding: {spk_dim} speakers")

    if "config" in checkpoint:
        config = checkpoint["config"]
        if len(config) > 17:
            print(f"  Sample Rate: {config[17]}Hz")

    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Run audio feedback diagnostics."""
    from rcwx.diagnose import main as diagnose_main

    diagnose_main()
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    """Show or open log files."""
    import os
    import subprocess

    log_dir = Path.home() / ".config" / "rcwx" / "logs"

    if not log_dir.exists():
        print("No log files found.")
        return 0

    log_files = sorted(log_dir.glob("rcwx_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not log_files:
        print("No log files found.")
        return 0

    if args.open:
        # Open the most recent log file
        latest = log_files[0]
        print(f"Opening: {latest}")
        if sys.platform == "win32":
            os.startfile(latest)
        elif sys.platform == "darwin":
            subprocess.run(["open", latest])
        else:
            subprocess.run(["xdg-open", latest])
        return 0

    if args.tail:
        # Show last N lines of most recent log
        latest = log_files[0]
        print(f"=== {latest.name} (last {args.tail} lines) ===\n")
        with open(latest, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-args.tail:]:
                print(line, end="")
        return 0

    # List log files
    print(f"Log directory: {log_dir}\n")
    print("Recent log files:")
    for log_file in log_files[:10]:
        size = log_file.stat().st_size
        mtime = datetime.fromtimestamp(log_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {log_file.name}  ({size:,} bytes, {mtime})")

    print(f"\nUse 'rcwx logs --open' to open the latest log")
    print(f"Use 'rcwx logs --tail 50' to show last 50 lines")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="rcwx",
        description="RCWX - RVC Real-time Voice Changer on Intel Arc (XPU)",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"rcwx {__version__}",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # GUI command (default)
    gui_parser = subparsers.add_parser("gui", help="Launch GUI application")
    gui_parser.set_defaults(func=cmd_gui)

    # Devices command
    devices_parser = subparsers.add_parser("devices", help="List available devices")
    devices_parser.set_defaults(func=cmd_devices)

    # Download command
    download_parser = subparsers.add_parser("download", help="Download required models")
    download_parser.add_argument(
        "--models-dir", "-d",
        help="Models directory",
    )
    download_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download",
    )
    download_parser.set_defaults(func=cmd_download)

    # Run command
    run_parser = subparsers.add_parser("run", help="Convert audio file")
    run_parser.add_argument("input", help="Input audio file")
    run_parser.add_argument("model", help="RVC model file (.pth)")
    run_parser.add_argument(
        "--output", "-o",
        help="Output audio file",
    )
    run_parser.add_argument(
        "--pitch", "-p",
        type=int,
        default=0,
        help="Pitch shift in semitones",
    )
    run_parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "xpu", "cuda", "cpu"],
        help="Compute device",
    )
    run_parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type",
    )
    run_parser.add_argument(
        "--f0-method",
        choices=["rmvpe", "fcpe", "swiftf0"],
        default="rmvpe",
        help="F0 extraction method (default: rmvpe)",
    )
    run_parser.add_argument(
        "--no-f0",
        action="store_true",
        help="Disable F0 extraction",
    )
    run_parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile",
    )
    run_parser.add_argument(
        "--index-rate",
        type=float,
        default=0.0,
        help="FAISS index blending rate (0=disabled, 0.5=balanced, 1=index only)",
    )
    run_parser.add_argument(
        "--pre-hubert-pitch",
        type=float,
        default=0.0,
        help="Pre-HuBERT pitch shift ratio (0.0=off, 1.0=full pitch shift before HuBERT)",
    )
    run_parser.add_argument(
        "--moe-boost",
        type=float,
        default=0.0,
        help="Moe voice style strength for F0 contour (0.0=off, 1.0=strong)",
    )
    run_parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.4,
        help="VAE noise coefficient (0=deterministic, 0.66666=RVC default, 0.4=RCWX default)",
    )
    run_parser.set_defaults(func=cmd_run)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("model", help="RVC model file (.pth)")
    info_parser.set_defaults(func=cmd_info)

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="View log files")
    logs_parser.add_argument(
        "--open", "-o",
        action="store_true",
        help="Open the latest log file in default editor",
    )
    logs_parser.add_argument(
        "--tail", "-t",
        type=int,
        metavar="N",
        help="Show last N lines of the latest log",
    )
    logs_parser.set_defaults(func=cmd_logs)

    # Diagnose command
    diagnose_parser = subparsers.add_parser("diagnose", help="Run audio feedback diagnostics")
    diagnose_parser.set_defaults(func=cmd_diagnose)

    # Parse arguments
    args = parser.parse_args()

    # Setup logging (with file output)
    log_file = setup_logging(args.verbose, log_to_file=True)

    # Default to GUI if no command
    if args.command is None:
        args.func = cmd_gui

    # Run command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
