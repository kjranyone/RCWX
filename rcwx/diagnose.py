"""Audio feedback diagnostic tool for RCWX."""

from __future__ import annotations

import sys
from pathlib import Path


def check_audio_devices():
    """Check audio device configuration."""
    print("=" * 60)
    print("RCWX Audio Diagnostic Tool")
    print("=" * 60)
    print()

    # Check sounddevice
    try:
        import sounddevice as sd

        print("[*] Available Audio Devices:")
        print()

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        # Build hostapi name mapping
        hostapi_names = {}
        for i, hostapi in enumerate(hostapis):
            name = hostapi["name"]
            # Simplify names
            if "WASAPI" in name:
                hostapi_names[i] = "WASAPI"
            elif "ASIO" in name:
                hostapi_names[i] = "ASIO"
            elif "DirectSound" in name:
                hostapi_names[i] = "DirectSound"
            elif "MME" in name:
                hostapi_names[i] = "MME"
            else:
                hostapi_names[i] = name

        input_devices = []
        output_devices = []

        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                input_devices.append((i, dev))
            if dev["max_output_channels"] > 0:
                output_devices.append((i, dev))

        print("=== Input Devices (Microphone) ===")
        for i, dev in input_devices:
            default = " (DEFAULT)" if i == sd.default.device[0] else ""
            hostapi = hostapi_names.get(dev["hostapi"], "Unknown")
            print(f"  [{i}] {dev['name']}{default}")
            print(f"      Driver: {hostapi}, Channels: {dev['max_input_channels']}, Sample Rate: {dev['default_samplerate']:.0f}Hz")

            # Check for potential loopback devices
            name_lower = dev["name"].lower()
            if any(x in name_lower for x in ["stereo mix", "what u hear", "loopback"]):
                print("      [!] WARNING: This may be a loopback device!")
                print("          Using this as input will route output back to input.")
        print()

        print("=== Output Devices (Speaker/Headphones) ===")
        for i, dev in output_devices:
            default = " (DEFAULT)" if i == sd.default.device[1] else ""
            hostapi = hostapi_names.get(dev["hostapi"], "Unknown")
            print(f"  [{i}] {dev['name']}{default}")
            print(f"      Driver: {hostapi}, Channels: {dev['max_output_channels']}, Sample Rate: {dev['default_samplerate']:.0f}Hz")
        print()

        # Check if input and output are on the same interface
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]

        if default_input is not None and default_output is not None:
            input_name = devices[default_input]["name"]
            output_name = devices[default_output]["name"]

            if "high definition audio" in input_name.lower() and "high definition audio" in output_name.lower():
                print("[!] NOTE: Input and output use the same audio interface (High Definition Audio).")
                print("    Some drivers have internal loopback features that may cause feedback.")
                print()

    except ImportError:
        print("[X] sounddevice is not installed")
        return
    except Exception as e:
        print(f"[X] Error checking audio devices: {e}")
        return

    # Check for common audio software
    print("[*] Windows Audio Settings to Check:")
    print()
    print("  Please verify these settings manually:")
    print()
    print("  1. Disable 'Listen to this device':")
    print("     - Win + R -> mmsys.cpl -> Enter")
    print("     - Select 'Recording' tab")
    print("     - Right-click Microphone -> Properties")
    print("     - In 'Listen' tab, UNCHECK 'Listen to this device'")
    print()
    print("  2. Disable Stereo Mix:")
    print("     - In the same 'Recording' tab")
    print("     - If 'Stereo Mix' exists, right-click -> Disable")
    print()
    print("  3. Realtek Audio Console:")
    print("     - Disable 'Stereo Mix' or 'Loopback' features")
    print()

    # Check for VoiceMeeter or similar
    print("[*] Virtual Audio Device Check:")
    print()

    virtual_audio_keywords = [
        "voicemeeter", "vb-cable", "virtual", "cable input", "cable output",
        "obs", "streamlabs"
    ]

    found_virtual = []
    for i, dev in enumerate(devices):
        name_lower = dev["name"].lower()
        for keyword in virtual_audio_keywords:
            if keyword in name_lower:
                found_virtual.append((i, dev["name"]))
                break

    if found_virtual:
        for i, name in found_virtual:
            print(f"  [!] Virtual audio device detected: [{i}] {name}")
        print()
        print("  Virtual audio devices found.")
        print("  If using VoiceMeeter or similar, check your routing settings.")
        print("  Make sure output is NOT being routed back to input.")
    else:
        print("  No virtual audio devices detected.")
    print()

    # Feedback test suggestion
    print("=" * 60)
    print("[*] Feedback Test:")
    print("=" * 60)
    print()
    print("  1. Start RCWX and set pitch shift to +5")
    print("  2. Start conversion")
    print("  3. Speak into the microphone")
    print("  4. If pitch keeps rising over time (+5, +10, +15...):")
    print("     -> Feedback loop is occurring")
    print("     -> Check the settings above")
    print()
    print("  5. If pitch stays constant (+5 only):")
    print("     -> Working correctly")
    print()


def check_torch_device():
    """Check PyTorch device availability."""
    print("=" * 60)
    print("[*] PyTorch Device Information:")
    print("=" * 60)
    print()

    try:
        import torch

        print(f"  PyTorch Version: {torch.__version__}")

        # XPU
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            print(f"  [OK] XPU Available: {torch.xpu.device_count()} device(s)")
            for i in range(torch.xpu.device_count()):
                props = torch.xpu.get_device_properties(i)
                print(f"       [{i}] {props.name}")
        else:
            print("  [--] XPU Not Available")

        # CUDA
        if torch.cuda.is_available():
            print(f"  [OK] CUDA Available: {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"       [{i}] {props.name}")
        else:
            print("  [--] CUDA Not Available")

        print("  [OK] CPU Available")
        print()

    except ImportError:
        print("  [X] PyTorch is not installed")
    except Exception as e:
        print(f"  [X] Error: {e}")


def main():
    """Run all diagnostics."""
    check_audio_devices()
    check_torch_device()

    print("=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)
    print()
    print("If issues persist, check the log files at:")
    print(f"  {Path.home() / '.config' / 'rcwx' / 'logs'}")
    print()


if __name__ == "__main__":
    main()
