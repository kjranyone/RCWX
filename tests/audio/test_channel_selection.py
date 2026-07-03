"""Channel selection normalization / routing / feedback detection tests.

No model or audio device required.  Covers the GUI display-string leak that
routed ASIO LOOPBACK output back into the input (digital-delay-like feedback
bug):

1. config normalizers accept both canonical and display strings
2. AudioConfig repairs corrupted (display-string) values on load
3. parse_output_channel_pair rejects display strings safely
4. select_channel: explicit index wins; unparseable falls back to auto
   WITH loopback exclusion honored
5. _max_normalized_lag_correlation detects delayed copies (feedback) that
   the old zero-lag check could never see
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rcwx.audio.input import _auto_select_channel, select_channel
from rcwx.audio.stream_base import parse_output_channel_pair
from rcwx.config import (
    AudioConfig,
    RCWXConfig,
    normalize_input_channel_selection,
    normalize_output_channel_selection,
)
from rcwx.pipeline.realtime_unified import _max_normalized_lag_correlation


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------

def test_normalize_input_canonical_passthrough():
    for v in ["auto", "average", "left", "right", "0", "3", "11"]:
        assert normalize_input_channel_selection(v) == v, v


def test_normalize_input_display_strings():
    cases = {
        "自動": "auto",
        "平均": "average",
        "Ch 1": "0",
        "Ch 3": "2",
        # Exact corrupted value observed in a real saved config
        "Ch 1: MIC/LINE/INST 1": "0",
        "Ch 12: LOOPBACK Right": "11",
    }
    for display, expected in cases.items():
        got = normalize_input_channel_selection(display)
        assert got == expected, f"{display!r}: expected {expected!r}, got {got!r}"


def test_normalize_input_garbage_falls_back_to_auto():
    for v in ["", "???", "channel one", "-1x"]:
        assert normalize_input_channel_selection(v) == "auto", v


def test_normalize_none_and_empty_are_auto():
    # None / empty / whitespace are "unset" and must normalize to auto
    for v in [None, "", "   ", "\t"]:
        assert normalize_input_channel_selection(v) == "auto", repr(v)
        assert normalize_output_channel_selection(v) == "auto", repr(v)


def test_normalize_output_canonical_passthrough():
    assert normalize_output_channel_selection("auto") == "auto"
    assert normalize_output_channel_selection("0,1") == "0,1"
    assert normalize_output_channel_selection("2, 3") == "2,3"


def test_normalize_output_display_strings():
    cases = {
        # Exact corrupted value observed in a real saved config
        "自動 (Ch 1-2)": "auto",
        "Ch 1-2": "0,1",
        "Ch 3-4": "2,3",
        "Ch 1-2: Main L / Main R": "0,1",
    }
    for display, expected in cases.items():
        got = normalize_output_channel_selection(display)
        assert got == expected, f"{display!r}: expected {expected!r}, got {got!r}"


def test_normalize_output_garbage_falls_back_to_auto():
    for v in ["", "???", "1-2-3", "Ch"]:
        assert normalize_output_channel_selection(v) == "auto", v


# ---------------------------------------------------------------------------
# AudioConfig repair on construction / load
# ---------------------------------------------------------------------------

def test_audio_config_repairs_display_strings():
    cfg = AudioConfig(
        input_channel_selection="Ch 1: MIC/LINE/INST 1",
        output_channel_selection="自動 (Ch 1-2)",
    )
    assert cfg.input_channel_selection == "0"
    assert cfg.output_channel_selection == "auto"


def test_config_load_repairs_corrupted_json():
    """A config.json saved by the buggy GUI must load with canonical values."""
    data = {
        "audio": {
            "input_channel_selection": "Ch 1: MIC/LINE/INST 1",
            "output_channel_selection": "自動 (Ch 1-2)",
        }
    }
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "config.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        cfg = RCWXConfig.load(p)
    assert cfg.audio.input_channel_selection == "0"
    assert cfg.audio.output_channel_selection == "auto"


# ---------------------------------------------------------------------------
# parse_output_channel_pair
# ---------------------------------------------------------------------------

def test_parse_output_channel_pair():
    assert parse_output_channel_pair("auto") is None
    assert parse_output_channel_pair("0,1") == (0, 1)
    assert parse_output_channel_pair("2,3") == (2, 3)
    # Display strings and garbage must NOT parse (callers treat None as auto)
    assert parse_output_channel_pair("自動 (Ch 1-2)") is None
    assert parse_output_channel_pair("Ch 1-2") is None
    assert parse_output_channel_pair("x,y") is None


# ---------------------------------------------------------------------------
# select_channel / _auto_select_channel
# ---------------------------------------------------------------------------

def _make_multichannel(frames: int = 4800) -> np.ndarray:
    """4ch input: ch0 = quiet mic, ch3 = loud 'loopback', others silent."""
    rng = np.random.default_rng(0)
    data = np.zeros((frames, 4), dtype=np.float32)
    data[:, 0] = rng.standard_normal(frames).astype(np.float32) * 0.05  # mic
    data[:, 3] = rng.standard_normal(frames).astype(np.float32) * 0.5  # loopback
    return data


def test_select_channel_explicit_index():
    data = _make_multichannel()
    out = select_channel(data, "0")
    assert np.array_equal(out, data[:, 0])


def test_auto_picks_loudest_without_exclusion():
    """Documents the dangerous default: auto picks the loud loopback ch."""
    data = _make_multichannel()
    out = _auto_select_channel(data)
    assert np.array_equal(out, data[:, 3])


def test_auto_respects_loopback_exclusion():
    data = _make_multichannel()
    out = _auto_select_channel(data, exclude_channels=frozenset({3}))
    assert not np.array_equal(out, data[:, 3])


def test_unparseable_selection_falls_back_to_auto_with_exclusion():
    """Regression: the display-string leak must no longer reach loopback."""
    data = _make_multichannel()
    out = select_channel(
        data, "Ch 1: MIC/LINE/INST 1", exclude_channels=frozenset({3})
    )
    assert not np.array_equal(out, data[:, 3])


def test_average_respects_exclusion():
    data = _make_multichannel()
    out = select_channel(data, "average", exclude_channels=frozenset({3}))
    expected = np.mean(data[:, [0, 1, 2]], axis=1).astype(np.float32)
    assert np.allclose(out, expected)


# ---------------------------------------------------------------------------
# Feedback detection (lag-scanning correlation)
# ---------------------------------------------------------------------------

def test_lag_correlation_detects_delayed_copy():
    """Feedback = input is a delayed copy of played output.  The old zero-lag
    check returned ~0 here; the lag scan must return ~1."""
    rng = np.random.default_rng(1)
    history = rng.standard_normal(11025).astype(np.float32) * 0.1
    delay = 5000  # ~450ms at 11025Hz — typical E2E latency
    needle = history[delay : delay + 2400].copy()
    corr = _max_normalized_lag_correlation(needle, history)
    assert corr > 0.9, f"Expected near-1 correlation for delayed copy, got {corr:.3f}"


def test_lag_correlation_low_for_unrelated_signals():
    rng = np.random.default_rng(2)
    history = rng.standard_normal(11025).astype(np.float32) * 0.1
    needle = rng.standard_normal(2400).astype(np.float32) * 0.1
    corr = _max_normalized_lag_correlation(needle, history)
    assert corr < 0.3, f"Expected low correlation for unrelated signals, got {corr:.3f}"


def test_lag_correlation_detects_attenuated_copy():
    """Feedback survives gain changes (normalized correlation)."""
    rng = np.random.default_rng(3)
    history = rng.standard_normal(11025).astype(np.float32) * 0.1
    needle = history[3000:5400] * 0.25  # attenuated repeat
    corr = _max_normalized_lag_correlation(needle, history)
    assert corr > 0.9, f"Expected near-1 correlation for attenuated copy, got {corr:.3f}"


def test_lag_correlation_degenerate_inputs():
    assert _max_normalized_lag_correlation(np.zeros(100, np.float32), np.zeros(1000, np.float32)) == 0.0
    assert _max_normalized_lag_correlation(np.ones(4, np.float32), np.ones(1000, np.float32)) == 0.0
    # needle longer than haystack
    assert _max_normalized_lag_correlation(np.ones(100, np.float32), np.ones(50, np.float32)) == 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_normalize_input_canonical_passthrough,
        test_normalize_input_display_strings,
        test_normalize_input_garbage_falls_back_to_auto,
        test_normalize_none_and_empty_are_auto,
        test_normalize_output_canonical_passthrough,
        test_normalize_output_display_strings,
        test_normalize_output_garbage_falls_back_to_auto,
        test_audio_config_repairs_display_strings,
        test_config_load_repairs_corrupted_json,
        test_parse_output_channel_pair,
        test_select_channel_explicit_index,
        test_auto_picks_loudest_without_exclusion,
        test_auto_respects_loopback_exclusion,
        test_unparseable_selection_falls_back_to_auto_with_exclusion,
        test_average_respects_exclusion,
        test_lag_correlation_detects_delayed_copy,
        test_lag_correlation_low_for_unrelated_signals,
        test_lag_correlation_detects_attenuated_copy,
        test_lag_correlation_degenerate_inputs,
    ]
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            print(f"Running {name}...")
            t()
            print("  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
