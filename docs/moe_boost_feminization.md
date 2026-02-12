# Moe Boost Feminization: Design Note

## Problem Statement

`moe_boost` should push low-register male F0 trajectories toward a brighter, game-style feminine contour while preserving intelligibility and avoiding harsh artifacts.

The key constraints are:
- no retraining,
- real-time compatibility,
- compatibility with existing RVC synthesis and post-filters.

## Signal Model

Let voiced F0 be `f(t)` in Hz and `l(t)=log2(f(t))`.

We decompose F0 into:
- phrase trend `m(t)` (low-frequency baseline),
- deviation `d(t)` in semitones.

`d(t) = 12 * (l(t) - m(t))`

This gives a controllable representation for:
- register shift (move baseline up),
- accent shaping (asymmetric gain for rising/falling motion),
- floor control (remove chesty low dips).

## Proposed Transform

For a given strength `s in [0, 1]`:

1. Short-gap interpolation:
- fill unvoiced gaps up to `2 + 6s` frames (100 fps basis).

2. Register target:
- target median: `f_target = 165 + 75s` Hz,
- upward shift only:
  `shift_st = clamp(12*log2(f_target / median(f_voiced)), 0, 3 + 9s)`.

3. Contour shaping:
- trend window: `w = odd(max(7, 7 + 14s))`,
- asymmetric gain:
  - upward: `1 + 0.80s`,
  - downward: `1 - 0.30s`.
- phrase bias: `0.20 + 0.90s` semitones.
- soft saturation for stability:
  `d <- d / (1 + (0.10 + 0.20s)*|d|)`.

4. Reconstruction:
- `l_out = m + (d + shift_st + bias_st) / 12`,
- `f_out = 2^(l_out)`.

5. Floor/ceiling constraints:
- relative floor: `f_target * (0.58 + 0.10s)`,
- absolute floor: `85 + 70s` Hz,
- applied floor: `max(relative, absolute)` clipped to `[85, 260]`,
- final clip to `[0, 940]` Hz.

## Why This Works

- Register normalization addresses the largest male/female perceptual gap directly.
- Asymmetric contour shaping boosts "cute" rising accents while reducing downward chest drops.
- Gap interpolation reduces raspy frame flicker often perceived as "older" timbre.
- Existing RCWX post-filters (`lowpass_f0`, octave-flip suppression, slew limit) then stabilize the stylized contour.

## Limitations

This is still an F0-only style transform.
It cannot directly perform formant relocation (vocal-tract resonance shift), which is another major component of true male-to-female conversion. For that, feature-space warping or model-level adaptation is required.

## Validation Added

`tests/models/test_moe_f0_style.py` validates:
- identity at `s=0`,
- male register lift and low-floor lift at high strength,
- short-gap fill while keeping long gaps,
- bounded behavior on already-high voices,
- monotonic register lift with strength.

`tests/integration/test_realtime_drift_control.py` validates runtime drift-control behavior introduced with recent callback-level fixes.
