//! F0 後処理チェーンの純 Rust 移植 (f0_processing.py と同一算法)。
//!
//! - short-hole fill (log2 補間)
//! - median spike removal (window 3, voiced のみ)
//! - zero-phase Butterworth 2次 lowpass (log2 輪郭, filtfilt 相当)
//! - octave flip 抑制 (ヒステリス 3fr)
//! - slew 制限 (ヒステリス 3fr)
//! - quantize_f0_to_pitch

pub const F0_CORRECTION_SUSTAIN_FRAMES: usize = 3;

/// short-hole fill: 両側有声の長さ ≤ max_gap の無声穴を log2 線形補間。
pub fn fill_short_unvoiced_gaps(f0: &mut [f32], max_gap_frames: usize) {
    if max_gap_frames == 0 || f0.len() < 3 {
        return;
    }
    let vidx: Vec<usize> = (0..f0.len()).filter(|&i| f0[i] > 0.0).collect();
    if vidx.len() < 2 {
        return;
    }
    for w in 0..vidx.len() - 1 {
        let left = vidx[w];
        let right = vidx[w + 1];
        let n = right - left;
        if n > 1 && n - 1 <= max_gap_frames {
            let lv = f0[left].log2();
            let rv = f0[right].log2();
            for k in 1..n {
                let t = k as f32 / n as f32;
                f0[left + k] = (lv * (1.0 - t) + rv * t).exp2();
            }
        }
    }
}

/// median (window 3) を有声フレームのみに適用。
pub fn smooth_f0_spikes(f0: &mut [f32]) {
    if f0.len() < 3 {
        return;
    }
    let mut out = f0.to_vec();
    for i in 0..f0.len() {
        if f0[i] > 0.0 {
            let (a, b, c) = (
                if i == 0 { f0[0] } else { f0[i - 1] } as f64,
                f0[i] as f64,
                if i + 1 >= f0.len() { f0[i] } else { f0[i + 1] } as f64,
            );
            // 3 要素の中央値
            let med = {
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                if c <= lo {
                    lo
                } else if c >= hi {
                    hi
                } else {
                    c
                }
            };
            out[i] = med as f32;
        }
    }
    f0.copy_from_slice(&out);
}

/// Butterworth 2次 lowpass 係数 (b, a) — scipy.signal.butter(2, Wn, 'low') と厳密一致。
fn butter2_lowpass(cutoff_hz: f64, sample_rate: f64) -> ([f64; 3], [f64; 3]) {
    let nyquist = sample_rate / 2.0;
    let wn = cutoff_hz / nyquist;
    let k = (std::f64::consts::PI * wn * 0.5).tan();
    let k2 = k * k;
    let denom = 1.0 + (2f64).sqrt() * k + k2;
    let b0 = k2 / denom;
    let b1 = 2.0 * b0;
    let b2 = b0;
    let a1 = 2.0 * (k2 - 1.0) / denom;
    let a2 = (1.0 - (2f64).sqrt() * k + k2) / denom;
    ([b0, b1, b2], [1.0, a1, a2])
}

/// filtfilt (zero-phase) — scipy と同様の odd extension padding (3*max(len(a),len(b)))。
fn filtfilt(b: &[f64; 3], a: &[f64; 3], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 4 {
        return x.to_vec();
    }
    let lfilter = |bb: &[f64; 3], aa: &[f64; 3], s: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0f64; s.len()];
        for i in 0..s.len() {
            let mut acc = bb[0] * s[i];
            if i >= 1 {
                acc += bb[1] * s[i - 1] - aa[1] * y[i - 1];
            }
            if i >= 2 {
                acc += bb[2] * s[i - 2] - aa[2] * y[i - 2];
            }
            y[i] = acc;
        }
        y
    };
    // scipy default padlen = 3 * max(len(a), len(b)) = 9 for 2nd order
    let padlen = 9usize.min(n - 1);
    // odd extension: pad[i] = 2*x[0] - x[padlen-i], pad[n+padlen+i] = 2*x[n-1] - x[n-2-i]
    let mut padded = Vec::with_capacity(n + 2 * padlen);
    for i in (1..=padlen).rev() {
        padded.push(2.0 * x[0] - x[i]);
    }
    padded.extend_from_slice(x);
    for i in 1..=padlen {
        let idx = n.saturating_sub(1 + i);
        padded.push(2.0 * x[n - 1] - x[idx]);
    }
    // forward
    let fwd = lfilter(b, a, &padded);
    // reverse
    let rev_in: Vec<f64> = fwd.iter().rev().copied().collect();
    let rev = lfilter(b, a, &rev_in);
    // un-reverse and trim
    (0..n)
        .map(|i| rev[rev.len() - 1 - (i + padlen)])
        .collect()
}

/// zero-phase lowpass を有声 log2 輪郭に適用 (有声 < 10 は素通し)。
pub fn lowpass_f0(f0: &mut [f32], cutoff_hz: f64, sample_rate: f64) {
    if f0.len() < 10 {
        return;
    }
    let n_voiced = f0.iter().filter(|v| **v > 0.0).count();
    if n_voiced <= 10 {
        return;
    }
    let (b, a) = butter2_lowpass(cutoff_hz, sample_rate);
    // log2 輪郭を有声区間で補間
    let n = f0.len();
    let mut contour = vec![0.0f64; n];
    let last_v = 0.0f64;
    let next_v = 0.0f64;
    // np.interp 相当: 有声 index で線形補間
    let vidx: Vec<usize> = (0..n).filter(|&i| f0[i] > 0.0).collect();
    for (k, &i) in vidx.iter().enumerate() {
        let _ = k;
        contour[i] = f0[i].log2() as f64;
    }
    // 先頭/末尾の無声は端の有声値で埋め、途中は線形補間
    for i in 0..n {
        if f0[i] > 0.0 {
            continue;
        }
        // 左の有声
        let li = vidx.iter().rfind(|&&v| v < i);
        let ri = vidx.iter().find(|&&v| v > i);
        match (li, ri) {
            (Some(&l), Some(&r)) => {
                let t = (i - l) as f64 / (r - l) as f64;
                contour[i] = contour[l] * (1.0 - t) + contour[r] * t;
            }
            (Some(&l), None) => contour[i] = contour[l],
            (None, Some(&r)) => contour[i] = contour[r],
            (None, None) => {}
        }
    }
    let _ = (last_v, next_v);
    let filtered = filtfilt(&b, &a, &contour);
    for i in 0..n {
        f0[i] = if f0[i] > 0.0 {
            (2.0f64).powf(filtered[i]) as f32
        } else {
            0.0
        };
    }
}

/// オクターブフリップ抑制 (ratio 2±0.16、持続 3fr で raw 復帰)。
pub fn suppress_octave_flips(f0: &mut [f32]) {
    if f0.len() < 2 {
        return;
    }
    let raw = f0.to_vec();
    let low = 2.0 - 0.16;
    let high = 2.0 + 0.16;
    let inv_low = 1.0 / high;
    let inv_high = 1.0 / low;
    let mut streak_dir: i32 = 0;
    let mut streak_start = 0usize;
    let mut streak_len = 0usize;
    for i in 1..f0.len() {
        let prev = f0[i - 1] as f64;
        let cur = raw[i] as f64;
        if prev <= 0.0 || cur <= 0.0 {
            streak_dir = 0;
            streak_len = 0;
            continue;
        }
        let ratio = cur / prev.max(1e-6);
        let direction: i32;
        if (low..=high).contains(&ratio) {
            direction = 1;
            f0[i] = (cur * 0.5) as f32;
        } else if (inv_low..=inv_high).contains(&ratio) {
            direction = -1;
            f0[i] = (cur * 2.0) as f32;
        } else {
            streak_dir = 0;
            streak_len = 0;
            continue;
        }
        if direction == streak_dir {
            streak_len += 1;
        } else {
            streak_dir = direction;
            streak_start = i;
            streak_len = 1;
        }
        if streak_len >= F0_CORRECTION_SUSTAIN_FRAMES {
            for k in streak_start..=i {
                f0[k] = raw[k];
            }
            streak_dir = 0;
            streak_len = 0;
        }
    }
}

/// slew 制限 (max_step_st 半音、持続 3fr で raw 復帰)。
pub fn limit_f0_slew(f0: &mut [f32], max_step_st: f64) {
    if f0.len() < 2 {
        return;
    }
    let raw = f0.to_vec();
    let max_ratio = 2.0f64.powf(max_step_st / 12.0);
    let mut streak_dir: i32 = 0;
    let mut streak_start = 0usize;
    let mut streak_len = 0usize;
    for i in 1..f0.len() {
        let prev = f0[i - 1] as f64;
        let cur = raw[i] as f64;
        if prev <= 0.0 || cur <= 0.0 {
            streak_dir = 0;
            streak_len = 0;
            continue;
        }
        let upper = prev * max_ratio;
        let lower = prev / max_ratio;
        let direction: i32;
        if cur > upper {
            direction = 1;
            f0[i] = upper as f32;
        } else if cur < lower {
            direction = -1;
            f0[i] = lower as f32;
        } else {
            f0[i] = cur as f32;
            streak_dir = 0;
            streak_len = 0;
            continue;
        }
        if direction == streak_dir {
            streak_len += 1;
        } else {
            streak_dir = direction;
            streak_start = i;
            streak_len = 1;
        }
        if streak_len >= F0_CORRECTION_SUSTAIN_FRAMES {
            for k in streak_start..=i {
                f0[k] = raw[k];
            }
            streak_dir = 0;
            streak_len = 0;
        }
    }
}

/// チェーン適用 (apply_f0_filter_chain 相当)。
#[allow(clippy::too_many_arguments)]
pub fn apply_f0_filter_chain(
    f0: &mut [f32],
    f0_lowpass_cutoff_hz: f64,
    enable_octave_flip_suppress: bool,
    enable_f0_slew_limit: bool,
    f0_slew_max_step_st: f64,
    f0_hole_fill_ms: f64,
) {
    if f0.is_empty() {
        return;
    }
    let max_gap = (f0_hole_fill_ms / 10.0).round() as usize;
    fill_short_unvoiced_gaps(f0, max_gap);
    smooth_f0_spikes(f0);
    lowpass_f0(f0, f0_lowpass_cutoff_hz, 100.0);
    if enable_octave_flip_suppress {
        suppress_octave_flips(f0);
    }
    if enable_f0_slew_limit {
        limit_f0_slew(f0, f0_slew_max_step_st);
    }
}

/// quantize_f0_to_pitch: mel 量子化 1..255 (有声)、無声は 1。
pub fn quantize_f0_to_pitch(f0: &[f32]) -> Vec<u32> {
    let f0_mel_min: f64 = 1127.0 * (1.0f64 + 50.0f64 / 700.0).ln();
    let f0_mel_max: f64 = 1127.0 * (1.0f64 + 1100.0f64 / 700.0).ln();
    f0.iter()
        .map(|&v| {
            let mel: f64 = 1127.0f64 * (1.0f64 + v as f64 / 700.0f64).ln();
            let norm = if mel > 0.0 {
                (mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0
            } else {
                0.0
            };
            norm.clamp(1.0, 255.0).round() as u32
        })
        .collect()
}
