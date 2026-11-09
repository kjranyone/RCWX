//! F0 のみの moe スタイリング (rcwx/pipeline/f0_processing.py apply_moe_f0_style 相当)。
//!
//! 低音域輪郭の引き上げ・上向きアクセント強調・短い無声穴の補間。
//! strength は係数の強さ、confidence は styled→raw のブレンド比 (別軸)。

const MAX_MOE_BOOST: f64 = 1.0;

/// 短い無声穴の線形補間。
fn fill_short_unvoiced_gaps(row: &mut [f64], max_gap_frames: usize) {
    if max_gap_frames == 0 || row.is_empty() {
        return;
    }
    let n = row.len();
    let mut i = 0;
    while i < n {
        if row[i] > 0.0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && row[i] <= 0.0 {
            i += 1;
        }
        let end = i;
        let gap = end - start;
        if gap > 0
            && gap <= max_gap_frames
            && start > 0
            && end < n
            && row[start - 1] > 0.0
            && row[end] > 0.0
        {
            let left = row[start - 1];
            let right = row[end];
            for (k, slot) in row[start..end].iter_mut().enumerate() {
                // torch.linspace(0,1,gap+2)[1:-1] → w = (k+1)/(gap+1)
                let w = (k + 1) as f64 / (gap + 1) as f64;
                *slot = left * (1.0 - w) + right * w;
            }
        }
    }
}

/// torch.median: 偶数長は下側中央値。
fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        v[n / 2 - 1]
    }
}

/// F0 (Hz, 無声=0) に moe スタイルを適用。入力を破壊的に更新して返す。
pub fn apply_moe_f0_style(f0: &mut [f32], strength: f64) {
    let strength = strength.clamp(0.0, MAX_MOE_BOOST);
    if strength <= 0.0 || f0.is_empty() {
        return;
    }

    let max_gap_frames = (2.0 + 4.0 * strength).round() as usize;
    let target_median_hz = 165.0 + 55.0 * strength;
    let max_up_shift_st = 1.5 + 4.5 * strength;
    let up_gain = 1.0 + 0.45 * strength;
    let down_gain = 1.0 - 0.25 * strength;
    let phrase_bias_st = 0.10 + 0.45 * strength;
    let sat_coef = 0.08 + 0.16 * strength;
    let mut window = (7.0 + 14.0 * strength) as usize;
    if window % 2 == 0 {
        window += 1;
    }
    let floor_rel = target_median_hz * (0.55 + 0.08 * strength);
    let floor_abs = 85.0 + 45.0 * strength;
    let floor_hz = floor_rel.max(floor_abs).clamp(85.0, 220.0);
    let max_dev_st = 22.0;

    let mut row: Vec<f64> = f0.iter().map(|v| *v as f64).collect();
    fill_short_unvoiced_gaps(&mut row, max_gap_frames);

    let voiced_idx: Vec<usize> = row.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, _)| i).collect();
    let n_voiced = voiced_idx.len();
    if n_voiced < 6 {
        for (i, v) in row.iter().enumerate() {
            f0[i] = *v as f32;
        }
        return;
    }

    let voiced_ratio = n_voiced as f64 / row.len().max(1) as f64;
    let confidence = ((voiced_ratio - 0.12) / 0.50).clamp(0.0, 1.0);
    if confidence <= 1e-4 {
        for (i, v) in row.iter().enumerate() {
            f0[i] = *v as f32;
        }
        return;
    }

    let raw_voiced: Vec<f64> = voiced_idx.iter().map(|&i| row[i].clamp(1e-5, 1400.0)).collect();
    let log2_f0: Vec<f64> = raw_voiced.iter().map(|v| v.log2()).collect();

    // ローカルトレンド: replicate pad + avg_pool1d(window)
    let half = window / 2;
    let nv = log2_f0.len();
    let trend: Vec<f64> = (0..nv)
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half).min(nv - 1);
            // replicate パディング相当: 範囲外は端値
            let mut sum = 0.0;
            for j in lo..=hi {
                sum += log2_f0[j];
            }
            // 端部: 窓の外側は端値で埋める (avg_pool1d の replicate pad)
            let pad_left = half.saturating_sub(i);
            let pad_right = half.saturating_sub(nv - 1 - i);
            (sum + log2_f0[lo] * pad_left as f64 + log2_f0[hi] * pad_right as f64) / window as f64
        })
        .collect();

    let mut med = raw_voiced.clone();
    let median_hz = median(&mut med);
    let raw_shift_st = 12.0 * (target_median_hz / median_hz.max(1e-5)).log2();
    let reg_shift_st = raw_shift_st.clamp(0.0, max_up_shift_st);

    for (k, &i) in voiced_idx.iter().enumerate() {
        let mut dev_st = (log2_f0[k] - trend[k]) * 12.0;
        dev_st = if dev_st >= 0.0 { dev_st * up_gain } else { dev_st * down_gain };
        let sat = 1.0 + sat_coef * dev_st.abs();
        dev_st = (dev_st / sat).clamp(-max_dev_st, max_dev_st);

        let shaped_log2 = trend[k] + (dev_st + reg_shift_st + phrase_bias_st) / 12.0;
        let mut shaped = shaped_log2.exp2();
        shaped = shaped.max(floor_hz).clamp(0.0, 940.0);

        row[i] = raw_voiced[k] * (1.0 - confidence) + shaped * confidence;
    }

    for (i, v) in row.iter().enumerate() {
        f0[i] = *v as f32;
    }
}
