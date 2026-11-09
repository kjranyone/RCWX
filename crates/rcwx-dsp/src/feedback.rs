//! フィードバック検出用の最大正規化遅延相互相関
//! (rcwx/pipeline/realtime_unified.py _max_normalized_lag_correlation 相当)。

/// needle が haystack のどの遅延位置で最も一致するか。戻り値 [0,1]。
pub fn max_normalized_lag_correlation(needle: &[f32], haystack: &[f32]) -> f64 {
    let n = needle.len();
    if n < 8 || haystack.len() <= n {
        return 0.0;
    }

    let needle_mean = needle.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    let needle_centered: Vec<f64> = needle.iter().map(|v| *v as f64 - needle_mean).collect();
    let needle_norm = needle_centered.iter().map(|v| v * v).sum::<f64>().sqrt();
    if needle_norm < 1e-8 {
        return 0.0;
    }

    let hl = haystack.len();
    let mut cumsum = vec![0.0f64; hl + 1];
    let mut cumsum_sq = vec![0.0f64; hl + 1];
    for i in 0..hl {
        let v = haystack[i] as f64;
        cumsum[i + 1] = cumsum[i] + v;
        cumsum_sq[i + 1] = cumsum_sq[i] + v * v;
    }

    let mut best = 0.0f64;
    for shift in 0..=(hl - n) {
        let mut dot = 0.0f64;
        let base = &haystack[shift..shift + n];
        for (w, &nc) in base.iter().zip(&needle_centered) {
            dot += *w as f64 * nc;
        }
        let ws = cumsum[shift + n] - cumsum[shift];
        let wss = cumsum_sq[shift + n] - cumsum_sq[shift];
        let norms_sq = (wss - ws * ws / n as f64).max(0.0);
        let norm = norms_sq.sqrt();
        if norm > 1e-8 {
            let c = (dot / (needle_norm * norm)).abs();
            if c > best {
                best = c;
            }
        }
    }
    best
}
