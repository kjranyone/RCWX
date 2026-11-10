//! フィードバック検出用の最大正規化遅延相互相関
//! (rcwx/pipeline/realtime_unified.py _max_normalized_lag_correlation 相当)。

/// FFT による高速版 (rustfft)。needle/hay 変換は毎回行う単発ユース向け。
/// スカラー版と同じ値 (f64 演算、窓ノルムは累積和) を O(L log L) で返す。
pub fn max_normalized_lag_correlation_fft(needle: &[f32], haystack: &[f32]) -> f64 {
    use rustfft::num_complex::Complex;
    let n = needle.len();
    if n < 8 || haystack.len() <= n {
        return 0.0;
    }
    let m = haystack.len();
    let l = (m).next_power_of_two();

    // needle 中心化
    let needle_mean = needle.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    let nc: Vec<f64> = needle.iter().map(|v| *v as f64 - needle_mean).collect();
    let needle_norm = nc.iter().map(|v| v * v).sum::<f64>().sqrt();
    if needle_norm < 1e-8 {
        return 0.0;
    }

    // 相関: dots[j] = Σ_i hay[j+i]·nc[i] — FFT(hay) × conj(FFT(nc)) の巡回相関
    let mut planner = rustfft::FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(l);
    let mut fh: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); l];
    for (i, v) in haystack.iter().enumerate() {
        fh[i].re = *v as f64;
    }
    fft.process(&mut fh);
    let mut fn_: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); l];
    for (i, v) in nc.iter().enumerate() {
        fn_[i].re = *v;
    }
    fft.process(&mut fn_);
    for i in 0..l {
        // conj(needle spectrum) を掛ける (正規化は IFFT 後に 1/l)
        let c = Complex::new(fn_[i].re, -fn_[i].im);
        fh[i] = fh[i] * c;
    }
    let ifft = planner.plan_fft_inverse(l);
    ifft.process(&mut fh);
    let scale = 1.0 / l as f64;

    // 窓ノルム (累積和)
    let mut cum = vec![0.0f64; m + 1];
    let mut cum2 = vec![0.0f64; m + 1];
    for i in 0..m {
        let v = haystack[i] as f64;
        cum[i + 1] = cum[i] + v;
        cum2[i + 1] = cum2[i] + v * v;
    }
    let shifts = m - n;
    let mut best = 0.0f64;
    for j in 0..=shifts {
        let dot = fh[j].re * scale;
        let ws = cum[j + n] - cum[j];
        let wss = cum2[j + n] - cum2[j];
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
