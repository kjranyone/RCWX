//! リサンプル: scipy.signal.resample_poly との厳密一致ポリフェーズ実装 + ストリーミング。
//!
//! - `resample_poly`: firwin(kaiser β=5) × up 増幅 + upfirdn + 中心合わせ trim
//! - `StatefulResampler`: overlap-save によるチャンク連続処理 (rcwx/audio/resample.py 相当)
//! - `resample_linear`: linspace 線形補間 (Python 側 linear 方式相当)

/// Kaiser 窓 (β) の i0 (第1種変形ベッセル関数)。
fn bessel_i0(x: f64) -> f64 {
    // 級数展開 (scipy.special.i0 と double 精度で一致)
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half = x / 2.0;
    for k in 1..64 {
        term *= (half / k as f64) * (half / k as f64);
        let next = sum + term;
        if next == sum {
            return sum;
        }
        sum = next;
    }
    sum
}

/// scipy.signal.firwin(numtaps, fc, window=('kaiser', beta)) (lowpass, scale=True) 相当。
fn firwin_kaiser(numtaps: usize, fc: f64, beta: f64) -> Vec<f64> {
    let alpha = (numtaps as f64 - 1.0) / 2.0;
    let mut h = vec![0.0f64; numtaps];
    let mut sum = 0.0f64;
    for (n, slot) in h.iter_mut().enumerate() {
        let m = n as f64 - alpha;
        let sinc = if m == 0.0 {
            1.0
        } else {
            (fc * m * std::f64::consts::PI).sin() / (fc * m * std::f64::consts::PI)
        };
        let w = bessel_i0(
            beta * (1.0 - (2.0 * n as f64 / (numtaps as f64 - 1.0) - 1.0).powi(2)).sqrt(),
        ) / bessel_i0(beta);
        let v = sinc * w;
        *slot = v;
        sum += v;
    }
    for v in h.iter_mut() {
        *v /= sum;
    }
    h
}

/// upfirdn 出力長: ((in_len-1)*up + len_h - 1)//down + 1
fn output_len(len_h: usize, in_len: usize, up: usize, down: usize) -> usize {
    (((in_len - 1) * up + len_h) - 1) / down + 1
}

/// scipy.signal.upfirdn(h, x, up, down) (mode=constant, cval=0) 相当。f64 演算。
fn upfirdn(h: &[f64], x: &[f64], up: usize, down: usize) -> Vec<f64> {
    let n_out = output_len(h.len(), x.len(), up, down);
    let mut y = vec![0.0f64; n_out];
    // y[j] = sum_k h[k] * xu[j*down - k], xu[i] = x[i/up] (i%up==0)
    // k = j*down - i*up in [0, len_h) → i = (j*down - k)/up
    let len_h = h.len();
    for (j, slot) in y.iter_mut().enumerate() {
        let pos = j * down;
        // i_min: k = pos - i*up <= len_h-1 → i >= (pos - len_h + 1)/up
        let i_min = pos.saturating_sub(len_h - 1).div_ceil(up);
        let i_max = pos / up; // k = pos - i*up >= 0
        let mut acc = 0.0f64;
        for i in i_min..=i_max.min(x.len().saturating_sub(1)) {
            let k = pos - i * up;
            if k < len_h {
                acc += h[k] * x[i];
            }
        }
        *slot = acc;
    }
    y
}

/// scipy.signal.resample_poly(x, up, down) 相当 (padtype 既定 constant/zero)。
pub fn resample_poly(x: &[f32], up: usize, down: usize) -> Vec<f32> {
    let g = gcd(up, down);
    let (up, down) = (up / g, down / g);
    if up == 1 && down == 1 {
        return x.to_vec();
    }
    let n_in = x.len();
    let n_out = n_in * up / down + usize::from(n_in * up % down != 0);

    let max_rate = up.max(down);
    let f_c = 1.0 / max_rate as f64;
    let half_len = 10 * max_rate;
    let mut h = firwin_kaiser(2 * half_len + 1, f_c, 5.0);
    for v in h.iter_mut() {
        *v *= up as f64;
    }

    let n_pre_pad = down - half_len % down;
    let mut n_post_pad = 0usize;
    let n_pre_remove = (half_len + n_pre_pad) / down;
    while output_len(h.len() + n_pre_pad + n_post_pad, n_in, up, down) < n_out + n_pre_remove {
        n_post_pad += 1;
    }
    let mut h_pad = vec![0.0f64; n_pre_pad];
    h_pad.extend_from_slice(&h);
    h_pad.extend(std::iter::repeat_n(0.0, n_post_pad));

    let xd: Vec<f64> = x.iter().map(|v| *v as f64).collect();
    let y = upfirdn(&h_pad, &xd, up, down);
    let end = (n_pre_remove + n_out).min(y.len());
    y[n_pre_remove..end].iter().map(|v| *v as f32).collect()
}

fn gcd(a: usize, b: usize) -> usize {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Python `resample(len→ratio)` 相当の線形補間 (linspace + interp)。
pub fn resample_linear(x: &[f32], orig_sr: u32, target_sr: u32) -> Vec<f32> {
    if orig_sr == target_sr || x.is_empty() {
        return x.to_vec();
    }
    let ratio = target_sr as f64 / orig_sr as f64;
    let target_len = (x.len() as f64 * ratio) as usize;
    if target_len == 0 {
        return Vec::new();
    }
    // np.linspace(0, len-1, target_len) + np.interp
    let n = x.len();
    let mut out = vec![0.0f32; target_len];
    if target_len == 1 {
        out[0] = x[0];
        return out;
    }
    let step = (n - 1) as f64 / (target_len - 1) as f64;
    for (i, slot) in out.iter_mut().enumerate() {
        let pos = i as f64 * step;
        let i0 = pos.floor() as usize;
        let i1 = (i0 + 1).min(n - 1);
        let fr = pos - i0 as f64;
        *slot = x[i0] + (x[i1] - x[i0]) * fr as f32;
    }
    out
}

/// overlap-save 方式のステートフル・リサンプラ (rcwx.audio.resample.StatefulResampler 相当)。
pub struct StatefulResampler {
    orig_sr: u32,
    target_sr: u32,
    up: usize,
    down: usize,
    overlap_samples: usize,
    overlap_buffer: Option<Vec<f32>>,
}

impl StatefulResampler {
    pub fn new(orig_sr: u32, target_sr: u32) -> Self {
        let g = gcd(orig_sr as usize, target_sr as usize);
        let up = target_sr as usize / g;
        let down = orig_sr as usize / g;
        let filter_half_input = 10 * up.max(down) / up + 1;
        let overlap_samples = 20.max(2 * filter_half_input);
        Self {
            orig_sr,
            target_sr,
            up,
            down,
            overlap_samples,
            overlap_buffer: None,
        }
    }

    pub fn resample_chunk(&mut self, chunk: &[f32]) -> Vec<f32> {
        if self.orig_sr == self.target_sr {
            return chunk.to_vec();
        }
        let actual_overlap = self.overlap_samples.min(chunk.len());
        let (out, tail) = match &self.overlap_buffer {
            None => {
                let mut extended = chunk.to_vec();
                extended.extend(std::iter::repeat_n(0.0f32, actual_overlap));
                let resampled = resample_poly(&extended, self.up, self.down);
                let expected = chunk.len() * self.up / self.down;
                (
                    resampled[..expected.min(resampled.len())].to_vec(),
                    chunk[chunk.len() - actual_overlap..].to_vec(),
                )
            }
            Some(buf) => {
                let mut extended = buf.clone();
                extended.extend_from_slice(chunk);
                let resampled = resample_poly(&extended, self.up, self.down);
                let overlap_out = buf.len() * self.up / self.down;
                let expected = chunk.len() * self.up / self.down;
                let end = (overlap_out + expected).min(resampled.len());
                (
                    resampled
                        .get(overlap_out..end)
                        .map(|s| s.to_vec())
                        .unwrap_or_default(),
                    chunk[chunk.len() - actual_overlap..].to_vec(),
                )
            }
        };
        self.overlap_buffer = Some(tail);
        out
    }

    pub fn reset(&mut self) {
        self.overlap_buffer = None;
    }
}
