//! 微分可能 STFT / mel スペクトログラム (損失計算用)。
//! rDFT 基底・mel 基底は外部 (golden / 前処理) で生成した同一のものを使う。

use rcwx_nn::engine::GpuTensor;
use rcwx_nn::engine::tape::{Tape, Var};

pub struct StftMel {
    pub n_fft: u32,
    pub hop: u32,
    pub basis_re: GpuTensor,  // [bins, n_fft]
    pub basis_im: GpuTensor,  // [bins, n_fft]
    pub mel_basis: GpuTensor, // [n_mels, bins]
    pub hann: GpuTensor,      // [n_fft]
    pub n_mels: u32,
}

impl StftMel {
    /// Rust 側で基底を生成 (n_fft=2048/hop=480/n_mels=128・Slaney mel)。
    /// golden 比較時は Python 側で生成した同一基底を差し込む。
    pub fn new_rust(
        eng: &rcwx_nn::engine::Engine,
        n_fft: u32,
        hop: u32,
        n_mels: u32,
        sr: u32,
    ) -> Self {
        let bins = n_fft / 2 + 1;
        let mut re = vec![0.0f32; (bins * n_fft) as usize];
        let mut im = vec![0.0f32; (bins * n_fft) as usize];
        for f in 0..bins {
            for k in 0..n_fft {
                let ang = -2.0 * std::f32::consts::PI * (f as f32) * (k as f32) / n_fft as f32;
                re[(f * n_fft + k) as usize] = ang.cos();
                im[(f * n_fft + k) as usize] = ang.sin();
            }
        }
        let mel = slaney_mel(n_mels, bins, sr, n_fft);
        let mut hann = vec![0.0f32; n_fft as usize];
        for k in 0..n_fft {
            hann[k as usize] =
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / n_fft as f32).cos();
        }
        Self {
            n_fft,
            hop,
            basis_re: eng.tensor_from_f32(&re, vec![bins as usize, n_fft as usize]),
            basis_im: eng.tensor_from_f32(&im, vec![bins as usize, n_fft as usize]),
            mel_basis: eng.tensor_from_f32(&mel, vec![n_mels as usize, bins as usize]),
            hann: eng.tensor_from_f32(&hann, vec![n_fft as usize]),
            n_mels,
        }
    }

    /// y [B·samples] (平坦 [B, 1, samples]) → log-mel [B·n_mels·frames]。
    /// torch.stft(center=False) + 両側 reflect pad (n_fft−hop)/2 相当。
    pub fn logmel(&self, tape: &mut Tape<'_>, y: Var, batch: u32, samples: u32) -> Var {
        let pad = (self.n_fft - self.hop) / 2;
        let y = tape.reflect_pad(y, pad);
        let padded = samples + 2 * pad;
        let n_frames = (padded.saturating_sub(self.n_fft)) / self.hop + 1;
        let frames = tape.im2col_frames(y, self.n_fft, self.hop, batch, padded, n_frames);
        let hann = tape.const_tensor(&self.hann, vec![self.n_fft as usize]);
        let frames = tape.mul_window(frames, hann, self.n_fft, n_frames);
        let bins = self.basis_re.len / self.n_fft as usize;
        let re = {
            let w = tape.const_tensor(&self.basis_re, vec![bins, self.n_fft as usize]);
            let out_len = (batch * bins as u32 * n_frames) as usize;
            tape.mm(
                w,
                frames,
                None,
                out_len,
                bins as u32,
                n_frames,
                self.n_fft,
                batch,
                self.n_fft,
                1,
                0,
                0,
                1,
                n_frames,
                0,
                self.n_fft * n_frames,
                n_frames,
                1,
                bins as u32 * n_frames,
                1.0,
            )
        };
        let im = {
            let w = tape.const_tensor(&self.basis_im, vec![bins, self.n_fft as usize]);
            let out_len = (batch * bins as u32 * n_frames) as usize;
            tape.mm(
                w,
                frames,
                None,
                out_len,
                bins as u32,
                n_frames,
                self.n_fft,
                batch,
                self.n_fft,
                1,
                0,
                0,
                1,
                n_frames,
                0,
                self.n_fft * n_frames,
                n_frames,
                1,
                bins as u32 * n_frames,
                1.0,
            )
        };
        let len = (batch * bins as u32 * n_frames) as u32;
        let mag = tape.mag(re, im, len, 2e-7);
        // mel = mel_basis · mag
        let mel = {
            let w = tape.const_tensor(&self.mel_basis, vec![self.n_mels as usize, bins]);
            let out_len = (batch * self.n_mels * n_frames) as usize;
            tape.mm(
                w,
                mag,
                None,
                out_len,
                self.n_mels,
                n_frames,
                bins as u32,
                batch,
                bins as u32,
                1,
                0,
                0,
                1,
                n_frames,
                0,
                bins as u32 * n_frames,
                n_frames,
                1,
                self.n_mels * n_frames,
                1.0,
            )
        };
        tape.log_clamp(mel, 2e-6)
    }
}

/// Slaney mel 基底 (librosa filters.mel sr/n_fft/n_mels/slaney 相当)。
pub fn slaney_mel(n_mels: u32, bins: u32, sr: u32, n_fft: u32) -> Vec<f32> {
    let fmin = 0.0f32;
    let fmax = sr as f32 / 2.0;
    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);
    let mut centers_mel = Vec::with_capacity((n_mels + 2) as usize);
    for i in 0..(n_mels + 2) {
        centers_mel.push(min_mel + (max_mel - min_mel) * i as f32 / (n_mels + 1) as f32);
    }
    let centers_hz: Vec<f32> = centers_mel.iter().map(|m| mel_to_hz(*m)).collect();
    let fft_freqs: Vec<f32> = (0..bins)
        .map(|k| k as f32 * sr as f32 / n_fft as f32)
        .collect();
    // Slaney: 三角フィルタ + 面積正規化 (enorm)
    let mut out = vec![0.0f32; (n_mels * bins) as usize];
    for m in 0..n_mels {
        let left = centers_hz[m as usize];
        let mid = centers_hz[m as usize + 1];
        let right = centers_hz[m as usize + 2];
        for k in 0..bins {
            let f = fft_freqs[k as usize];
            let v = if f <= left || f >= right {
                0.0
            } else if f <= mid {
                (f - left) / (mid - left)
            } else {
                (right - f) / (right - mid)
            };
            // enorm = 2.0 / (right - left) * v (slaney norm)
            out[(m * bins + k) as usize] = v * 2.0 / (right - left);
        }
    }
    out
}

fn hz_to_mel(f: f32) -> f32 {
    // Slaney: f_sp=200/3, min_log_hz=1000, min_log_mel=(1000-0)/200*3=15, logstep=log(6.4)/27
    let f_sp = 200.0f32 / 3.0;
    let min_log_hz = 1000.0f32;
    let min_log_mel = 15.0f32;
    let logstep = (6.4f32).ln() / 27.0;
    if f < min_log_hz {
        f / f_sp
    } else {
        min_log_mel + ((f / min_log_hz).ln()) / logstep
    }
}

fn mel_to_hz(m: f32) -> f32 {
    let f_sp = 200.0f32 / 3.0;
    let min_log_hz = 1000.0f32;
    let min_log_mel = 15.0f32;
    let logstep = (6.4f32).ln() / 27.0;
    if m < min_log_mel {
        m * f_sp
    } else {
        min_log_hz * ((m - min_log_mel) * logstep).exp()
    }
}
