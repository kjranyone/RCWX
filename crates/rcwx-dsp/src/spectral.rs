//! ストリーミングスペクトルゲート (rcwx/audio/denoise.py StreamingSpectralGate 相当)。
//!
//! sqrt-Hann 解析+合成 (n_fft=512 / hop=256)。COLA 成立で確定サンプルは正規化不要。
//! ノイズプロファイルは非対称 1 極追跡 (最小統計)。最新 ≤16ms は dry クロスフェード近似。
//! dtype 流儀は numpy に追従: FFT/プロファイル/gain 演算は f64、蓄積バッファは f32。

use rustfft::num_complex::Complex;

pub const N_FFT: usize = 512;
pub const HOP: usize = 256;
const N_BINS: usize = N_FFT / 2 + 1;

pub struct StreamingSpectralGate {
    window: Vec<f32>, // sqrt-Hann
    mag_smooth_coeff: f64,
    rise_factor: f64,
    bootstrap_frames: usize,
    g_attack: f64,
    g_release: f64,
    noise_floor_seed: f64,
    threshold_mult: f64,
    reduction_mult: f64,
    // ストリーム状態
    noise_profile: Option<Vec<f64>>,
    mag_smooth: Option<Vec<f64>>,
    frames_seen: usize,
    prev_gain: Vec<f32>,
    buf: Vec<f32>,
    pos: usize,
    ola_start: usize,
    ola: Vec<f32>,
    win: Vec<f32>,
    next_out: usize,
}

impl StreamingSpectralGate {
    pub fn new(sample_rate: u32, threshold_db: f64, reduction_db: f64) -> Self {
        // sqrt(np.hanning(512)) — 対称 Hann
        let m = (N_FFT - 1) as f64;
        let window: Vec<f32> = (0..N_FFT)
            .map(|n| {
                let h = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / m).cos();
                (h as f32).sqrt()
            })
            .collect();
        let hop_sec = HOP as f64 / sample_rate as f64;
        let reduction_db = reduction_db.clamp(-60.0, -3.0);
        Self {
            window,
            mag_smooth_coeff: (-hop_sec / 0.25).exp(),
            rise_factor: 10f64.powf(6.0 / 20.0 * hop_sec),
            bootstrap_frames: (0.25 / hop_sec).round().max(1.0) as usize,
            g_attack: (-hop_sec / 0.01).exp(),
            g_release: (-hop_sec / 0.05).exp(),
            noise_floor_seed: 1e-4,
            threshold_mult: 10f64.powf(threshold_db / 20.0),
            reduction_mult: 10f64.powf(reduction_db / 20.0),
            noise_profile: None,
            mag_smooth: None,
            frames_seen: 0,
            prev_gain: vec![1.0; N_BINS],
            buf: vec![0.0; HOP],
            pos: 0,
            ola_start: 0,
            ola: Vec::new(),
            win: Vec::new(),
            next_out: HOP,
        }
    }

    pub fn configure(&mut self, threshold_db: f64, reduction_db: f64) {
        self.threshold_mult = 10f64.powf(threshold_db / 20.0);
        let reduction_db = reduction_db.clamp(-60.0, -3.0);
        self.reduction_mult = 10f64.powf(reduction_db / 20.0);
    }

    pub fn reset(&mut self) {
        self.noise_profile = None;
        self.mag_smooth = None;
        self.frames_seen = 0;
        self.prev_gain.fill(1.0);
        self.buf = vec![0.0; HOP];
        self.pos = 0;
        self.ola_start = 0;
        self.ola.clear();
        self.win.clear();
        self.next_out = HOP;
    }

    fn update_profile(&mut self, magnitude: &[f64]) {
        self.frames_seen += 1;
        match &mut self.mag_smooth {
            None => self.mag_smooth = Some(magnitude.to_vec()),
            Some(ms) => {
                let b = self.mag_smooth_coeff;
                for i in 0..N_BINS {
                    ms[i] = b * ms[i] + (1.0 - b) * magnitude[i];
                }
            }
        }
        let ms = self.mag_smooth.as_deref().unwrap();
        match &mut self.noise_profile {
            None => {
                self.noise_profile = Some(ms.iter().map(|v| v.max(self.noise_floor_seed)).collect());
            }
            Some(np) => {
                for i in 0..N_BINS {
                    let risen = np[i] * self.rise_factor;
                    np[i] = ms[i].min(risen).max(self.noise_floor_seed);
                }
            }
        }
    }

    /// ゲート gain (f64 演算 → conv → f32)。
    fn compute_gain(&self, magnitude: &[f64]) -> Vec<f32> {
        if self.noise_profile.is_none() || self.frames_seen < self.bootstrap_frames {
            return vec![1.0; N_BINS];
        }
        let np = self.noise_profile.as_deref().unwrap();
        let mut gain = vec![0.0f64; N_BINS];
        for i in 0..N_BINS {
            let threshold = np[i] * self.threshold_mult;
            let ratio = magnitude[i] / (threshold + 1e-10);
            let soft = 0.5 * (1.0 + (2.0 * (ratio - 1.0)).tanh());
            gain[i] = self.reduction_mult + soft * (1.0 - self.reduction_mult);
        }
        // np.convolve(gain, ones(3)/3, mode="same"): 端はゼロ埋め
        let mut out = vec![0.0f32; N_BINS];
        for (i, slot) in out.iter_mut().enumerate() {
            let s = match i {
                0 => 0.0 + gain[0] + gain[1],
                i if i == N_BINS - 1 => gain[i - 1] + gain[i] + 0.0,
                i => gain[i - 1] + gain[i] + gain[i + 1],
            };
            *slot = (s / 3.0) as f32;
        }
        out
    }

    /// 1 フレーム (n_fft) をゲート。窓適用済み出力フレーム (f64) を返す。
    fn gate_frame(&mut self, frame: &[f32], learn: bool) -> Vec<f64> {
        let mut planner = rustfft::FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let mut spec = vec![Complex::<f64>::new(0.0, 0.0); N_FFT];
        for i in 0..N_FFT {
            spec[i].re = (frame[i] * self.window[i]) as f64;
        }
        fft.process(&mut spec);
        let magnitude: Vec<f64> = spec[..N_BINS].iter().map(|c| (c.re * c.re + c.im * c.im).sqrt()).collect();
        if learn {
            self.update_profile(&magnitude);
        }
        let gain = self.compute_gain(&magnitude);
        // 時間平滑 (attack/release)。speculative フレームは prev_gain を進めない
        let mut smoothed = vec![0.0f64; N_BINS];
        for i in 0..N_BINS {
            let (g, p) = (gain[i] as f64, self.prev_gain[i] as f64);
            smoothed[i] = if gain[i] > self.prev_gain[i] {
                self.g_attack * p + (1.0 - self.g_attack) * g
            } else {
                self.g_release * p + (1.0 - self.g_release) * g
            };
        }
        if learn {
            for i in 0..N_BINS {
                self.prev_gain[i] = smoothed[i] as f32;
            }
        }
        // 逆 FFT (rfft の共役対称復元)
        let mut full = vec![Complex::<f64>::new(0.0, 0.0); N_FFT];
        for i in 0..N_BINS {
            let g = smoothed[i];
            full[i] = Complex::new(spec[i].re * g, spec[i].im * g);
            if i > 0 && i < N_FFT - i {
                full[N_FFT - i] = Complex::new(spec[i].re * g, -spec[i].im * g);
            }
        }
        let ifft = planner.plan_fft_inverse(N_FFT);
        ifft.process(&mut full);
        let scale = 1.0 / N_FFT as f64;
        full.iter()
            .zip(&self.window)
            .map(|(c, w)| c.re * scale * *w as f64)
            .collect()
    }

    /// 1 hop をデノイズ。入力と同長を返す。
    pub fn process(&mut self, audio: &[f32]) -> Vec<f32> {
        let n = audio.len();
        if n == 0 {
            return audio.to_vec();
        }
        self.buf.extend_from_slice(audio);
        let s_end = self.pos + self.buf.len();
        let emit_start = self.next_out;

        // 確定フレーム
        while self.buf.len() >= N_FFT {
            let enhanced = self.gate_frame(&self.buf[..N_FFT].to_vec(), true);
            let lo = self.pos - self.ola_start;
            let needed = lo + N_FFT;
            if self.ola.len() < needed {
                self.ola.resize(needed, 0.0);
                self.win.resize(needed, 0.0);
            }
            for i in 0..N_FFT {
                // f32 バッファへの in-place 加算 (numpy と同じ丸め)
                self.ola[lo + i] = (self.ola[lo + i] as f64 + enhanced[i]) as f32;
                let w2 = self.window[i] * self.window[i];
                self.win[lo + i] += w2;
            }
            self.buf.drain(..HOP);
            self.pos += HOP;
        }

        let exact_end = self.pos.min(s_end);
        let exact: Vec<f32> = if exact_end > emit_start {
            let lo = emit_start - self.ola_start;
            self.ola[lo..lo + (exact_end - emit_start)].to_vec()
        } else {
            Vec::new()
        };

        // 投機的テール (ゼロ埋め・状態を進めない) → dry クロスフェード
        let spec_lo = self.pos.max(emit_start);
        let spec_len = s_end.saturating_sub(spec_lo);
        let mut spec = vec![0.0f32; spec_len];
        if spec_len > 0 {
            let base = self.pos - self.ola_start;
            let mut spec_ola = self.ola[base..].to_vec();
            let mut spec_win = self.win[base..].to_vec();
            let mut spec_pos = self.pos;
            let gain_snapshot = self.prev_gain.clone();
            while spec_pos < s_end {
                let offset = spec_pos - self.pos;
                let mut frame = vec![0.0f32; N_FFT];
                for (i, slot) in frame.iter_mut().enumerate() {
                    *slot = self.buf.get(offset + i).copied().unwrap_or(0.0);
                }
                let enhanced = self.gate_frame(&frame, false);
                let lo = spec_pos - self.pos;
                let needed = lo + N_FFT;
                if spec_ola.len() < needed {
                    spec_ola.resize(needed, 0.0);
                    spec_win.resize(needed, 0.0);
                }
                for i in 0..N_FFT {
                    spec_ola[lo + i] = (spec_ola[lo + i] as f64 + enhanced[i]) as f32;
                    let w2 = self.window[i] * self.window[i];
                    spec_win[lo + i] += w2;
                }
                spec_pos += HOP;
            }
            self.prev_gain = gain_snapshot;
            let lo = spec_lo - self.pos;
            for j in 0..spec_len {
                let coverage = spec_win
                    .get(lo + j)
                    .copied()
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0) as f64;
                let wet = spec_ola.get(lo + j).copied().unwrap_or(0.0) as f64;
                let dry = self.buf.get(lo + j).copied().unwrap_or(0.0) as f64;
                spec[j] = (wet * coverage + dry * (1.0 - coverage)) as f32;
            }
        }

        // 送り出し分を破棄
        let keep = self.pos - self.ola_start;
        self.ola.drain(..keep);
        self.win.drain(..keep);
        self.ola_start = self.pos;
        self.next_out = s_end;

        let mut out = exact;
        out.extend_from_slice(&spec);
        out
    }
}
