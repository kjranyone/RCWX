//! ポストプロセス (rcwx/audio/postprocess.py 相当):
//! トレブルブースト (ピーキング biquad) + EMA RMS ノーマライザ + ルックアヘッドリミッタ。

#[derive(Clone)]
pub struct PostprocessConfig {
    pub enabled: bool,
    pub treble_boost_db: f64,
    pub treble_cutoff_hz: f64,
    pub limiter_threshold_db: f64,
    pub limiter_release_ms: f64,
    pub limiter_lookahead_ms: f64,
    pub normalizer_enabled: bool,
    pub normalizer_target_rms: f64,
    pub normalizer_ema_alpha: f64,
    pub normalizer_max_gain_db: f64,
    pub normalizer_min_gain_db: f64,
}

impl Default for PostprocessConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            treble_boost_db: 4.0,
            treble_cutoff_hz: 2800.0,
            limiter_threshold_db: -1.0,
            limiter_release_ms: 80.0,
            limiter_lookahead_ms: 5.0,
            normalizer_enabled: true,
            normalizer_target_rms: 0.1,
            normalizer_ema_alpha: 0.15,
            normalizer_max_gain_db: 12.0,
            normalizer_min_gain_db: -12.0,
        }
    }
}

/// ピーキングEQ (scipy lfilter 互換 transposed DF-II、warm-start 付き)。
pub struct TrebleBoost {
    sample_rate: u32,
    cfg: PostprocessConfig,
    b: [f32; 3],
    a: [f32; 2], // a1, a2 (正規化後)
    zi: [f32; 2],
    warm_start: bool,
}

impl TrebleBoost {
    pub fn new(sample_rate: u32, cfg: PostprocessConfig) -> Self {
        let mut t = Self {
            sample_rate,
            cfg,
            b: [1.0, 0.0, 0.0],
            a: [0.0, 0.0],
            zi: [0.0, 0.0],
            warm_start: true,
        };
        t.design_filter();
        t
    }

    fn design_filter(&mut self) {
        if self.cfg.treble_boost_db <= 0.0 {
            self.b = [1.0, 0.0, 0.0];
            self.a = [0.0, 0.0];
            self.warm_start = true;
            return;
        }
        let nyquist = self.sample_rate as f64 / 2.0;
        let cutoff = self.cfg.treble_cutoff_hz.min(nyquist * 0.49);
        let gain_db = self.cfg.treble_boost_db;
        let a = 10f64.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f64::consts::PI * cutoff / self.sample_rate as f64;
        let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / 0.9 - 1.0) + 2.0).sqrt();

        let b0 = a * ((a + 1.0) + (a - 1.0) * w0.cos() + 2.0 * a.sqrt() * alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * w0.cos());
        let b2 = a * ((a + 1.0) + (a - 1.0) * w0.cos() - 2.0 * a.sqrt() * alpha);
        let a0 = (a + 1.0) - (a - 1.0) * w0.cos() + 2.0 * a.sqrt() * alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * w0.cos());
        let a2 = (a + 1.0) - (a - 1.0) * w0.cos() - 2.0 * a.sqrt() * alpha;

        self.b = [(b0 / a0) as f32, (b1 / a0) as f32, (b2 / a0) as f32];
        self.a = [(a1 / a0) as f32, (a2 / a0) as f32];
        self.warm_start = true;
        // lfilter_zi: 定常入力 x=1 に対する状態
        self.zi = self.compute_zi();
    }

    fn compute_zi(&self) -> [f32; 2] {
        let y_ss = (self.b[0] + self.b[1] + self.b[2]) / (1.0 + self.a[0] + self.a[1]);
        let z1 = self.b[2] - self.a[1] * y_ss;
        let z0 = self.b[1] - self.a[0] * y_ss + z1;
        [z0, z1]
    }

    pub fn reset(&mut self) {
        self.zi = self.compute_zi();
        self.warm_start = true;
    }

    pub fn process(&mut self, audio: &[f32]) -> Vec<f32> {
        if self.cfg.treble_boost_db <= 0.0 || audio.is_empty() {
            return audio.to_vec();
        }
        let (b0, b1, b2) = (self.b[0], self.b[1], self.b[2]);
        let (a1, a2) = (self.a[0], self.a[1]);
        let mut zi = if self.warm_start {
            self.warm_start = false;
            [self.zi[0] * audio[0], self.zi[1] * audio[0]]
        } else {
            self.zi
        };
        let mut out = vec![0.0f32; audio.len()];
        for (o, &x) in out.iter_mut().zip(audio) {
            let y = b0 * x + zi[0];
            zi[0] = b1 * x - a1 * y + zi[1];
            zi[1] = b2 * x - a2 * y;
            *o = y;
        }
        self.zi = zi;
        out
    }
}

/// EMA スムージング RMS ノーマライザ (スローエージング AGC)。
pub struct RmsNormalizer {
    sample_rate: u32,
    cfg: PostprocessConfig,
    ema_rms: f64,
    prev_gain: f64,
    ramp_samples: usize,
}

impl RmsNormalizer {
    const MIN_RMS: f64 = 0.005;
    const ALPHA_REF_CHUNK_SEC: f64 = 0.3;

    pub fn new(sample_rate: u32, cfg: PostprocessConfig) -> Self {
        Self {
            sample_rate,
            cfg,
            ema_rms: 0.0,
            prev_gain: 1.0,
            ramp_samples: ((sample_rate as f64 * 0.005) as usize).max(1),
        }
    }

    pub fn reset(&mut self) {
        self.ema_rms = 0.0;
        self.prev_gain = 1.0;
    }

    pub fn process(&mut self, audio: &[f32]) -> Vec<f32> {
        if !self.cfg.normalizer_enabled || audio.is_empty() {
            return audio.to_vec();
        }
        let rms = (audio.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / audio.len() as f64).sqrt();
        if rms < Self::MIN_RMS {
            return audio.to_vec();
        }

        if self.ema_rms <= 0.0 {
            self.ema_rms = rms;
        } else {
            let chunk_sec = audio.len() as f64 / self.sample_rate as f64;
            let alpha = 1.0 - (1.0 - self.cfg.normalizer_ema_alpha).powf(chunk_sec / Self::ALPHA_REF_CHUNK_SEC);
            self.ema_rms += alpha * (rms - self.ema_rms);
        }

        let mut gain = if self.ema_rms > Self::MIN_RMS {
            self.cfg.normalizer_target_rms / self.ema_rms
        } else {
            1.0
        };
        let max_gain = 10f64.powf(self.cfg.normalizer_max_gain_db / 20.0);
        let min_gain = 10f64.powf(self.cfg.normalizer_min_gain_db / 20.0);
        gain = gain.clamp(min_gain, max_gain);

        let ramp_len = self.ramp_samples.min(audio.len());
        let mut out = vec![0.0f32; audio.len()];
        if (gain - self.prev_gain).abs() > 1e-6 && ramp_len > 1 {
            for (i, o) in out.iter_mut().enumerate() {
                let x = audio[i] as f64;
                if i < ramp_len {
                    let t = i as f64 / ramp_len as f64;
                    let ramp = self.prev_gain + (gain - self.prev_gain) * 0.5 * (1.0 - (std::f64::consts::PI * t).cos());
                    *o = (x * ramp) as f32;
                } else {
                    *o = (x * gain) as f32;
                }
            }
        } else {
            for (o, &x) in out.iter_mut().zip(audio) {
                *o = (x as f64 * gain) as f32;
            }
        }
        self.prev_gain = gain;
        out
    }
}

/// ルックアヘッドリミッタ (即時アタック・指数リリース)。
pub struct LookaheadLimiter {
    lookahead: usize,
    threshold: f64,
    release_coeff: f64,
    envelope_state: f64,
    delay_buffer: Option<Vec<f32>>,
}

impl LookaheadLimiter {
    pub fn new(sample_rate: u32, cfg: &PostprocessConfig) -> Self {
        let lookahead = ((cfg.limiter_lookahead_ms * sample_rate as f64 / 1000.0) as usize).clamp(1, 256);
        let release_samples = cfg.limiter_release_ms * sample_rate as f64 / 1000.0;
        Self {
            lookahead,
            threshold: 10f64.powf(cfg.limiter_threshold_db / 20.0),
            release_coeff: (-1.0 / release_samples.max(1.0)).exp(),
            envelope_state: 1.0,
            delay_buffer: None,
        }
    }

    pub fn reset(&mut self) {
        self.envelope_state = 1.0;
        self.delay_buffer = None;
    }

    pub fn process(&mut self, audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return audio.to_vec();
        }
        let lookahead = self.lookahead;
        let n = audio.len();
        if self.delay_buffer.is_none() {
            self.delay_buffer = Some(vec![0.0f32; lookahead]);
        }

        // combined = [delay_buf, audio]; 先頭 n が出力、残りが次の delay
        let db = self.delay_buffer.take().unwrap();
        let mut combined = Vec::with_capacity(lookahead + n);
        combined.extend_from_slice(&db);
        combined.extend_from_slice(audio);
        let delayed = combined[..n].to_vec();
        self.delay_buffer = Some(combined[n..].to_vec());

        let mut env = self.envelope_state;
        let rc = self.release_coeff;
        let mut gain = vec![1.0f32; n];
        for i in 0..n {
            let v = audio[i].abs() as f64;
            env = if v > env { v } else { rc * env + (1.0 - rc) * v };
            if env > self.threshold {
                gain[i] = (self.threshold / env) as f32;
            }
        }
        self.envelope_state = env;

        delayed
            .iter()
            .zip(&gain)
            .map(|(d, g)| d * g)
            .collect()
    }
}

pub struct Postprocessor {
    cfg: PostprocessConfig,
    treble: TrebleBoost,
    normalizer: RmsNormalizer,
    limiter: LookaheadLimiter,
}

impl Postprocessor {
    pub fn new(sample_rate: u32, cfg: PostprocessConfig) -> Self {
        Self {
            treble: TrebleBoost::new(sample_rate, cfg.clone()),
            normalizer: RmsNormalizer::new(sample_rate, cfg.clone()),
            limiter: LookaheadLimiter::new(sample_rate, &cfg),
            cfg,
        }
    }

    pub fn reset(&mut self) {
        self.treble.reset();
        self.normalizer.reset();
        self.limiter.reset();
    }

    pub fn process(&mut self, audio: &[f32]) -> Vec<f32> {
        if !self.cfg.enabled || audio.is_empty() {
            return audio.to_vec();
        }
        let mut out = audio.to_vec();
        if self.cfg.treble_boost_db > 0.0 {
            out = self.treble.process(&out);
        }
        out = self.normalizer.process(&out);
        out = self.limiter.process(&out);
        out
    }
}
