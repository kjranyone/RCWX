//! リアルタイム設定と統計。

#[derive(Clone)]
pub struct RtConfig {
    /// 入力デバイス実レート (実行時に上書きされる)
    pub mic_sample_rate: u32,
    /// 出力デバイス実レート
    pub output_sample_rate: u32,
    /// 1 hop の長さ (秒)。20ms Aggressive / 60ms Normal 相当
    pub chunk_sec: f64,
    /// 出力ゲイン (dB)
    pub input_gain_db: f64,
    pub output_gain_db: f64,
    /// pitch shift (半音)
    pub pitch: f64,
    /// f0 手法: swiftf0 / rmvpe
    pub f0_method: String,
    /// denoise: none / spectral
    pub denoise: String,
    pub denoise_strength: f64,
    /// 入力ノイズゲート (自動しきい値)
    pub noise_gate: bool,
    /// HuBERT コンテキスト上限 (秒)。Aggressive 0.56 相当
    pub hubert_context_sec: f64,
    /// F0 コンテキスト (秒)。SwiftF0 Aggressive 0.10 相当
    pub f0_context_sec: f64,
    /// SOLA
    pub crossfade_sec: f64,
    pub sola_search_sec: f64,
    /// IVF index ブレンド率 (0 = off)
    pub index_rate: f32,
    pub index_k: usize,
    /// moe f0 スタイル強度
    pub moe_boost: f64,
    /// ポストプロセス
    pub postprocess: bool,
    /// 出力リング上限 (秒) — ドリフトトリムの判定基準
    pub buffer_margin: f64,
    pub prebuffer_chunks: u32,
    /// フィードバック検出
    pub feedback_detect: bool,
    /// 入力チャンネル (None = モノラル平均)
    pub input_channel: Option<usize>,
    /// 出力チャンネル (None = 両ch)
    pub output_channel: Option<usize>,
}

impl Default for RtConfig {
    fn default() -> Self {
        Self {
            mic_sample_rate: 48000,
            output_sample_rate: 48000,
            chunk_sec: 0.06,
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            pitch: 0.0,
            f0_method: "swiftf0".into(),
            denoise: "spectral".into(),
            denoise_strength: 1.0,
            noise_gate: false,
            hubert_context_sec: 0.56,
            f0_context_sec: 0.10,
            crossfade_sec: 0.012,
            sola_search_sec: 0.010,
            index_rate: 0.0,
            index_k: 4,
            moe_boost: 0.0,
            postprocess: true,
            buffer_margin: 0.25,
            prebuffer_chunks: 2,
            feedback_detect: true,
            input_channel: None,
            output_channel: None,
        }
    }
}

/// hop あたり 16k サンプル数 (320 境界に固定)。
impl RtConfig {
    pub fn hop_16k(&self) -> usize {
        (self.chunk_sec * 16000.0).round() as usize / 320 * 320
    }
    pub fn hop_mic(&self) -> usize {
        (self.chunk_sec * self.mic_sample_rate as f64).round() as usize
    }
    /// HuBERT コンテキスト (16k サンプル・320 境界)
    pub fn hubert_ctx_samples(&self) -> usize {
        let s = (self.hubert_context_sec * 16000.0).round() as usize;
        s / 320 * 320
    }
    pub fn f0_ctx_samples(&self) -> usize {
        (self.f0_context_sec * 16000.0).round() as usize
    }
}

/// 統計 (表示/テレメトリ用)。
#[derive(Default, Clone)]
pub struct RtStats {
    pub hops: u64,
    pub queue_full: u64,
    pub underruns: u64,
    pub inference_ms: Vec<f64>,
    pub last_inference_ms: f64,
    pub solo_offset: usize,
    pub feedback_corr: f64,
    pub ring_peak: usize,
}

impl RtStats {
    pub fn record_inference(&mut self, ms: f64) {
        self.last_inference_ms = ms;
        self.inference_ms.push(ms);
        if self.inference_ms.len() > 256 {
            self.inference_ms.remove(0);
        }
    }
    pub fn percentile(&self, p: f64) -> f64 {
        if self.inference_ms.is_empty() {
            return 0.0;
        }
        let mut v = self.inference_ms.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((p / 100.0 * v.len() as f64).ceil() as usize).saturating_sub(1).min(v.len() - 1);
        v[idx]
    }
}
