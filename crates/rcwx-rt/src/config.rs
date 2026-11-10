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
    /// f0 手法: swiftf0 / rmvpe / none (F0なし)
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
    /// formant shift (半音)。ピッチを保ったまま声質を移動 (upstream rtrvc と同一手法)
    pub formant_shift: f64,
    /// 無声フレーム保護 (0..0.5)。index 混入前の特徴へブレンドして子音・息を守る
    pub protect: f64,
    /// 音量エンベロープ追従 (upstream rms_mix_rate: 1 = off, 0 = 完全追従)
    pub rms_mix_rate: f64,
    /// 出力側スペクトルゲート (decoder 残留ノイズ低減)
    pub output_denoise: bool,
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
    /// 入力ノイズゲート感度 0=低 / 1=中 / 2=高
    pub noise_gate_sensitivity: u8,
    /// ポスト処理詳細 (旧 GUI のポスト処理セクション相当)
    pub treble_boost_db: f64,
    pub treble_cutoff_hz: f64,
    pub limiter_threshold_db: f64,
    pub limiter_release_ms: f64,
    pub normalizer_enabled: bool,
    pub normalizer_target_rms: f64,
    /// 1 推論あたりにまとめる hop 数 (入力 hop 粒度は変わらず推論をバッチ。
    /// decoder マージンを償却し GPU 形状を大きくして持続レートを稼ぐ)
    pub batch_hops: u32,
    /// 変換後ボイスゲート: off / strict / expand / energy
    pub voice_gate_mode: String,
    pub energy_threshold: f64,
    /// F0 オクターブ飛び抑制とスルー制限
    pub octave_suppress: bool,
    pub f0_slew_limit: bool,
    pub f0_slew_max_step: f64,
    pub f0_hole_fill_ms: f64,
    /// マイクの代わりにループ再生する WAV。None なら入力デバイス。
    pub input_wav: Option<String>,
    /// cpal ホスト名。空なら既定ホスト。
    pub host_name: String,
    /// RVC モデル .pth。Some なら synth.safetensors の代わりに直接ロード。
    pub model_path: Option<String>,
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
            index_k: 8,
            formant_shift: 0.0,
            protect: 0.0,
            rms_mix_rate: 1.0,
            output_denoise: false,
            moe_boost: 0.0,
            postprocess: true,
            buffer_margin: 0.25,
            prebuffer_chunks: 2,
            feedback_detect: true,
            input_channel: None,
            output_channel: None,
            noise_gate_sensitivity: 1,
            treble_boost_db: 4.0,
            treble_cutoff_hz: 2800.0,
            limiter_threshold_db: -1.0,
            limiter_release_ms: 80.0,
            normalizer_enabled: true,
            normalizer_target_rms: 0.1,
            batch_hops: 1,
            voice_gate_mode: "off".into(),
            energy_threshold: 0.2,
            octave_suppress: true,
            f0_slew_limit: true,
            f0_slew_max_step: 3.6,
            f0_hole_fill_ms: 30.0,
            input_wav: None,
            host_name: String::new(),
            model_path: None,
        }
    }
}

/// 実行中に hop ごとに読み出すパラメータ。チャンク長・F0 方式・デバイスは含まない。
#[derive(Clone)]
pub struct LiveParams {
    pub pitch: f64,
    pub moe_boost: f64,
    pub input_gain_db: f64,
    pub output_gain_db: f64,
    pub index_rate: f32,
    pub index_k: usize,
    pub formant_shift: f64,
    pub protect: f64,
    pub rms_mix_rate: f64,
    pub output_denoise: bool,
    pub denoise: String,
    pub denoise_strength: f64,
    pub noise_gate: bool,
    pub noise_gate_sensitivity: u8,
    pub postprocess: bool,
    pub treble_boost_db: f64,
    pub treble_cutoff_hz: f64,
    pub limiter_threshold_db: f64,
    pub limiter_release_ms: f64,
    pub normalizer_enabled: bool,
    pub normalizer_target_rms: f64,
    pub octave_suppress: bool,
    pub f0_slew_limit: bool,
    pub f0_slew_max_step: f64,
    pub voice_gate_mode: String,
    pub energy_threshold: f64,
    pub input_channel: Option<usize>,
    pub output_channel: Option<usize>,
}

impl LiveParams {
    pub fn from_config(cfg: &RtConfig) -> Self {
        Self {
            pitch: cfg.pitch,
            moe_boost: cfg.moe_boost,
            input_gain_db: cfg.input_gain_db,
            output_gain_db: cfg.output_gain_db,
            index_rate: cfg.index_rate,
            index_k: cfg.index_k,
            formant_shift: cfg.formant_shift,
            protect: cfg.protect.clamp(0.0, 0.5),
            rms_mix_rate: cfg.rms_mix_rate.clamp(0.0, 1.0),
            output_denoise: cfg.output_denoise,
            denoise: cfg.denoise.clone(),
            denoise_strength: cfg.denoise_strength,
            noise_gate: cfg.noise_gate,
            noise_gate_sensitivity: cfg.noise_gate_sensitivity,
            postprocess: cfg.postprocess,
            treble_boost_db: cfg.treble_boost_db,
            treble_cutoff_hz: cfg.treble_cutoff_hz,
            limiter_threshold_db: cfg.limiter_threshold_db,
            limiter_release_ms: cfg.limiter_release_ms,
            normalizer_enabled: cfg.normalizer_enabled,
            normalizer_target_rms: cfg.normalizer_target_rms,
            octave_suppress: cfg.octave_suppress,
            f0_slew_limit: cfg.f0_slew_limit,
            f0_slew_max_step: cfg.f0_slew_max_step,
            voice_gate_mode: cfg.voice_gate_mode.clone(),
            energy_threshold: cfg.energy_threshold,
            input_channel: cfg.input_channel,
            output_channel: cfg.output_channel,
        }
    }
}

impl RtConfig {
    pub fn apply_live(&mut self, live: &LiveParams) {
        self.pitch = live.pitch;
        self.moe_boost = live.moe_boost;
        self.input_gain_db = live.input_gain_db;
        self.output_gain_db = live.output_gain_db;
        self.index_rate = live.index_rate;
        self.index_k = live.index_k.max(1);
        self.formant_shift = live.formant_shift;
        self.protect = live.protect.clamp(0.0, 0.5);
        self.rms_mix_rate = live.rms_mix_rate.clamp(0.0, 1.0);
        self.output_denoise = live.output_denoise;
        self.denoise = live.denoise.clone();
        self.denoise_strength = live.denoise_strength;
        self.noise_gate = live.noise_gate;
        self.noise_gate_sensitivity = live.noise_gate_sensitivity;
        self.postprocess = live.postprocess;
        self.treble_boost_db = live.treble_boost_db;
        self.treble_cutoff_hz = live.treble_cutoff_hz;
        self.limiter_threshold_db = live.limiter_threshold_db;
        self.limiter_release_ms = live.limiter_release_ms;
        self.normalizer_enabled = live.normalizer_enabled;
        self.normalizer_target_rms = live.normalizer_target_rms;
        self.octave_suppress = live.octave_suppress;
        self.f0_slew_limit = live.f0_slew_limit;
        self.f0_slew_max_step = live.f0_slew_max_step;
        self.voice_gate_mode = live.voice_gate_mode.clone();
        self.energy_threshold = live.energy_threshold;
        self.input_channel = live.input_channel;
        self.output_channel = live.output_channel;
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

/// GUI へ 1 秒ごとに渡すスナップショット。
#[derive(Clone, Copy, Default)]
pub struct RtReport {
    pub hops: u64,
    pub underruns: u64,
    pub queue_full: u64,
    pub trims: u64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub last_ms: f64,
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
        let idx = ((p / 100.0 * v.len() as f64).ceil() as usize)
            .saturating_sub(1)
            .min(v.len() - 1);
        v[idx]
    }
}
