//! ストリーミング推論エンジン: 1 hop → 48k 変換音声。
//!
//! hop ごとの流れ (realtime_unified.py 相当の独自ストリーミング):
//! 1) 入力ゲイン → mic→16k リサンプル (poly・overlap-save)
//! 2) denoise (spectral) / noise gate
//! 3) HuBERT 窓 [context | new] → 新規 50fps フレームを特徴キャッシュへ
//!    (オーバーラップ領域は sigmoid ブレンド)
//! 4) F0 窓 [context | new] → 条件付け (shift+moe) → キャッシュ →
//!    フィルタ chain (末尾新規分のみ確定)
//! 5) IVF ブレンド (新規フレーム) → 100fps 補間
//! 6) TextEncoder/Flow 窓 → z、sinegen (窓内) → decoder (末尾 2フレームは
//!    decoder overlap として未確定) → SOLA (hold-back) → postprocess
//!
//! HuBERT 窓長は毎 hop 同一 → plan キャッシュが 1 エントリで定常化する。

use anyhow::Result;

use rcwx_dsp::noise_gate::{NoiseGate, Sensitivity};
use rcwx_dsp::postprocess::{PostprocessConfig, Postprocessor};
use rcwx_dsp::resample::StatefulResampler;
use rcwx_dsp::sola::{sola_crossfade, SolaState};
use rcwx_dsp::spectral::StreamingSpectralGate;

use crate::config::{RtConfig, RtStats};
use rcwx_nn::engine::{Backend, Engine, GpuTensor, Plan};
use rcwx_nn::f0;
use rcwx_nn::hubert::Hubert;
use rcwx_nn::ivf::IvfIndex;
use rcwx_nn::rmvpe::Rmvpe;
use rcwx_nn::swiftf0::SwiftF0;
use rcwx_nn::synth::Synth;

const MODEL_SR: u32 = 48000; // kurumi (synth.json の sr)
const ZC: usize = MODEL_SR as usize / 100; // 480 samples/frame @100fps
const DECODER_OVERLAP_FRAMES: usize = 3;
/// 特徴/F0 キャッシュの上限 (秒)
const CACHE_SEC: f64 = 4.0;

/// synth の形状別プランキャッシュ (encode/decode)。
struct SynthPlans {
    n100: usize,
    n_dec: usize,
    feats_in: GpuTensor,
    pitch_emb_in: GpuTensor,
    z_out: GpuTensor,
    har_in: GpuTensor,
    encode_plan: Plan,
    decode_plan: Plan,
    audio_out: GpuTensor,
}

fn sigmoid_blend(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = (i as f32 + 0.5) / n as f32;
            1.0 / (1.0 + (-8.0 * (t - 0.5)).exp())
        })
        .collect()
}

pub struct RtEngine {
    eng: Engine,
    hubert: Hubert,
    rmvpe: Option<Rmvpe>,
    swiftf0: Option<SwiftF0>,
    synth: Synth,
    ivf: Option<IvfIndex>,
    pub cfg: RtConfig,
    // 入力側
    resampler_in: StatefulResampler,
    resampler_out: Option<StatefulResampler>,
    spectral: Option<StreamingSpectralGate>,
    gate: Option<NoiseGate>,
    // 16k 履歴 (HuBERT 窓 = hubert_ctx + hop)
    hist: Vec<f32>,
    // 特徴キャッシュ (blended, 50fps, flat [T,768])
    feat_cache: Vec<f32>,
    // F0 (条件付け済み・フィルタ前) 100fps
    f0_cond: Vec<f32>,
    // F0 (フィルタ済み) 100fps — synth 用
    f0_filt: Vec<f32>,
    sola: SolaState,
    post: Postprocessor,
    out_hist: Vec<f32>,
    plans: Option<Box<SynthPlans>>,
    // hop バッチング: 入力 hop を貯めて batch_hops 分を一度に推論
    batch_pending: Vec<f32>,
    batch_count: u32,
    warmups: u32,
    pub stats: RtStats,
    pub warn: Vec<String>,
}

impl RtEngine {
    pub fn new(backend: Backend, cfg: RtConfig) -> Result<Self> {
        let dir = rcwx_nn::rmvpe::default_rmvpe_dir();
        let eng = Engine::new(backend)?;
        let hubert = Hubert::load(&eng, &dir.join("hubert.safetensors"))?;
        let synth = Synth::load(&eng, &dir)?;
        let (rmvpe, swiftf0) = if cfg.f0_method == "rmvpe" {
            (Some(Rmvpe::load(&eng, &dir)?), None)
        } else {
            (
                None,
                Some(SwiftF0::load(
                    &eng,
                    &rcwx_nn::swiftf0::default_swiftf0_path(),
                )?),
            )
        };
        let ivf = if cfg.index_rate > 0.0 {
            IvfIndex::load(&dir).ok()
        } else {
            None
        };

        let hop = cfg.hop_16k();
        let ctx = cfg.hubert_ctx_samples();
        let spectral = if cfg.denoise == "spectral" {
            Some(StreamingSpectralGate::new(16000, 6.0, -24.0 * cfg.denoise_strength.clamp(0.5, 2.0)))
        } else {
            None
        };
        let gate = if cfg.noise_gate {
            let mut g = NoiseGate::new(16000, -40.0);
            g.auto_threshold = true;
            g.set_sensitivity(Sensitivity::Mid);
            Some(g)
        } else {
            None
        };

        Ok(Self {
            resampler_in: StatefulResampler::new(cfg.mic_sample_rate, 16000),
            resampler_out: (cfg.output_sample_rate != MODEL_SR)
                .then(|| StatefulResampler::new(MODEL_SR, cfg.output_sample_rate)),
            spectral,
            gate,
            hist: vec![0.0; ctx], // ゼロ充填履歴 (prime 相当)
            feat_cache: Vec::new(),
            f0_cond: Vec::new(),
            f0_filt: Vec::new(),
            sola: SolaState::new(
                (cfg.crossfade_sec * MODEL_SR as f64).round() as usize,
                (cfg.sola_search_sec * MODEL_SR as f64).round() as usize,
            ),
            post: Postprocessor::new(MODEL_SR, PostprocessConfig::default()),
            out_hist: Vec::new(),
            plans: None,
            batch_pending: Vec::new(),
            batch_count: 0,
            warmups: 0,
            eng,
            hubert,
            rmvpe,
            swiftf0,
            synth,
            ivf,
            cfg,
            stats: RtStats::default(),
            warn: Vec::new(),
        })
        .map(|mut e| {
            let _ = hop;
            e
        })
    }

    /// hop (mic レート) → 変換音声 (出力レート)。
    /// batch_hops > 1 のときは貯めてから一括推論し、非最終 hop は空 Vec を返す。
    /// 戻り値の平均長は hop あたり new100*zc (出力レート)。
    pub fn process_hop(&mut self, hop_mic: &[f32]) -> Vec<f32> {
        if self.cfg.batch_hops > 1 {
            self.batch_pending.extend_from_slice(hop_mic);
            self.batch_count += 1;
            if self.batch_count < self.cfg.batch_hops {
                return Vec::new();
            }
            let batched = std::mem::take(&mut self.batch_pending);
            self.batch_count = 0;
            let n = self.cfg.batch_hops as usize;
            let out = self.process_batch(&batched, n);
            return out;
        }
        self.process_batch(hop_mic, 1)
    }

    /// hop_mic: (batch_n hop 分を連結した) 入力。batch_n=1 で単一 hop。
    fn process_batch(&mut self, hop_mic: &[f32], batch_n: usize) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let mut stage_t = std::time::Instant::now();
        let mut stage = |name: &str, t: &mut std::time::Instant| {
            if std::env::var("RCWX_RT_STAGES").is_ok() {
                eprintln!("  [stage] {name}: {:.2}ms", t.elapsed().as_secs_f64() * 1000.0);
            }
            *t = std::time::Instant::now();
        };
        let hop16 = self.cfg.hop_16k() * batch_n.max(1);
        let new50 = hop16 / 320;
        let new100 = new50 * 2;

        // 1) 入力ゲイン + リサンプル
        let gain = 10f64.powf(self.cfg.input_gain_db / 20.0) as f32;
        let scaled: Vec<f32> = hop_mic.iter().map(|v| v * gain).collect();
        let mut chunk16 = if self.cfg.mic_sample_rate == 16000 {
            scaled
        } else {
            self.resampler_in.resample_chunk(&scaled)
        };
        if chunk16.len() < hop16 {
            chunk16.resize(hop16, 0.0);
        }
        chunk16.truncate(hop16);

        // 2) denoise / gate
        if let Some(sp) = self.spectral.as_mut() {
            chunk16 = sp.process(&chunk16);
        }
        if let Some(g) = self.gate.as_mut() {
            chunk16 = g.process(&chunk16);
        }

        stage("input+denoise", &mut stage_t);
        // 3) HuBERT 窓
        self.hist.extend_from_slice(&chunk16);
        let win_len = self.hist.len();
        let wav_t = self.eng.tensor_from_f32(&self.hist, vec![win_len]);
        let feats = self.hubert.features(&self.eng, &wav_t, win_len as u32);
        let feats_data = self.eng.readback_f32(&feats);
        let t50_win = feats_data.len() / 768;

        // 新規フレーム → キャッシュ (sigmoid ブレンド)
        if t50_win > new50 {
            let (feat_win, _) = feats_data.split_at(t50_win * 768);
            stage("hubert+blend", &mut stage_t);
        let cache_t50 = self.feat_cache.len() / 768;
            // 窓先頭の (t50_win - new50) フレームがキャッシュ末尾と同一時刻
            let overlap = (t50_win - new50).min(cache_t50);
            let blend = sigmoid_blend(overlap);
            for r in 0..overlap {
                let cache_off = self.feat_cache.len() - overlap * 768;
                for c in 0..768 {
                    let ci = cache_off + r * 768 + c;
                    let ni = r * 768 + c;
                    let a = blend[r];
                    self.feat_cache[ci] = a * self.feat_cache[ci] + (1.0 - a) * feat_win[ni];
                }
            }
            let new_off = (t50_win - new50) * 768;
            self.feat_cache
                .extend_from_slice(&feat_win[new_off..t50_win * 768]);
        } else {
            self.warn.push(format!("HuBERT window too short: {t50_win} <= {new50}"));
        }
        let cache_cap50 = (CACHE_SEC * 50.0) as usize * 768;
        if self.feat_cache.len() > cache_cap50 {
            let drop = self.feat_cache.len() - cache_cap50;
            self.feat_cache.drain(..drop / 768 * 768);
        }
        stage("hubert+blend", &mut stage_t);
        let cache_t50 = self.feat_cache.len() / 768;

        // 4) F0 窓 [ctx | new]
        let f0_ctx_samples = self.cfg.f0_ctx_samples();
        let f0_win_samples = f0_ctx_samples + hop16;
        let win_start = self.hist.len().saturating_sub(f0_win_samples);
        let f0_audio = self.hist[win_start..].to_vec();
        let mut f0_win = match (&self.swiftf0, &self.rmvpe) {
            (Some(sf), _) => sf.detect(&self.eng, &f0_audio),
            (_, Some(rm)) => rm.infer(&self.eng, &f0_audio),
            _ => vec![0.0; f0_win_samples / 160],
        };
        let want = f0_win_samples / 160;
        f0_win.resize(want, 0.0);

        // 条件付け: pitch shift + moe (窓全体)
        condition_f0(&mut f0_win, self.cfg.pitch);
        if self.cfg.moe_boost > 0.0 {
            rcwx_dsp::moe::apply_moe_f0_style(&mut f0_win, self.cfg.moe_boost);
        }
        // キャッシュ (sigmoid ブレンド) + 新規確定
        // 窓先頭の (want - new) フレームが f0_cond 末尾と同一時刻
        let new_in_win = new100;
        if want > new_in_win {
            let overlap = (want - new_in_win).min(self.f0_cond.len()).min(32);
            let blend = sigmoid_blend(overlap);
            for r in 0..overlap {
                let ci = self.f0_cond.len() - overlap + r;
                let a = blend[r];
                self.f0_cond[ci] = a * self.f0_cond[ci] + (1.0 - a) * f0_win[r];
            }
            self.f0_cond.extend_from_slice(&f0_win[want - new_in_win..]);
        } else {
            self.warn.push("f0 window too short".into());
        }
        let cap100 = (CACHE_SEC * 100.0) as usize;
        if self.f0_cond.len() > cap100 {
            self.f0_cond.drain(..self.f0_cond.len() - cap100);
        }

        // フィルタ chain: 窓 [filter_ctx | new] でかけて新規分を確定
        let filter_ctx = 24.min(self.f0_cond.len().saturating_sub(new100));
        let fs = self.f0_cond.len() - filter_ctx - new100;
        let mut f0_to_filter = self.f0_cond[fs..].to_vec();
        rcwx_nn::f0::apply_f0_filter_chain(
            &mut f0_to_filter,
            16.0,
            true,
            true,
            3.6,
            30.0,
        );
        self.f0_filt
            .extend_from_slice(&f0_to_filter[filter_ctx..]);
        if self.f0_filt.len() > cap100 {
            self.f0_filt.drain(..self.f0_filt.len() - cap100);
        }

        stage("f0", &mut stage_t);
        // 5) IVF ブレンド + 100fps 化 + decoder 窓
        // encode 窓 = hubert ctx (0.56s, Python Aggressive と同じ) + new。
        // decoder の受容野マージンは n_dec 側で確保するため含めない。
        let enc_ctx50 = 28;
        let n100 = {
            let avail100 = self.f0_filt.len();
            let enc_ctx100 = enc_ctx50 * 2;
            (enc_ctx100 + new100 + DECODER_OVERLAP_FRAMES).min(avail100)
        };
        let win100_start = self.f0_filt.len().saturating_sub(n100);
        let f0_synth_win = self.f0_filt[win100_start..].to_vec();

        // 特徴窓 (50fps): n100/2 相当
        let win50 = n100.div_ceil(2);
        let win50_start = cache_t50.saturating_sub(win50);
        let mut feats_win = self.feat_cache[win50_start * 768..].to_vec();
        let feats_win_frames = feats_win.len() / 768;
        feats_win.truncate(feats_win_frames * 768);
        if let Some(ivf) = self.ivf.as_ref() {
            if feats_win_frames > 0 {
                ivf.blend(&mut feats_win, feats_win_frames, self.cfg.index_rate, self.cfg.index_k);
            }
        }

        // 100fps 補間 (align_corners=False 相当)
        let t50 = feats_win_frames;
        let mut feats100 = vec![0.0f32; n100 * 768];
        for r in 0..n100 {
            let pos = r as f32 * 0.5 - 0.25;
            let i0 = (pos.floor() as i32).clamp(0, t50 as i32 - 1) as usize;
            let i1 = (i0 + 1).min(t50.saturating_sub(1));
            let fr = (pos - i0 as f32).clamp(0.0, 1.0);
            for c in 0..768 {
                let a = feats_win[i0 * 768 + c];
                let b = feats_win[i1 * 768 + c];
                feats100[r * 768 + c] = a + (b - a) * fr;
            }
        }
        let pitch: Vec<u32> = f0::quantize_f0_to_pitch(&f0_synth_win);

        // 6) encode → decode → SOLA (固定形状はプランをキャッシュして再生)
        //    TextEncoder/Flow は全窓で attention コンテキストを取るが、
        //    デコーダは受容野マージン分 (8fr) の z で十分 → 計算量 1/5 に削減
        let n_dec = (new100 + 4 + DECODER_OVERLAP_FRAMES).min(n100);
        let pitch_emb = self.synth.pitch_embedding(&pitch, n100 as u32);
        let f0_dec = {
            let s = self.f0_filt.len().saturating_sub(n_dec);
            self.f0_filt[s..].to_vec()
        };
        let har = self.synth.sinegen(&f0_dec, n_dec as u32);
        let mut audio48: Vec<f32> = {
            let RtEngine {
                eng,
                synth,
                plans,
                ..
            } = self;
            // 形状が変わったらプラン再構築 (ウォームアップ中のみ)
            if plans.as_ref().map(|p| (p.n100, p.n_dec)) != Some((n100, n_dec)) {
                let feats_in = eng.tensor_uninit(n100 * 768);
                let pitch_emb_in = eng.tensor_uninit(synth.cfg.hidden_channels as usize * n100);
                // encode plan
                let mut g = eng.graph();
                let z_out = synth.encode_graph(eng, &mut g, &feats_in, &pitch_emb_in, n100 as u32);
                let encode_plan = g.into_plan().expect("encode plan");
                // decode plan: encode の z を GPU 直結 (slice_ch で末尾抽出)
                let har_in = eng.tensor_uninit(har.len());
                let z_link = z_out.clone_handle();
                let mut g = eng.graph();
                let audio = synth.decode_graph(eng, &mut g, &z_link, &har_in, n100 as u32, n_dec as u32);
                let decode_plan = g.into_plan().expect("decode plan");
                *plans = Some(Box::new(SynthPlans {
                    n100,
                    n_dec,
                    feats_in,
                    pitch_emb_in,
                    z_out,
                    har_in,
                    encode_plan,
                    decode_plan,
                    audio_out: audio,
                }));
            }
            let p = plans.as_ref().unwrap();
            eng.write_f32(&p.feats_in, &feats100);
            eng.write_f32(&p.pitch_emb_in, &pitch_emb);
            let dbg = std::env::var("RCWX_RT_STAGES").is_ok();
            let mut tt = std::time::Instant::now();
            eng.replay_inputs(&p.encode_plan, &[]);
            if dbg { eprintln!("    [synth] encode: {:.2}ms", tt.elapsed().as_secs_f64()*1000.0); tt = std::time::Instant::now(); }
            eng.write_f32(&p.har_in, &har);
            eng.replay_inputs(&p.decode_plan, &[]);
            if dbg { eprintln!("    [synth] decode replay: {:.2}ms", tt.elapsed().as_secs_f64()*1000.0); tt = std::time::Instant::now(); }
            let out = eng.readback_f32(&p.audio_out);
            if dbg { eprintln!("    [synth] readback: {:.2}ms", tt.elapsed().as_secs_f64()*1000.0); }
            out
        };

        stage("synth", &mut stage_t);
        // 末尾 decoder overlap 2 フレームは未確定 → SOLA で消費
        let out_frames = n100 - DECODER_OVERLAP_FRAMES;
        // 出力対象はこの hop の新規分 = 末尾から new100+overlap 手前 ... 実際は
        // 「前 hop で未確定だった分 + 新規」だが、hold-back 設計では
        // 各 hop の確定出力 = 末尾 (new100 + overlap) フレーム分の音声から
        // SOLA で target_len = new100*zc を取り出す。
        let tail_frames = (new100 + DECODER_OVERLAP_FRAMES).min(n100);
        let take_from = audio48.len() - tail_frames * ZC;
        let raw: Vec<f32> = audio48[take_from..].to_vec();
        audio48.clear();

        // soft clip
        let clipped: Vec<f32> = raw.iter().map(|v| soft_clip(*v)).collect();
        let target = new100 * ZC;
        let mut out = sola_crossfade(&clipped, &mut self.sola, target);
        self.stats.solo_offset = self.sola.last_offset;

        // 出力レート変換
        if let Some(rs) = self.resampler_out.as_mut() {
            out = rs.resample_chunk(&out);
        }
        let out_gain = 10f64.powf(self.cfg.output_gain_db / 20.0) as f32;
        if out_gain != 1.0 {
            for v in out.iter_mut() {
                *v *= out_gain;
            }
        }
        if self.cfg.postprocess {
            out = self.post.process(&out);
        }

        // フィードバック検出 (10 hop ごと)
        if self.cfg.feedback_detect {
            self.out_hist.extend_from_slice(&out);
            let keep = MODEL_SR as usize; // 1s
            if self.out_hist.len() > keep * 2 {
                let drop = self.out_hist.len() - keep * 2;
                self.out_hist.drain(..drop);
            }
            self.stats.hops += 1;
            if self.stats.hops % 10 == 0 && self.out_hist.len() > keep + out.len() {
                let hay_start = self.out_hist.len() - keep - out.len();
                let corr = rcwx_dsp::feedback::max_normalized_lag_correlation(
                    &out,
                    &self.out_hist[hay_start..hay_start + keep],
                );
                self.stats.feedback_corr = corr as f64;
            }
        } else {
            self.stats.hops += 1;
        }

        stage("sola+post", &mut stage_t);
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / batch_n.max(1) as f64;
        // 初回 2 推論 (プラン構築) はウォームアップとして統計から除外
        self.warmups += 1;
        if self.warmups > 2 {
            self.stats.record_inference(ms);
        }

        // 履歴を窓長に縮める
        let win_target = self.cfg.hubert_ctx_samples() + hop16;
        if self.hist.len() > win_target {
            let drop = self.hist.len() - win_target;
            self.hist.drain(..drop);
        }
        out
    }
}

/// Python の soft clip 相当 (ソフトニー): |x|≤0.8 は通し、超過分は tanh で飽和。
fn soft_clip(x: f32) -> f32 {
    let a = x.abs();
    if a <= 0.8 {
        x
    } else {
        x.signum() * (0.8 + 0.2 * (((a - 0.8) / 0.2).tanh()))
    }
}

/// pitch shift は有声のみ (CLI と同じ式)。
pub fn condition_f0(f0v: &mut [f32], shift_st: f64) {
    if shift_st.abs() > 0.01 {
        let r = 2.0f64.powf(shift_st / 12.0) as f32;
        for v in f0v.iter_mut() {
            if *v > 0.0 {
                *v *= r;
            }
        }
    }
}
