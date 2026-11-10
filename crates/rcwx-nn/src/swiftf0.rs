//! SwiftF0 (ONNX MIT, 95.8K params) の独自エンジン移植。
//!
//! 重み: tools/convert_swiftf0.py → ~/.cache/rcwx/models/rust/swiftf0.safetensors
//! - pad 384 両端 → STFT(1024, hop 256, periodic-hann) → magnitude bins 3..135
//! - log(mag+1e-8) → [1, 132, T]
//! - conv2d 5x5 pad2 (SAME_UPPER) ×5 (1→8→16→32→64→1) + relu — WGSL im2col+matmul
//! - freq_projection 1x1 (132→200) → softmax → argmax + ±9bin 加重平均 — CPU
//! - hop 256→160 補間 (linear align_corners=False / voiced nearest) — CPU
//! - オクターブ解決 Viterbi (波形周期性) — CPU
//!
//! 実行時の Python/ort 依存ゼロ。

use anyhow::{Context, Result};

use crate::engine::{Engine, GpuTensor, Plan};

const N_FFT: usize = 1024;
const STFT_HOP: usize = 256;
const PAD: usize = 384;
const FREQ_BINS: u32 = 132; // STFT bins 3..135
const N_PITCH_BINS: usize = 200;
const BIN_LO: usize = 3;

/// RCWX 既定の検出パラメータ (rcwx/models/swiftf0.py 準拠)。
pub const CONFIDENCE_THRESHOLD: f32 = 0.35;
pub const F0_MIN: f64 = 50.0;
pub const F0_MAX: f64 = 1100.0;
/// voicing 判定に使うモデル側レート制限 (46.875–2093.75) は F0_MIN/F0_MAX に包含される。
const MODEL_FMIN: f64 = 46.875;
const MODEL_FMAX: f64 = 2093.75;

const TARGET_HOP: usize = 160;

struct SwiftPlan {
    input: GpuTensor,
    proj: GpuTensor,
    plan: Plan,
}

pub struct SwiftF0 {
    w: std::collections::HashMap<String, GpuTensor>,
    pub window: Vec<f32>,  // [1024]
    pub centers: Vec<f32>, // [200]
    /// n_frames ごとに GPU プランを再利用し、毎 hop のバッファ破棄を避ける。
    plans: std::cell::RefCell<std::collections::HashMap<u32, SwiftPlan>>,
}

pub fn default_swiftf0_path() -> std::path::PathBuf {
    crate::rust_models_dir().join("swiftf0.safetensors")
}

impl SwiftF0 {
    pub fn load(eng: &Engine, path: &std::path::Path) -> Result<Self> {
        let raw = std::fs::read(path)?;
        let st = safetensors::SafeTensors::deserialize(&raw)?;
        let mut w = std::collections::HashMap::new();
        let mut cpu = std::collections::HashMap::new();
        for name in st.names() {
            let v = st.tensor(name)?;
            let shape = v.shape().to_vec();
            let n: usize = shape.iter().product();
            let mut out = Vec::with_capacity(n);
            for c in v.data().chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            if matches!(name, "window" | "centers") {
                cpu.insert(name.to_string(), out.clone());
            }
            w.insert(name.to_string(), eng.tensor_from_f32(&out, shape));
        }
        Ok(Self {
            w,
            window: cpu.get("window").context("window")?.clone(),
            centers: cpu.get("centers").context("centers")?.clone(),
            plans: std::cell::RefCell::new(std::collections::HashMap::new()),
        })
    }

    fn gpu_plan(&self, eng: &Engine, t: u32) -> SwiftPlan {
        let mut g = eng.graph();
        let input = eng.tensor_uninit((FREQ_BINS as usize) * t as usize);
        let mut cur = input.clone_handle();
        let (mut cin, layers) = (1u32, [(1u32, 8u32), (8, 16), (16, 32), (32, 64), (64, 1)]);
        for (i, &(c_in, c_out)) in layers.iter().enumerate() {
            debug_assert_eq!(cin, c_in);
            let ic = eng.tensor_uninit((cin as usize) * 25 * FREQ_BINS as usize * t as usize);
            g.im2col2d_k(&cur, &ic, cin, FREQ_BINS, t, 5, 2);
            let out = eng.tensor_uninit((c_out as usize) * FREQ_BINS as usize * t as usize);
            g.matmul_ex(
                self.g(&format!("conv{i}.weight")),
                &ic,
                Some(self.g(&format!("conv{i}.bias"))),
                &out,
                c_out,
                FREQ_BINS * t,
                cin * 25,
                1,
                cin * 25,
                1,
                0,
                0,
                1,
                FREQ_BINS * t,
                0,
                0,
                FREQ_BINS * t,
                1,
                0,
                0,
                1.0,
                1,
            );
            g.relu(&out, c_out * FREQ_BINS * t);
            cur = out;
            cin = c_out;
        }
        let proj = eng.tensor_uninit(N_PITCH_BINS * t as usize);
        g.matmul_ex(
            self.g("proj.weight"),
            &cur,
            Some(self.g("proj.bias")),
            &proj,
            N_PITCH_BINS as u32,
            t,
            FREQ_BINS,
            1,
            FREQ_BINS,
            1,
            0,
            0,
            1,
            t,
            0,
            0,
            t,
            1,
            0,
            0,
            1.0,
            1,
        );
        let plan = g.into_plan().expect("swiftf0 plan");
        SwiftPlan { input, proj, plan }
    }

    fn g(&self, name: &str) -> &GpuTensor {
        self.w.get(name).unwrap_or_else(|| panic!("weight {name}"))
    }

    /// STFT と GPU conv を投入する。読み戻しは呼び出し側。無音や空は None。
    pub fn launch_gpu(&self, eng: &Engine, audio: &[f32]) -> Option<u32> {
        let t = self.prepare_gpu(eng, audio)?;
        let plans = self.plans.borrow();
        let entry = plans.get(&t).unwrap();
        eng.replay(&entry.plan, None);
        Some(t)
    }

    /// STFT (CPU) + 入力書き込み + プラン構築まで。submit は呼び出し側で
    /// 統合する (リアルタイム経路)。無音や空は None。
    pub fn prepare_gpu(&self, eng: &Engine, audio: &[f32]) -> Option<u32> {
        let n = audio.len();
        let n_frames = 1 + (n + 2 * PAD).saturating_sub(N_FFT) / STFT_HOP;
        if n == 0 || n_frames == 0 {
            return None;
        }
        if audio.iter().fold(0.0f32, |m, v| m.max(v.abs())) < 1e-6 {
            return None;
        }

        // ---- STFT (CPU fp32, rustfft) → log magnitude [132, T] ----
        let padded_len = n + 2 * PAD;
        let mut buf = vec![0.0f32; padded_len];
        buf[PAD..PAD + n].copy_from_slice(audio);
        let mut logmag = vec![0.0f32; FREQ_BINS as usize * n_frames];
        let mut planner = rustfft::FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let mut frame = vec![rustfft::num_complex::Complex::<f32>::new(0.0, 0.0); N_FFT];
        for f in 0..n_frames {
            for i in 0..N_FFT {
                frame[i] = rustfft::num_complex::Complex {
                    re: buf[f * STFT_HOP + i] * self.window[i],
                    im: 0.0,
                };
            }
            fft.process(&mut frame);
            for (b, slot) in logmag
                .iter_mut()
                .skip(f)
                .step_by(n_frames)
                .take(FREQ_BINS as usize)
                .enumerate()
            {
                let c = frame[BIN_LO + b];
                let mag = (c.re * c.re + c.im * c.im).sqrt();
                *slot = (mag + 1e-8).ln();
            }
        }

        // ---- conv スタック (GPU) + freq_projection → probs [200, T] ----
        let t = n_frames as u32;
        if !self.plans.borrow().contains_key(&t) {
            let built = self.gpu_plan(eng, t);
            self.plans.borrow_mut().insert(t, built);
        }
        let plans = self.plans.borrow();
        let entry = plans.get(&t).unwrap();
        eng.write_f32(&entry.input, &logmag);
        Some(t)
    }

    /// キャッシュ済み plan と proj 出力のクローン (統合 submit 用)。
    pub fn plan_and_proj_clone(&self, t: u32) -> (GpuTensor, crate::engine::Plan) {
        let plans = self.plans.borrow();
        let entry = plans.get(&t).unwrap();
        (entry.proj.clone_handle(), entry.plan.clone())
    }

    pub fn proj_handle(&self, t: u32) -> GpuTensor {
        self.plans.borrow().get(&t).unwrap().proj.clone_handle()
    }

    /// probs は [200, T] (bin 主導)。
    pub fn pitch_from_probs(&self, probs: &[f32], t: u32) -> (Vec<f32>, Vec<f32>) {
        let mut pitch = vec![0.0f32; t as usize];
        let mut conf = vec![0.0f32; t as usize];
        let mut row = vec![0.0f32; N_PITCH_BINS];
        for f in 0..t as usize {
            for (b, slot) in row.iter_mut().enumerate() {
                *slot = probs[b * t as usize + f];
            }
            let mut mx = f32::NEG_INFINITY;
            let mut am = 0usize;
            for (b, &v) in row.iter().enumerate() {
                if v > mx {
                    mx = v;
                    am = b;
                }
            }
            let mut esum = 0.0f32;
            let mut exps = [0.0f32; N_PITCH_BINS];
            for (b, &v) in row.iter().enumerate() {
                let e = (v - mx).exp();
                exps[b] = e;
                esum += e;
            }
            let mut wsum = 0.0f32;
            let mut psum = 0.0f32;
            let lo = am.saturating_sub(9);
            let hi = (am + 9).min(N_PITCH_BINS - 1);
            for b in lo..=hi {
                let p = exps[b] / esum;
                wsum += p;
                psum += p * self.centers[b];
            }
            pitch[f] = psum / (wsum + 1e-7);
            conf[f] = wsum;
        }
        (pitch, conf)
    }

    /// 内部 hop 256 での生検出。(pitch_hz, confidence) を返す。
    pub fn infer_raw(&self, eng: &Engine, audio: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = audio.len();
        let n_frames = 1 + (n + 2 * PAD).saturating_sub(N_FFT) / STFT_HOP;
        if n == 0 || n_frames == 0 {
            return (Vec::new(), Vec::new());
        }
        let n_target = n / TARGET_HOP;
        if audio.iter().fold(0.0f32, |m, v| m.max(v.abs())) < 1e-6 {
            return (vec![0.0; n_target.max(1)], vec![0.0; n_frames]);
        }
        let Some(t) = self.launch_gpu(eng, audio) else {
            return (Vec::new(), Vec::new());
        };
        let probs = {
            let plans = self.plans.borrow();
            eng.readback_f32(&plans.get(&t).unwrap().proj)
        };
        self.pitch_from_probs(&probs, t)
    }

    /// RCWX 相当の検出: 100fps (hop 160) への補間 + オクターブ解決済み F0。
    pub fn detect(&self, eng: &Engine, audio: &[f32]) -> Vec<f32> {
        let n = audio.len();
        let n_target = n / TARGET_HOP;
        if n == 0 || n_target == 0 {
            return Vec::new();
        }
        let timing = std::env::var("RCWX_TIMING").is_ok();
        let t0 = std::time::Instant::now();
        let (pitch, conf) = self.infer_raw(eng, audio);
        if timing {
            eprintln!(
                "[timing] swiftf0.infer_raw ({}s audio): {:.2}s",
                n as f64 / 16000.0,
                t0.elapsed().as_secs_f64()
            );
        }
        let t1 = std::time::Instant::now();
        let out = self.finish(audio, &pitch, &conf);
        if timing {
            eprintln!(
                "[timing] swiftf0.finish: {:.2}s",
                t1.elapsed().as_secs_f64()
            );
        }
        out
    }

    /// GPU 読み戻し済みの (pitch, confidence) から 100fps F0 を作る。
    pub fn finish(&self, audio: &[f32], pitch: &[f32], conf: &[f32]) -> Vec<f32> {
        let n = audio.len();
        let n_target = n / TARGET_HOP;
        if n == 0 || n_target == 0 {
            return Vec::new();
        }
        let n_swift = pitch.len();
        if n_swift == 0 {
            return vec![0.0; n_target];
        }

        // voicing (モデル域 + RCWX 域の積)
        let voiced: Vec<bool> = pitch
            .iter()
            .zip(conf.iter())
            .map(|(&p, &c)| {
                c > CONFIDENCE_THRESHOLD
                    && (p as f64) >= MODEL_FMIN
                    && (p as f64) <= MODEL_FMAX
                    && (p as f64) >= F0_MIN
                    && (p as f64) <= F0_MAX
            })
            .collect();
        let mut f0 = vec![0.0f32; n_swift];
        for i in 0..n_swift {
            f0[i] = if voiced[i] { pitch[i] } else { 0.0 };
        }

        // hop 256 → 160 補間 (linear align_corners=False / voiced nearest)
        let (f0_r, conf_r, voiced_r): (Vec<f32>, Vec<f32>, Vec<bool>) = if n_swift == n_target {
            (f0.clone(), conf.to_vec(), voiced.clone())
        } else {
            let scale = n_swift as f32 / n_target as f32;
            let mut fo = vec![0.0f32; n_target];
            let mut co = vec![0.0f32; n_target];
            let mut vo = vec![false; n_target];
            for r in 0..n_target {
                // linear align_corners=False
                let src = ((r as f32 + 0.5) * scale - 0.5).max(0.0);
                let i0 = src.floor() as usize;
                let i1 = (i0 + 1).min(n_swift - 1);
                let fr = src - i0 as f32;
                fo[r] = f0[i0] + (f0[i1] - f0[i0]) * fr;
                co[r] = (conf[i0] + (conf[i1] - conf[i0]) * fr).clamp(0.0, 1.0);
                // nearest (torch upsample_nearest: floor(r*scale))
                let vi = ((r as f32 * scale) as usize).min(n_swift - 1);
                vo[r] = voiced[vi];
            }
            (fo, co, vo)
        };
        let mut f0_out = f0_r.clone();
        for r in 0..n_target {
            if !voiced_r[r] {
                f0_out[r] = 0.0;
            }
        }

        resolve_octave_from_waveform(&mut f0_out, &voiced_r, audio, &conf_r);
        f0_out
    }
}

/// 有声セグメントの列挙 (rcwx/models/swiftf0.py _iter_voiced_segments 相当)。
fn voiced_segments(mask: &[bool]) -> Vec<(usize, usize)> {
    let mut segs = Vec::new();
    let mut i = 0;
    while i < mask.len() {
        while i < mask.len() && !mask[i] {
            i += 1;
        }
        let start = i;
        while i < mask.len() && mask[i] {
            i += 1;
        }
        if start < i {
            segs.push((start, i));
        }
    }
    segs
}

/// 反射パディング付き解析フレーム抽出 (64ms, hop 160, 平均除去)。
fn reflected_frames(
    audio: &[f32],
    frame_count: usize,
    frame_size: usize,
    hop: usize,
) -> Vec<Vec<f32>> {
    let n = audio.len();
    let half = frame_size / 2;
    let mut out = Vec::with_capacity(frame_count);
    if n <= 1 {
        return vec![vec![0.0; frame_size]; frame_count];
    }
    let get = |i: i64| -> f32 {
        // reflect (周期 2n-2, 端点不含む)
        let period = (2 * n - 2) as i64;
        let mut m = i.rem_euclid(period);
        if m >= n as i64 {
            m = period - m;
        }
        audio[m as usize]
    };
    for f in 0..frame_count {
        let center = ((f as f64 + 0.5) * hop as f64) as i64;
        let mut fr = Vec::with_capacity(frame_size);
        let mut sum = 0.0f64;
        for j in 0..frame_size {
            let v = get(center - half as i64 + j as i64);
            fr.push(v);
            sum += v as f64;
        }
        let mean = (sum / frame_size as f64) as f32;
        for v in fr.iter_mut() {
            *v -= mean;
        }
        out.push(fr);
    }
    out
}

/// 正規化自己相関スコア。
fn normalized_periodicity(frame: &[f32], lag: usize) -> f64 {
    if lag == 0 || lag >= frame.len() - 1 {
        return -1.0;
    }
    let mut dot = 0.0f64;
    let mut ea = 0.0f64;
    let mut eb = 0.0f64;
    for i in 0..frame.len() - lag {
        let a = frame[i] as f64;
        let b = frame[i + lag] as f64;
        dot += a * b;
        ea += a * a;
        eb += b * b;
    }
    let energy = ea * eb;
    if energy < 1e-16 {
        return -1.0;
    }
    dot / energy.sqrt()
}

/// 波形周期性 + Viterbi によるオクターブ誤検出の解決。
fn resolve_octave_from_waveform(
    f0: &mut [f32],
    voiced: &[bool],
    audio: &[f32],
    confidence: &[f32],
) {
    let n_frames = f0.len();
    if n_frames == 0 {
        return;
    }
    for (i, &v) in voiced.iter().enumerate() {
        if !v {
            f0[i] = 0.0;
        }
    }
    if !voiced.iter().any(|&v| v) {
        return;
    }

    let frame_size = {
        let fs = (16000.0f64 * 0.064).round() as usize;
        let fs = fs.max(256);
        if fs % 2 != 0 { fs + 1 } else { fs }
    };
    let analysis = reflected_frames(audio, n_frames, frame_size, TARGET_HOP);

    let multipliers = [0.25f64, 0.5, 1.0, 2.0];
    let prior_weight = 0.08f64;
    let trans_weight = 0.28;
    let state_weight = 0.06;

    for (seg_start, seg_end) in voiced_segments(voiced) {
        // states[t] = [(f0, mult, emission), ...]
        let mut states: Vec<Vec<(f64, f64, f64)>> = Vec::with_capacity(seg_end - seg_start);
        for &idx in &(seg_start..seg_end).collect::<Vec<_>>() {
            let raw = f0[idx].max(0.0) as f64;
            if raw <= 0.0 {
                states.push(vec![(0.0, 1.0, -1.0)]);
                continue;
            }
            let conf_i = confidence.get(idx).copied().unwrap_or(0.5).clamp(0.0, 1.0) as f64;
            let mut cand = Vec::new();
            for &mult in &multipliers {
                let cand_f0 = raw * mult;
                if cand_f0 < F0_MIN || cand_f0 > F0_MAX {
                    continue;
                }
                let lag = (16000.0 / cand_f0.max(1e-6)).round() as usize;
                let periodicity = normalized_periodicity(&analysis[idx], lag);
                let prior_penalty = prior_weight * conf_i * mult.log2().abs();
                cand.push((cand_f0, mult, periodicity - prior_penalty));
            }
            if cand.is_empty() {
                cand.push((raw, 1.0, -1.0));
            }
            states.push(cand);
        }

        let seg_len = states.len();
        if seg_len == 0 {
            continue;
        }
        let mut dp_prev: Vec<f32> = states[0].iter().map(|s| s.2 as f32).collect();
        let mut back_ptrs: Vec<Vec<i32>> = vec![vec![-1; states[0].len()]; 1];

        for t in 1..seg_len {
            let prev = &states[t - 1];
            let cur = &states[t];
            let mut dp_cur = vec![-1e9f32; cur.len()];
            let mut back_cur = vec![-1i32; cur.len()];
            for (j, &(f_j, m_j, e_j)) in cur.iter().enumerate() {
                let mut best = -1e9f64;
                let mut best_k = -1i32;
                for (k, &(f_k, m_k, _)) in prev.iter().enumerate() {
                    let trans_cost = if f_j <= 0.0 || f_k <= 0.0 {
                        0.0
                    } else {
                        let log_step = (f_j.max(1e-6) / f_k.max(1e-6)).log2().abs();
                        let state_step = (m_j.max(1e-6) / m_k.max(1e-6)).log2().abs();
                        -(trans_weight * log_step + state_weight * state_step)
                    };
                    let score = dp_prev[k] as f64 + e_j + trans_cost;
                    if score > best {
                        best = score;
                        best_k = k as i32;
                    }
                }
                dp_cur[j] = best as f32;
                back_cur[j] = best_k;
            }
            dp_prev = dp_cur;
            back_ptrs.push(back_cur);
        }

        let mut state_idx = dp_prev
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        for t in (0..seg_len).rev() {
            let (f_sel, _, _) = states[t][state_idx];
            f0[seg_start + t] = f_sel as f32;
            if t > 0 {
                let prev_idx = back_ptrs[t][state_idx];
                state_idx = if prev_idx >= 0 { prev_idx as usize } else { 0 };
            }
        }
    }

    for (i, &v) in voiced.iter().enumerate() {
        if !v {
            f0[i] = 0.0;
        }
    }
}
