//! rcwx-cli — 完全 Rust 単一バイナリ音声変換 (wav in → F0+HuBERT+Synth → wav out)。
//!
//! 実行時の Python/torch/ort 依存ゼロ (Issue #6 セクション5 のオフライン経路)。
//!
//! 使い方:
//!   rcwx-cli run input.wav -o output.wav [--pitch 0] [--moe 0.45] [--noise 0.0]

mod commands;
mod pth_info;
mod rt_cmd;

use anyhow::{Context, Result, bail};
use rcwx_nn::engine::{Backend, Engine};
use rcwx_nn::f0;
use rcwx_nn::hubert::Hubert;
use rcwx_nn::rmvpe::Rmvpe;
use rcwx_nn::synth::Synth;

fn weights_dir() -> std::path::PathBuf {
    rcwx_nn::rust_models_dir()
}

/// 線形伸縮 (出力長厳密一致)。Python `resample(len→len)` 相当。
fn resample_to(x: &[f32], out_len: usize) -> Vec<f32> {
    if x.is_empty() || out_len == 0 {
        return vec![0.0; out_len];
    }
    if x.len() == out_len {
        return x.to_vec();
    }
    let mut out = vec![0.0f32; out_len];
    if out_len == 1 {
        out[0] = x[0];
        return out;
    }
    let step = (x.len() - 1) as f32 / (out_len - 1) as f32;
    for (i, o) in out.iter_mut().enumerate() {
        let pos = i as f32 * step;
        let i0 = pos.floor() as usize;
        let i1 = (i0 + 1).min(x.len() - 1);
        let fr = pos - i0 as f32;
        *o = x[i0] + (x[i1] - x[i0]) * fr;
    }
    out
}

fn resample_linear(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let out_len = (x.len() as u64 * to as u64 / from as u64) as usize;
    resample_to(x, out_len)
}

/// 5次 Butterworth ハイパス (scipy butter(5) 係数・odd padding + filtfilt)。
fn highpass_filter(x: &[f32], _sr: u32, _cutoff: f64) -> Vec<f32> {
    if x.len() < 100 {
        return x.to_vec();
    }
    const B: [f64; 6] = [
        0.9699606451838447,
        -4.849803225919223,
        9.699606451838447,
        -9.699606451838447,
        4.849803225919223,
        -0.9699606451838447,
    ];
    const A: [f64; 6] = [
        1.0,
        -4.939001819168364,
        9.757863526739543,
        -9.639544849413458,
        4.761506797356209,
        -0.9408236532054606,
    ];

    // direct form IIR (scipy lfilter と同一出力)
    let lfilter = |s: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0f64; s.len()];
        for i in 0..s.len() {
            let mut acc = B[0] * s[i];
            for k in 1..6 {
                if i >= k {
                    acc += B[k] * s[i - k] - A[k] * y[i - k];
                }
            }
            y[i] = acc;
        }
        y
    };

    let xd: Vec<f64> = x.iter().map(|v| *v as f64).collect();
    let n = xd.len();
    // 十分な odd padding (scipy デフォルト 18 + startup transient 余裕)
    let padlen = 100usize;

    let mut padded = Vec::with_capacity(n + 2 * padlen);
    // odd extension: 2*x[0] - x[padlen..0] (逆順)
    for i in (1..=padlen).rev() {
        let idx = i.min(n - 1);
        padded.push(2.0 * xd[0] - xd[idx]);
    }
    padded.extend_from_slice(&xd);
    // right odd
    for i in 1..=padlen {
        let idx = n.saturating_sub(1 + i);
        padded.push(2.0 * xd[n - 1] - xd[idx]);
    }

    // forward
    let fwd = lfilter(&padded);
    // reverse
    let rev_in: Vec<f64> = fwd.iter().rev().copied().collect();
    let rev = lfilter(&rev_in);
    // un-reverse and trim
    (0..n)
        .map(|i| rev[rev.len() - 1 - (i + padlen)] as f32)
        .collect()
}

/// pitch shift は有声のみ。
fn condition_f0(f0v: &mut [f32], shift_st: f64) {
    if shift_st.abs() > 0.01 {
        let r = 2.0f64.powf(shift_st / 12.0);
        for v in f0v.iter_mut() {
            if *v > 0.0 {
                *v = (*v as f64 * r) as f32;
            }
        }
    }
}

/// stabilize_f0_boundaries (f0_processing.py と同一): 端の有声フレームを中央値へ ramp blend。
/// 長尺入力の F0 抽出を窓分割する (RMVPE の dispatch グリッド上限対策)。
/// 25s 窓 + 1s 文脈で中央のフレームのみ採録。100fps (160 samples/frame)。
fn windowed_f0(wav: &[f32], n_total: usize, want_frames: usize, infer: impl Fn(&[f32]) -> Vec<f32>) -> Vec<f32> {
    let win = 400_000usize;
    let ctx = 16_000usize;
    if n_total <= win + ctx {
        return infer(wav);
    }
    let mut out: Vec<f32> = Vec::with_capacity(want_frames);
    let mut seg_start = 0usize;
    while seg_start < n_total {
        let seg_end = (seg_start + win).min(n_total);
        let w_start = seg_start.saturating_sub(ctx);
        let w_end = (seg_end + ctx).min(n_total);
        let f = infer(&wav[w_start..w_end]);
        let ctx_f = (seg_start - w_start) / 160;
        let lo = if seg_start == 0 { 0 } else { ctx_f };
        let hi = if seg_end == n_total {
            f.len()
        } else {
            (ctx_f + (seg_end - seg_start) / 160).min(f.len())
        };
        out.extend_from_slice(&f[lo..hi]);
        seg_start = seg_end;
    }
    out.resize(want_frames, 0.0);
    out
}

fn stabilize_f0_boundaries(f0: &mut [f32], edge_frames: usize) {
    let n = f0.len();
    if n < 4 || edge_frames == 0 {
        return;
    }
    let voiced_idx: Vec<usize> = (0..n).filter(|&i| f0[i] > 0.0).collect();
    if voiced_idx.len() < 4 {
        return;
    }
    let first = voiced_idx[0];
    let last = *voiced_idx.last().unwrap();

    let median_of = |vals: &[f32]| -> f32 {
        let mut v: Vec<f32> = vals.iter().filter(|x| **x > 0.0).copied().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() { 0.0 } else { v[v.len() / 2] }
    };

    if first <= edge_frames {
        let search_end = (first + edge_frames * 3).min(n);
        let target = median_of(&f0[first..search_end]);
        if target > 0.0 {
            let blend_end = (first + edge_frames).min(n);
            if blend_end > first {
                let m = blend_end - first;
                for k in 0..m {
                    let w = k as f32 / m as f32; // linspace(0,1,m)
                    let i = first + k;
                    f0[i] = target * (1.0 - w) + f0[i] * w;
                }
            }
        }
    }

    if last >= n.saturating_sub(1 + edge_frames) {
        let search_start = last.saturating_sub(edge_frames * 3 - 1);
        let target = median_of(&f0[search_start..=last]);
        if target > 0.0 {
            let blend_start = last.saturating_sub(edge_frames - 1);
            if last + 1 > blend_start {
                let m = last + 1 - blend_start;
                for k in 0..m {
                    let w = k as f32 / m as f32;
                    let i = blend_start + k;
                    f0[i] = f0[i] * (1.0 - w) + target * w;
                }
            }
        }
    }
}

/// 端 8ms エッジフェード。
fn edge_fade(x: &mut [f32], sr: u32) {
    let n = (sr as f64 * 0.008) as usize;
    if x.len() < 2 * n || n == 0 {
        return;
    }
    for i in 0..n {
        let w = i as f32 / n as f32;
        x[i] *= w;
        x[x.len() - 1 - i] *= w;
    }
}

struct Args {
    input: String,
    output: String,
    pitch: f64,
    noise: f64,
    f0_method: String,
    backend: String,
    moe: f64,
    index_rate: f64,
    index_k: usize,
    /// RVC モデル .pth (指定時は synth.safetensors の代わりに直接ロード)
    model: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        input: String::new(),
        output: "output.wav".into(),
        pitch: 0.0,
        noise: 0.0,
        f0_method: "rmvpe".into(),
        backend: "dx12".into(),
        moe: 0.0,
        index_rate: 0.0,
        index_k: 4,
        model: None,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "run" => {}
            "--pitch" | "-p" => {
                i += 1;
                a.pitch = args.get(i).context("--pitch value")?.parse()?;
            }
            "--model" | "-m" => {
                i += 1;
                a.model = Some(args.get(i).context("--model value")?.to_string());
            }
            "--noise" => {
                i += 1;
                a.noise = args.get(i).context("--noise value")?.parse()?;
            }
            "--moe" => {
                i += 1;
                a.moe = args.get(i).context("--moe value")?.parse()?;
            }
            "--index-rate" => {
                i += 1;
                a.index_rate = args.get(i).context("--index-rate value")?.parse()?;
            }
            "--index-k" => {
                i += 1;
                a.index_k = args.get(i).context("--index-k value")?.parse()?;
            }
            "--f0-method" => {
                i += 1;
                a.f0_method = args.get(i).context("--f0-method value")?.to_string();
            }
            "--backend" => {
                i += 1;
                a.backend = args.get(i).context("--backend value")?.to_string();
            }
            "-o" | "--output" => {
                i += 1;
                a.output = args.get(i).context("-o value")?.to_string();
            }
            other if a.input.is_empty() && !other.starts_with('-') => a.input = other.to_string(),
            other => bail!("unknown arg: {other}"),
        }
        i += 1;
    }
    if a.input.is_empty() {
        bail!(
            "usage: rcwx-cli run input.wav [--model voice.pth] [-o output.wav] [--pitch N] [--moe 0.45] [--index-rate 0.15]"
        );
    }
    Ok(a)
}

fn main() -> Result<()> {
    // サブコマンド: run / info / devices / download / logs
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    match raw_args.first().map(|s| s.as_str()) {
        Some("info") => return commands::cmd_info(&raw_args[1..]),
        Some("devices") => return commands::cmd_devices(),
        Some("download") => return commands::cmd_download(&raw_args[1..]),
        Some("logs") => return commands::cmd_logs(&raw_args[1..]),
        Some("rt") => return rt_cmd::cmd_rt(&raw_args[1..]),
        _ => {}
    }
    let args = parse_args()?;
    let dir = weights_dir();
    // 共通重み (.pth 直読み時は synth バンドル不要)
    let mut need = vec!["hubert.safetensors", "rmvpe_folded.safetensors"];
    if args.model.is_none() {
        need.extend(["synth.safetensors", "synth.json"]);
    }
    let missing: Vec<&str> = need
        .iter()
        .filter(|f| !dir.join(f).exists())
        .copied()
        .collect();
    if !missing.is_empty() {
        bail!(
            "共通重みが未準備です: {} ({})\n  `rcwx-cli download` でダウンロードと変換を実行してください。",
            missing.join(", "),
            dir.display()
        );
    }
    if let Some(m) = &args.model {
        if !std::path::Path::new(m).exists() {
            bail!("モデルファイルがありません: {m}");
        }
    }

    // ---- wav 読み込み ----
    let mut reader =
        hound::WavReader::open(&args.input).with_context(|| format!("open {}", args.input))?;
    let sr = reader.spec().sample_rate;
    let ch = reader.spec().channels;
    let samples: Vec<f32> = reader
        .samples::<i32>()
        .map(|s| s.map(|v| v as f32 / 32768.0).unwrap_or(0.0))
        .collect();
    let mut audio: Vec<f32> = if ch > 1 {
        (0..samples.len() / ch as usize)
            .map(|i| {
                (0..ch as usize)
                    .map(|c| samples[i * ch as usize + c])
                    .sum::<f32>()
                    / ch as f32
            })
            .collect()
    } else {
        samples
    };
    println!(
        "input: {} ({}Hz, ch={}, {} samples)",
        args.input,
        sr,
        ch,
        audio.len()
    );

    // ---- 16k 化 + HPF ----
    audio = resample_linear(&audio, sr, 16000);
    audio = highpass_filter(&audio, 16000, 48.0);
    if std::env::var("RCWX_CLI_DUMP").is_ok() {
        let _ = std::fs::write(
            "target/golden/cli_rust_hp.bin",
            bytemuck::cast_slice(&audio),
        );
    }

    // ---- padding (batch 相当) ----
    let hubert_hop = 320usize;
    let input_length = audio.len();
    let base_pad = 800usize;
    let min_input = (64usize / 2 + 2) * hubert_hop;
    let total = base_pad + input_length + base_pad;
    let t_pad = if total < min_input {
        base_pad + (min_input - total + 1) / 2
    } else {
        base_pad
    };
    let padded = t_pad + input_length + t_pad;
    let extra_pad = (hubert_hop - padded % hubert_hop) % hubert_hop;
    let n_total = padded + extra_pad;
    let mut wav_pad = vec![0.0f32; n_total];
    for i in 0..t_pad.min(input_length) {
        wav_pad[t_pad - 1 - i] = audio[(i + 1).min(input_length - 1)];
    }
    wav_pad[t_pad..t_pad + input_length].copy_from_slice(&audio);
    for i in 0..(t_pad + extra_pad) {
        let src = input_length
            .saturating_sub(2 + i)
            .min(input_length.saturating_sub(1));
        wav_pad[t_pad + input_length + i] = audio[src];
    }
    if std::env::var("RCWX_CLI_DUMP").is_ok() {
        let _ = std::fs::write(
            "target/golden/cli_rust_wavpad.bin",
            bytemuck::cast_slice(&wav_pad),
        );
    }
    let t50 = (n_total / hubert_hop).saturating_sub(1);
    let t100 = t50 * 2;

    // ---- engine / models ----
    let backend = match args.backend.as_str() {
        "vulkan" => Backend::Vulkan,
        _ => Backend::Dx12,
    };
    let eng = Engine::new(backend).context("engine init")?;
    println!(
        "adapter: {} ({:?})",
        eng.adapter_info.name, eng.adapter_info.backend
    );
    let hubert = Hubert::load(&eng, &dir.join("hubert.safetensors")).context("HuBERT load")?;
    let rmvpe = Rmvpe::load(&eng, &dir).context("RMVPE load")?;
    let synth = match &args.model {
        Some(m) => Synth::load_from_pth(&eng, std::path::Path::new(m))
            .with_context(|| format!("モデル {m} の読み込み"))?,
        None => Synth::load(&eng, &dir).context("Synth load")?,
    };
    let model_version = synth.cfg.version;
    let fd = synth.cfg.in_channels as usize;
    println!(
        "model: RVC v{model_version} ({fd}-dim features, sr={})",
        synth.cfg.sr
    );

    // ---- HuBERT → [T50, fd] (v1=256-dim/layer9, v2=768-dim/layer12) ----
    // full-context の attention は T² のスコア行列をレイヤー分保持するため
    // 約束 40s (T50=2000) を超える入力は窓分割する (両側 2s コンテキスト付き、
    // 中央のみ採録)。窓内は完全 full-context 相当。
    // RCWX_HUBERT_CHUNK=1 で短尺でも分割パスを強制 (品質比較用)。
    let force_hubert_chunk = std::env::var("RCWX_HUBERT_CHUNK").ok().as_deref() == Some("1");
    let ctx_samples = 64_000usize; // 2s @16k
    let win_samples = 480_000usize; // 30s @16k
    let mut feats_data: Vec<f32> = if t50 <= 2000 && !force_hubert_chunk {
        let wav_t = eng.tensor_from_f32(&wav_pad, vec![n_total]);
        let feats = hubert.features_ver(&eng, &wav_t, n_total as u32, model_version);
        eng.readback_f32(&feats)
    } else {
        // wav_pad 全体 (パッド込み) を窓分割。採録フレーム数を full-context と
        // 同一 (t50) に揃えるため、先頭窓は左端から、末尾窓は右端まで採る。
        let hop = hubert_hop;
        let mut out: Vec<f32> = Vec::with_capacity(t50 * fd);
        println!("hubert: chunked ({t50} frames > 2000)");
        let mut seg_start = 0usize;
        while seg_start < n_total {
            let seg_end = (seg_start + win_samples).min(n_total);
            let w_start = seg_start.saturating_sub(ctx_samples);
            let w_end = (seg_end + ctx_samples).min(n_total);
            let w = &wav_pad[w_start..w_end];
            let w_t = eng.tensor_from_f32(w, vec![w.len()]);
            let feats = hubert.features_ver(&eng, &w_t, w.len() as u32, model_version);
            let f = eng.readback_f32(&feats);
            let f_frames = f.len() / fd;
            let ctx_f = (seg_start - w_start) / hop;
            let lo = if seg_start == 0 { 0 } else { ctx_f };
            let hi = if seg_end == n_total {
                f_frames
            } else {
                (ctx_f + (seg_end - seg_start) / hop).min(f_frames)
            };
            out.extend_from_slice(&f[lo * fd..hi.min(f_frames) * fd]);
            seg_start = seg_end;
        }
        if out.len() / fd != t50 {
            // 保険: フレーム数がずれたらトリム/延長 (ここを通るのは異常時のみ)
            eprintln!(
                "hubert chunked frame mismatch: {} vs t50={t50} — adjusting",
                out.len() / fd
            );
            out.resize(t50 * fd, 0.0);
        }
        out
    };
    if std::env::var("RCWX_CLI_DUMP").is_ok() {
        let _ = std::fs::write(
            "target/golden/cli_rust_feats.bin",
            bytemuck::cast_slice(&feats_data),
        );
    }
    // ---- IVF index ブレンド (補間前・Python と同じ位置) ----
    if args.index_rate > 0.0 {
        match rcwx_nn::ivf::IvfIndex::load(&dir) {
            Ok(ivf) if ivf.dim == fd => {
                ivf.blend(&mut feats_data, t50, args.index_rate as f32, args.index_k);
                println!(
                    "index: rate={} k={} lists={}",
                    args.index_rate, args.index_k, ivf.nlist
                );
            }
            Ok(ivf) => eprintln!(
                "index: ivf dim {} != model feat dim {fd} — skipping blend",
                ivf.dim
            ),
            Err(e) => eprintln!("index: ivf.safetensors not found ({e}); skipping blend"),
        }
    }

    // ---- F0 ----
    let mut f0v = match args.f0_method.as_str() {
        // RMVPE は conv の dispatch グリッド上限 (x≤65535) に当たるため長尺は
        // 窓分割 (25s 窓 + 1s 文脈、中央のみ採録)。SwiftF0 も同条件で統一。
        "rmvpe" => windowed_f0(&wav_pad, n_total, t100, |w| rmvpe.infer(&eng, w)),
        "swiftf0" => {
            let sf0 =
                rcwx_nn::swiftf0::SwiftF0::load(&eng, &rcwx_nn::swiftf0::default_swiftf0_path())
                    .context("SwiftF0 load")?;
            windowed_f0(&wav_pad, n_total, t100, |w| sf0.detect(&eng, w))
        }
        _ => vec![0.0f32; t100],
    };
    f0v.resize(t100, 0.0);
    if std::env::var("RCWX_CLI_DUMP").is_ok() {
        let _ = std::fs::write(
            "target/golden/cli_rust_f0raw.bin",
            bytemuck::cast_slice(&f0v),
        );
    }
    condition_f0(&mut f0v, args.pitch);
    if args.moe > 0.0 {
        rcwx_dsp::moe::apply_moe_f0_style(&mut f0v, args.moe);
    }
    f0::apply_f0_filter_chain(&mut f0v, 16.0, true, true, 3.6, 30.0);
    stabilize_f0_boundaries(&mut f0v, 10);
    let pitch = f0::quantize_f0_to_pitch(&f0v);
    println!(
        "f0: {} frames (voiced {})",
        t100,
        f0v.iter().filter(|v| **v > 0.0).count()
    );
    if std::env::var("RCWX_CLI_DUMP").is_ok() {
        let dump_dir = "target/golden";
        let _ = std::fs::write(
            format!("{dump_dir}/cli_rust_f0.bin"),
            bytemuck::cast_slice(&f0v),
        );
        let _ = std::fs::write(
            format!("{dump_dir}/cli_rust_pitch.bin"),
            bytemuck::cast_slice(&pitch),
        );
        let _ = std::fs::write(
            format!("{dump_dir}/cli_rust_ntotal.bin"),
            bytemuck::cast_slice(&[n_total as u32]),
        );
    }

    // ---- 100fps 補間 (上の HuBERT FIRST で抽出・IVF ブレンド済みの feats_data から) ----
    // (かつてここで再抽出していたため IVF ブレンドが破棄されていた — M4 以降の不具合を修正)
    let mut feats100 = vec![0.0f32; t100 * fd];
    for r in 0..t100 {
        // PyTorch align_corners=False: src = (r + 0.5) * (in/out) - 0.5 = r*0.5 - 0.25
        let pos = (r as f32) * 0.5 - 0.25;
        let i0 = (pos.floor() as i32).clamp(0, t50 as i32 - 1) as usize;
        let i1 = (i0 + 1).min(t50.saturating_sub(1));
        let fr = (pos - i0 as f32).clamp(0.0, 1.0);
        for c in 0..fd {
            let a = feats_data[i0 * fd + c];
            let b = feats_data[i1 * fd + c];
            feats100[r * fd + c] = a + (b - a) * fr;
        }
    }

    // ---- Synth (noise_scale=0 決定論) ----
    let feats_t = eng.tensor_from_f32(&feats100, vec![]);
    // T が大きいとデコーダ中間バッファが 128MB バインド上限を超えるため、
    // 96 フレーム窓 + 8 フレーム受容野マージンで分割生成
    // (RCWX_CHUNK=1 で短尺でもチャンクパスを強制して検証できる)
    let force_chunk = std::env::var("RCWX_CHUNK").ok().as_deref() == Some("1");
    let mut out: Vec<f32> = if t100 > 96 || force_chunk {
        synth.infer_chunked(&eng, &feats_t, &pitch, &f0v, t100 as u32, 96, 8)
    } else {
        let audio_out = synth.infer(&eng, &feats_t, &pitch, &f0v, t100 as u32);
        eng.readback_f32(&audio_out)
    };
    println!("synth out: {} samples", out.len());
    if std::env::var("RCWX_CLI_DUMP").is_ok() {
        let _ = std::fs::write(
            "target/golden/cli_rust_synthout.bin",
            bytemuck::cast_slice(&out),
        );
    }

    // ---- trim & 長さ補正 ----
    let model_sr = if synth.cfg.sr > 0 {
        synth.cfg.sr
    } else {
        48000
    };
    let zc = (model_sr / 100) as usize;
    let t_pad_tgt = t_pad * model_sr as usize / 16000;
    let extra_pad_tgt = extra_pad * model_sr as usize / 16000;
    let hubert_deficit = 2 * zc;
    let trim_start = t_pad_tgt;
    let trim_end = (t_pad_tgt + extra_pad_tgt).saturating_sub(hubert_deficit);
    if out.len() > trim_start + trim_end {
        out = out[trim_start..out.len() - trim_end].to_vec();
    }
    let expected = input_length * model_sr as usize / 16000;
    if out.len() != expected {
        out = resample_to(&out, expected);
    }
    edge_fade(&mut out, model_sr);

    // ---- wav 出力 ----
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: model_sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&args.output, spec)
        .with_context(|| format!("create {}", args.output))?;
    for v in &out {
        w.write_sample((v.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    println!(
        "done: {} ({} samples @{}Hz, {:.2}s)",
        args.output,
        out.len(),
        model_sr,
        out.len() as f64 / model_sr as f64
    );
    let _ = args.noise; // noise_scale=0 決定論 (将来拡張)
    Ok(())
}
