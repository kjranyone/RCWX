//! 学習系サブコマンド (Issue #8): train-preprocess / train / train-export / train-index。

use anyhow::{Context, Result};
use rcwx_nn::engine::{Backend, Engine};
use rcwx_nn::hubert::Hubert;
use rcwx_nn::rmvpe::Rmvpe;
use rcwx_train::data::preprocess::{PreprocessCfg, load_f32, preprocess_wav};
use rcwx_train::index::build_ivf;
use rcwx_train::modules::synth_train::{SynthTrain, SynthTrainCfg};
use rcwx_train::train_loop::{Batch, StepInputs, Trainer};

fn train_root() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap_or_default())
        .join(".cache/rcwx/train")
}

fn v2_cfg_from_json(dir: &std::path::Path) -> Result<SynthTrainCfg> {
    let p = dir.join("cfg.json");
    if p.exists() {
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
        let g = |k: &str, d: u32| j.get(k).and_then(|v| v.as_u64()).unwrap_or(d as u64) as u32;
        let gv = |k: &str, d: Vec<u32>| {
            j.get(k)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect())
                .unwrap_or(d)
        };
        let gd = |k: &str, d: Vec<Vec<u32>>| {
            j.get(k)
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .map(|v| {
                            v.as_array()
                                .map(|x| x.iter().map(|y| y.as_u64().unwrap_or(0) as u32).collect())
                                .unwrap_or_default()
                        })
                        .collect()
                })
                .unwrap_or(d)
        };
        return Ok(SynthTrainCfg {
            in_channels: 768,
            inter_channels: g("inter_channels", 192),
            hidden_channels: g("hidden_channels", 192),
            filter_channels: g("filter_channels", 768),
            n_heads: g("n_heads", 2),
            n_layers: g("n_layers", 6),
            kernel_size: g("kernel_size", 3),
            window_size: 10,
            resblock_kernel_sizes: gv("resblock_kernel_sizes", vec![3, 7, 11]),
            resblock_dilation_sizes: gd("resblock_dilation_sizes", vec![vec![1, 3, 5]; 3]),
            upsample_rates: gv("upsample_rates", vec![12, 10, 2, 2]),
            upsample_initial_channel: g("upsample_initial_channel", 512),
            upsample_kernel_sizes: gv("upsample_kernel_sizes", vec![24, 20, 4, 4]),
            gin_channels: g("gin_channels", 256),
            n_speakers: 1,
            sr: g("sr", 48000),
            n_flows: 4,
            flow_wn_layers: 3,
            flow_kernel: 5,
            encq_wn_layers: 16,
            encq_kernel: 5,
        });
    }
    Ok(SynthTrainCfg::v2_48k())
}

/// rcwx-cli train-preprocess <wav_dir_or_file> --exp NAME
pub fn cmd_train_preprocess(args: &[String]) -> Result<()> {
    let mut input = String::new();
    let mut exp = "exp1".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--exp" if i + 1 < args.len() => {
                exp = args[i + 1].clone();
                i += 1;
            }
            a if !a.starts_with("--") => input = a.to_string(),
            _ => {}
        }
        i += 1;
    }
    if input.is_empty() {
        anyhow::bail!("使い方: rcwx-cli train-preprocess <wav_dir_or_file> --exp NAME");
    }
    let eng = Engine::new(Backend::Vulkan).context("engine")?;
    let hubert = Hubert::load(&eng, &rcwx_nn::hubert::default_weights_path()).context("hubert")?;
    let rmvpe = Rmvpe::load(&eng, &rcwx_nn::rmvpe::default_rmvpe_dir()).context("rmvpe")?;
    let out_root = train_root().join(&exp);
    let cfg = PreprocessCfg {
        out_dir: out_root.clone(),
        ..Default::default()
    };
    let input_path = std::path::Path::new(&input);
    let mut wavs: Vec<std::path::PathBuf> = Vec::new();
    if input_path.is_dir() {
        for e in std::fs::read_dir(input_path)? {
            let p = e?.path();
            if p.extension().map(|e| e == "wav").unwrap_or(false) {
                wavs.push(p);
            }
        }
    } else {
        wavs.push(input_path.to_path_buf());
    }
    wavs.sort();
    println!("前処理: {} files → {}", wavs.len(), out_root.display());
    for (i, w) in wavs.iter().enumerate() {
        let r = preprocess_wav(w, &cfg, &hubert, &rmvpe, &eng, Backend::Vulkan)?;
        println!(
            "  [{}/{}] {}: {} chunks",
            i + 1,
            wavs.len(),
            w.display(),
            r.wav_paths.len()
        );
    }
    Ok(())
}

/// rcwx-cli train --exp NAME [--base-g G.pth] [--base-d D.pth] [--epochs 20]
pub fn cmd_train(args: &[String]) -> Result<()> {
    let mut exp = "exp1".to_string();
    let mut base_g: Option<String> = None;
    let mut base_d: Option<String> = None;
    let mut epochs = 20usize;
    let mut resume = false;
    let mut mel_loss = "mono".to_string();
    let mut start_ep = 0usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--exp" if i + 1 < args.len() => exp = args[i + 1].clone(),
            "--base-g" if i + 1 < args.len() => base_g = Some(args[i + 1].clone()),
            "--base-d" if i + 1 < args.len() => base_d = Some(args[i + 1].clone()),
            "--epochs" if i + 1 < args.len() => epochs = args[i + 1].parse().unwrap_or(20),
            "--mel-loss" if i + 1 < args.len() => mel_loss = args[i + 1].clone(),
            "--resume" => resume = true,
            _ => {}
        }
        i += 1;
    }
    if mel_loss != "mono" && mel_loss != "multi" {
        anyhow::bail!("--mel-loss は mono | multi (got {mel_loss})");
    }
    let root = train_root().join(&exp);
    let feat_dir = root.join("3_feature768");
    let spec_dir = root.join("spec_cache");
    let f0_dir = root.join("2a_f0");
    let f0nsf_dir = root.join("2b_f0nsf");
    let gt_dir = root.join("0_gt_wavs");
    let names: Vec<String> = std::fs::read_dir(&feat_dir)?
        .filter_map(|e| {
            let p = e.ok()?.path();
            let n = p.file_name()?.to_string_lossy().to_string();
            n.strip_suffix(".safetensors").map(|s| s.to_string())
        })
        .collect();
    if names.is_empty() {
        anyhow::bail!(
            "前処理結果がありません: {} (train-preprocess を先に)",
            feat_dir.display()
        );
    }
    println!("データ: {} utterances", names.len());
    let eng = Engine::new(Backend::Vulkan).context("engine")?;
    let cfg = v2_cfg_from_json(&root)?;
    // 重み: base .pth があればそれを使う (weight_v/g 保持)
    let ckpt = root.join("ckpt");
    let mut dw_ckpt: Option<rcwx_train::modules::mpd::MpdWeights> = None;
    let gw = if resume && ckpt.join("meta.json").exists() {
        let (w, d, ep) = load_checkpoint(&ckpt)?;
        start_ep = ep;
        dw_ckpt = d;
        println!("resume: epoch {start_ep} から再開");
        w
    } else {
        match &base_g {
            Some(p) => {
                rcwx_train::data::preprocess::pth_loader::load_pth_weights(std::path::Path::new(p))?
            }
            None => {
                anyhow::bail!("--base-g <f0G48k_v2.pth 相当> は必須 (スクラッチ学習は未サポート)")
            }
        }
    };
    // D: checkpoint の学習状態 > --base-d (旧 checkpoint は D 未保存なので base-d に戻る)
    let dw_from_ckpt = dw_ckpt.is_some();
    let dw = match dw_ckpt.or(match &base_d {
        Some(p) => Some(rcwx_train::data::preprocess::pth_loader::load_pth_weights(
            std::path::Path::new(p),
        )?),
        None => None,
    }) {
        Some(w) => {
            if dw_from_ckpt {
                println!("D: checkpoint から復元");
            }
            w
        }
        None => Default::default(),
    };
    let mut trainer = Trainer::new(&eng, cfg.clone(), &gw, &dw)?;
    if mel_loss == "multi" {
        trainer = trainer.with_multi_mel();
        println!("mel loss: multi (7 スケール)");
    }
    let batch = 1usize;
    let seg_t = 36usize; // segment 17280 samples
    let upp = cfg.upp() as usize;
    let seg_samples = seg_t * upp;
    // エポックごとに全発話を 1 サンプルずつ
    let mut lcg: u64 = 1234;
    let mut rnd = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 33) as f32 / (1u64 << 33) as f32 - 0.5) * 2.0
    };
    // EMA 過学習検出: mel 損失の EMA (α=0.8) が 5 epoch 改善しなければ警告
    let mut mel_ema: Option<f32> = None;
    let mut mel_best = f32::INFINITY;
    let mut stale_epochs = 0usize;
    for ep in start_ep..epochs {
        let t0 = std::time::Instant::now();
        let mut last = None;
        for (si, name) in names.iter().enumerate() {
            let feats = load_f32(&feat_dir.join(format!("{name}.safetensors")))?;
            let spec = load_f32(&spec_dir.join(format!("{name}.safetensors")))?;
            let pitch = load_f32(&f0_dir.join(format!("{name}.safetensors")))?;
            let pitchf = load_f32(&f0nsf_dir.join(format!("{name}.safetensors")))?;
            // spec bins = 1025
            let spec_bins = 1025usize;
            let t_len = spec.len() / spec_bins;
            if t_len <= seg_t + 1 {
                continue;
            }
            // 全入力 (phone / spec / wave / pitchf) を同一の 10ms グリッド開始位置で
            // スライスする (整合ズレが学習を壊すため)
            let start = (si * 7) % (t_len - seg_t);
            let f_dim = 768usize;
            let ph_t = feats.len() / f_dim;
            let st = start.min(ph_t.saturating_sub(t_len));
            if st + t_len > ph_t {
                continue;
            }
            let phone: Vec<f32> = feats[st * f_dim..(st + t_len) * f_dim].to_vec();
            let pitch_b: Vec<f32> = pitch[st..st + t_len].to_vec();
            let mut spec_b = Vec::with_capacity(spec_bins * t_len);
            for f in 0..t_len {
                spec_b.extend_from_slice(&spec[f * spec_bins..(f + 1) * spec_bins]);
            }
            if phone.len() != t_len * f_dim || pitch_b.len() != t_len {
                continue;
            }
            // wave セグメント
            let (sr, wav) =
                rcwx_train::data::preprocess::read_wav_mono(&gt_dir.join(format!("{name}.wav")))?;
            if sr != cfg.sr {
                continue;
            }
            let w_start = st * (cfg.sr as usize / 100);
            if w_start + seg_samples > wav.len() {
                continue;
            }
            let wave: Vec<f32> = wav[w_start..w_start + seg_samples].to_vec();
            // pitchf セグメント + 励振 (sinegen)
            let pf: Vec<f32> = (0..seg_t)
                .map(|t| pitchf.get(st + t).copied().unwrap_or(0.0))
                .collect();
            let har = rcwx_train::sinegen::sinegen_train(
                &rcwx_train::sinegen::SineGenCfg::default(),
                &pf,
                1,
                seg_t,
                upp,
                &mut rcwx_train::sinegen::LcgRng(999 + si as u64),
            );
            // randn
            let randn: Vec<f32> = (0..batch * cfg.inter_channels as usize * t_len)
                .map(|_| rnd())
                .collect();
            let starts = vec![st as u32; batch];
            let b = Batch {
                phone,
                pitch: pitch_b
                    .iter()
                    .map(|v| v.round().clamp(1.0, 255.0) as u32)
                    .collect(),
                spec: spec_b,
                wave,
                lengths: vec![t_len as u32; batch],
                t_len: t_len as u32,
                seg_t: seg_t as u32,
                spec_bins: spec_bins as u32,
            };
            let inp = StepInputs {
                randn: &randn,
                har: &har,
                start_frames: &starts,
            };
            last = Some(trainer.step(&b, &inp)?);
        }
        if let Some(l) = last {
            println!(
                "epoch {}/{}: disc={:.4} gen={:.4} fm={:.4} mel={:.4} kl={:.4} ({:.1}s)",
                ep + 1,
                epochs,
                l.disc,
                l.gen_adv,
                l.fm,
                l.mel,
                l.kl,
                t0.elapsed().as_secs_f32()
            );
            let m = mel_ema.map(|e| 0.8 * e + 0.2 * l.mel).unwrap_or(l.mel);
            mel_ema = Some(m);
            if m < mel_best - 1e-4 {
                mel_best = m;
                stale_epochs = 0;
            } else {
                stale_epochs += 1;
                if stale_epochs == 5 {
                    println!(
                        "[OVERFIT?] mel EMA {m:.4} が 5 epoch 改善なし (best {mel_best:.4})。\
                         学習打ち切りや --epochs 短縮を検討"
                    );
                }
            }
        }
        // エポックごとにチェックポイント (device loss 時は --resume で再開)
        save_checkpoint(&eng, &trainer, &root, ep + 1)?;
    }
    // チェックポイント保存 (WN v/g のまま)
    save_checkpoint(&eng, &trainer, &root, epochs)?;
    // native エクスポート
    let out_dir = root.join("model");
    rcwx_train::export::export_native(&eng, &trainer.synth, &cfg, &out_dir)?;
    rcwx_train::export::export_pth_intermediate(&eng, &trainer.synth, &cfg, &root.join("pth_out"))?;
    println!("エクスポート: {}", out_dir.display());
    // 学習完了後 index 自動生成 (Applio QoL — エクスポート直後に rt/GUI で使える)
    build_index(&root)?;
    println!(
        "RVC .pth 生成: uv run python tools/export_pth.py --dir {} --out {}/model.pth",
        root.join("pth_out").display(),
        root.display()
    );
    Ok(())
}

/// 特徴キャッシュから IVF index を model/ へ生成 (train-index と学習完了後の共用)。
fn build_index(root: &std::path::Path) -> Result<()> {
    let feat_dir = root.join("3_feature768");
    if !feat_dir.exists() {
        anyhow::bail!("特徴キャッシュなし: {}", feat_dir.display());
    }
    let mut all: Vec<f32> = Vec::new();
    for e in std::fs::read_dir(&feat_dir)? {
        let p = e?.path();
        if p.extension().map(|e| e == "safetensors").unwrap_or(false) {
            all.extend(load_f32(&p)?);
        }
    }
    let dim = 768;
    let n = all.len() / dim;
    println!("index: {n} x {dim}");
    let out = root.join("model");
    let n_ivf = build_ivf(&all, dim, &out, 0, 42)?;
    println!("IVF nlist={n_ivf} → {}", out.display());
    Ok(())
}

fn save_checkpoint(
    eng: &Engine,
    trainer: &Trainer,
    root: &std::path::Path,
    epoch: usize,
) -> Result<()> {
    // 全パラメータ (v/g 含む) を safetensors 風に保存 (簡易: meta+bin)。
    // G (synth) はそのままのキー、D (mpd) は "d." prefix で同ファイルへ —
    // resume 時に D も学習状態から再開できる (無い場合は base-d 相当にフォールバック)
    let mut meta = Vec::new();
    let mut data: Vec<u8> = Vec::new();
    let mut stores: Vec<(bool, &rcwx_nn::engine::tape::ParamStore)> =
        { vec![(false, &trainer.synth.store), (true, &trainer.mpd.store)] };
    for (is_d, store) in stores.drain(..) {
        let mut names: Vec<String> = store.entries.iter().map(|e| e.name.clone()).collect();
        names.sort();
        for n in &names {
            let e = store.entries.iter().find(|e| &e.name == n).unwrap();
            let off = data.len() / 4;
            let v = eng.readback_f32(&e.value);
            for x in &v {
                data.extend_from_slice(&x.to_le_bytes());
            }
            meta.push(serde_json::json!({
                "name": if is_d { format!("d.{n}") } else { n.clone() },
                "shape": e.value.shape, "offset": off, "dtype": "F32",
            }));
        }
    }
    std::fs::create_dir_all(root.join("ckpt"))?;
    std::fs::write(root.join("ckpt/weights.f32"), &data)?;
    std::fs::write(
        root.join("ckpt/meta.json"),
        serde_json::to_string_pretty(&serde_json::json!({ "tensors": meta, "epoch": epoch }))?,
    )?;
    Ok(())
}

/// checkpoint から重みと進行 epoch を読む。
/// D 重み ("d." prefix) があれば戻り値の第 2 要素に入る。
fn load_checkpoint(
    ckpt: &std::path::Path,
) -> Result<(
    rcwx_train::modules::synth_train::Weights,
    Option<rcwx_train::modules::mpd::MpdWeights>,
    usize,
)> {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(ckpt.join("meta.json"))?)?;
    let raw = std::fs::read(ckpt.join("weights.f32"))?;
    let mut gw = rcwx_train::modules::synth_train::Weights::new();
    let mut dw = rcwx_train::modules::mpd::MpdWeights::new();
    for t in meta["tensors"].as_array().unwrap() {
        let name = t["name"].as_str().unwrap().to_string();
        let shape: Vec<usize> = t["shape"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect())
            .unwrap_or_default();
        let off = t["offset"].as_u64().unwrap_or(0) as usize;
        let n: usize = shape.iter().product();
        let d: Vec<f32> = raw[off * 4..(off + n) * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if let Some(dn) = name.strip_prefix("d.") {
            dw.insert(dn.to_string(), (d, shape));
        } else {
            gw.insert(name, (d, shape));
        }
    }
    let epoch = meta["epoch"].as_u64().unwrap_or(0) as usize;
    let dw = if dw.is_empty() { None } else { Some(dw) };
    Ok((gw, dw, epoch))
}

/// rcwx-cli train-export --exp NAME
pub fn cmd_train_export(args: &[String]) -> Result<()> {
    let mut exp = "exp1".to_string();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--exp" && i + 1 < args.len() {
            exp = args[i + 1].clone();
        }
        i += 1;
    }
    let root = train_root().join(&exp);
    let eng = Engine::new(Backend::Vulkan).context("engine")?;
    // ckpt から再構築
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("ckpt/meta.json"))?)?;
    let raw = std::fs::read(root.join("ckpt/weights.f32"))?;
    let mut gw = rcwx_train::modules::synth_train::Weights::new();
    for t in meta["tensors"].as_array().unwrap() {
        let name = t["name"].as_str().unwrap().to_string();
        let shape: Vec<usize> = t["shape"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect())
            .unwrap_or_default();
        let off = t["offset"].as_u64().unwrap_or(0) as usize;
        let n: usize = shape.iter().product();
        let d: Vec<f32> = raw[off * 4..(off + n) * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        gw.insert(name, (d, shape));
    }
    let cfg = v2_cfg_from_json(&root)?;
    let synth = SynthTrain::new(&eng, cfg.clone(), &gw)?;
    let out_dir = root.join("model");
    rcwx_train::export::export_native(&eng, &synth, &cfg, &out_dir)?;
    rcwx_train::export::export_pth_intermediate(&eng, &synth, &cfg, &root.join("pth_out"))?;
    println!("エクスポート: {}", out_dir.display());
    Ok(())
}

/// rcwx-cli train-index --exp NAME
pub fn cmd_train_index(args: &[String]) -> Result<()> {
    let mut exp = "exp1".to_string();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--exp" && i + 1 < args.len() {
            exp = args[i + 1].clone();
        }
        i += 1;
    }
    let root = train_root().join(&exp);
    build_index(&root)
}
