//! 重みの実行時変換 (.pt / .pth → safetensors)。
//!
//! 従来 tools/convert_*.py (Python) で行っていた変換を Rust で実行時に行い、
//! 頒布バイナリ単体で重式を準備できるようにする:
//! - HuBERT: fairseq hubert_base.pt → hubert.safetensors
//!   (pos_conv weight_norm fold + QKV fusion + 名前簡約)
//! - RMVPE:  rmvpe.pt → rmvpe_folded.safetensors
//!   (Conv+BN fold + mel_basis/hann/cents_mapping 計算)
//! - SwiftF0: swift-f0 wheel 内 model.onnx → swiftf0.safetensors
//!   (ONNX initializer 抽出)
//!
//! 数値処理はすべて閉形式のテンソル演算のみ (nn forward 不要)。

use crate::pth::{PthFile, Value};
use anyhow::{Context, bail};
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
use std::collections::BTreeMap;
use std::path::Path;

type Tensor = (Vec<f32>, Vec<usize>);

fn write_safetensors(out: &Path, tensors: &BTreeMap<String, Tensor>) -> anyhow::Result<()> {
    let mut views = Vec::new();
    let mut bufs: Vec<Vec<u8>> = Vec::new();
    for (data, _) in tensors.values() {
        bufs.push(data.iter().flat_map(|v| v.to_le_bytes()).collect());
    }
    for ((name, (_, shape)), bytes) in tensors.iter().zip(bufs.iter()) {
        views.push((
            name.clone(),
            TensorView::new(Dtype::F32, shape.clone(), bytes)
                .with_context(|| format!("tensor {name}"))?,
        ));
    }
    serialize_to_file(views, None, out).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

/// state_dict (名前→TensorMeta) を取り出す。RVC .pth は `weight` キー内、
/// fairseq .pt は `model` キー内 (どちらも無ければトップレベルを state_dict とみなす)。
fn state_dict(top: &Value) -> &Vec<(Value, Value)> {
    for key in ["weight", "model"] {
        if let Some(Value::Dict(d)) = top.get(key)
            && !d.is_empty()
        {
            return d;
        }
    }
    top.as_dict().expect("checkpoint root is not a dict")
}

fn get_meta<'a>(sd: &'a [(Value, Value)], name: &str) -> Option<&'a crate::pth::TensorMeta> {
    sd.iter()
        .find(|(k, _)| k.as_str() == Some(name))
        .and_then(|(_, v)| match v {
            Value::Tensor(t) => Some(t),
            _ => None,
        })
}

fn read(f: &mut PthFile, sd: &[(Value, Value)], name: &str) -> anyhow::Result<Tensor> {
    let meta = get_meta(sd, name).ok_or_else(|| anyhow::anyhow!("tensor {name} not found"))?;
    let size: Vec<usize> = meta.size.iter().map(|&d| d as usize).collect();
    let data = f
        .read_f32(meta)
        .with_context(|| format!("read tensor {name}"))?;
    Ok((data, size))
}

// ---------------------------------------------------------------------------
// HuBERT
// ---------------------------------------------------------------------------

/// fairseq `hubert_base.pt` → `hubert.safetensors` (tools/convert_weights.py と同一出力)。
pub fn convert_hubert(pt_path: &Path, out_dir: &Path) -> anyhow::Result<()> {
    let mut f = PthFile::open(pt_path).context("open hubert pt")?;
    let top = f.parse().context("parse hubert pt")?;
    let sd = state_dict(&top);

    // 層数 (hubert_base = 12)
    let n_layers = sd
        .iter()
        .filter_map(|(k, _)| k.as_str())
        .filter_map(|k| k.strip_prefix("encoder.layers."))
        .filter_map(|rest| rest.split('.').next())
        .filter_map(|s| s.parse::<usize>().ok())
        .max()
        .context("no encoder layers")?
        + 1;

    let mut t: BTreeMap<String, Tensor> = BTreeMap::new();
    let mut put = |name: String, v: Tensor| {
        t.insert(name, v);
    };

    // conv frontend: conv_layers.{i}.0 (i は飛び飛稀の可能性 → 出現順に通し番号)
    let mut conv_srcs: Vec<i64> = sd
        .iter()
        .filter_map(|(k, _)| {
            let k = k.as_str()?;
            let rest = k.strip_prefix("feature_extractor.conv_layers.")?;
            let (idx, tail) = rest.split_once('.')?;
            if tail == "0.weight" {
                Some(idx.parse::<i64>().ok()?)
            } else {
                None
            }
        })
        .collect();
    conv_srcs.sort_unstable();
    for (seq, src) in conv_srcs.iter().enumerate() {
        let w = read(
            &mut f,
            sd,
            &format!("feature_extractor.conv_layers.{src}.0.weight"),
        )?;
        put(format!("conv.{seq}.weight"), w);
        if get_meta(sd, &format!("feature_extractor.conv_layers.{src}.0.bias")).is_some() {
            let b = read(
                &mut f,
                sd,
                &format!("feature_extractor.conv_layers.{src}.0.bias"),
            )?;
            put(format!("conv.{seq}.bias"), b);
        }
    }
    // GN は第 1 段のみ (transformers group mode)
    for part in ["weight", "bias"] {
        let src = format!("feature_extractor.conv_layers.0.2.{part}");
        if get_meta(sd, &src).is_some() {
            let v = read(&mut f, sd, &src)?;
            put(format!("gn0.{part}"), v);
        }
    }
    // feature projection
    for (src, dst) in [("layer_norm", "fp_ln"), ("post_extract_proj", "fp_proj")] {
        for part in ["weight", "bias"] {
            let s = format!("{src}.{part}");
            if get_meta(sd, &s).is_some() {
                let v = read(&mut f, sd, &s)?;
                put(format!("{dst}.{part}"), v);
            }
        }
    }
    // pos_conv: weight_norm (dim=2) fold
    {
        let g = read(&mut f, sd, "encoder.pos_conv.0.weight_g")?;
        let v = read(&mut f, sd, "encoder.pos_conv.0.weight_v")?;
        let b = read(&mut f, sd, "encoder.pos_conv.0.bias")?;
        put("pos_conv.weight".into(), fold_wn_dim2(&g, &v));
        put("pos_conv.bias".into(), b);
    }
    // encoder 最終 LN
    for part in ["weight", "bias"] {
        let v = read(&mut f, sd, &format!("encoder.layer_norm.{part}"))?;
        put(format!("enc_ln.{part}"), v);
    }
    // transformer 層
    for i in 0..n_layers {
        let p = format!("encoder.layers.{i}");
        // QKV fusion (cat dim=0)
        let mut qw = read(&mut f, sd, &format!("{p}.self_attn.q_proj.weight"))?;
        let kw = read(&mut f, sd, &format!("{p}.self_attn.k_proj.weight"))?;
        let vw = read(&mut f, sd, &format!("{p}.self_attn.v_proj.weight"))?;
        qw.0.extend(kw.0);
        qw.0.extend(vw.0);
        qw.1 = vec![qw.1[0] * 3, qw.1[1]];
        put(format!("layers.{i}.qkv.weight"), qw);
        let mut qb = read(&mut f, sd, &format!("{p}.self_attn.q_proj.bias"))?;
        let kb = read(&mut f, sd, &format!("{p}.self_attn.k_proj.bias"))?;
        let vb = read(&mut f, sd, &format!("{p}.self_attn.v_proj.bias"))?;
        qb.0.extend(kb.0);
        qb.0.extend(vb.0);
        qb.1 = vec![qb.1[0] * 3];
        put(format!("layers.{i}.qkv.bias"), qb);
        for (src, dst) in [
            ("self_attn.out_proj", "o"),
            ("self_attn_layer_norm", "ln_a"),
            ("fc1", "ff1"),
            ("fc2", "ff2"),
            ("final_layer_norm", "ln_f"),
        ] {
            for part in ["weight", "bias"] {
                let v = read(&mut f, sd, &format!("{p}.{src}.{part}"))?;
                put(format!("layers.{i}.{dst}.{part}"), v);
            }
        }
    }
    // final_proj (ContentVec 用)
    for part in ["weight", "bias"] {
        let s = format!("final_proj.{part}");
        if get_meta(sd, &s).is_some() {
            let v = read(&mut f, sd, &s)?;
            put(format!("final_proj.{part}"), v);
        }
    }

    std::fs::create_dir_all(out_dir)?;
    let out = out_dir.join("hubert.safetensors");
    write_safetensors(&out, &t)?;
    let n: usize = t.values().map(|(d, _)| d.len()).sum();
    println!(
        "converted {} ({} tensors, {n} params)",
        out.display(),
        t.len()
    );
    Ok(())
}

/// weight_norm (dim=2): g [1,1,K] / v [C,I,K] → w [C,I,K]。
/// torch の norm_except_dim(v, 2, dim) は「dim 以外の全次元」を縮約する:
///   norm[k] = ‖v[:,:,k]‖₂ (c,i 全体) → w[c,i,k] = v[c,i,k]·g[k]/max(norm[k],1e-12)
fn fold_wn_dim2(g: &Tensor, v: &Tensor) -> Tensor {
    let (gv, _) = g;
    let (vv, vs) = v;
    let k = vs[2];
    let mut norm2 = vec![0.0f32; k];
    for (idx, &x) in vv.iter().enumerate() {
        norm2[idx % k] += x * x;
    }
    let norm: Vec<f32> = norm2.iter().map(|&n| n.sqrt().max(1e-12)).collect();
    let out = vv
        .iter()
        .enumerate()
        .map(|(idx, &x)| x * gv[idx % k] / norm[idx % k])
        .collect();
    (out, vs.clone())
}

/// weight_norm (dim=0): g [C] / v [C, ...] → w (Rust synth/.pth 用と共通)。
/// w = v * (g[c] / max(‖v[c,:]‖₂, 1e-12))
pub fn fold_wn_dim0(g: &[f32], v: &[f32], shape: &[usize]) -> Vec<f32> {
    let c = shape[0];
    let inner = v.len() / c.max(1);
    let mut out = vec![0.0f32; v.len()];
    for (ci, gv) in g.iter().take(c).enumerate() {
        let base = ci * inner;
        let mut norm = 0.0f32;
        for k in 0..inner {
            let x = v[base + k];
            norm += x * x;
        }
        norm = norm.sqrt().max(1e-12);
        let s = gv / norm;
        for k in 0..inner {
            out[base + k] = v[base + k] * s;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// RMVPE
// ---------------------------------------------------------------------------

/// `rmvpe.pt` → `rmvpe_folded.safetensors`。
/// Conv(bias 無し)+BN を fold し、mel_basis/hann/cents を計算して同梱する。
pub fn convert_rmvpe(pt_path: &Path, out_dir: &Path) -> anyhow::Result<()> {
    let mut f = PthFile::open(pt_path).context("open rmvpe pt")?;
    let top = f.parse().context("parse rmvpe pt")?;
    let sd = state_dict(&top);

    let names: Vec<String> = sd
        .iter()
        .filter_map(|(k, _)| k.as_str().map(|s| s.to_string()))
        .collect();
    let has = |n: &str| get_meta(sd, n).is_some();

    // fold 対象を先に収集: 「<conv>.weight」+「<bn>.{weight,running_var}」のペア
    //   a) *.conv.{b}.conv.{0,3}.weight + *.conv.{b}.conv.{1,4}
    //      (ConvBlockRes の Sequential 内 Conv2d 2 本と直後の BN)
    //   b) unet.decoder.layers.{i}.conv1.0.weight + conv1.1 (ConvTranspose2d)
    let mut folds: Vec<(String, String, bool)> = Vec::new(); // (conv weight, bn prefix, is_ct)
    let mut skip: Vec<String> = Vec::new(); // fold 済みとして流さない名前
    for name in &names {
        let (bn, is_ct) = if let Some(p) = name.strip_suffix(".conv.0.weight") {
            (format!("{p}.conv.1"), false)
        } else if let Some(p) = name.strip_suffix(".conv.3.weight") {
            // ConvBlockRes の Sequential: Conv2d, BN, ReLU, Conv2d, BN, ReLU
            // 2 本目 (index 3) の BN は index 4
            (format!("{p}.conv.4"), false)
        } else if name.starts_with("unet.decoder.layers.") {
            match name.strip_suffix(".conv1.0.weight") {
                Some(p) => (format!("{p}.conv1.1"), true),
                None => continue,
            }
        } else {
            continue;
        };
        if has(&format!("{bn}.running_var")) && has(&format!("{bn}.weight")) {
            folds.push((name.clone(), bn.clone(), is_ct));
            skip.push(name.clone());
            for part in ["weight", "bias", "running_mean", "running_var"] {
                skip.push(format!("{bn}.{part}"));
            }
        }
    }

    let mut t: BTreeMap<String, Tensor> = BTreeMap::new();
    // fold に吸収されるもの以外をコピー
    for name in &names {
        if name.ends_with(".num_batches_tracked") || skip.contains(name) {
            continue;
        }
        let v = read(&mut f, sd, name).with_context(|| format!("rmvpe tensor {name}"))?;
        t.insert(name.clone(), v);
    }

    // Conv+BN fold: s[c]=γ[c]/√(var+eps), W'=W·s, b'=β-μ·s (eps=1e-5)
    for (conv_name, bn_pfx, is_ct) in &folds {
        let w = read(&mut f, sd, conv_name)?;
        let gamma = read(&mut f, sd, &format!("{bn_pfx}.weight"))?;
        let beta = read(&mut f, sd, &format!("{bn_pfx}.bias"))?;
        let mean = read(&mut f, sd, &format!("{bn_pfx}.running_mean"))?;
        let var = read(&mut f, sd, &format!("{bn_pfx}.running_var"))?;
        let (mut wv, ws) = w;
        // Conv2d [Cout,Cin,kh,kw]: cout は最外側
        // ConvTranspose2d [Cin,Cout,kh,kw]: cout は内側から 2 番目
        let cout = if *is_ct { ws[1] } else { ws[0] };
        let scale: Vec<f32> = (0..cout)
            .map(|c| gamma.0[c] / (var.0[c] + 1e-5).sqrt())
            .collect();
        let bias: Vec<f32> = (0..cout)
            .map(|c| beta.0[c] - mean.0[c] * scale[c])
            .collect();
        if *is_ct {
            let kk = ws[2] * ws[3];
            let cin = ws[0];
            for i_in in 0..cin {
                for (c, sc) in scale.iter().enumerate().take(cout) {
                    let base = (i_in * cout + c) * kk;
                    for j in 0..kk {
                        wv[base + j] *= sc;
                    }
                }
            }
        } else {
            let inner = wv.len() / cout.max(1);
            for c in 0..cout {
                for j in 0..inner {
                    wv[c * inner + j] *= scale[c];
                }
            }
        }
        t.insert(conv_name.clone(), (wv, ws));
        t.insert(
            conv_name.trim_end_matches(".weight").to_string() + ".bias",
            (bias, vec![cout]),
        );
    }

    // mel_basis / hann / cents (librosa htk=True 相当 — rmvpe.py の手実装移植)
    t.insert(
        "mel_basis".into(),
        (
            mel_basis_htk(16000, 1024, 128, 30.0, 8000.0),
            vec![128, 513],
        ),
    );
    t.insert(
        "hann_window".into(),
        (
            (0..1024)
                .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / 1024.0).cos()))
                .collect(),
            vec![1024],
        ),
    );
    t.insert(
        "cents_mapping".into(),
        (
            (0..360).map(|i| 20.0 * i as f32 + 1997.379_4_f32).collect(),
            vec![360],
        ),
    );

    std::fs::create_dir_all(out_dir)?;
    let out = out_dir.join("rmvpe_folded.safetensors");
    write_safetensors(&out, &t)?;
    println!("converted {} ({} tensors)", out.display(), t.len());
    Ok(())
}

/// htk スケール三角窓 mel フィルタバンク [n_mels, n_fft/2+1]。
/// librosa.filters.mel(htk=True) と同一 (rmvpe.py の手実装と同式)。
fn mel_basis_htk(sr: u32, n_fft: usize, n_mels: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |m: f32| 700.0 * (10f32.powf(m / 2595.0) - 1.0);
    let mel_fmin = hz_to_mel(fmin);
    let mel_fmax = hz_to_mel(fmax);
    let n_freq = n_fft / 2 + 1;
    let hz_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_to_hz(mel_fmin + (mel_fmax - mel_fmin) * i as f32 / (n_mels + 1) as f32))
        .collect();
    let fft_freqs: Vec<f32> = (0..n_freq)
        .map(|k| k as f32 * sr as f32 / n_fft as f32)
        .collect();
    let mut basis = vec![0.0f32; n_mels * n_freq];
    for i in 0..n_mels {
        let (left, center, right) = (hz_points[i], hz_points[i + 1], hz_points[i + 2]);
        let enorm = 2.0 / (hz_points[i + 2] - hz_points[i]);
        for (k, &fq) in fft_freqs.iter().enumerate() {
            let v = if fq >= left && fq < center {
                (fq - left) / (center - left)
            } else if fq >= center && fq <= right {
                (right - fq) / (right - center)
            } else {
                0.0
            };
            basis[i * n_freq + k] = v * enorm;
        }
    }
    basis
}

// ---------------------------------------------------------------------------
// SwiftF0 (ONNX)
// ---------------------------------------------------------------------------

/// swift-f0 の model.onnx → `swiftf0.safetensors`。
/// initializer 名はエクスポート時の自動命名依存のため、Conv ノード列挙で解決する。
pub fn convert_swiftf0_onnx(onnx_path: &Path, out_dir: &Path) -> anyhow::Result<()> {
    let raw = std::fs::read(onnx_path).context("read model.onnx")?;
    let graph = onnx_graph(&raw).context("parse onnx graph")?;
    let mut t: BTreeMap<String, Tensor> = BTreeMap::new();

    // initializer を名前→データ化
    let mut inits: BTreeMap<String, Tensor> = BTreeMap::new();
    for (name, dims, data) in &graph.initializers {
        inits.insert(name.clone(), (data.clone(), dims.clone()));
    }
    // STFT window: STFT ノードの入力 (X, frame_step, window, frame_length) の [2]
    if let Some((wname, _, _)) = graph
        .nodes
        .iter()
        .find(|n| n.op_type == "STFT" && n.inputs.len() >= 3)
        .and_then(|n| {
            graph
                .initializers
                .iter()
                .find(|(nm, _, _)| *nm == n.inputs[2])
        })
    {
        let v = inits.get(wname).cloned().unwrap_or_default();
        t.insert("window".into(), v);
    }
    // pitch_bin_centers は固定名
    if let Some(v) = inits.get("pitch_bin_centers").cloned() {
        t.insert("centers".into(), v);
    }
    // Conv ノード (graph 順) → conv{i}.weight/bias。freq_projection は proj として別処理。
    let mut conv_i = 0usize;
    for node in &graph.nodes {
        if node.op_type != "Conv" {
            continue;
        }
        let is_proj = node.inputs.iter().any(|i| i == "freq_projection.weight");
        if is_proj {
            continue;
        }
        // input[1]=weight, input[2]=bias (あれば)
        if node.inputs.len() >= 2 {
            if let Some(w) = inits.get(&node.inputs[1]).cloned() {
                t.insert(format!("conv{conv_i}.weight"), w);
            }
            if node.inputs.len() >= 3
                && let Some(b) = inits.get(&node.inputs[2]).cloned()
            {
                t.insert(format!("conv{conv_i}.bias"), b);
            }
        }
        conv_i += 1;
    }
    // freq_projection (1x1 Conv) → proj: [200,132,1,1] を reshape しないで保存し、
    // Rust 側ローダは [200,132] を期待 → ここで squeeze
    if let Some(node) = graph
        .nodes
        .iter()
        .find(|n| n.inputs.len() >= 2 && n.inputs.iter().any(|i| i == "freq_projection.weight"))
    {
        if let Some(mut w) = inits.get("freq_projection.weight").cloned() {
            w.1.retain(|&d| d != 1); // squeeze [200,132,1,1] → [200,132]
            t.insert("proj.weight".into(), w);
            let _ = node;
        }
        if let Some(b) = inits.get("freq_projection.bias").cloned() {
            t.insert("proj.bias".into(), b);
        }
    }

    // 検証: 必要 13 テンソル
    for need in ["window", "centers", "proj.weight", "proj.bias"] {
        if !t.contains_key(need) {
            bail!("onnx missing initializer for {need}");
        }
    }
    if conv_i != 5 {
        bail!("expected 5 conv layers, got {conv_i}");
    }

    std::fs::create_dir_all(out_dir)?;
    let out = out_dir.join("swiftf0.safetensors");
    write_safetensors(&out, &t)?;
    println!("converted {} ({} tensors)", out.display(), t.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// RVC Synth (.pth 直読み)
// ---------------------------------------------------------------------------

/// 名前付きテンソル (名前, f32 平坦データ, shape)。
pub type NamedTensor = (String, Vec<f32>, Vec<usize>);

/// RVC .pth から Synth 用テンソル群 + SynthConfig を取り出す。
/// detect_model_type (rcwx/models/synthesizer.py) と同一の判定で
/// version/has_f0/sr を決定し、weight_norm fold (dim=0) と enc_q 除外を行う。
pub fn synth_tensors_from_pth(
    pth_path: &Path,
) -> anyhow::Result<(
    Vec<(String, Vec<f32>, Vec<usize>)>,
    crate::synth::SynthConfig,
)> {
    let mut f = PthFile::open(pth_path).context("open RVC pth")?;
    let top = f.parse().context("parse RVC pth")?;
    let sd = state_dict(&top);

    let meta_shape =
        |name: &str| -> Option<Vec<i64>> { get_meta(sd, name).map(|t| t.size.clone()) };

    // ---- detect_model_type 相当 ----
    let version = match top.get("version").and_then(|v| v.as_str()) {
        Some(s) if s.contains('2') => 2,
        Some(s) if s.contains('1') => 1,
        Some(_) => 2,
        _ => {
            // emb_phone.weight [hidden, in] の in で判定 (v1: 256 / v2: 768)
            let in_ch = meta_shape("enc_p.emb_phone.weight").and_then(|s| s.get(1).copied());
            match in_ch {
                Some(256) => 1,
                Some(768) => 2,
                _ => {
                    // さらに fallback: config 長 (synthesizer.py と同じ)
                    let n = top
                        .get("config")
                        .and_then(|v| match v {
                            Value::List(l) => Some(l.len()),
                            _ => None,
                        })
                        .unwrap_or(18);
                    if n > 18 { 2 } else { 1 }
                }
            }
        }
    };
    let has_f0 = match top.get("f0").and_then(|v| v.as_i64()) {
        Some(v) => v != 0,
        None => get_meta(sd, "dec.m_source.l_linear.weight").is_some(),
    };
    let n_speakers = meta_shape("emb_g.weight")
        .and_then(|s| s.first().copied())
        .unwrap_or(1) as u32;
    // sr: int または "40k"/"48k" 等の文字列
    let sr = match top.get("sr") {
        Some(Value::Int(i)) => *i as u32,
        Some(Value::Str(s)) => parse_sr_str(s).context("parse sr")?,
        _ => 40000,
    };

    // ---- config 18 keys (RVC WebUI 順・ネストリスト含む) ----
    // [spec, segment, inter, hidden, filter, heads, layers, kernel, dropout,
    //  resblock(str), rb_kernels, rb_dilations, up_rates, up_init_ch, up_kernels,
    //  spk_dim, gin, sr]
    let empty: Vec<Value> = Vec::new();
    let conf: &[Value] = match top.get("config") {
        Some(Value::List(items)) => items,
        _ => &empty,
    };
    let cnum = |i: usize, d: u32| -> u32 {
        conf.get(i)
            .and_then(|v| v.as_i64())
            .map(|v| v as u32)
            .unwrap_or(d)
    };
    let cvec = |i: usize, d: &[u32]| -> Vec<u32> {
        match conf.get(i) {
            Some(Value::List(items)) => items
                .iter()
                .filter_map(|v| v.as_i64().map(|x| x as u32))
                .collect(),
            _ => d.to_vec(),
        }
    };
    // resblock は文字列 "1"/"2" で入る
    let resblock = match conf.get(9) {
        Some(Value::Str(s)) => s.trim().parse::<u32>().unwrap_or(1),
        Some(v) => v.as_i64().unwrap_or(1) as u32,
        None => 1,
    };
    if resblock == 2 {
        anyhow::bail!("ResBlock2 モデルは未対応です (config resblock=2)");
    }
    let cfg = crate::synth::SynthConfig {
        version,
        in_channels: if version == 1 { 256 } else { 768 },
        random_test: false,
        has_f0,
        inter_channels: cnum(2, 192),
        hidden_channels: cnum(3, 192),
        filter_channels: cnum(4, 768),
        n_heads: cnum(5, 2),
        n_layers: cnum(6, 6),
        kernel_size: cnum(7, 3),
        upsample_rates: cvec(12, &[10, 10, 2, 2]),
        upsample_initial_channel: cnum(13, 512),
        upsample_kernel_sizes: cvec(14, &[16, 16, 4, 4]),
        resblock_kernel_sizes: cvec(10, &[3, 7, 11]),
        gin_channels: cnum(16, 256),
        sr,
        n_speakers,
    };

    // ---- テンソル構築: enc_q 除外 + weight_norm fold (dim=0) ----
    let mut out: Vec<NamedTensor> = Vec::new();
    for (k, v) in sd {
        let Some(name) = k.as_str() else { continue };
        if name.starts_with("enc_q.") {
            continue;
        }
        if let Some(base) = name.strip_suffix(".weight_g") {
            let v_name = format!("{base}.weight_v");
            let (g, _) = read(&mut f, sd, name).with_context(|| format!("read {name}"))?;
            let (vv, vs) = read(&mut f, sd, &v_name).with_context(|| format!("read {v_name}"))?;
            let w = fold_wn_dim0(&g, &vv, &vs);
            out.push((format!("{base}.weight"), w, vs));
            continue;
        }
        if name.ends_with(".weight_v") {
            continue; // weight_g 側で fold 済み (weight_g 無しの孤立 v は無い)
        }
        if let Value::Tensor(tm) = v {
            let data = f.read_f32(tm).with_context(|| format!("read {name}"))?;
            let sz: Vec<usize> = tm.size.iter().map(|&d| d as usize).collect();
            out.push((name.to_string(), data, sz));
        }
    }
    Ok((out, cfg))
}

/// "40k" / "48k" / "32k" → u32
fn parse_sr_str(s: &str) -> Option<u32> {
    let t = s.trim();
    t.strip_suffix('k')
        .and_then(|n| n.parse::<f64>().ok())
        .map(|n| (n * 1000.0) as u32)
        .or_else(|| t.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn home() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap())
    }

    fn tmp_out(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("rcwx-verify/{tag}"));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// 実変換 (開発機のみ): cargo test -p rcwx-nn --lib convert_real -- --ignored --nocapture
    #[test]
    #[ignore = "requires local model files"]
    fn convert_real_hubert() {
        let out = tmp_out("hubert");
        convert_hubert(
            &home().join(".cache/rcwx/models/hubert/hubert_base.pt"),
            &out,
        )
        .unwrap();
        println!("out: {}", out.join("hubert.safetensors").display());
    }

    #[test]
    #[ignore = "requires local model files"]
    fn convert_real_rmvpe() {
        let out = tmp_out("rmvpe");
        convert_rmvpe(&home().join(".cache/rcwx/models/rmvpe/rmvpe.pt"), &out).unwrap();
        println!("out: {}", out.join("rmvpe_folded.safetensors").display());
    }

    #[test]
    #[ignore = "requires local onnx"]
    fn convert_real_swiftf0() {
        let onnx = std::path::Path::new(
            "C:/lib/github/grand2-products/RCWX/.venv/Lib/site-packages/swift_f0/model.onnx",
        );
        if !onnx.exists() {
            eprintln!("model.onnx not found, skip");
            return;
        }
        let out = tmp_out("swiftf0");
        convert_swiftf0_onnx(onnx, &out).unwrap();
        println!("out: {}", out.join("swiftf0.safetensors").display());
    }

    /// mel_basis の数値検証: librosa (htk=True) が使える環境で比較する
    /// (Python 側スクリプト tools/check_mel.py と併用)。
    #[test]
    #[ignore = "requires librosa"]
    fn mel_basis_matches_librosa() {
        let b = mel_basis_htk(16000, 1024, 128, 30.0, 8000.0);
        let path = std::env::temp_dir().join("rcwx-verify/mel_rust.bin");
        std::fs::write(&path, bytemuck::cast_slice(&b)).unwrap();
        println!("wrote {}", path.display());
    }

    /// RVC .pth の detect + テンソル抽出 (v1/v2 実モデル)。
    /// あわせて safetensors を吐き tools/convert_synth.py 出力と比較する。
    #[test]
    #[ignore = "requires local model files"]
    fn synth_detect_v1_v2() {
        for (p, want_ver, want_sr) in [
            (
                "C:/lib/github/grand2-products/RCWX/rvc_models/NORA01-少女Girl/NORA01-少女Girl.pth",
                1u32,
                48000u32,
            ),
            (
                "C:/lib/github/grand2-products/RCWX/rvc_models/kana/voice.pth",
                2,
                40000,
            ),
        ] {
            let (tensors, cfg) = synth_tensors_from_pth(Path::new(p)).unwrap();
            println!(
                "{p}: {} tensors, version={} has_f0={} sr={} spk={} n_layers={}",
                tensors.len(),
                cfg.version,
                cfg.has_f0,
                cfg.sr,
                cfg.n_speakers,
                cfg.n_layers
            );
            assert_eq!(cfg.version, want_ver);
            assert_eq!(cfg.sr, want_sr);
            assert!(cfg.has_f0);
            assert!(tensors.iter().any(|(n, _, _)| n == "dec.ups.0.weight"));
            assert!(!tensors.iter().any(|(n, _, _)| n.ends_with(".weight_g")));
            assert!(!tensors.iter().any(|(n, _, _)| n.starts_with("enc_q.")));
            // 比較用に safetensors を吐く
            let map: BTreeMap<String, Tensor> = tensors
                .clone()
                .into_iter()
                .map(|(n, d, s)| (n, (d, s)))
                .collect();
            let out =
                std::env::temp_dir().join(format!("rcwx-verify/synth-pth-v{want_ver}.safetensors"));
            write_safetensors(&out, &map).unwrap();
            println!("wrote {}", out.display());
        }
    }
}

struct PbReader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> PbReader<'a> {
    fn varint(&mut self) -> anyhow::Result<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let byte = *self
                .b
                .get(self.pos)
                .ok_or_else(|| anyhow::anyhow!("pb eof"))?;
            self.pos += 1;
            v |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift > 63 {
                bail!("pb varint too long");
            }
        }
        Ok(v)
    }
    fn field(&mut self) -> anyhow::Result<(u32, u8)> {
        let key = self.varint()?;
        Ok(((key >> 3) as u32, (key & 7) as u8))
    }
    fn bytes(&mut self) -> anyhow::Result<&'a [u8]> {
        let len = self.varint()? as usize;
        if self.pos + len > self.b.len() {
            bail!("pb bytes eof");
        }
        let s = &self.b[self.pos..self.pos + len];
        self.pos += len;
        Ok(s)
    }
    fn skip(&mut self, wire: u8) -> anyhow::Result<()> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => self.pos += 8,
            2 | 5 => {
                self.bytes()?;
            }
            _ => bail!("pb wire {wire}"),
        }
        Ok(())
    }
}

struct OnnxNode {
    op_type: String,
    inputs: Vec<String>,
}

struct OnnxGraph {
    nodes: Vec<OnnxNode>,
    initializers: Vec<(String, Vec<usize>, Vec<f32>)>,
}

fn onnx_graph(model: &[u8]) -> anyhow::Result<OnnxGraph> {
    let mut r = PbReader { b: model, pos: 0 };
    let mut graph: Option<&[u8]> = None;
    while r.pos < r.b.len() {
        let (field, wire) = r.field()?;
        if field == 7 && wire == 2 {
            graph = Some(r.bytes()?);
        } else {
            r.skip(wire)?;
        }
    }
    let graph = graph.context("onnx graph not found")?;

    let mut nodes = Vec::new();
    let mut initializers = Vec::new();
    let mut gr = PbReader { b: graph, pos: 0 };
    while gr.pos < gr.b.len() {
        let (field, wire) = gr.field()?;
        if wire != 2 {
            gr.skip(wire)?;
            continue;
        }
        let payload = gr.bytes()?;
        match field {
            1 => {
                // NodeProto: input=1 (repeated string), op_type=4
                let mut nr = PbReader { b: payload, pos: 0 };
                let mut op_type = String::new();
                let mut inputs = Vec::new();
                while nr.pos < nr.b.len() {
                    let (nf, nw) = nr.field()?;
                    match (nf, nw) {
                        (1, 2) => inputs.push(String::from_utf8_lossy(nr.bytes()?).into_owned()),
                        (4, 2) => op_type = String::from_utf8_lossy(nr.bytes()?).into_owned(),
                        _ => nr.skip(nw)?,
                    }
                }
                nodes.push(OnnxNode { op_type, inputs });
            }
            5 => {
                // TensorProto: dims=1 (packed int64), data_type=2, float_data=4 (packed),
                // name=8, raw_data=9
                let mut tr = PbReader { b: payload, pos: 0 };
                let mut name = String::new();
                let mut dims: Vec<usize> = Vec::new();
                let mut floats: Vec<f32> = Vec::new();
                let mut raw: Vec<u8> = Vec::new();
                let mut dtype = 1i64; // FLOAT
                while tr.pos < tr.b.len() {
                    let (tf, tw) = tr.field()?;
                    match (tf, tw) {
                        (1, 2) => {
                            let packed = tr.bytes()?;
                            let mut pr = PbReader { b: packed, pos: 0 };
                            while pr.pos < pr.b.len() {
                                dims.push(pr.varint()? as usize);
                            }
                        }
                        (1, 0) => dims.push(tr.varint()? as usize),
                        (2, 0) => dtype = tr.varint()? as i64,
                        (4, 2) => {
                            let packed = tr.bytes()?;
                            let mut pr = PbReader { b: packed, pos: 0 };
                            while pr.pos < pr.b.len() {
                                floats.push(pr.varint()? as u32 as f32);
                            }
                        }
                        (4, 5) => {
                            // 固定 32bit (非 packed)
                            let b = tr.bytes()?;
                            floats.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32);
                        }
                        (8, 2) => name = String::from_utf8_lossy(tr.bytes()?).into_owned(),
                        (9, 2) => raw = tr.bytes()?.to_vec(),
                        _ => tr.skip(tw)?,
                    }
                }
                // FLOAT (dtype=1) のみ抽出。INT64 等の shape 定数はスキップ。
                if dtype != 1 {
                    continue;
                }
                let data = if !raw.is_empty() {
                    raw.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                } else {
                    floats
                };
                initializers.push((name, dims, data));
            }
            _ => {}
        }
    }
    Ok(OnnxGraph {
        nodes,
        initializers,
    })
}
