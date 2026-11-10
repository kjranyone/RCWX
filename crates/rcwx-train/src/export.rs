//! 学習済みモデルのエクスポート (Issue #8 M-T4)。
//! WN fold → (1) 自前形式 (synth.safetensors + synth.json — rcwx run/rt/GUI が直接読める)
//! (2) RVC 互換 .pth 用の中間生成物 (weights.f16 + meta.json → tools/export_pth.py)。

use std::collections::HashMap;

use anyhow::Result;

use crate::modules::synth_train::{SynthTrain, SynthTrainCfg};

/// WN を fold した state_dict (PyTorch キー名) を CPU 側で組み立て。
pub fn folded_state_dict(
    eng: &rcwx_nn::engine::Engine,
    synth: &SynthTrain,
) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let mut out = HashMap::new();
    for e in &synth.store.entries {
        let name = &e.name;
        if name.ends_with(".weight_v") || name.ends_with(".weight_g") {
            continue;
        }
        let v = eng.readback_f32(&e.value);
        out.insert(name.clone(), (v, e.value.shape.clone()));
    }
    // WN: w = v·g/||v_r|| を行単位で計算
    for wn in &synth.store.wns {
        let v = eng.readback_f32(&synth.store.entries[wn.v_idx].value);
        let g = eng.readback_f32(&synth.store.entries[wn.g_idx].value);
        let shape = &synth.store.entries[wn.v_idx].value.shape;
        let prefix = synth.store.entries[wn.v_idx]
            .name
            .trim_end_matches(".weight_v")
            .to_string();
        let rows = wn.rows as usize;
        let row_len = wn.row_len as usize;
        let mut w = vec![0.0f32; v.len()];
        for r in 0..rows {
            let n = (0..row_len)
                .map(|j| v[r * row_len + j] * v[r * row_len + j])
                .sum::<f32>()
                .sqrt();
            for j in 0..row_len {
                w[r * row_len + j] = v[r * row_len + j] * g[r] / n;
            }
        }
        out.insert(format!("{prefix}.weight"), (w, shape.clone()));
    }
    Ok(out)
}

/// f32 → f16 bits。
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let frac = (x & 0x007f_ffff) as i32;
    if exp == 0 {
        return sign;
    }
    if exp == 0xff {
        return sign | 0x7c00 | u16::from(frac != 0) * 0x200;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // f16 では表現困難な小さい値 → 0 (PyTorch 変換と同等の実用丸め)
        return sign;
    }
    sign | ((e as u16) << 10) | (frac >> 13) as u16
}

/// 自前形式で保存。
pub fn export_native(
    eng: &rcwx_nn::engine::Engine,
    synth: &SynthTrain,
    cfg: &SynthTrainCfg,
    out_dir: &std::path::Path,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let sd = folded_state_dict(eng, synth)?;
    let mut entries: Vec<(String, Vec<usize>, Vec<f32>)> = sd
        .iter()
        .map(|(k, (d, s))| (k.clone(), s.clone(), d.clone()))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut offsets = Vec::new();
    let mut off = 0usize;
    for (_, _, d) in &entries {
        offsets.push((off, off + d.len() * 4));
        off += d.len() * 4;
    }
    let header_obj: std::collections::BTreeMap<String, serde_json::Value> = entries
        .iter()
        .enumerate()
        .map(|(i, (k, s, _))| {
            (
                k.clone(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": s,
                    "data_offsets": [offsets[i].0, offsets[i].1],
                }),
            )
        })
        .collect();
    let header = serde_json::to_string(&header_obj)?;
    use std::io::Write;
    let path = out_dir.join("synth.safetensors");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&path)?);
    // safetensors 仕様: hlen は padding 込み (データ開始を 8B 境界に)
    let pad = ((8 - (header.len() as u64 + 8) % 8) % 8) as usize;
    let hlen = (header.len() + pad) as u64;
    w.write_all(&hlen.to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    w.write_all(&vec![b' '; pad])?;
    for (_, _, d) in &entries {
        let bytes: Vec<u8> = d.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.write_all(&bytes)?;
    }
    w.flush()?;
    let cfg_json = serde_json::json!({
        "version": if cfg.in_channels == 256 { 1 } else { 2 },
        "has_f0": true,
        "sr": cfg.sr,
        "n_speakers": cfg.n_speakers,
        "inter_channels": cfg.inter_channels,
        "hidden_channels": cfg.hidden_channels,
        "filter_channels": cfg.filter_channels,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "kernel_size": cfg.kernel_size,
        "resblock_kernel_sizes": cfg.resblock_kernel_sizes,
        "resblock_dilation_sizes": cfg.resblock_dilation_sizes,
        "upsample_rates": cfg.upsample_rates,
        "upsample_initial_channel": cfg.upsample_initial_channel,
        "upsample_kernel_sizes": cfg.upsample_kernel_sizes,
        "gin_channels": cfg.gin_channels,
    });
    std::fs::write(
        out_dir.join("synth.json"),
        serde_json::to_string_pretty(&cfg_json)?,
    )?;
    Ok(())
}

/// RVC 互換 .pth 用中間生成物 (weights.f16 + meta.json)。
/// tools/export_pth.py が torch.save 形式にパックする。
pub fn export_pth_intermediate(
    eng: &rcwx_nn::engine::Engine,
    synth: &SynthTrain,
    cfg: &SynthTrainCfg,
    out_dir: &std::path::Path,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    // RVC の checkpoint は WN を _v/_g 保持で保存する (deployed .pth も同様)。
    // fold 済み .weight を書くと load_state_dict(strict=False) で WN パラメータが
    // ランダム初期化のまま残るため、学習ストアの生パラメータ (v/g) をそのまま出す。
    let mut sd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for e in &synth.store.entries {
        let v = eng.readback_f32(&e.value);
        let mut shape = e.value.shape.clone();
        // weight_g は PyTorch の dim=0 reparam 形状 [rows, 1, 1, ...] に戻す
        if e.name.ends_with(".weight_g") && shape.len() == 1 {
            // 対応する weight_v の形状から復元
            let v_name = e.name.replace(".weight_g", ".weight_v");
            if let Some(ve) = synth.store.entries.iter().find(|x| x.name == v_name) {
                shape = ve.value.shape.clone();
                for d in shape.iter_mut().skip(1) {
                    *d = 1;
                }
            }
        }
        sd.insert(e.name.clone(), (v, shape));
    }
    let mut names: Vec<&String> = sd.keys().collect();
    names.sort();
    let mut data: Vec<u8> = Vec::new();
    let mut tensors = Vec::new();
    for n in &names {
        let (d, s) = &sd[*n];
        let off = data.len() / 2;
        for v in d {
            data.extend_from_slice(&f32_to_f16_bits(*v).to_le_bytes());
        }
        tensors.push(serde_json::json!({
            "name": n,
            "shape": s,
            "offset": off,
            "dtype": "F16",
        }));
    }
    std::fs::write(out_dir.join("weights.f16"), &data)?;
    let meta = serde_json::json!({
        "tensors": tensors,
        "config": rvc_config_list(cfg),
        "info": "rcwx-train",
        "sr": format!("{}k", cfg.sr / 1000),
        "f0": 1,
        "version": if cfg.in_channels == 256 { "v1" } else { "v2" },
    });
    std::fs::write(
        out_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;
    Ok(())
}

/// RVC の 18-key config リスト (SynthesizerTrn コンストラクタ引数)。
/// 混合型 (int/float/str/list) は serde_json::Value。
fn rvc_config_list(cfg: &SynthTrainCfg) -> serde_json::Value {
    serde_json::json!([
        1025,
        17280,
        cfg.inter_channels,
        cfg.hidden_channels,
        cfg.filter_channels,
        cfg.n_heads,
        cfg.n_layers,
        cfg.kernel_size,
        0.1,
        "1",
        cfg.resblock_kernel_sizes,
        cfg.resblock_dilation_sizes,
        cfg.upsample_rates,
        cfg.upsample_initial_channel,
        cfg.upsample_kernel_sizes,
        109,
        cfg.gin_channels,
        cfg.sr,
    ])
}
