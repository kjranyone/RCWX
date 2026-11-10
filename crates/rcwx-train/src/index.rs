//! 特徴 index 構築 (Issue #8 M-T5)。HuBERT 特徴から k-means (省略可) + IVF を
//! 作り既存 rcwx-nn ivf.rs 形式 (ivf.safetensors + ivf.slots.bin + ivf.json) で出力。

use anyhow::Result;

/// k-means (Lloyd, 簡易)。rows > max_rows ならサブサンプル。
pub fn kmeans(data: &[f32], dim: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let n = data.len() / dim;
    if n == 0 || k == 0 {
        return vec![];
    }
    // 初期中心: 線形 stride サンプリング (決定的)
    let mut centers = vec![0.0f32; k * dim];
    for (i, c) in centers.iter_mut().enumerate().take(k * dim) {
        let _ = i;
        *c = 0.0;
    }
    let mut lcg = seed;
    let rnd = |lcg: &mut u64| -> f64 {
        *lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*lcg >> 33) as f64) / (1u64 << 33) as f64
    };
    for ci in 0..k {
        let src = ((rnd(&mut lcg) * n as f64) as usize).min(n - 1);
        centers[ci * dim..(ci + 1) * dim].copy_from_slice(&data[src * dim..(src + 1) * dim]);
    }
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        // 割当
        for r in 0..n {
            let mut best = 0usize;
            let mut bd = f32::INFINITY;
            for ci in 0..k {
                let mut d = 0.0f32;
                for j in 0..dim {
                    let diff = data[r * dim + j] - centers[ci * dim + j];
                    d += diff * diff;
                }
                if d < bd {
                    bd = d;
                    best = ci;
                }
            }
            assign[r] = best;
        }
        // 更新
        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0usize; k];
        for r in 0..n {
            let ci = assign[r];
            counts[ci] += 1;
            for j in 0..dim {
                sums[ci * dim + j] += data[r * dim + j];
            }
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                for j in 0..dim {
                    centers[ci * dim + j] = sums[ci * dim + j] / counts[ci] as f32;
                }
            }
        }
    }
    centers
}

/// IVF 割当 + 既存形式で保存。
/// n_ivf = max(1, min(16√N, N/39)), nprobe=1。
pub fn build_ivf(
    features: &[f32], // [N, dim]
    dim: usize,
    out_dir: &std::path::Path,
    kmeans_clusters: usize, // >0 で事前 k-means (N>200k 時)
    _seed: u64,
) -> Result<usize> {
    let n = features.len() / dim;
    if n == 0 {
        anyhow::bail!("no features");
    }
    // 大規模時は k-means で代表点に圧縮
    let (vecs_ref, n_vec): (&[f32], usize) = if kmeans_clusters > 0 && n > 200_000 {
        // centroids をそのままベクトル扱い (上流 RVC と同じ)
        // 簡易: そのまま (k-means は centroids を返すのでここに格納)
        // 注: 実装簡略化 — 大規模時は centroids を vecs にする
        (features, n) // TODO: kmeans 結果に差し替え
    } else {
        (features, n)
    };
    let n_ivf = 1
        .max((16.0 * (n_vec as f64).sqrt()) as usize)
        .min(n_vec / 39)
        .max(1);
    // 中心点: IVF 用に stride サンプリング
    let mut centroids = vec![0.0f32; n_ivf * dim];
    for ci in 0..n_ivf {
        let src = ci * n_vec / n_ivf;
        centroids[ci * dim..(ci + 1) * dim].copy_from_slice(&vecs_ref[src * dim..(src + 1) * dim]);
    }
    // 割当
    let mut slots: Vec<Vec<u32>> = vec![Vec::new(); n_ivf];
    for r in 0..n_vec {
        let mut best = 0usize;
        let mut bd = f32::INFINITY;
        for ci in 0..n_ivf {
            let mut d = 0.0f32;
            for j in 0..dim {
                let diff = vecs_ref[r * dim + j] - centroids[ci * dim + j];
                d += diff * diff;
            }
            if d < bd {
                bd = d;
                best = ci;
            }
        }
        slots[best].push(r as u32);
    }
    std::fs::create_dir_all(out_dir)?;
    // ivf.safetensors: centroids [nlist, dim]。packed_features は **ID 順テーブル**
    // (行 = 元ベクトル ID)。ローダ (rcwx-nn ivf.rs) は slot 値をこの行番号として
    // 引くため、リスト連結順ではならない (tools/convert_ivf.py と同一形式)。
    let packed_features: Vec<f32> = vecs_ref[..n_vec * dim].to_vec();
    let mut entries = vec![
        ("centroids".to_string(), vec![n_ivf, dim], centroids.clone()),
        (
            "packed_features".to_string(),
            vec![n_vec, dim],
            packed_features,
        ),
    ];
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
    let mut w = std::io::BufWriter::new(std::fs::File::create(out_dir.join("ivf.safetensors"))?);
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
    // slots.bin: [nlist, cap] の i64 行列 (リスト毎に cap まで -1 パディング。
    // ローダは固定ストライド cap で読む)
    let cap = slots.iter().map(|s| s.len()).max().unwrap_or(1);
    let mut slot_mat: Vec<i64> = vec![-1i64; n_ivf * cap];
    for (ci, s) in slots.iter().enumerate() {
        for (j, id) in s.iter().enumerate() {
            slot_mat[ci * cap + j] = *id as i64;
        }
    }
    let bin: Vec<u8> = slot_mat.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out_dir.join("ivf.slots.bin"), &bin)?;
    // ivf.json ("d" は推論側ローダのキー名に合わせる)
    let meta = serde_json::json!({
        "nlist": n_ivf,
        "d": dim,
        "cap": cap,
        "ntotal": n_vec,
    });
    std::fs::write(
        out_dir.join("ivf.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;
    Ok(n_ivf)
}
