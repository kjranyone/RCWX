//! FAISS IndexIVFFlat 相当の IVF 特徴検索 (L2, nprobe=1)。
//!
//! 重み: tools/convert_ivf.py → ivf.safetensors (centroids/packed_features) +
//! ivf.slots.bin (固定長 256 スロットの候補 ID) + ivf.json (メタ)。
//! 検索は coarse argmin (L2) → 単一リスト内 top-k → (1/(d²+1e-6))² 重み付き平均。
//! CPU 実装 (20ms hop の新規フレーム数は僅かで十分間に合う)。

use anyhow::{Context, Result};

pub struct IvfIndex {
    pub nlist: usize,
    pub dim: usize,
    pub cap: usize,
    pub ntotal: usize,
    centroids: Vec<f32>,      // [nlist, dim]
    centroid_norms: Vec<f32>, // [nlist]
    packed: Vec<f32>,         // [N, dim]
    packed_norms: Vec<f32>,   // [N]
    slots: Vec<i64>,          // [nlist, cap]
}

pub fn default_ivf_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap_or_default())
        .join(".cache/rcwx/models/rust")
}

impl IvfIndex {
    pub fn load(dir: &std::path::Path) -> Result<Self> {
        let raw = std::fs::read(dir.join("ivf.safetensors"))?;
        let st = safetensors::SafeTensors::deserialize(&raw)?;
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("ivf.json"))?)?;
        let nlist = meta["nlist"].as_u64().context("nlist")? as usize;
        let dim = meta["d"].as_u64().context("d")? as usize;
        let cap = meta["cap"].as_u64().context("cap")? as usize;
        let ntotal = meta["ntotal"].as_u64().context("ntotal")? as usize;

        let get = |name: &str| -> Vec<f32> {
            let v = st.tensor(name).unwrap();
            v.data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let centroids = get("centroids");
        let packed = get("packed_features");
        let slots_raw = std::fs::read(dir.join("ivf.slots.bin"))?;
        let slots: Vec<i64> = slots_raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();

        let centroid_norms: Vec<f32> = (0..nlist)
            .map(|i| {
                (0..dim)
                    .map(|j| centroids[i * dim + j] * centroids[i * dim + j])
                    .sum()
            })
            .collect();
        let n_packed = packed.len() / dim;
        let packed_norms: Vec<f32> = (0..n_packed)
            .map(|i| {
                (0..dim)
                    .map(|j| packed[i * dim + j] * packed[i * dim + j])
                    .sum()
            })
            .collect();

        Ok(Self {
            nlist,
            dim,
            cap,
            ntotal,
            centroids,
            centroid_norms,
            packed,
            packed_norms,
            slots,
        })
    }

    /// 1 クエリ [dim] に対する重み付き平均取得特徴 [dim] を書き込む。
    pub fn retrieve(&self, query: &[f32], k: usize, out: &mut [f32]) {
        let dim = self.dim;
        debug_assert_eq!(query.len(), dim);
        // coarse: argmin(||c||² - 2 q·c) (||q||² は定数)
        let mut best_list = 0usize;
        let mut best_score = f32::INFINITY;
        for i in 0..self.nlist {
            let mut dot = 0.0f32;
            let c = &self.centroids[i * dim..(i + 1) * dim];
            for j in 0..dim {
                dot += c[j] * query[j];
            }
            let s = self.centroid_norms[i] - 2.0 * dot;
            if s < best_score {
                best_score = s;
                best_list = i;
            }
        }

        // リスト内 top-k (二乗 L2 距離)
        let q_norm: f32 = query.iter().map(|v| v * v).sum();
        let list_slots = &self.slots[best_list * self.cap..(best_list + 1) * self.cap];
        // (distance, packed_idx) の上位 k を選択 (単純挿入ソート、k は小さい)
        let mut best_d = vec![f32::INFINITY; k];
        let mut best_i = vec![usize::MAX; k];
        for &slot in list_slots {
            if slot < 0 || slot as usize >= self.ntotal {
                continue; // パディング
            }
            let pidx = slot as usize;
            let f = &self.packed[pidx * dim..(pidx + 1) * dim];
            let mut dot = 0.0f32;
            for j in 0..dim {
                dot += f[j] * query[j];
            }
            let dist = (self.packed_norms[pidx] + q_norm - 2.0 * dot).max(0.0);
            // top-k 挿入
            let mut m = k;
            while m > 0 && dist < best_d[m - 1] {
                m -= 1;
            }
            if m < k {
                for t in (m + 1..k).rev() {
                    best_d[t] = best_d[t - 1];
                    best_i[t] = best_i[t - 1];
                }
                best_d[m] = dist;
                best_i[m] = pidx;
            }
        }

        let mut weights = vec![0.0f64; k];
        let mut wsum = 0.0f64;
        for t in 0..k {
            weights[t] = (1.0 / (best_d[t] as f64 + 1e-6)).powi(2);
            wsum += weights[t];
        }
        for slot in out.iter_mut() {
            *slot = 0.0;
        }
        if wsum <= 0.0 {
            return;
        }
        for t in 0..k {
            if best_i[t] == usize::MAX {
                continue;
            }
            let w = (weights[t] / wsum) as f32;
            let f = &self.packed[best_i[t] * dim..(best_i[t] + 1) * dim];
            for j in 0..dim {
                out[j] += w * f[j];
            }
        }
    }

    /// features [T, dim] に index ブレンドを破壊的に適用。
    /// blended = rate·retrieved + (1-rate)·original。
    pub fn blend(&self, features: &mut [f32], t: usize, rate: f32, k: usize) {
        let dim = self.dim;
        let mut retrieved = vec![0.0f32; dim];
        for f in 0..t {
            let q = &features[f * dim..(f + 1) * dim];
            let q = q.to_vec();
            self.retrieve(&q, k, &mut retrieved);
            for j in 0..dim {
                features[f * dim + j] = rate * retrieved[j] + (1.0 - rate) * features[f * dim + j];
            }
        }
    }
}
