//! チャンク化 (Applio Simple 方式: 固定長 + overlap) と
//! 静音ベース分割 (Automatic 方式の簡易版)。

/// 静音検出パラメータ (未使用 — Simple が既定)。
pub struct SlicerCfg {
    pub threshold_db: f32,
    pub min_length_ms: u32,
    pub min_interval_ms: u32,
    pub hop_ms: u32,
    pub max_sil_kept_ms: u32,
}

/// チャンク化: per 秒 / overlap 秒 (静音検出なし — Applio Simple 相当)。
/// 長さ per 未満のチャンクは出力しない (学習時 seg 17280 samples 要件)。
pub fn slice_and_chunk(
    x: &[f32],
    sr: u32,
    _cfg: &SlicerCfg,
    per: f32,
    overlap: f32,
) -> Vec<Vec<f32>> {
    let per_n = (per * sr as f32) as usize;
    let ovl_n = (overlap * sr as f32) as usize;
    let mut out = Vec::new();
    if x.len() <= per_n {
        if x.len() > sr as usize {
            out.push(x.to_vec());
        }
        return out;
    }
    let step = per_n - ovl_n.min(per_n - 1);
    let mut i = 0usize;
    while i + per_n <= x.len() {
        out.push(x[i..i + per_n].to_vec());
        i += step;
    }
    if x.len() - i > per_n / 2 {
        out.push(x[x.len() - per_n..].to_vec());
    }
    out
}
