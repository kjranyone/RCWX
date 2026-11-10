//! RCWX ホスト DSP (Issue #6 M4): Python 実装と厳密一致する純 Rust ポート。

pub mod drift;
pub mod feedback;
pub mod moe;
pub mod noise_gate;
pub mod postprocess;
pub mod resample;
pub mod sola;
pub mod spectral;
pub mod voice_gate;

/// 比較用 cosine 類似度。
pub fn spectral_corr(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt())
}
