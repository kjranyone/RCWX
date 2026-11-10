//! RCWX 独自 wgpu+WGSL 推論エンジン (Issue #6)。

pub mod engine;
pub mod hubert;
pub mod synth;
pub mod rmvpe;
pub mod swiftf0;
pub mod ivf;
pub mod f0;

/// safetensors から名前付きテンソルを f32 の平坦配列 + shape で読むヘルパ。
pub fn read_safetensor_f32(
    path: &std::path::Path,
    name: &str,
) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
    let raw = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&raw)?;
    let view = st.tensor(name)?;
    let shape = view.shape().to_vec();
    let n: usize = shape.iter().product();
    let mut out = Vec::with_capacity(n);
    for c in view.data().chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok((out, shape))
}

/// フラット化した cosine 類似度 (f64 累積)。
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
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
