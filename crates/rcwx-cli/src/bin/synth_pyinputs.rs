//! 段階検証: Python が synth に与えた入力 (feats/pitch/pitchf ダンプ) を
//! Rust synth に直接入れて出力を比較用に保存する。
//! mode:
//!   py       = py feats + py pitch + py pitchf (全 py 入力 → synth 無罪証明)
//!   mix-f0   = rust feats (×2) + py pitch/pitchf  (f0 差の寄与を除去)
//!   mix-feats= py feats + rust pitch + rust f0chain (特徴差の寄与を除去)
//! 使い方: cargo run --release -p rcwx-cli --bin synth_pyinputs -- <model_dir|model.pth> [mode] [frames]

use rcwx_nn::engine::{Backend, Engine};
use rcwx_nn::synth::Synth;

fn load_f32(path: &str) -> Vec<f32> {
    let raw = std::fs::read(path).expect("dump read");
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load_u32(path: &str) -> Vec<u32> {
    let raw = std::fs::read(path).expect("dump read");
    raw.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        anyhow::bail!("usage: synth_pyinputs <model_dir_or_pth> [max_frames]");
    }
    let model = &args[1];
    let mode = args.get(2).map(|s| s.as_str()).unwrap_or("py").to_string();
    let max_frames: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    // py 側入力 (sy_feats は 100fps の synth 入力)
    let py_feats = load_f32("target/golden/cli_py_sy_feats.bin");
    let py_pitchf = load_f32("target/golden/cli_py_sy_pitchf.bin");
    let py_pitch = load_f32("target/golden/cli_py_sy_pitch.bin");
    let fd = if py_pitchf.is_empty() {
        768
    } else {
        py_feats.len() / py_pitchf.len().max(1)
    };

    // rust 側入力 (cli_rust_feats = 50fps → ×2。cli_rust_f0 = 100fps chain 済み)
    let ru_feats50 = load_f32("target/golden/cli_rust_feats.bin");
    let ru_f0 = load_f32("target/golden/cli_rust_f0.bin");
    let ru_pitch = load_u32("target/golden/cli_rust_pitch.bin");

    let (feats, pitch, pitchf): (Vec<f32>, Vec<u32>, Vec<f32>) = match mode.as_str() {
        "mix-f0" => {
            let t100_target = py_pitchf.len().min(max_frames);
            let n50 = ru_feats50.len() / fd;
            let mut f2 = Vec::with_capacity(n50 * 2 * fd);
            for i in 0..n50 {
                f2.extend_from_slice(&ru_feats50[i * fd..(i + 1) * fd]);
                f2.extend_from_slice(&ru_feats50[i * fd..(i + 1) * fd]);
            }
            let m = (t100_target * fd).min(f2.len());
            let p: Vec<u32> = py_pitch
                .iter()
                .take(t100_target)
                .map(|v| v.round().clamp(1.0, 255.0) as u32)
                .collect();
            (f2[..m].to_vec(), p, py_pitchf[..t100_target].to_vec())
        }
        "mix-feats" => {
            let t100_target = py_pitchf.len().min(max_frames).min(ru_f0.len());
            let m = (t100_target * fd).min(py_feats.len());
            let p: Vec<u32> = ru_pitch
                .iter()
                .take(t100_target)
                .map(|v| (*v).clamp(1, 255))
                .collect();
            (py_feats[..m].to_vec(), p, ru_f0[..t100_target].to_vec())
        }
        _ => {
            let t100 = py_pitchf.len().min(max_frames);
            let p: Vec<u32> = py_pitch
                .iter()
                .take(t100)
                .map(|v| v.round().clamp(1.0, 255.0) as u32)
                .collect();
            (
                py_feats[..(t100 * fd).min(py_feats.len())].to_vec(),
                p,
                py_pitchf[..t100].to_vec(),
            )
        }
    };
    let t100 = pitchf.len();
    let feats = feats[..(t100 * fd).min(feats.len())].to_vec();
    println!("mode={mode} frames={t100} fd={fd}");

    let eng = Engine::new(Backend::Vulkan)?;
    let synth = if model.ends_with(".pth") {
        Synth::load_from_pth(&eng, std::path::Path::new(model))?
    } else {
        let dir = std::path::Path::new(model);
        Synth::load(&eng, dir)?
    };
    let feats_t = eng.tensor_from_f32(&feats, vec![]);
    let out = if t100 > 96 {
        synth.infer_chunked(&eng, &feats_t, &pitch, &pitchf, t100 as u32, 96, 8)
    } else {
        let o = synth.infer(&eng, &feats_t, &pitch, &pitchf, t100 as u32);
        eng.readback_f32(&o)
    };
    println!("out: {} samples", out.len());
    std::fs::write(
        "target/golden/cli_rust_pysynth.bin",
        bytemuck::cast_slice(&out),
    )?;
    println!("saved target/golden/cli_rust_pysynth.bin");
    Ok(())
}
