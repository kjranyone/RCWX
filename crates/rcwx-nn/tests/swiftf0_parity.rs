//! SwiftF0 パリティ: golden (onnxruntime + RCWX ラッパー) との一致検証。

use rcwx_nn::engine::{Backend, Engine};
use rcwx_nn::swiftf0::SwiftF0;

fn golden_path() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/golden/swiftf0_golden.npz")
        .to_string_lossy()
        .into_owned()
}

/// npz から平坦 f32 配列を読む最小パーサ (zip 内 1 エントリ = .npy)。
fn read_npz_f32(path: &str, name: &str) -> Vec<f32> {
    let file = std::fs::File::open(path).expect("npz open");
    let mut zip = zip::ZipArchive::new(file).expect("zip");
    let mut entry = zip.by_name(&format!("{name}.npy")).expect("npy entry");
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut entry, &mut buf).expect("read");
    // npy header: magic \x93NUMPY, ver, header len, dict, padding, then data
    let hlen = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let data = &buf[10 + hlen..];
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn swiftf0_matches_onnxruntime() {
    let gp = golden_path();
    if !std::path::Path::new(&gp).exists() {
        eprintln!("golden missing ({gp}); run tools/convert_swiftf0.py first");
        return;
    }
    let audio = read_npz_f32(&gp, "audio");
    let f0_ref = read_npz_f32(&gp, "f0_final");

    let eng = Engine::new(Backend::Dx12).expect("engine");
    let model = SwiftF0::load(&eng, &rcwx_nn::swiftf0::default_swiftf0_path()).expect("load");

    let (pitch_raw, conf_raw) = model.infer_raw(&eng, &audio);
    let pitch_ref = read_npz_f32(&gp, "pitch_raw");
    let conf_ref = read_npz_f32(&gp, "conf_raw");
    assert_eq!(pitch_raw.len(), pitch_ref.len(), "internal hop frame count");
    // 生 pitch/confidence (softmax 加重平均) は厳密比較に近い一致が必要
    let max_pitch_diff = pitch_raw
        .iter()
        .zip(&pitch_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_conf_diff = conf_raw
        .iter()
        .zip(&conf_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("raw pitch max diff: {max_pitch_diff:.4} Hz, conf max diff: {max_conf_diff:.5}");
    assert!(max_pitch_diff < 1.0, "raw pitch diff {max_pitch_diff}");
    assert!(max_conf_diff < 0.02, "conf diff {max_conf_diff}");

    // 最終出力 (hop 補間 + オクターブ解決込み)
    let f0 = model.detect(&eng, &audio);
    assert_eq!(f0.len(), f0_ref.len(), "target frame count");
    let voiced_ref: Vec<bool> = f0_ref.iter().map(|v| *v > 0.0).collect();
    let voiced: Vec<bool> = f0.iter().map(|v| *v > 0.0).collect();
    let agree = voiced
        .iter()
        .zip(&voiced_ref)
        .filter(|(a, b)| a == b)
        .count();
    let cos = rcwx_nn::cosine(&f0, &f0_ref);
    let voiced_diff = f0
        .iter()
        .zip(&f0_ref)
        .filter(|(_, r)| **r > 0.0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "final f0: cosine={cos:.5} voiced_agree={}/{} max_voiced_diff={voiced_diff:.2}Hz",
        agree,
        voiced_ref.len()
    );
    assert!(cos > 0.99, "final cosine {cos}");
    assert!(voiced_diff < 15.0, "voiced f0 diff {voiced_diff}");
}
