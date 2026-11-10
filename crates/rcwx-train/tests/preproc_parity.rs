//! M-T3 前処理 golden パリティ: butter5 HPF (scipy lfilter) / spectrogram
//! (torch.stft) / slice_and_chunk (鏡像実装)。
//! 事前生成: uv run python tools/dump_preproc_golden.py

use rcwx_train::data::preprocess::{butter5_hpf, spectrogram};
use rcwx_train::data::slicer::{SlicerCfg, slice_and_chunk};

fn load_golden() -> Option<safetensors::SafeTensors<'static>> {
    let path = std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap())
        .join(".cache/rcwx/train/golden/preproc_golden.safetensors");
    if !path.exists() {
        eprintln!("skipping: run `uv run python tools/dump_preproc_golden.py` first");
        return None;
    }
    let raw: &'static [u8] = Box::leak(std::fs::read(&path).unwrap().into_boxed_slice());
    Some(safetensors::SafeTensors::deserialize(raw).unwrap())
}

fn tensor<'a>(st: &'a safetensors::SafeTensors<'a>, name: &str) -> Vec<f32> {
    let v = st.tensor(name).unwrap();
    v.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn preproc_golden_parity() {
    let Some(st) = load_golden() else { return };
    let x = tensor(&st, "x");
    assert!(x.len() >= 100, "golden x too short for HPF");

    // ---- butter5 HPF vs scipy.signal.lfilter: maxdiff 1e-5 ----
    let want_hpf = tensor(&st, "hpf");
    let got_hpf = butter5_hpf(&x, 48000, 48.0);
    assert_eq!(got_hpf.len(), want_hpf.len());
    let mut worst = 0.0f32;
    for (g, w) in got_hpf.iter().zip(want_hpf.iter()) {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
        }
    }
    println!("hpf maxdiff = {worst:.3e}");
    assert!(worst < 1e-5, "HPF maxdiff {worst:.3e} >= 1e-5");

    // ---- spectrogram vs torch.stft: rtol 1e-4 ----
    let want_spec = tensor(&st, "spec");
    let got_spec = spectrogram(&x, 2048, 480);
    assert_eq!(
        got_spec.len(),
        want_spec.len(),
        "spec len mismatch: {} vs {}",
        got_spec.len(),
        want_spec.len()
    );
    let mut worst_rel = 0.0f32;
    for (g, w) in got_spec.iter().zip(want_spec.iter()) {
        let denom = w.abs().max(1e-6);
        let r = (g - w).abs() / denom;
        if r > worst_rel {
            worst_rel = r;
        }
    }
    println!("spec worst rtol = {worst_rel:.3e}");
    assert!(worst_rel < 1e-4, "spec rtol {worst_rel:.3e} >= 1e-4");

    // ---- slicer vs 鏡像実装: 境界完全一致 ----
    let want_sl = tensor(&st, "slice");
    let cfg = SlicerCfg {
        threshold_db: -40.0,
        min_length_ms: 3000,
        min_interval_ms: 100,
        hop_ms: 10,
        max_sil_kept_ms: 5000,
    };
    let got = slice_and_chunk(&x, 48000, &cfg, 3.0, 0.3);
    let want_pairs: Vec<(f32, f32)> = want_sl.chunks_exact(2).map(|c| (c[0], c[1])).collect();
    assert_eq!(
        got.len(),
        want_pairs.len(),
        "chunk count mismatch: {} vs {}",
        got.len(),
        want_pairs.len()
    );
    for (i, (g, (ws, wl))) in got.iter().zip(want_pairs.iter()).enumerate() {
        // 境界一致 = 長さ一致 + 先頭要素一致 (chunk 内容は x の部分集合なので)
        assert_eq!(g.len(), *wl as usize, "chunk {i} len");
        let expect_start = *ws as usize;
        assert_eq!(&g[0], &x[expect_start], "chunk {i} starts at wrong sample");
        assert_eq!(&g[g.len() - 1], &x[expect_start + g.len() - 1]);
    }
    println!("slicer: {} chunks 境界完全一致", got.len());
}
