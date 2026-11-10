//! Formant shift 経路 (return_length2 / z・har 時間伸長 decode) の golden パリティ。
//! golden はリサンプル戻し前の伸長状態 [T2*zc] と比較する。
//! 事前に: uv run python tools/dump_synth_formant.py <model.pth>

use rcwx_nn::engine::{Backend, Engine};
use rcwx_nn::synth::{Synth, default_synth_dir};

fn parity(dir: &std::path::Path, label: &str) {
    let golden = dir.join("synth_formant_pos.safetensors");
    let golden_neg = dir.join("synth_formant_neg.safetensors");
    if !dir.join("synth.safetensors").exists() || !golden.exists() || !golden_neg.exists() {
        eprintln!(
            "skipping {label}: run tools/dump_synth_formant.py first ({})",
            dir.display()
        );
        return;
    }
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let synth = Synth::load(&eng, dir).expect("load");

    let gate = if synth.cfg.random_test { 0.995 } else { 0.999 };
    for tag in ["pos", "neg"] {
        let path = dir.join(format!("synth_formant_{tag}.safetensors"));
        let (feats, fshape) = rcwx_nn::read_safetensor_f32(&path, "feats").unwrap();
        let (pitch, _) = rcwx_nn::read_safetensor_f32(&path, "pitch").unwrap();
        let (pitchf, _) = rcwx_nn::read_safetensor_f32(&path, "pitchf").unwrap();
        let (want, _) = rcwx_nn::read_safetensor_f32(&path, "audio").unwrap();
        let (factor_v, _) = rcwx_nn::read_safetensor_f32(&path, "factor").unwrap();
        let factor = factor_v[0] as f64;
        let t = fshape[1] as u32;

        assert_eq!(
            fshape[2] as u32, synth.cfg.in_channels,
            "{label}/{tag}: dim"
        );

        let feats_t = eng.tensor_from_f32(&feats, vec![feats.len()]);
        let pitch_idx: Vec<u32> = pitch.iter().map(|v| *v as u32).collect();
        let got = synth.infer_formant_stretched(&eng, &feats_t, &pitch_idx, &pitchf, t, factor);
        println!(
            "{label}/{tag} (factor {factor:.5}): len got={} want={}  cosine={:.7}",
            got.len(),
            want.len(),
            rcwx_nn::cosine(&got, &want)
        );
        assert_eq!(
            got.len(),
            want.len(),
            "{label}/{tag}: audio length mismatch"
        );
        assert!(
            rcwx_nn::cosine(&got, &want) >= gate,
            "{label}/{tag}: parity failed (gate {gate})"
        );
    }
}

#[test]
fn synth_formant_parity() {
    parity(&default_synth_dir(), "synth-formant");
}

/// v1 モデルの formant パリティ (RCWX_MODELS_DIR_V1 または rust-v1 があるときのみ)。
#[test]
fn synth_formant_parity_v1() {
    let dir = std::env::var("RCWX_MODELS_DIR_V1")
        .ok()
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap_or_default())
                .join(".cache/rcwx/models/rust-v1")
        });
    parity(&dir, "synth-formant-v1");
}
