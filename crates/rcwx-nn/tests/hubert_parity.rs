//! HuBERT golden パリティ (独自 wgpu エンジン vs transformers 参照)。
//!
//! 事前に生成: uv run python tools/convert_weights.py hubert

use rcwx_nn::engine::{Backend, Engine};
use rcwx_nn::hubert::Hubert;

fn home_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("USERPROFILE").expect("USERPROFILE"))
}

fn golden_path() -> std::path::PathBuf {
    home_dir().join(".cache/rcwx/models/rust/hubert_golden.safetensors")
}

#[test]
fn hubert_parity_wgsl() {
    let weights = rcwx_nn::hubert::default_weights_path();
    if !weights.exists() || !golden_path().exists() {
        eprintln!("skipping: run `uv run python tools/convert_weights.py hubert` first");
        return;
    }
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    println!(
        "# adapter: {} ({:?})",
        eng.adapter_info.name, eng.adapter_info.backend
    );
    let model = Hubert::load(&eng, &weights).expect("load");

    for (name, n) in [("aggr", 10880u32), ("norm", 17920u32)] {
        let (wav, wshape) =
            rcwx_nn::read_safetensor_f32(&golden_path(), &format!("{name}.wav")).unwrap();
        assert_eq!(wshape[1] as u32, n);
        let wav_t = eng.tensor_from_f32(&wav, vec![n as usize]);

        let out_v2 = model.features(&eng, &wav_t, n);
        let got_v2 = eng.readback_f32(&out_v2);
        let (want_v2, _) = rcwx_nn::read_safetensor_f32(&golden_path(), &format!("{name}.v2")).unwrap();
        eprintln!("dbg {name}: got_v2 len={} want len={}", got_v2.len(), want_v2.len());
        let c2 = rcwx_nn::cosine(&got_v2, &want_v2);

        let out_v1 = model.extract(&eng, &wav_t, n, 9, 256);
        let got_v1 = eng.readback_f32(&out_v1);
        let (want_v1, _) = rcwx_nn::read_safetensor_f32(&golden_path(), &format!("{name}.v1")).unwrap();
        let c1 = rcwx_nn::cosine(&got_v1, &want_v1);

        println!("{name}: v2 cosine={c2:.7}  v1 cosine={c1:.7}");
        assert!(c2 >= 0.9999, "{name} v2 parity failed: {c2}");
        assert!(c1 >= 0.9999, "{name} v1 parity failed: {c1}");
    }
}
