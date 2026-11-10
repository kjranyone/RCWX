//! RMVPE golden パリティ。
//! 事前に: uv run python tools/convert_rmvpe.py (folded 版は別スクリプトで生成済み)

use rcwx_nn::engine::{Backend, Engine};
use rcwx_nn::rmvpe::{Rmvpe, default_rmvpe_dir};

fn home_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("USERPROFILE").expect("USERPROFILE"))
}

#[test]
fn rmvpe_parity_wgsl() {
    let dir = default_rmvpe_dir();
    let golden = dir.join("rmvpe_golden.safetensors");
    if !dir.join("rmvpe_folded.safetensors").exists() || !golden.exists() {
        eprintln!("skipping: run tools/convert_rmvpe.py first");
        return;
    }
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let model = Rmvpe::load(&eng, &dir).expect("load");

    let (audio, _) = rcwx_nn::read_safetensor_f32(&golden, "audio").unwrap();
    let (want, wshape) = rcwx_nn::read_safetensor_f32(&golden, "f0").unwrap();
    let got = model.infer(&eng, &audio);

    println!(
        "f0: got len={} want len={} (T={})",
        got.len(),
        want.len(),
        wshape[1]
    );
    let mut got_v = Vec::new();
    let mut want_v = Vec::new();
    for t in 0..got.len().min(want.len()) {
        if want[t] > 0.0 {
            got_v.push(got[t]);
            want_v.push(want[t]);
        }
    }
    let cos = rcwx_nn::cosine(&got_v, &want_v);
    println!("voiced frames: {}  cosine={:.7}", got_v.len(), cos);
    println!("got[:8]  = {:?}", &got[..8.min(got.len())]);
    println!("want[:8] = {:?}", &want[..8.min(want.len())]);
    // 一致率: 5% 以内のフレーム比率
    let ok = got_v
        .iter()
        .zip(want_v.iter())
        .filter(|(a, b)| (**a - **b).abs() / **b < 0.05)
        .count();
    println!("within-5% ratio: {}/{}", ok, got_v.len());
    assert_eq!(got.len(), want.len(), "frame count mismatch");
    assert!(ok * 100 >= got_v.len() * 90, "within-5% below 90%");
}
