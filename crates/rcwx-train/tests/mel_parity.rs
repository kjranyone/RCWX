//! 学習 mel 損失 (logmel) の PyTorch golden パリティ (Issue #8 M-T4)。
//! 事前生成: uv run python tools/dump_mel_golden.py

use rcwx_nn::engine::tape::Tape;
use rcwx_nn::engine::{Backend, Engine};
use rcwx_train::modules::stft_mel::StftMel;

#[test]
fn mel_golden_parity() {
    let path = std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap())
        .join(".cache/rcwx/train/golden/mel_golden.safetensors");
    if !path.exists() {
        eprintln!("skipping: run `uv run python tools/dump_mel_golden.py` first");
        return;
    }
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let raw = std::fs::read(&path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&raw).unwrap();
    let read = |name: &str| -> Vec<f32> {
        let v = st.tensor(name).unwrap();
        v.data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let y = read("y");
    let want = read("mel");
    let n_frames = want.len() / 128;

    // Rust 側: 同一基底 (golden の y を使い mel_basis も自前生成)
    let stft = StftMel::new_rust(&eng, 2048, 480, 128, 48000);
    let mut tape = Tape::new(&eng);
    let yv = tape.input_f32s(&y, vec![y.len()]);
    let mel = stft.logmel(&mut tape, yv, 1, y.len() as u32);
    let mel_t = tape.var_tensor(mel);
    tape.submit().expect("submit");
    let got = eng.readback_f32(&mel_t);

    assert_eq!(
        got.len(),
        want.len(),
        "frame count mismatch: {} vs {}",
        got.len(),
        want.len()
    );
    // 要素ごと rtol (log 値なので絶対差で見る)。librosa 版と自前 slaney 基底の
    // 微小差が低エネルギービンで log 増幅されるため分布で評価:
    // cos 厳密 (0.9999) + 98% 以上のビンが 0.05 以内。
    let mut worst = 0.0f32;
    let mut at = 0usize;
    let mut n_bad = 0usize;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        if d > 0.05 {
            n_bad += 1;
        }
        if d > worst {
            worst = d;
            at = i;
        }
    }
    let cos = rcwx_train::cosine_pub(&got, &want);
    let frac_bad = n_bad as f32 / got.len() as f32;
    println!(
        "mel parity: cos={cos:.6} worst_absdiff={worst:.4} at (mel={}, frame={}) \
         n_frames={n_frames} bins>0.05: {n_bad} ({frac_bad:.3}%)",
        at % n_frames,
        at / n_frames
    );
    assert!(cos >= 0.9999, "mel cos {cos:.6} < 0.9999");
    assert!(
        frac_bad < 0.02,
        "mel bins over 0.05: {n_bad}/{} ({frac_bad:.4})",
        got.len()
    );
}
