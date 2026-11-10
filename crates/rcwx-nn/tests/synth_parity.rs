//! Synth golden パリティ (noise_scale=0, 励振ノイズ 0) + 段階検証。
//! 事前に: uv run python tools/convert_synth.py model/<model>/<model>.pth
//!         uv run python tools/dump_synth_stages.py

use rcwx_nn::engine::{Backend, Engine};
use rcwx_nn::synth::{Synth, default_synth_dir};

#[test]
fn synth_parity_wgsl() {
    let dir = default_synth_dir();
    let golden = dir.join("synth_golden.safetensors");
    if !dir.join("synth.safetensors").exists() || !golden.exists() {
        eprintln!("skipping: run tools/convert_synth.py first");
        return;
    }
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let synth = Synth::load(&eng, &dir).expect("load");

    let (feats, fshape) = rcwx_nn::read_safetensor_f32(&golden, "feats").unwrap();
    let (pitch, _) = rcwx_nn::read_safetensor_f32(&golden, "pitch").unwrap();
    let (pitchf, _) = rcwx_nn::read_safetensor_f32(&golden, "pitchf").unwrap();
    let (want, _) = rcwx_nn::read_safetensor_f32(&golden, "audio").unwrap();
    let t = fshape[1] as u32;

    let feats_t = eng.tensor_from_f32(&feats, vec![feats.len()]);
    let pitch_idx: Vec<u32> = pitch.iter().map(|v| *v as u32).collect();
    synth.enable_debug();
    let out = synth.infer(&eng, &feats_t, &pitch_idx, &pitchf, t);
    let got = eng.readback_f32(&out);

    // 段階検証 (あれば)
    let stages = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/golden/synth_stages16.safetensors");
    if stages.exists() {
        for (name, buf) in synth.take_debug() {
            let name2: &'static str = Box::leak(name.to_string().into_boxed_str());
            let _ = name2;
            if let Ok((want_s, _)) = rcwx_nn::read_safetensor_f32(&stages, name) {
                let got_s = eng.readback_f32(&buf);
                if got_s.len() != want_s.len() {
                    if name == "wn6_acts0" && got_s.len() == 2 * want_s.len() {
                        // with_cond [384, T] の前半 192 行 = acts
                        let t = 33usize;
                        let mut acts = Vec::with_capacity(want_s.len());
                        for c in 0..192usize {
                            for r in 0..t {
                                acts.push(got_s[c * t + r]);
                            }
                        }
                        println!("{name:<12}: cos={:.7}", rcwx_nn::cosine(&acts, &want_s));
                        continue;
                    }
                    println!("{name:<12}: skip (len {} vs {})", got_s.len(), want_s.len());
                    continue;
                }
                // python 側 [B, C, T] → engine [C, T]: flat 同一 (B=1)
                if name == "p_attn0" {
                    std::fs::write(
                        "../../target/golden/p_attn_got.bin",
                        bytemuck::cast_slice(&got_s),
                    )
                    .unwrap();
                }
                if name == "ffn0" {
                    // want との差分パターン
                    let mut bad = 0;
                    let mut first_bad = usize::MAX;
                    for i in 0..got_s.len().min(want_s.len()) {
                        if (got_s[i] - want_s[i]).abs() > 1e-3 {
                            bad += 1;
                            first_bad = first_bad.min(i);
                        }
                    }
                    println!("ffn0 bad={bad}/{} first_bad={first_bad}", got_s.len());
                    if first_bad < usize::MAX {
                        let ch = first_bad / 33;
                        println!(
                            "bad starts ch={ch} t={} got={:?} want={:?}",
                            first_bad % 33,
                            got_s[first_bad],
                            want_s[first_bad]
                        );
                    }
                }
                if name == "wn6_rs0" {
                    let t = 33usize;
                    println!(
                        "rs got[ch0,t0..2]={:?} want={:?}",
                        &got_s[0..3],
                        &want_s[0..3]
                    );
                    println!(
                        "rs got[ch1,t0..2]={:?} want={:?}",
                        &got_s[t..t + 3],
                        &want_s[t..t + 3]
                    );
                }
                if name == "wn6_skip0" {
                    println!(
                        "skip0 got[0..3]={:?} want[0..3]={:?}",
                        &got_s[0..3],
                        &want_s[0..3]
                    );
                }
                println!("{name:<12}: cos={:.7}", rcwx_nn::cosine(&got_s, &want_s));
            }
        }
    }

    println!(
        "audio: len got={} want={}  cosine={:.7}",
        got.len(),
        want.len(),
        rcwx_nn::cosine(&got, &want)
    );
    println!("got[:4]  = {:?}", &got[..4.min(got.len())]);
    println!("want[:4] = {:?}", &want[..4.min(want.len())]);
    assert_eq!(got.len(), want.len(), "audio length mismatch");
    assert!(rcwx_nn::cosine(&got, &want) >= 0.999);
}
