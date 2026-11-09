//! f0_processing チェーンのパリティ (Python 参照 vs 純Rust)。
//! 事前に: uv run python tools/dump_f0_ref.py

#[test]
fn f0_chain_parity() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/golden/f0_ref.safetensors");
    if !dir.exists() {
        eprintln!("skipping: run tools/dump_f0_ref.py first");
        return;
    }
    for case in ["steady", "octave_flip", "slew", "holes", "mixed"] {
        let (inp, _) = rcwx_nn::read_safetensor_f32(&dir, &format!("{case}_in")).unwrap();
        let (want, _) = rcwx_nn::read_safetensor_f32(&dir, &format!("{case}_out")).unwrap();
        let mut got = inp.clone();
        rcwx_nn::f0::apply_f0_filter_chain(
            &mut got,
            16.0, true, true, 3.6, 30.0,
        );
        let voiced_want: Vec<f32> = want.iter().copied().filter(|v| *v > 0.0).collect();
        let voiced_got: Vec<f32> = got.iter().copied().filter(|v| *v > 0.0).collect();
        let cos = rcwx_nn::cosine(&voiced_got, &voiced_want);
        println!("{case:12}: cos={cos:.5} (voiced {}/{})", voiced_got.len(), want.len());
        assert!(cos >= 0.98, "{case} cosine {cos} < 0.98");
    }
    // quantize
    let (f0, _) = rcwx_nn::read_safetensor_f32(&dir, "quant_in").unwrap();
    let (q_want, _) = rcwx_nn::read_safetensor_f32(&dir, "quant_out").unwrap();
    let q = rcwx_nn::f0::quantize_f0_to_pitch(&f0);
    let mism = q.iter().zip(q_want.iter()).filter(|(a, b)| **a as f32 != **b).count();
    println!("quantize mismatches: {}/{}", mism, q.len());
    assert!(mism * 100 <= q.len(), "quantize mismatch > 1%");
}
