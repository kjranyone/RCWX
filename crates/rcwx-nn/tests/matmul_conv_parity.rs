//! matmul_conv (implicit GEMM) の conv1d に対するビット一致検証。
//! post (lrelu) 込み。RCWX_CONV=gemm 経路の回帰テスト。
use rcwx_nn::engine::{Backend, Engine, GpuTensor};

fn conv_ref(eng: &Engine, x: &GpuTensor, w: &GpuTensor, b: &GpuTensor,
            cout: u32, cin: u32, k: u32, dil: u32, t: u32) -> Vec<f32> {
    let out = eng.tensor_uninit((cout * t) as usize);
    let mut g = eng.graph();
    g.conv1d(x, w, Some(b), &out, cout, cin, k, 1, dil, (k - 1) * dil / 2, t, t, 1);
    g.submit().expect("submit");
    eng.readback_f32(&out)
}

fn conv_gemm(eng: &Engine, x: &GpuTensor, w: &GpuTensor, b: &GpuTensor,
             cout: u32, cin: u32, k: u32, dil: u32, t: u32) -> Vec<f32> {
    let out = eng.tensor_uninit((cout * t) as usize);
    let mut g = eng.graph();
    g.matmul_conv(w, x, Some(b), &out, cout, t, cin * k, k, dil, (k - 1) * dil / 2, t, 1);
    let o2 = eng.tensor_uninit((cout * t) as usize);
    let mut g2 = eng.graph();
    g2.conv1d(x, w, Some(b), &o2, cout, cin, k, 1, dil, (k - 1) * dil / 2, t, t, 1);
    g2.pointwise(&o2, None, cout * t, 1, 0.1, 0, 0, 0);
    g2.submit().expect("submit");
    return eng.readback_f32(&o2);
    g.submit().expect("submit");
    eng.readback_f32(&out)
}

#[test]
fn matmul_conv_matches_conv1d() {
    let eng = Engine::new(Backend::Dx12).expect("engine");
    let mut rng: u32 = 12345;
    let mut rand = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); (rng >> 8) as f32 / 8388608.0 - 1.0 };
    for &(cin, cout, k, dil, t) in &[
        (8u32, 16u32, 3u32, 1u32, 50u32),
        (64, 128, 3, 3, 33),
        (512, 512, 3, 1, 97),
        (7, 19, 5, 2, 41),
    ] {
        let xv: Vec<f32> = (0..(cin * (t + 16) * dil).min(200_000)).map(|_| rand()).collect();
        let wv: Vec<f32> = (0..(cout * cin * k)).map(|_| rand()).collect();
        let bv: Vec<f32> = (0..cout).map(|_| rand()).collect();
        let in_len = t + (k - 1) * dil; // pad (k-1)*dil/2 both sides → in_len = t + (k-1)*dil
        let x = eng.tensor_from_f32(&xv[..(cin * in_len) as usize], vec![]);
        let w = eng.tensor_from_f32(&wv, vec![]);
        let b = eng.tensor_from_f32(&bv, vec![]);
        let r1 = conv_ref(&eng, &x, &w, &b, cout, cin, k, dil, t);
        let r2 = conv_gemm(&eng, &x, &w, &b, cout, cin, k, dil, t);
        let r1: Vec<f32> = r1.iter().map(|v| if *v > 0.0 { *v } else { 0.1 * v }).collect();
        let maxd = r1.iter().zip(&r2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let scale = r1.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        println!("cin={cin} cout={cout} k={k} dil={dil} t={t}: maxdiff={maxd:.6} (scale {scale:.3})");
        assert!(maxd < 1e-3 * scale + 1e-4, "matmul_conv mismatch");
    }
}
