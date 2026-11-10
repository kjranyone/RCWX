//! matmul_tiled vs matmul_tiled_kahan の A/B (Tape 経路で kahan が効くか)。
use rcwx_nn::engine::tape::Tape;
use rcwx_nn::engine::{Backend, Engine};

#[test]
fn kahan_dump_neumaier() {
    let k = rcwx_nn::engine::shaders::matmul_tiled_kahan_src();
    let i = k.find("var y = a0 * b0").expect("Neumaier 置換なし");
    println!("{}", &k[i - 40..i + 640]);
}

#[test]
fn kahan_accumulate_present() {
    let k = rcwx_nn::engine::shaders::matmul_tiled_kahan_src();
    assert!(
        k.contains("var y = a0 * b0 - cc00"),
        "Kahan 累積置換が適用されていない"
    );
}

#[test]
fn kahan_ab_differs() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    // k = 64 (tiled 経路) の mm を Tape (kahan) と素の Graph で比較
    let m = 64usize;
    let k = 64usize;
    let n = 96usize;
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32 * 0.137) % 1.0) - 0.5)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32 * 0.271) % 1.0) - 0.5)
        .collect();

    // Tape 経路 (kahan)
    let mut tape = Tape::new(&eng);
    let av = tape.const_f32s(&a, vec![m, k]);
    let bv = tape.const_f32s(&b, vec![k, n]);
    let out = tape.mm(
        av,
        bv,
        None,
        m * n,
        m as u32,
        n as u32,
        k as u32,
        1,
        k as u32,
        1,
        0,
        0,
        1,
        k as u32,
        0,
        0,
        n as u32,
        1,
        m as u32 * n as u32,
        1.0,
        0,
    );
    let ot = tape.var_tensor(out);
    tape.submit().expect("submit");
    let got_kahan = eng.readback_f32(&ot);

    // 素 Graph (非 kahan)
    let mut g = eng.graph();
    assert!(!g.kahan);
    let at = eng.tensor_from_f32(&a, vec![m, k]);
    let bt = eng.tensor_from_f32(&b, vec![k, n]);
    let ct = eng.tensor_uninit(m * n);
    g.matmul_ex(
        &at,
        &bt,
        None,
        &ct,
        m as u32,
        n as u32,
        k as u32,
        1,
        k as u32,
        1,
        0,
        0,
        1,
        k as u32,
        0,
        0,
        n as u32,
        1,
        0,
        m as u32 * n as u32,
        1.0,
        0,
    );
    g.submit().expect("submit2");
    let got_plain = eng.readback_f32(&ct);

    // 生 Graph + kahan フラグ (Tape を経由しない)
    let mut g2 = eng.graph();
    g2.kahan = true;
    let ct2 = eng.tensor_uninit(m * n);
    g2.matmul_ex(
        &at,
        &bt,
        None,
        &ct2,
        m as u32,
        n as u32,
        k as u32,
        1,
        k as u32,
        1,
        0,
        0,
        1,
        k as u32,
        0,
        0,
        n as u32,
        1,
        0,
        m as u32 * n as u32,
        1.0,
        0,
    );
    g2.submit().expect("submit3");
    let got_k2 = eng.readback_f32(&ct2);
    let n2 = got_k2
        .iter()
        .zip(got_plain.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    println!("raw-graph kahan vs plain: {n2}/{} differ", got_k2.len());

    let mut n_diff = 0usize;
    for (x, y) in got_kahan.iter().zip(got_plain.iter()) {
        if x.to_bits() != y.to_bits() {
            n_diff += 1;
        }
    }
    println!(
        "kahan vs plain: {n_diff}/{} elements differ",
        got_kahan.len()
    );
    println!("kahan[:4] = {:?}", &got_kahan[..4]);
    println!("plain[:4] = {:?}", &got_plain[..4]);

    // CPU f64 参照でどちらが正確か
    let mut ref_out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a[i * k + kk] as f64 * b[kk * n + j] as f64;
            }
            ref_out[i * n + j] = acc;
        }
    }
    let err_kahan: f64 = got_kahan
        .iter()
        .zip(ref_out.iter())
        .map(|(x, r)| (*x as f64 - r).abs())
        .sum();
    let err_plain: f64 = got_plain
        .iter()
        .zip(ref_out.iter())
        .map(|(x, r)| (*x as f64 - r).abs())
        .sum();
    println!("err kahan={err_kahan:.6e} plain={err_plain:.6e}");
    let cpu4: Vec<f32> = ref_out[..4].iter().map(|x| *x as f32).collect();
    println!("cpu  [:4] = {cpu4:?}");
    assert!(err_kahan <= err_plain, "kahan は同等以上に正確なこと");
}

/// 大型 k での Kahan 効果検証 (k=4096, CPU f64 参照)。
#[test]
fn kahan_big_k() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (m, k, n) = (64usize, 4096usize, 128usize);
    let a: Vec<f32> = (0..m * k).map(|i| (((i as f64) * 0.6180339887) % 1.0 - 0.5) as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (((i as f64) * 0.31415926) % 1.0 - 0.5) as f32).collect();

    let run = |kahan: bool| -> Vec<f32> {
        let mut g = eng.graph();
        g.kahan = kahan;
        let at = eng.tensor_from_f32(&a, vec![m, k]);
        let bt = eng.tensor_from_f32(&b, vec![k, n]);
        let ct = eng.tensor_uninit(m * n);
        // A[m,k] (m-stride=k, k-stride=1), W[k,n] (n-stride=1, k-stride=n)
        g.matmul_ex(
            &at, &bt, None, &ct, m as u32, n as u32, k as u32, 1, k as u32, 1, 0, 0, 1,
            n as u32, 0, 0, n as u32, 1, 0, m as u32 * n as u32, 1.0, 0,
        );
        g.submit().expect("submit");
        eng.readback_f32(&ct)
    };
    let got_plain = run(false);
    let got_kahan = run(true);

    let mut ref_out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a[i * k + kk] as f64 * b[kk * n + j] as f64;
            }
            ref_out[i * n + j] = acc;
        }
    }
    let err = |got: &[f32]| -> f64 {
        got.iter()
            .zip(ref_out.iter())
            .map(|(x, r)| (f64::from(*x) - r).abs())
            .sum()
    };
    let n_diff = got_plain
        .iter()
        .zip(got_kahan.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let ep = err(&got_plain);
    let ek = err(&got_kahan);
    println!("big-k: bit-diff {n_diff}/{}", got_plain.len());
    println!("err plain={ep:.6e} kahan={ek:.6e} ratio={}", ep / ek.max(1e-300));
    assert!(ek <= ep, "kahan should not be worse");
}
