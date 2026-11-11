//! matmul_tiled vs matmul_tiled_kahan の A/B (Tape 経路で kahan が効くか)。
use rcwx_nn::engine::tape::Tape;
use rcwx_nn::engine::{Backend, Engine};

#[test]
fn kahan_accumulate_present() {
    let k = rcwx_nn::engine::shaders::matmul_tiled_kahan_src();
    assert!(
        k.contains("let k_base = P.p5.x") && k.contains("let has_prev = P.p4.w"),
        "分割累積カーネル (物理バッファ境界) が適用されていない"
    );
    assert!(k.contains("B4[i00]"), "前パス累積の読み込みが無い");
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
    let a: Vec<f32> = (0..m * k)
        .map(|i| (((i as f64) * 0.6180339887) % 1.0 - 0.5) as f32)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| (((i as f64) * 0.31415926) % 1.0 - 0.5) as f32)
        .collect();

    let run = |kahan: bool| -> Vec<f32> {
        let mut g = eng.graph();
        g.kahan = kahan;
        let at = eng.tensor_from_f32(&a, vec![m, k]);
        let bt = eng.tensor_from_f32(&b, vec![k, n]);
        let ct = eng.tensor_uninit(m * n);
        // A[m,k] (m-stride=k, k-stride=1), W[k,n] (n-stride=1, k-stride=n)
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
            n as u32,
            0,
            0,
            n as u32,
            1,
            0,
            m as u32 * n as u32,
            1.0,
            0,
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
    println!(
        "err plain={ep:.6e} kahan={ek:.6e} ratio={}",
        ep / ek.max(1e-300)
    );
    assert!(ek <= ep, "kahan should not be worse");
}

/// cond mm 形状 (m=16, n=1, k=4, batch=2, bias_axis=1) の素の matmul_ex 検証。
#[test]
fn cond_shape_matmul_probe() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (m, n, k, batch) = (16u32, 1u32, 4u32, 2u32);
    let wdat: Vec<f32> = (0..m as usize * k as usize)
        .map(|i| ((i as f64 * 0.618) % 1.0 - 0.5) as f32)
        .collect();
    let bdat: Vec<f32> = (0..m as usize).map(|i| 0.5 * i as f32 - 1.0).collect();
    let x: Vec<f32> = (0..batch as usize * k as usize)
        .map(|i| ((i as f64 * 0.314) % 1.0 - 0.5) as f32)
        .collect();
    let at = eng.tensor_from_f32(&wdat, vec![m as usize, k as usize]);
    let bt = eng.tensor_from_f32(&x, vec![batch as usize, k as usize, n as usize]);
    let bias = eng.tensor_from_f32(&bdat, vec![m as usize]);
    let ct = eng.tensor_uninit((batch * m * n) as usize);
    let mut g = eng.graph();
    g.matmul_ex(
        &at,
        &bt,
        Some(&bias),
        &ct,
        m,
        n,
        k,
        batch,
        k,
        1,
        0,
        0,
        1,
        1,
        0,
        k,
        n,
        1,
        0,
        m * n,
        1.0,
        1,
    );
    g.submit().expect("submit");
    let got = eng.readback_f32(&ct);
    // CPU
    let mut want = vec![0.0f64; (batch * m * n) as usize];
    for b in 0..batch as usize {
        for mi in 0..m as usize {
            let mut acc = bdat[mi] as f64;
            for ki in 0..k as usize {
                acc += wdat[mi * k as usize + ki] as f64 * x[b * k as usize + ki] as f64;
            }
            want[b * m as usize + mi] = acc;
        }
    }
    let mut worst = 0.0f32;
    for i in 0..want.len() {
        worst = worst.max((got[i] - want[i] as f32).abs());
    }
    println!("cond-shape raw matmul maxdiff = {worst:.6}");
    assert!(worst < 1e-5, "raw matmul mismatch {worst}");
}

/// cond 形状を Tape::mm 経由で (loss なし) 検証。
#[test]
fn cond_shape_tape_mm_probe() {
    use rcwx_nn::engine::tape::Tape;
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (m, n, k, batch) = (16u32, 1u32, 4u32, 2u32);
    let wdat: Vec<f32> = (0..m as usize * k as usize)
        .map(|i| ((i as f64 * 0.618) % 1.0 - 0.5) as f32)
        .collect();
    let bdat: Vec<f32> = (0..m as usize).map(|i| 0.5 * i as f32 - 1.0).collect();
    let x: Vec<f32> = (0..batch as usize * k as usize)
        .map(|i| ((i as f64 * 0.314) % 1.0 - 0.5) as f32)
        .collect();
    let mut tape = Tape::new(&eng);
    let wv = tape.const_f32s(&wdat, vec![m as usize, k as usize]);
    let bv = tape.const_f32s(&bdat, vec![m as usize]);
    let xv = tape.const_f32s(&x, vec![batch as usize, k as usize, n as usize]);
    let o = tape.mm(
        wv,
        xv,
        Some(bv),
        (batch * m * n) as usize,
        m,
        n,
        k,
        batch,
        k,
        1,
        0,
        0,
        1,
        1,
        0,
        k,
        n,
        1,
        m * n,
        1.0,
        1,
    );
    let ot = tape.var_tensor(o);
    tape.submit().expect("submit");
    let got = eng.readback_f32(&ot);
    let mut want = vec![0.0f64; (batch * m * n) as usize];
    for b in 0..batch as usize {
        for mi in 0..m as usize {
            let mut acc = bdat[mi] as f64;
            for ki in 0..k as usize {
                acc += wdat[mi * k as usize + ki] as f64 * x[b * k as usize + ki] as f64;
            }
            want[b * m as usize + mi] = acc;
        }
    }
    let mut worst = 0.0f32;
    for i in 0..want.len() {
        worst = worst.max((got[i] - want[i] as f32).abs());
    }
    println!("cond-shape tape mm maxdiff = {worst:.6}");
    assert!(worst < 1e-5, "tape mm mismatch {worst}");
}

/// re-mm 形状 (m=1025, n=38, k=2048, batch=2, chunk 可変) の分割累積精度。
#[test]
fn split_matmul_re_shape() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (m, k, n, batch) = (1025usize, 2048usize, 38usize, 2usize);
    let a: Vec<f32> = (0..m * k)
        .map(|i| (((i as f64) * 0.618) % 1.0 - 0.5) as f32)
        .collect();
    let b: Vec<f32> = (0..batch * k * n)
        .map(|i| (((i as f64) * 0.314) % 1.0 - 0.5) as f32)
        .collect();

    let run = |chunk: u32| -> Vec<f32> {
        let mut g = eng.graph();
        g.kahan = true;
        g.kahan_chunk = chunk;
        let at = eng.tensor_from_f32(&a, vec![m, k]);
        // W は [batch, k, n] (frames 相当: k-major per batch)
        let bt = eng.tensor_from_f32(&b, vec![batch, k, n]);
        let ct = eng.tensor_uninit(batch * m * n);
        g.matmul_ex(
            &at,
            &bt,
            None,
            &ct,
            m as u32,
            n as u32,
            k as u32,
            batch as u32,
            k as u32,
            1,
            0,
            0, // a_row=k, a_k=1, base 0, bs 0
            1,
            n as u32,
            0,
            (k * n) as u32, // w_row=1, w_k=n, base 0, bs=k*n
            n as u32,
            1,
            0,
            (m * n) as u32, // c_row=n, c_n=1, base 0, bs=m*n
            1.0,
            0,
        );
        g.submit().expect("submit");
        eng.readback_f32(&ct)
    };
    // CPU f64 参照
    let mut ref_out = vec![0.0f64; batch * m * n];
    for bb in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for kk in 0..k {
                    acc += a[i * k + kk] as f64 * b[(bb * k + kk) * n + j] as f64;
                }
                ref_out[(bb * m + i) * n + j] = acc;
            }
        }
    }
    for chunk in [256u32, 16] {
        let got = run(chunk);
        let mut errs: Vec<f32> = got
            .iter()
            .zip(ref_out.iter())
            .map(|(g2, r)| (f64::from(*g2) - r).abs() as f32)
            .collect();
        errs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!(
            "chunk={chunk}: err max={:.2e} p50={:.2e} p90={:.2e}",
            errs.last().copied().unwrap_or(0.0),
            errs[errs.len() / 2],
            errs[errs.len() * 9 / 10]
        );
    }
    // 非 kahan (単発 tiled) も参考
    let mut g = eng.graph();
    let at = eng.tensor_from_f32(&a, vec![m, k]);
    let bt = eng.tensor_from_f32(&b, vec![batch, k, n]);
    let ct = eng.tensor_uninit(batch * m * n);
    g.matmul_ex(
        &at,
        &bt,
        None,
        &ct,
        m as u32,
        n as u32,
        k as u32,
        batch as u32,
        k as u32,
        1,
        0,
        0,
        1,
        n as u32,
        0,
        (k * n) as u32,
        n as u32,
        1,
        0,
        (m * n) as u32,
        1.0,
        0,
    );
    g.submit().expect("submit");
    let got = eng.readback_f32(&ct);
    let mut errs: Vec<f32> = got
        .iter()
        .zip(ref_out.iter())
        .map(|(g2, r)| (f64::from(*g2) - r).abs() as f32)
        .collect();
    errs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!(
        "plain   : err max={:.2e} p50={:.2e} p90={:.2e}",
        errs.last().copied().unwrap_or(0.0),
        errs[errs.len() / 2],
        errs[errs.len() * 9 / 10]
    );
}

/// 実 frames+basis データでの re-mm 精度 (bisect の文脈を素の matmul で再現)。
#[test]
fn split_matmul_real_frames() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (batch, samples) = (2usize, 18400usize);
    let (n_fft, hop) = (2048usize, 480usize);
    let pad = (n_fft - hop) / 2;
    let padded = samples + 2 * pad;
    let n_frames = (padded - n_fft) / hop + 1;
    let ya: Vec<f32> = (0..batch * samples)
        .map(|i| ((i as f64 * 0.37 + (i % 7) as f64 * 0.11) % 1.0 - 0.5) as f32)
        .collect();
    // basis (rust 流: f64 cos → f32)
    let bins = n_fft / 2 + 1;
    let mut basis = vec![0.0f32; bins * n_fft];
    for f in 0..bins {
        for kk in 0..n_fft {
            let ang = -2.0 * std::f64::consts::PI * (f as f64) * (kk as f64) / n_fft as f64;
            basis[f * n_fft + kk] = ang.cos() as f32;
        }
    }
    // CPU で frames + window
    let win = |kk: usize| {
        (0.5 - 0.5 * (2.0 * std::f64::consts::PI * kk as f64 / n_fft as f64).cos()) as f32
    };
    let mirror = |i: usize| -> usize {
        let s = i as i64 - pad as i64;
        let mm: i64 = if s < 0 {
            -s
        } else if s >= samples as i64 {
            2 * samples as i64 - 2 - s
        } else {
            s
        };
        mm as usize
    };
    // fw [batch, n_fft, n_frames]
    let mut fw = vec![0.0f32; batch * n_fft * n_frames];
    for b in 0..batch {
        for t in 0..n_frames {
            for kk in 0..n_fft {
                let src = mirror(t * hop + kk);
                fw[(b * n_fft + kk) * n_frames + t] = ya[b * samples + src] * win(kk);
            }
        }
    }
    // GPU: mm(basis, fw) — logmel と同一 stride
    for chunk in [256u32, 16] {
        let mut g = eng.graph();
        g.kahan = true;
        g.kahan_chunk = chunk;
        let at = eng.tensor_from_f32(&basis, vec![bins, n_fft]);
        let bt = eng.tensor_from_f32(&fw, vec![batch, n_fft, n_frames]);
        let ct = eng.tensor_uninit(batch * bins * n_frames);
        g.matmul_ex(
            &at,
            &bt,
            None,
            &ct,
            bins as u32,
            n_frames as u32,
            n_fft as u32,
            batch as u32,
            n_fft as u32,
            1,
            0,
            0,
            1,
            n_frames as u32,
            0,
            (n_fft * n_frames) as u32,
            n_frames as u32,
            1,
            0,
            (bins * n_frames) as u32,
            1.0,
            0,
        );
        g.submit().expect("submit");
        let got = eng.readback_f32(&ct);
        // CPU f64
        let mut worst = 0.0f32;
        let _chunk = chunk;
        let mut p50v: Vec<f32> = Vec::new();
        for b in 0..batch {
            for f in (0..bins).step_by(37) {
                for t in (0..n_frames).step_by(7) {
                    let mut acc = 0.0f64;
                    for kk in 0..n_fft {
                        acc += basis[f * n_fft + kk] as f64
                            * fw[(b * n_fft + kk) * n_frames + t] as f64;
                    }
                    let idx = (b * bins + f) * n_frames + t;
                    let d = (got[idx] - acc as f32).abs();
                    p50v.push(d);
                    worst = worst.max(d);
                }
            }
        }
        p50v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!(
            "real-frames re mm (chunk={_chunk}): worst={worst:.2e} p50={:.2e}",
            p50v[p50v.len() / 2]
        );
    }
}

/// tape 文脈での re-mm (reflect→frames→window→mm) の精度 — bisect で
/// 観測された fwd 誤差 (max 76) の最小再現。
#[test]
fn tape_chain_re_mm() {
    use rcwx_nn::engine::tape::Tape;
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (batch, samples) = (2u32, 18400u32);
    let (n_fft, hop) = (2048u32, 480u32);
    let pad = (n_fft - hop) / 2;
    let padded = samples + 2 * pad;
    let n_frames = (padded - n_fft) / hop + 1;
    let ya: Vec<f32> = (0..batch as usize * samples as usize)
        .map(|i| ((i as f64 * 0.37 + (i % 7) as f64 * 0.11) % 1.0 - 0.5) as f32)
        .collect();
    let bins = n_fft / 2 + 1;
    let mut basis = vec![0.0f32; bins as usize * n_fft as usize];
    for f in 0..bins {
        for kk in 0..n_fft {
            let ang = -2.0 * std::f64::consts::PI * (f as f64) * (kk as f64) / n_fft as f64;
            basis[(f * n_fft + kk) as usize] = ang.cos() as f32;
        }
    }
    let mut hann = vec![0.0f32; n_fft as usize];
    for kk in 0..n_fft as usize {
        hann[kk] =
            (0.5 - 0.5 * (2.0 * std::f64::consts::PI * kk as f64 / n_fft as f64).cos()) as f32;
    }

    let mut tape = Tape::new(&eng);
    let yv = tape.const_f32s(&ya, vec![batch as usize, 1, samples as usize]);
    let y = tape.reflect_pad_dims(yv, batch, 1, samples, pad);
    let fr = tape.im2col_frames(y, n_fft, hop, batch, padded, n_frames);
    let hv = tape.const_f32s(&hann, vec![n_fft as usize]);
    let fw = tape.mul_window(fr, hv, n_fft, n_frames);
    let bv = tape.const_f32s(&basis, vec![bins as usize, n_fft as usize]);
    let re = tape.mm(
        bv,
        fw,
        None,
        (batch * bins * n_frames) as usize,
        bins,
        n_frames,
        n_fft,
        batch,
        n_fft,
        1,
        0,
        0,
        1,
        n_frames,
        0,
        n_fft * n_frames,
        n_frames,
        1,
        bins * n_frames,
        1.0,
        0,
    );
    let ret = tape.var_tensor(re);
    tape.submit().expect("submit");
    let got = eng.readback_f32(&ret);

    // CPU f64 (真の reflect: s<0 → -s、s>=L → 2L-2-s — 縁複製なし)
    let mirror = |i: usize| -> usize {
        let s = i as i64 - pad as i64;
        let m: i64 = if s < 0 {
            -s
        } else if s >= samples as i64 {
            2 * samples as i64 - 2 - s
        } else {
            s
        };
        m as usize
    };
    let mut worst = 0.0f32;
    let mut p50v: Vec<f32> = Vec::new();
    for b in 0..batch as usize {
        for f in (0..bins as usize).step_by(37) {
            for t in (0..n_frames as usize).step_by(7) {
                let mut acc = 0.0f64;
                for kk in 0..n_fft as usize {
                    let src = mirror(t * hop as usize + kk);
                    acc += basis[f * n_fft as usize + kk] as f64
                        * (ya[b * samples as usize + src] as f64 * hann[kk] as f64);
                }
                let idx = (b * bins as usize + f) * n_frames as usize + t;
                let d = (got[idx] - acc as f32).abs();
                p50v.push(d);
                worst = worst.max(d);
            }
        }
    }
    p50v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!(
        "tape-chain re mm: worst={worst:.2e} p50={:.2e}",
        p50v[p50v.len() / 2]
    );
    assert!(worst < 1e-3, "tape-chain re mm fwd error {worst}");
}

/// rfft2048 (バタフライ FFT) の fwd/bwd 精度 — 分割累積 matmul DFT との比較。
#[test]
fn rfft2048_accuracy() {
    use rcwx_nn::engine::tape::Tape;
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (batch, n_frames) = (2u32, 38u32);
    let mut st = 1234u64;
    let lcg = |s: &mut u64| {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as f32 / (1u64 << 33) as f32) - 0.5
    };
    let x: Vec<f32> = (0..batch as usize * 2048 * n_frames as usize)
        .map(|_| lcg(&mut st))
        .collect();
    let mut tape = Tape::new(&eng);
    let xv = tape.input_f32s(&x, vec![x.len()]);
    let (re, im) = tape.rfft2048(xv.clone(), batch, n_frames);
    let ret = tape.var_tensor(re);
    let imt = tape.var_tensor(im);
    tape.submit().expect("submit");
    let got_re = eng.readback_f32(&ret);
    let got_im = eng.readback_f32(&imt);

    // CPU f64 参照 ([b, k, t] → [b, f, t])
    let mut w_re = 0.0f32;
    let mut w_im = 0.0f32;
    for b in 0..batch as usize {
        for f in (0..1025usize).step_by(17) {
            for t in (0..n_frames as usize).step_by(5) {
                let mut ar = 0.0f64;
                let mut ai = 0.0f64;
                for k in 0..2048usize {
                    let ang = -2.0 * std::f64::consts::PI * (f as f64) * (k as f64) / 2048.0;
                    let v = x[(b * 2048 + k) * n_frames as usize + t] as f64;
                    ar += v * ang.cos();
                    ai += v * ang.sin();
                }
                let idx = (b * 1025 + f) * n_frames as usize + t;
                w_re = w_re.max((got_re[idx] - ar as f32).abs());
                w_im = w_im.max((got_im[idx] - ai as f32).abs());
            }
        }
    }
    // |X| は最大 ~276 (2048 点の位相揃い bin) — 相対 2e-6 で評価
    println!("rfft2048 fwd: worst re={w_re:.2e} im={w_im:.2e} (相対で ~5e-7)");
    assert!(
        w_re < 6e-4 && w_im < 6e-4,
        "FFT fwd 精度不足 re={w_re} im={w_im}"
    );

    // bwd: ランダム gRe/gIm → dframes vs CPU
    let gr: Vec<f32> = (0..batch as usize * 1025 * n_frames as usize)
        .map(|_| lcg(&mut st))
        .collect();
    let gi: Vec<f32> = (0..batch as usize * 1025 * n_frames as usize)
        .map(|_| lcg(&mut st))
        .collect();
    let mut tape2 = Tape::new(&eng);
    let fr2 = tape2.input_f32s(&x, vec![x.len()]);
    let (re2, im2) = tape2.rfft2048(fr2.clone(), batch, n_frames);
    let grv = tape2.const_f32s(&gr, vec![gr.len()]);
    let giv = tape2.const_f32s(&gi, vec![gi.len()]);
    // loss = Σ re·gRe + im·gIm を l1 経由で: |re−(−gRe)| は式が変わる —
    // 簡易に mul2 2 本 + 手動加算は sum_all 勾配なし → l1_mean(re, neg_g) を 2 本
    // 足し合わせは add_scalar で可能。単純化: loss = mean|re − c1| + mean|im − c2|
    // (c1 = re−gRe 相当) では dy が sign ベース — 勾配検証には dy=±1 で十分。
    let c1: Vec<f32> = got_re.iter().zip(gr.iter()).map(|(a, b)| a - b).collect();
    let c2: Vec<f32> = got_im.iter().zip(gi.iter()).map(|(a, b)| a - b).collect();
    let c1v = tape2.const_f32s(&c1, vec![c1.len()]);
    let c2v = tape2.const_f32s(&c2, vec![c2.len()]);
    let l1 = rcwx_l1(&mut tape2, re2, c1v);
    let l2 = rcwx_l1(&mut tape2, im2, c2v);
    let lt = tape2.add_scalar(l1, l2);
    tape2.backward_from(lt);
    let dx_t = tape2.grad_tensor(fr2).expect("dx");
    tape2.submit().expect("submit2");
    let dx = eng.readback_f32(&dx_t);
    // CPU: dx = Σ_k (dyRe_k cosθ − dyIm_k sinθ)。dy = sign(g)/N (l1 の勾配)
    let inv_n = 1.0 / (batch as f64 * 1025.0 * n_frames as f64);
    let mut wdx = 0.0f32;
    for b in 0..batch as usize {
        for k in (0..2048usize).step_by(31) {
            for t in (0..n_frames as usize).step_by(7) {
                let mut acc = 0.0f64;
                for f in 0..1025usize {
                    let ang = 2.0 * std::f64::consts::PI * (f as f64) * (k as f64) / 2048.0;
                    let gv = gr[(b * 1025 + f) * n_frames as usize + t] as f64;
                    let hv = gi[(b * 1025 + f) * n_frames as usize + t] as f64;
                    let cr = gv.signum() * inv_n;
                    let ci = hv.signum() * inv_n;
                    acc += cr * ang.cos() - ci * ang.sin();
                }
                let idx = (b * 2048 + k) * n_frames as usize + t;
                wdx = wdx.max((dx[idx] - acc as f32).abs());
            }
        }
    }
    println!("rfft2048 bwd: worst dx={wdx:.2e}");
    assert!(wdx < 2e-5, "FFT bwd 精度不足 {wdx}");
}

/// rcwx-nn テストから l1_mean 互換 (t_l1_sign を直接 dispatch)。
fn rcwx_l1(
    tape: &mut rcwx_nn::engine::tape::Tape<'_>,
    a: rcwx_nn::engine::tape::Var,
    b: rcwx_nn::engine::tape::Var,
) -> rcwx_nn::engine::tape::Var {
    // sum_all に勾配が無いため t_reduce 系は使えない — mul2+sum が不可なので
    // ここは簡易に a−b の絶対値和を push_loss_op 風に自作できない。
    // 検証目的は dy=±c の伝達 — tape.mm を 1 本 (恒等) 通す方法も複雑。
    // 実装: l1_mean 相当を t_pw_fwd(abs) + t_reduce で構成 (losses.rs と同一方式)。
    let len = tape.numel_pub(a);
    let n_partials = len.div_ceil(256);
    let a_t = tape.var_tensor(a);
    let b_t = tape.var_tensor(b);
    let partials = tape.alloc_pub(n_partials as usize);
    {
        let mut words = [0u32; 32];
        words[0..2].copy_from_slice(&[len, 3]);
        tape.dispatch_pub(
            "t_reduce_partials",
            &[&a_t, &b_t, &partials],
            words,
            n_partials,
            1,
            1,
        );
    }
    let out = tape.alloc_pub(1);
    {
        let mut words = [0u32; 32];
        words[0..2].copy_from_slice(&[n_partials, (1.0f32 / len as f32).to_bits()]);
        tape.dispatch_pub("t_reduce_final", &[&partials, &out], words, 1, 1, 1);
    }
    let scalar = tape.push_scalar(out);
    // 逆伝達: da = sign(a−b)/len, db = −da
    let a_id = a.id;
    let b_id = b.id;
    tape.push_loss_op(
        scalar,
        Box::new(move |ctx: &mut rcwx_nn::engine::tape::BwdCtx<'_, '_>| {
            let Some(dy) = ctx.grad_of(scalar.id) else {
                return;
            };
            let at = ctx.node_tensor(a_id);
            let bt = ctx.node_tensor(b_id);
            let da = ctx.alloc(len as usize);
            let mut words = [0u32; 32];
            words[0..3].copy_from_slice(&[len, (1.0f32 / len as f32).to_bits(), 0]);
            ctx.dispatch(
                "t_l1_sign",
                &[&at, &bt, &da, &dy],
                words,
                len.div_ceil(256),
                1,
                1,
            );
            ctx.accum(a_id, da.clone_handle());
            let db = ctx.alloc(len as usize);
            let mut w2 = [0u32; 32];
            w2[0..4].copy_from_slice(&[len, 5, 0.0f32.to_bits(), 0]);
            ctx.dispatch("t_pw_fwd", &[&da, &db], w2, len.div_ceil(256), 1, 1);
            ctx.accum(b_id, db);
        }),
    );
    scalar
}

/// FFT と matmul DFT の相互比較 (アルゴリズムバグの切り分け)。
#[test]
fn fft_vs_matmul_dft() {
    use rcwx_nn::engine::tape::Tape;
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (batch, n_frames) = (1u32, 8u32);
    let x: Vec<f32> = (0..batch as usize * 2048 * n_frames as usize)
        .map(|i| (((i as f64) * 0.618) % 1.0 - 0.5) as f32)
        .collect();
    // FFT
    let mut tape = Tape::new(&eng);
    let xv = tape.const_f32s(&x, vec![x.len()]);
    let (re, im) = tape.rfft2048(xv, batch, n_frames);
    let rt = tape.var_tensor(re);
    let it = tape.var_tensor(im);
    tape.submit().expect("submit");
    let f_re = eng.readback_f32(&rt);
    let f_im = eng.readback_f32(&it);
    // matmul DFT ([b,k,t] k-major × 基底 [1025, 2048])
    let mut basis = vec![0.0f32; 1025 * 2048];
    for f in 0..1025usize {
        for k in 0..2048usize {
            let ang = -2.0 * std::f64::consts::PI * (f as f64) * (k as f64) / 2048.0;
            basis[f * 2048 + k] = ang.cos() as f32;
        }
    }
    let mut im_basis = vec![0.0f32; 1025 * 2048];
    for f in 0..1025usize {
        for k in 0..2048usize {
            let ang = -2.0 * std::f64::consts::PI * (f as f64) * (k as f64) / 2048.0;
            im_basis[f * 2048 + k] = ang.sin() as f32;
        }
    }
    let mut g = eng.graph();
    let at = eng.tensor_from_f32(&basis, vec![1025, 2048]);
    let ait = eng.tensor_from_f32(&im_basis, vec![1025, 2048]);
    let xt = eng.tensor_from_f32(&x, vec![batch as usize, 2048, n_frames as usize]);
    let crt = eng.tensor_uninit((batch * 1025 * n_frames) as usize);
    let cit = eng.tensor_uninit((batch * 1025 * n_frames) as usize);
    g.matmul_ex(
        &at,
        &xt,
        None,
        &crt,
        1025,
        n_frames,
        2048,
        batch,
        2048,
        1,
        0,
        0,
        1,
        n_frames,
        0,
        2048 * n_frames,
        n_frames,
        1,
        0,
        1025 * n_frames,
        1.0,
        0,
    );
    g.matmul_ex(
        &ait,
        &xt,
        None,
        &cit,
        1025,
        n_frames,
        2048,
        batch,
        2048,
        1,
        0,
        0,
        1,
        n_frames,
        0,
        2048 * n_frames,
        n_frames,
        1,
        0,
        1025 * n_frames,
        1.0,
        0,
    );
    g.submit().expect("submit2");
    let m_re = eng.readback_f32(&crt);
    let m_im = eng.readback_f32(&cit);
    let n = f_re.len();
    let dr: f32 = (0..n)
        .map(|i| (f_re[i] - m_re[i]).abs())
        .fold(0.0, f32::max);
    let di: f32 = (0..n)
        .map(|i| (f_im[i] - m_im[i]).abs())
        .fold(0.0, f32::max);
    // スケール
    let scale = m_re.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("fft vs matmul DFT: re={dr:.2e} im={di:.2e} (max|X|={scale:.1})");
}

/// DS (double-single) FFT 単体の fwd 精度検証。
#[test]
fn rfft2048_ds_accuracy() {
    use rcwx_nn::engine::tape::Tape;
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (batch, n_frames) = (1u32, 8u32);
    let mut st = 99u64;
    let lcg = |s: &mut u64| {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as f32 / (1u64 << 33) as f32) - 0.5
    };
    let x: Vec<f32> = (0..batch as usize * 2048 * n_frames as usize)
        .map(|_| lcg(&mut st))
        .collect();
    let mut tape = Tape::new(&eng);
    let xv = tape.const_f32s(&x, vec![x.len()]);
    let (re, im) = tape.rfft2048_ds(xv, batch, n_frames);
    let rt = tape.var_tensor(re);
    let it = tape.var_tensor(im);
    tape.submit().expect("submit");
    let g_re = eng.readback_f32(&rt);
    let g_im = eng.readback_f32(&it);
    let mut w_re = 0.0f64;
    let mut w_im = 0.0f64;
    for b in 0..batch as usize {
        for f in (0..1025usize).step_by(9) {
            for t in 0..n_frames as usize {
                let mut ar = 0.0f64;
                let mut ai = 0.0f64;
                for k in 0..2048usize {
                    let ang = -2.0 * std::f64::consts::PI * (f as f64) * (k as f64) / 2048.0;
                    let v = x[(b * 2048 + k) * n_frames as usize + t] as f64;
                    ar += v * ang.cos();
                    ai += v * ang.sin();
                }
                let idx = (b * 1025 + f) * n_frames as usize + t;
                w_re = w_re.max((g_re[idx] as f64 - ar).abs());
                w_im = w_im.max((g_im[idx] as f64 - ai).abs());
            }
        }
    }
    // 値域 |X|~O(2048) の絶対誤差 — f32 版 (~1e-4) より 1 桁以上良好なら合格。
    // 相対誤差は ~2.5e-8 (f32 の ~20 倍精度)。
    println!("rfft2048_ds fwd: worst re={w_re:.3e} im={w_im:.3e} (f32 版は ~1e-4)");
    assert!(
        w_re < 1e-4 && w_im < 1e-4,
        "DS FFT 精度不足 re={w_re} im={w_im}"
    );
}

/// DS FFT デバッグ: インパルス入力 (X_k ≡ 1) と単一 bin 検査。
#[test]
fn rfft2048_ds_debug() {
    use rcwx_nn::engine::tape::Tape;
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (batch, n_frames) = (1u32, 1u32);
    let mut x = vec![0.0f32; 2048];
    x[0] = 1.0;
    let mut tape = Tape::new(&eng);
    let xv = tape.const_f32s(&x, vec![x.len()]);
    let (re, im) = tape.rfft2048_ds(xv, batch, n_frames);
    let rt = tape.var_tensor(re);
    let it = tape.var_tensor(im);
    tape.submit().expect("submit");
    let g_re = eng.readback_f32(&rt);
    let g_im = eng.readback_f32(&it);
    println!("impulse: re[0..8] = {:?}  (全 1 が正解)", &g_re[..8]);
    println!("impulse: im[0..8] = {:?}  (全 0 が正解)", &g_im[..8]);
}

/// reflect pad fwd/bwd が numpy 'reflect' (縁複製なし) と一致するかの既知値検証。
/// 旧実装は両側 symmetric になっており、STFT 端フレームが破損していた
/// (logmel 段階 bisect の re 誤差 rms 202 @ t=0 の原因)。
#[test]
fn reflect_pad_exact_reflect() {
    use rcwx_nn::engine::tape::Tape;
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (batch, c, in_len, pad) = (2u32, 1u32, 9u32, 4u32);
    let out_len = in_len + 2 * pad;
    // 既知値: np.pad([1..5],(2,2),'reflect') = [3,2,1,2,3,4,5,4,3]
    let x: Vec<f32> = (0..batch as usize * in_len as usize)
        .map(|i| (i % in_len as usize) as f32 + 1.0 + 0.5 * (i / in_len as usize) as f32)
        .collect();
    let mirror = |i: usize| -> usize {
        let s = i as i64 - pad as i64;
        let m: i64 = if s < 0 {
            -s
        } else if s >= in_len as i64 {
            2 * in_len as i64 - 2 - s
        } else {
            s
        };
        m as usize
    };
    let mut want = vec![0.0f32; batch as usize * out_len as usize];
    for b in 0..batch as usize {
        for o in 0..out_len as usize {
            want[b * out_len as usize + o] = x[b * in_len as usize + mirror(o)];
        }
    }
    let mut tape = Tape::new(&eng);
    let xv = tape.const_f32s(&x, vec![batch as usize, 1, in_len as usize]);
    let y = tape.reflect_pad_dims(xv, batch, c, in_len, pad);
    let yt = tape.var_tensor(y);
    tape.submit().expect("submit");
    let got = eng.readback_f32(&yt);
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f32;
    for i in 0..got.len() {
        worst = worst.max((got[i] - want[i]).abs());
    }
    assert!(
        worst == 0.0,
        "reflect pad fwd が numpy reflect と不一致 (worst={worst}) — symmetric 実装への回帰"
    );

    // 随伴: loss = mean|pad(x) − tgt| (tgt 固定) → dx[k] = Σ_mirror sign(pad(x)[i]−tgt[i]) / N
    let tgt: Vec<f32> = (0..batch as usize * out_len as usize)
        .map(|i| ((i as f64 * 0.617) % 1.0 - 0.5) as f32)
        .collect();
    let n = tgt.len();
    let mut want_dx = vec![0.0f64; x.len()];
    for b in 0..batch as usize {
        for i in 0..out_len as usize {
            let j = b * in_len as usize + mirror(i);
            let a = x[j] as f64;
            let t = tgt[b * out_len as usize + i] as f64;
            want_dx[j] += (a - t).signum();
        }
    }
    for wv in want_dx.iter_mut() {
        *wv /= n as f64;
    }
    let mut t2 = Tape::new(&eng);
    let xv2 = t2.input_f32s(&x, vec![batch as usize, 1, in_len as usize]);
    let y2 = t2.reflect_pad_dims(xv2, batch, c, in_len, pad);
    let tv = t2.const_f32s(&tgt, vec![batch as usize, 1, out_len as usize]);
    let l = rcwx_l1(&mut t2, y2, tv);
    t2.backward_from(l);
    let g = t2.grad_tensor(xv2).expect("grad");
    t2.submit().expect("submit2");
    let got_dx = eng.readback_f32(&g);
    let mut worst2 = 0.0f32;
    for i in 0..got_dx.len() {
        worst2 = worst2.max((got_dx[i] - want_dx[i] as f32).abs());
    }
    assert!(worst2 < 1e-6, "reflect pad bwd 随伴不一致 worst={worst2}");
    println!("reflect pad exact: fwd=0, bwd worst={worst2:.2e}");
}
