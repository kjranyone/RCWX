//! MPD 用 conv (conv1d_grouped / conv2d) の有限差分 gradcheck。
//! step_parity の D delta cos が崩れる問題の切り分け用。

use rcwx_nn::engine::tape::Tape;
use rcwx_nn::engine::{Backend, Engine};

fn fd_grad_conv2d() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    // x [1, 2, 10, 3], w [3, 2, 3, 1] k(3,1) s(1,1) pad(1,0)
    let (c_out, c_in, kh, kw, sh, sw, ph, pw, h, w) = (3u32, 2u32, 3, 1, 2, 1, 1, 0, 10, 3);
    let batch = 1u32;
    let x0: Vec<f32> = (0..batch as usize * c_in as usize * h as usize * w as usize)
        .map(|i| ((i as f32 * 0.37) % 1.0) - 0.5)
        .collect();
    let w0: Vec<f32> = (0..c_out as usize * c_in as usize * kh as usize * kw as usize)
        .map(|i| ((i as f32 * 0.61) % 1.0) - 0.5)
        .collect();

    let fwd = |xa: &[f32], wa: &[f32]| -> f32 {
        let mut tape = Tape::new(&eng);
        let xv = tape.const_f32s(xa, vec![xa.len()]);
        let wv = tape.const_f32s(wa, vec![wa.len()]);
        let out = tape.conv2d(
            xv, wv, None, c_out, c_in, kh, kw, sh, sw, ph, pw, batch, h, w,
        );
        let s = tape.sum_all(out);
        let st2 = tape.var_tensor(s);
        tape.submit().expect("submit");
        eng.readback_f32(&st2)[0]
    };

    // 解析勾配
    let mut tape = Tape::new(&eng);
    let xv = tape.input_f32s(&x0, vec![x0.len()]);
    let wv = tape.input_f32s(&w0, vec![w0.len()]);
    let out = tape.conv2d(
        xv.clone(),
        wv.clone(),
        None,
        c_out,
        c_in,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        batch,
        h,
        w,
    );
    // 出力 Var から直接 backward (Σ 播種と同一)
    tape.backward_from(out);
    let dx = tape.grad_tensor(xv).expect("dx");
    let dw = tape.grad_tensor(wv).expect("dw");
    tape.submit().expect("submit");
    let dx = eng.readback_f32(&dx);
    let dw = eng.readback_f32(&dw);

    // FD (dX)
    let eps = 1e-3f32;
    let mut worst_x = 0.0f32;
    for i in [0usize, 5, 17, 33, x0.len() - 1] {
        let mut xp = x0.clone();
        xp[i] += eps;
        let fp = fwd(&xp, &w0);
        xp[i] -= 2.0 * eps;
        let fm = fwd(&xp, &w0);
        let fd = (fp - fm) / (2.0 * eps);
        let d = (fd - dx[i]).abs() / fd.abs().max(1e-3);
        println!("  dX[{i}]: fd={fd:.5} analytic={:.5}", dx[i]);
        worst_x = worst_x.max(d);
    }
    let mut worst_w = 0.0f32;
    for i in [0usize, 3, 9, w0.len() - 1] {
        let mut wp = w0.clone();
        wp[i] += eps;
        let fp = fwd(&x0, &wp);
        wp[i] -= 2.0 * eps;
        let fm = fwd(&x0, &wp);
        let fd = (fp - fm) / (2.0 * eps);
        let d = (fd - dw[i]).abs() / fd.abs().max(1e-3);
        worst_w = worst_w.max(d);
    }
    println!("conv2d FD: worst dX rel={worst_x:.5} worst dW rel={worst_w:.5}");
    assert!(worst_x < 2e-2, "conv2d dX FD mismatch");
    assert!(worst_w < 2e-2, "conv2d dW FD mismatch");
}

fn fd_grad_conv1d_grouped() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    // c_in 4 → c_out 8, groups 2, k 3, stride 1
    let (c_out, c_in, k, s, g, t) = (8u32, 4u32, 3, 4, 2, 21);
    let batch = 1u32;
    let x0: Vec<f32> = (0..batch as usize * c_in as usize * t as usize)
        .map(|i| ((i as f32 * 0.29) % 1.0) - 0.5)
        .collect();
    let w0: Vec<f32> = (0..c_out as usize * (c_in / g) as usize * k as usize)
        .map(|i| ((i as f32 * 0.53) % 1.0) - 0.5)
        .collect();
    let out_len = (t - k) / s + 1;

    let fwd = |xa: &[f32], wa: &[f32]| -> f32 {
        let mut tape = Tape::new(&eng);
        let xv = tape.const_f32s(xa, vec![xa.len()]);
        let wv = tape.const_f32s(wa, vec![wa.len()]);
        let out = tape.conv1d_grouped(xv, wv, None, c_out, c_in, k, s, 0, g, batch, t, out_len);
        let s = tape.sum_all(out);
        let st2 = tape.var_tensor(s);
        tape.submit().expect("submit");
        eng.readback_f32(&st2)[0]
    };

    let mut tape = Tape::new(&eng);
    let xv = tape.input_f32s(&x0, vec![x0.len()]);
    let wv = tape.input_f32s(&w0, vec![w0.len()]);
    let out = tape.conv1d_grouped(
        xv.clone(),
        wv.clone(),
        None,
        c_out,
        c_in,
        k,
        s,
        0,
        g,
        batch,
        t,
        out_len,
    );
    tape.backward_from(out);
    let dx_t = tape.grad_tensor(xv).expect("dx");
    let dw_t = tape.grad_tensor(wv).expect("dw");
    tape.submit().expect("submit");
    let dx = eng.readback_f32(&dx_t);
    let dw = eng.readback_f32(&dw_t);

    let eps = 1e-3f32;
    let mut worst_x = 0.0f32;
    for i in [0usize, 7, 21, x0.len() - 1] {
        let mut xp = x0.clone();
        xp[i] += eps;
        let fp = fwd(&xp, &w0);
        xp[i] -= 2.0 * eps;
        let fm = fwd(&xp, &w0);
        let fd = (fp - fm) / (2.0 * eps);
        let d = (fd - dx[i]).abs() / fd.abs().max(1e-3);
        worst_x = worst_x.max(d);
    }
    let mut worst_w = 0.0f32;
    for i in [0usize, 4, 11, w0.len() - 1] {
        let mut wp = w0.clone();
        wp[i] += eps;
        let fp = fwd(&x0, &wp);
        wp[i] -= 2.0 * eps;
        let fm = fwd(&x0, &wp);
        let fd = (fp - fm) / (2.0 * eps);
        let d = (fd - dw[i]).abs() / fd.abs().max(1e-3);
        worst_w = worst_w.max(d);
    }
    println!("conv1d_grouped FD: worst dX rel={worst_x:.5} worst dW rel={worst_w:.5}");
    assert!(worst_x < 2e-2, "conv1d_grouped dX FD mismatch");
    assert!(worst_w < 2e-2, "conv1d_grouped dW FD mismatch");
}

#[test]
fn mpd_conv_gradcheck() {
    fd_grad_conv2d();
    fd_grad_conv1d_grouped();
}

/// 2 層 grouped conv チェーン (disc_s と同じ構造) の FD。
#[test]
fn grouped_chain_gradcheck() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let t = 200u32;
    let batch = 1u32;
    let x0: Vec<f32> = (0..batch as usize * t as usize)
        .map(|i| ((i as f32 * 0.31) % 1.0) - 0.5)
        .collect();
    let w1: Vec<f32> = (0..8 * 1 * 15)
        .map(|i| ((i as f32 * 0.17) % 1.0) - 0.5)
        .collect();
    let b1: Vec<f32> = vec![0.05; 8];
    let o1 = t - 15 + 1;
    let w2: Vec<f32> = (0..16 * 2 * 41)
        .map(|i| ((i as f32 * 0.13) % 1.0) - 0.5)
        .collect();
    let b2: Vec<f32> = vec![0.02; 16];
    let o2 = (o1 - 41) / 4 + 1;

    let run = |x: Option<&[f32]>,
               w1p: Option<&[f32]>,
               w2p: Option<&[f32]>|
     -> (f32, Option<Vec<f32>>, Option<Vec<f32>>) {
        let xa = x.unwrap_or(&x0);
        let w1a = w1p.unwrap_or(&w1);
        let w2a = w2p.unwrap_or(&w2);
        let mut tape = Tape::new(&eng);
        let xv = tape.input_f32s(xa, vec![xa.len()]);
        let w1v = tape.input_f32s(w1a, vec![w1a.len()]);
        let w2v = tape.input_f32s(w2a, vec![w2a.len()]);
        let bv1 = tape.const_f32s(&b1, vec![8]);
        let bv2 = tape.const_f32s(&b2, vec![16]);
        let h = tape.conv1d_grouped(
            xv.clone(),
            w1v.clone(),
            Some(bv1),
            8,
            1,
            15,
            1,
            0,
            1,
            batch,
            t,
            o1,
        );
        let h = tape.lrelu(h, 0.1);
        let out = tape.conv1d_grouped(h, w2v.clone(), Some(bv2), 16, 8, 41, 4, 0, 4, batch, o1, o2);
        let s = tape.sum_all(out);
        let st = tape.var_tensor(s);
        tape.backward_from(out);
        let gx = tape.grad_tensor(xv);
        let g1 = tape.grad_tensor(w1v);
        let g2 = tape.grad_tensor(w2v);
        tape.submit().expect("submit");
        let l = eng.readback_f32(&st)[0];
        println!(
            "  grads present: x={} w1={} w2={}",
            gx.is_some(),
            g1.is_some(),
            g2.is_some()
        );
        (
            l,
            g1.map(|g| eng.readback_f32(&g)),
            g2.map(|g| eng.readback_f32(&g)),
        )
    };

    let (_, gw1, gw2) = run(None, None, None);
    let gw1 = gw1.expect("g1");
    let gw2 = gw2.expect("g2");
    let eps = 1e-3f32;
    let mut worst1 = 0.0f32;
    for i in [0usize, 40, 77, w1.len() - 1] {
        let mut wp = w1.clone();
        wp[i] += eps;
        let (fp, _, _) = run(None, Some(&wp), None);
        wp[i] -= 2.0 * eps;
        let (fm, _, _) = run(None, Some(&wp), None);
        let fd = (fp - fm) / (2.0 * eps);
        let d = (fd - gw1[i]).abs() / fd.abs().max(1e-3);
        println!("  chain w1[{i}]: fd={fd:.5} analytic={:.5}", gw1[i]);
        worst1 = worst1.max(d);
    }
    let mut worst2 = 0.0f32;
    let step = (w2.len() / 24).max(1);
    let idxs: Vec<usize> = (0..24).map(|k| k * step).collect();
    for i in idxs {
        let mut wp = w2.clone();
        wp[i] += eps;
        let (fp, _, _) = run(None, None, Some(&wp));
        wp[i] -= 2.0 * eps;
        let (fm, _, _) = run(None, None, Some(&wp));
        let fd = (fp - fm) / (2.0 * eps);
        let d = (fd - gw2[i]).abs() / fd.abs().max(1e-3);
        if d > 0.05 {
            // w2 layout [o=16, cin_g=2, k=41]: idx = (o*2 + c)*41 + kk
            let o = i / (2 * 41);
            let rem = i % (2 * 41);
            println!(
                "  w2[{i}] (o={o},c={},k={}): fd={fd:.4} an={:.4}",
                gw2[i],
                rem / 41,
                rem % 41
            );
        }
        worst2 = worst2.max(d);
    }
    println!("chain FD: w1 worst={worst1:.5} w2 worst={worst2:.5}");
    assert!(worst1 < 2e-2, "chain w1 FD mismatch");
    assert!(worst2 < 2e-2, "chain w2 FD mismatch");
}

/// 大型次元 (tiled matmul + colsum > 65535) の conv2d dW/dX FD。
/// disc_p 深層の崩れ (colsum 列上限) の回帰。
#[test]
fn conv2d_large_fd() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    // out_c·ck = 1024·2560 = 2.6M > 65535 (colsum 修正のターゲット)
    let (c_out, c_in, kh, kw, sh, sw, ph, pw, h, w) = (
        1024u32, 512u32, 5u32, 1u32, 3u32, 1u32, 2u32, 0u32, 300u32, 2u32,
    );
    let batch = 2u32;
    let oh = (h + 2 * ph - kh) / sh + 1;
    let x0: Vec<f32> = (0..batch as usize * c_in as usize * h as usize * w as usize)
        .map(|i| ((i as f32 * 0.371) % 1.0) - 0.5)
        .collect();
    let w0: Vec<f32> = (0..c_out as usize * c_in as usize * kh as usize)
        .map(|i| ((i as f32 * 0.613) % 1.0) - 0.5)
        .collect();
    let b0: Vec<f32> = vec![0.0; c_out as usize];

    let fwd = |xa: &[f32], wa: &[f32]| -> f32 {
        let mut tape = Tape::new(&eng);
        let xv = tape.const_f32s(xa, vec![xa.len()]);
        let wv = tape.const_f32s(wa, vec![wa.len()]);
        let bv = tape.const_f32s(&b0, vec![b0.len()]);
        let out = tape.conv2d(
            xv,
            wv,
            Some(bv),
            c_out,
            c_in,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            batch,
            h,
            w,
        );
        let s = tape.sum_all(out);
        let st = tape.var_tensor(s);
        tape.submit().expect("submit");
        eng.readback_f32(&st)[0]
    };
    let mut tape = Tape::new(&eng);
    let xv = tape.input_f32s(&x0, vec![x0.len()]);
    let wv = tape.input_f32s(&w0, vec![w0.len()]);
    let out = tape.conv2d(
        xv.clone(),
        wv.clone(),
        None,
        c_out,
        c_in,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        batch,
        h,
        w,
    );
    tape.backward_from(out);
    let dx_t = tape.grad_tensor(xv).expect("dx");
    let dw_t = tape.grad_tensor(wv).expect("dw");
    tape.submit().expect("submit");
    let dx = eng.readback_f32(&dx_t);
    let dw = eng.readback_f32(&dw_t);

    // CPU 参照 (dy=1): dW[o, (c,j)] = Σ_b Σ_oy Σ_ox x[b, c, oy·sh+j−ph, ox]
    let idx_w = |o: usize, c: usize, j: usize| (o * c_in as usize + c) * kh as usize + j;
    let idx_x = |b: usize, c: usize, y: usize, x: usize| {
        ((b * c_in as usize + c) * h as usize + y) * w as usize + x
    };
    let mut worst_w = 0.0f32;
    for (o, c, j) in [
        (0usize, 0usize, 0usize),
        (0, 200, 0),
        (3, 101, 4),
        (512, 0, 0),
        (1023, 511, 4),
    ] {
        let mut acc = 0.0f64;
        for b in 0..batch as usize {
            for oy in 0..oh as usize {
                for ox in 0..w as usize {
                    let wy = oy * sh as usize + j;
                    if wy >= ph as usize && wy < ph as usize + h as usize {
                        acc += x0[idx_x(b, c, wy - ph as usize, ox)] as f64;
                    }
                }
            }
        }
        let want = acc as f32;
        let got = dw[idx_w(o, c, j)];
        let d = (got - want).abs() / want.abs().max(1.0);
        println!("  big w[({o},{c},{j})]: cpu={want:.4} analytic={got:.4}");
        worst_w = worst_w.max(d);
    }
    // dX スポット CPU 参照
    let mut worst_x = 0.0f32;
    for (b, c, y, x) in [
        (0usize, 0usize, 0usize, 0usize),
        (1, 100, 250, 1),
        (1, 511, 299, 1),
    ] {
        let mut acc = 0.0f64;
        for o in 0..c_out as usize {
            for j in 0..kh as usize {
                let wy = y + j;
                let oy = (wy + ph as usize).saturating_sub(j); // = y+ph-j → oy·sh+j = y+ph+j? 逆算
                let _ = oy;
                // x 位置 y に寄与する出力: oy·sh + j = y + ph → oy = (y + ph - j) / sh が整数かつ範囲内
                if (y + ph as usize) >= j && ((y + ph as usize - j) % sh as usize) == 0 {
                    let oy = (y + ph as usize - j) / sh as usize;
                    if oy < oh as usize {
                        acc += w0[idx_w(o, c, j)] as f64;
                    }
                }
            }
        }
        let want = acc as f32;
        let got = dx[idx_x(b, c, y, x)];
        let d = (got - want).abs() / want.abs().max(1.0);
        println!("  big x[({b},{c},{y},{x})]: cpu={want:.4} analytic={got:.4}");
        worst_x = worst_x.max(d);
    }
    println!("conv2d large CPU-ref: dW worst={worst_w:.5} dX worst={worst_x:.5} oh={oh}");
    assert!(worst_w < 1e-3, "large dW CPU-ref {worst_w}");
    assert!(worst_x < 1e-3, "large dX CPU-ref {worst_x}");
}

/// gather_frames (start=5, B=2) の全要素 CPU 参照比較。
#[test]
fn gather_frames_batch2() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (batch, ch, t_len, seg, start) = (2u32, 6u32, 2310u32, 2300u32, 5u32);
    let x: Vec<f32> = (0..batch as usize * ch as usize * t_len as usize)
        .map(|i| i as f32 * 0.001)
        .collect();
    let mut tape = Tape::new(&eng);
    let xv = tape.const_f32s(&x, vec![batch as usize, ch as usize, t_len as usize]);
    let got = tape.gather_frames_pub(xv, start, ch, t_len, seg);
    let gt = tape.var_tensor(got);
    tape.submit().expect("submit");
    let g = eng.readback_f32(&gt);
    let mut worst = 0.0f32;
    for b in 0..batch as usize {
        for c in 0..ch as usize {
            for i in 0..seg as usize {
                let want =
                    x[b * ch as usize * t_len as usize + c * t_len as usize + start as usize + i];
                let got_v = g[b * ch as usize * seg as usize + c * seg as usize + i];
                worst = worst.max((got_v - want).abs());
            }
        }
    }
    println!("gather B=2 worst = {worst:.6}");
    assert!(worst < 1e-5, "gather_frames B=2 mismatch {worst}");
}

/// resblock 相当の dilated conv1d (k=7 dil=3, 中規模チャネル) の dW/dX FD。
#[test]
fn conv1d_dilated_fd() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    // in=64 out=32 k=7 dil=3 stride=1 (resblock convs2 相当)
    let (c_out, c_in, k, s, dil, t) = (32u32, 64u32, 7u32, 1u32, 3u32, 500u32);
    let batch = 2u32;
    let out_len = (t - dil * (k - 1) - 1) / s + 1;
    let x0: Vec<f32> = (0..batch as usize * c_in as usize * t as usize)
        .map(|i| ((i as f32 * 0.37) % 1.0) - 0.5)
        .collect();
    let w0: Vec<f32> = (0..c_out as usize * c_in as usize * k as usize)
        .map(|i| ((i as f32 * 0.61) % 1.0) - 0.5)
        .collect();
    let b0: Vec<f32> = vec![0.0; c_out as usize];

    let fwd = |xa: &[f32], wa: &[f32]| -> f32 {
        let mut tape = Tape::new(&eng);
        let xv = tape.const_f32s(xa, vec![xa.len()]);
        let wv = tape.const_f32s(wa, vec![wa.len()]);
        let bv = tape.const_f32s(&b0, vec![b0.len()]);
        let out = tape.conv1d(
            xv,
            wv,
            Some(bv),
            c_out,
            c_in,
            k,
            s,
            dil,
            0,
            batch,
            t,
            out_len,
        );
        let sm = tape.sum_all(out);
        let st = tape.var_tensor(sm);
        tape.submit().expect("submit");
        eng.readback_f32(&st)[0]
    };
    let mut tape = Tape::new(&eng);
    let xv = tape.input_f32s(&x0, vec![x0.len()]);
    let wv = tape.input_f32s(&w0, vec![w0.len()]);
    let out = tape.conv1d(
        xv.clone(),
        wv.clone(),
        None,
        c_out,
        c_in,
        k,
        s,
        dil,
        0,
        batch,
        t,
        out_len,
    );
    tape.backward_from(out);
    let dx_t = tape.grad_tensor(xv).expect("dx");
    let dw_t = tape.grad_tensor(wv).expect("dw");
    tape.submit().expect("submit");
    let dx = eng.readback_f32(&dx_t);
    let dw = eng.readback_f32(&dw_t);

    // CPU 参照: loss=Σout (dy=1)
    let mut ref_dw = vec![0.0f64; w0.len()];
    let mut ref_dx = vec![0.0f64; x0.len()];
    for b in 0..batch as usize {
        for o in 0..c_out as usize {
            for ot in 0..out_len as usize {
                for c in 0..c_in as usize {
                    for j in 0..k as usize {
                        let src = ot + j * dil as usize;
                        let wv = w0[((o * c_in as usize + c) * k as usize + j)] as f64;
                        let xv = x0[((b * c_in as usize + c) * t as usize + src)] as f64;
                        ref_dw[(o * c_in as usize + c) * k as usize + j] += xv;
                        ref_dx[(b * c_in as usize + c) * t as usize + src] += wv;
                    }
                }
            }
        }
    }
    let worst = |got: &[f32], want: &[f64]| {
        got.iter()
            .zip(want)
            .map(|(g, w)| (f64::from(*g) - w).abs())
            .fold(0.0f64, f64::max)
    };
    let wdw = worst(&dw, &ref_dw);
    let wdx = worst(&dx, &ref_dx);
    println!("dilated conv1d: dW worst={wdw:.6} dX worst={wdx:.6} out_len={out_len}");
    assert!(wdw < 1e-3, "dilated dW {wdw}");
    assert!(wdx < 1e-3, "dilated dX {wdx}");
}

/// logmel 逆伝達の FD 検証 (mel 経路の G 勾配残差の原因特定)。
/// 損失は学習と同じ l1_mean(logmel(a), logmel(b)) — a 側のみ FD する。
#[test]
fn logmel_backward_fd() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let stft = rcwx_train::modules::stft_mel::StftMel::new_rust(&eng, 2048, 480, 128, 48000);
    let batch = 2u32;
    let samples = 18400u32;
    let ya: Vec<f32> = (0..batch as usize * samples as usize)
        .map(|i| ((i as f64 * 0.37 + (i % 7) as f64 * 0.11) % 1.0 - 0.5) as f32)
        .collect();
    let yb: Vec<f32> = (0..batch as usize * samples as usize)
        .map(|i| ((i as f64 * 0.71 + (i % 5) as f64 * 0.13) % 1.0 - 0.5) as f32)
        .collect();

    let run = |ya: &[f32]| -> f32 {
        let mut tape = Tape::new(&eng);
        let av = tape.const_f32s(ya, vec![ya.len()]);
        let bv = tape.const_f32s(&yb, vec![yb.len()]);
        let ma = stft.logmel(&mut tape, av, batch, samples);
        let mb = stft.logmel(&mut tape, bv, batch, samples);
        let l = rcwx_train::losses::mel_l1_loss(&mut tape, ma, mb, 1.0);
        let lt = tape.var_tensor(l);
        tape.submit().expect("submit");
        eng.readback_f32(&lt)[0]
    };

    // 解析 dy (a 側)
    let mut tape = Tape::new(&eng);
    let av = tape.input_f32s(&ya, vec![ya.len()]);
    let bv = tape.const_f32s(&yb, vec![yb.len()]);
    let ma = stft.logmel(&mut tape, av, batch, samples);
    let mb = stft.logmel(&mut tape, bv, batch, samples);
    let l = rcwx_train::losses::mel_l1_loss(&mut tape, ma, mb, 1.0);
    tape.backward_from(l);
    let dy_t = tape.grad_tensor(av).expect("dy");
    tape.submit().expect("submit");
    let dy = eng.readback_f32(&dy_t);

    // FD: |解析 dy| が大きい上位 30 点のみ (小値は FD ノイズに埋もれる)
    let eps = 0.5f32;
    let mut idxs: Vec<usize> = (0..dy.len()).collect();
    idxs.sort_by(|&a, &b| dy[a].abs().partial_cmp(&dy[b].abs()).unwrap());
    let picks: Vec<usize> = idxs[dy.len() - 30..].to_vec();
    let mut worst = 0.0f64;
    let mut worst_i = 0usize;
    let mut n_big = 0usize;
    for &i in &picks {
        let mut yp = ya.clone();
        yp[i] += eps;
        let fp = run(&yp);
        yp[i] -= 2.0 * eps;
        let fm = run(&yp);
        let fd = (f64::from(fp) - f64::from(fm)) / f64::from(2.0 * eps);
        let d = ((fd - f64::from(dy[i])) / fd.abs().max(1e-2)).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
        n_big += 1;
        println!("  dy[{i}]: fd={fd:.6} an={:.6}", dy[i]);
    }
    println!("entries checked: {n_big}");
    println!(
        "logmel bwd FD worst rel = {worst:.4} at {worst_i} (fd dy={:.5}, an={:.5})",
        (run(&ya.clone()) - run(&ya)) as f64 / 1.0, // dummy to keep types
        dy[worst_i]
    );
    assert!(worst < 0.05, "logmel backward FD mismatch {worst} at {worst_i}");
}
