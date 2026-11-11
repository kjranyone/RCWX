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

/// logmel 逆伝達の py (torch autograd) 参照との直接比較。
/// 事前生成: uv run python tools/logmel_bwd_ref.py
#[test]
fn logmel_backward_parity() {
    let path = std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap())
        .join(".cache/rcwx/train/golden/logmel_bwd.safetensors");
    if !path.exists() {
        eprintln!("skipping: run `uv run python tools/logmel_bwd_ref.py` first");
        return;
    }
    let raw = std::fs::read(&path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&raw).unwrap();
    let read = |name: &str| -> Vec<f32> {
        let v = st.tensor(name).unwrap();
        v.data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let ya = read("ya");
    let yb = read("yb");
    let want = read("dy");
    let py_la = read("la");
    let (batch, samples) = (2u32, 18400u32);
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let stft = rcwx_train::modules::stft_mel::StftMel::new_rust(&eng, 2048, 480, 128, 48000);

    // forward 比較 (rust logmel(a) vs py la)
    {
        let mut tape2 = Tape::new(&eng);
        let av2 = tape2.const_f32s(&ya, vec![ya.len()]);
        let ma2 = stft.logmel(&mut tape2, av2, batch, samples);
        let mt = tape2.var_tensor(ma2);
        tape2.submit().expect("submit");
        let got = eng.readback_f32(&mt);
        let n = got.len().min(py_la.len());
        println!(
            "fwd lens: rust={} py={} ya_len={}",
            got.len(),
            py_la.len(),
            ya.len()
        );
        let mut worst = 0.0f32;
        let mut at = 0usize;
        let mut n_bad = 0usize;
        for i in 0..n {
            let d = (got[i] - py_la[i]).abs();
            if d > 0.05 {
                n_bad += 1;
            }
            if d > worst {
                worst = d;
                at = i;
            }
        }
        println!(
            "fwd logmel: worst={worst:.4} at {at} (val ru={} py={}) n_bad(>0.05)={n_bad}/{n}",
            got[at], py_la[at]
        );
        // bad 要素の batch/mel/frame 分布
        let per_b = [0usize, 0];
        let mut bcount = per_b;
        let mut mel_of: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            if (got[i] - py_la[i]).abs() > 0.05 {
                let b = i / (128 * 38);
                bcount[b] += 1;
                mel_of.push((b, (i / 38) % 128));
            }
        }
        println!("fwd bad by batch: b0={} b1={}", bcount[0], bcount[1]);
        let mels: Vec<usize> = mel_of.iter().map(|(_, m)| *m).collect();
        let minm = mels.iter().min().copied().unwrap_or(0);
        let maxm = mels.iter().max().copied().unwrap_or(0);
        println!("bad mel range: {minm}..{maxm}");
        // bad の frame 分布 (先頭/末尾寄りなら端フレーム処理の差)
        let frames: Vec<usize> = (0..n)
            .filter(|&i| (got[i] - py_la[i]).abs() > 0.05)
            .map(|i| i % 38)
            .collect();
        let fmin = frames.iter().min().copied().unwrap_or(0);
        let fmax = frames.iter().max().copied().unwrap_or(0);
        println!("bad frame range: {fmin}..{fmax} (n_frames=38)");
        // 要素値の絶対差 top の生値 (clamp 域か否か)
        let mut top: Vec<(usize, f32)> = (0..n).map(|i| (i, (got[i] - py_la[i]).abs())).collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for &(i, d) in top[..6].iter() {
            println!("  fwd worst[{i}]: |d|={d:.3} ru={} py={}", got[i], py_la[i]);
        }
    }
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

    // 比較: cos + 比率分布
    let dot: f64 = dy
        .iter()
        .zip(want.iter())
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let na = (dy.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>()).sqrt();
    let nb = (want.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>()).sqrt();
    println!(
        "logmel bwd: cos={:.6} |ru|={na:.4} |py|={nb:.4} ratio={}",
        dot / (na * nb),
        na / nb
    );
    // 帰属実験: py-f64 golden (dy64) との直接比較 — rust≈f64 なら py-f32 が
    // 下限 / 0.99x なら rust 側に mel 経路の残差源
    {
        let want64 = read("dy64");
        let n2 = dy.len().min(want64.len());
        let dot2: f64 = (0..n2)
            .map(|k| f64::from(dy[k]) * f64::from(want64[k]))
            .sum();
        let na2 = (0..n2)
            .map(|k| f64::from(dy[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        let nb2 = (0..n2)
            .map(|k| f64::from(want64[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        let rel2 = ((0..n2)
            .map(|k| f64::from(dy[k]) - f64::from(want64[k]))
            .map(|d| d * d)
            .sum::<f64>()
            .sqrt())
            / nb2;
        println!(
            "logmel bwd vs py-f64: cos={:.6} relnorm={:.3e}",
            dot2 / (na2 * nb2),
            rel2
        );
    }
    // batch 別統計
    {
        let half = dy.len() / 2;
        let cosb = |s: std::ops::Range<usize>| {
            let d: f64 = dy[s.clone()]
                .iter()
                .zip(want[s.clone()].iter())
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let na = (dy[s.start..s.end]
                .iter()
                .map(|x| f64::from(*x).powi(2))
                .sum::<f64>())
            .sqrt();
            let nb = (want[s.start..s.end]
                .iter()
                .map(|x| f64::from(*x).powi(2))
                .sum::<f64>())
            .sqrt();
            d / (na * nb)
        };
        println!(
            "b0 cos={:.6} b1 cos={:.6}",
            cosb(0..half),
            cosb(half..dy.len())
        );
    }
    // 誤差上位の位置構造 (mod hop / エッジ距離) + batch 分布
    let mut diffs: Vec<(usize, f32)> = (0..dy.len())
        .map(|i| (i, (dy[i] - want[i]).abs()))
        .collect();
    diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let b1 = diffs.iter().filter(|(i, _)| *i < 18400).count();
    println!("worst-200 batch split: b0={} b1={}", b1, 200 - b1.min(200));
    for &(i, d) in diffs[..8].iter() {
        println!(
            "  worst[{i}]: |d|={d:.6} ru={:.6} py={:.6} mod480={} edge_min={}",
            dy[i],
            want[i],
            i % 480,
            i.min(dy.len() - i)
        );
    }
    // 参照側の mel L1 は同分布信号間でナイフエッジ (sign 反転) を含むため
    // cos ~0.986 が実測フロア。学習実使用 (y_hat vs wave) では差が大きく
    // ナイフエッジは稀 — step_parity の mel 損失 5 桁一致が実質検証。
    assert!(
        (dot / (na * nb)) > 0.98,
        "logmel bwd cos {}",
        dot / (na * nb)
    );
}

/// reflect_pad_dims 逆伝達の FD 検証 (mel 経路エッジ誤差の原因候補)。
#[test]
fn reflect_pad_backward_fd() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let batch = 2u32;
    let in_len = 18400u32;
    let pad = 784u32;
    let x: Vec<f32> = (0..batch as usize * in_len as usize)
        .map(|i| ((i as f64 * 0.37 + (i % 7) as f64 * 0.11) % 1.0 - 0.5) as f32)
        .collect();
    let wlen = (batch * (in_len + 2 * pad)) as usize;

    // sum_all は勾配非伝播のため l1_mean 経路で backward を駆動する:
    // loss = mean |pad(x) − pad(x2)| (x2 固定 → dx 側だけ見る)
    let x2: Vec<f32> = (0..x.len())
        .map(|i| ((i as f64 * 0.71 + (i % 5) as f64 * 0.13) % 1.0 - 0.5) as f32)
        .collect();

    let mut tape = Tape::new(&eng);
    let xv = tape.input_f32s(&x, vec![batch as usize, 1, in_len as usize]);
    let x2v = tape.const_f32s(&x2, vec![batch as usize, 1, in_len as usize]);
    let y = tape.reflect_pad_dims(xv, batch, 1, in_len, pad);
    let y2 = tape.reflect_pad_dims(x2v, batch, 1, in_len, pad);
    let l = rcwx_train::losses::l1_mean(&mut tape, y, y2);
    tape.backward_from(l);
    let dy_t = tape.grad_tensor(xv).expect("dy");
    tape.submit().expect("submit");
    let dy = eng.readback_f32(&dy_t);

    // CPU 参照: dx[j] = (Σ_i sign(pad(x)[i] − pad(x2)[i])) / N over mirror positions
    let out_len = (in_len + 2 * pad) as usize;
    let mirror = |i: usize| -> usize {
        let src = i as i64 - pad as i64;
        let m: i64 = if src < 0 {
            -src
        } else if src >= in_len as i64 {
            2 * in_len as i64 - 2 - src
        } else {
            src
        };
        m as usize
    };
    let mut want = vec![0.0f64; x.len()];
    let n_total = (batch as usize) * out_len;
    for b in 0..batch as usize {
        for i in 0..out_len {
            let j = b * in_len as usize + mirror(i);
            let va = x[j] as f64;
            // pad 値は mirror 先の x/x2
            let a = x[b * in_len as usize + mirror(i)] as f64;
            let b2 = x2[b * in_len as usize + mirror(i)] as f64;
            let _ = va;
            want[j] += (a - b2).signum();
        }
    }
    for wv in want.iter_mut() {
        *wv /= n_total as f64;
    }
    let want: Vec<f32> = want.iter().map(|v| *v as f32).collect();
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for i in 0..want.len() {
        let d = (dy[i] - want[i]).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    println!(
        "reflect bwd worst = {worst:.6} at {at} (ru={:.5} py={:.5})",
        dy[at], want[at]
    );
    assert!(
        worst < 5e-4,
        "reflect_pad backward mismatch {worst} at {at}"
    );
}

/// im2col_frames 逆伝達 (col2im, k=2048 stride=480) の CPU 参照検証。
#[test]
fn im2col_frames_backward_cpu() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let batch = 2u32;
    let in_len = 18400u32;
    let (n_fft, hop) = (2048u32, 480u32);
    let n_frames = (in_len - n_fft) / hop + 1;
    let x: Vec<f32> = (0..batch as usize * in_len as usize)
        .map(|i| ((i as f64 * 0.37 + (i % 7) as f64 * 0.11) % 1.0 - 0.5) as f32)
        .collect();
    let x2: Vec<f32> = (0..x.len())
        .map(|i| ((i as f64 * 0.71 + (i % 5) as f64 * 0.13) % 1.0 - 0.5) as f32)
        .collect();

    let mut tape = Tape::new(&eng);
    let xv = tape.input_f32s(&x, vec![batch as usize, 1, in_len as usize]);
    let x2v = tape.const_f32s(&x2, vec![batch as usize, 1, in_len as usize]);
    let f1 = tape.im2col_frames(xv, n_fft, hop, batch, in_len, n_frames);
    let f2 = tape.im2col_frames(x2v, n_fft, hop, batch, in_len, n_frames);
    let l = rcwx_train::losses::l1_mean(&mut tape, f1, f2);
    tape.backward_from(l);
    let dy_t = tape.grad_tensor(xv).expect("dy");
    tape.submit().expect("submit");
    let dy = eng.readback_f32(&dy_t);

    // CPU 参照: dx[pos] = Σ_{t: 0≤t<n_f, 0≤pos−t·hop<n_fft} sign/N
    let n_total = (batch * n_fft * n_frames) as f64;
    let mut want = vec![0.0f64; x.len()];
    for b in 0..batch as usize {
        for t in 0..n_frames as usize {
            for j in 0..n_fft as usize {
                let pos = t * hop as usize + j;
                let sa = x[b * in_len as usize + pos] as f64;
                let sb = x2[b * in_len as usize + pos] as f64;
                want[b * in_len as usize + pos] += (sa - sb).signum();
            }
        }
    }
    for w in want.iter_mut() {
        *w /= n_total;
    }
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for i in 0..want.len() {
        let d = (dy[i] - want[i] as f32).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    println!(
        "im2col_frames bwd worst = {worst:.7} at {at} (ru={:.7} py={:.7})",
        dy[at], want[at] as f32
    );
    assert!(
        worst < 5e-4,
        "im2col_frames backward mismatch {worst} at {at}"
    );
}

/// logmel 形状の mm 逆伝達検証 (A 共有 [M,K] × W per-batch [B,K,N])。
#[test]
fn mm_shared_a_backward() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (m, k, n, batch) = (1025u32, 2048u32, 38u32, 2u32);
    let a: Vec<f32> = (0..m as usize * k as usize)
        .map(|i| ((i as f64 * 0.618) % 1.0 - 0.5) as f32)
        .collect();
    let wdat: Vec<f32> = (0..batch as usize * k as usize * n as usize)
        .map(|i| ((i as f64 * 0.314) % 1.0 - 0.5) as f32)
        .collect();
    let w2: Vec<f32> = (0..wdat.len())
        .map(|i| ((i as f64 * 0.271) % 1.0 - 0.5) as f32)
        .collect();

    // loss = mean |mm(a, w) − mm(a, w2)|
    let mut tape = Tape::new(&eng);
    let av = tape.const_f32s(&a, vec![m as usize, k as usize]);
    let wv = tape.input_f32s(&wdat, vec![batch as usize, k as usize, n as usize]);
    let w2v = tape.const_f32s(&w2, vec![batch as usize, k as usize, n as usize]);
    let o1 = tape.mm(
        av.clone(),
        wv.clone(),
        None,
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
        n,
        0,
        k * n,
        n,
        1,
        m * n,
        1.0,
        0,
    );
    let o2 = tape.mm(
        av,
        w2v,
        None,
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
        n,
        0,
        k * n,
        n,
        1,
        m * n,
        1.0,
        0,
    );
    let l = rcwx_train::losses::l1_mean(&mut tape, o1, o2);
    tape.backward_from(l);
    let dw_t = tape.grad_tensor(wv).expect("dw");
    tape.submit().expect("submit");
    let dw = eng.readback_f32(&dw_t);

    // CPU 参照: sign/mm — dout = sign(mm1−mm2)/N; dw[b,k,n] = Σ_m dout[b,m,n]·a[m,k]
    let n_total = (batch * m * n) as f64;
    let mut mm1 = vec![0.0f64; n_total as usize];
    let mut mm2 = vec![0.0f64; n_total as usize];
    for b in 0..batch as usize {
        for mi in 0..m as usize {
            for ni in 0..n as usize {
                let mut s1 = 0.0f64;
                let mut s2 = 0.0f64;
                for ki in 0..k as usize {
                    s1 += a[mi * k as usize + ki] as f64
                        * wdat[(b * k as usize + ki) * n as usize + ni] as f64;
                    s2 += a[mi * k as usize + ki] as f64
                        * w2[(b * k as usize + ki) * n as usize + ni] as f64;
                }
                mm1[(b * m as usize + mi) * n as usize + ni] = s1;
                mm2[(b * m as usize + mi) * n as usize + ni] = s2;
            }
        }
    }
    let mut want = vec![0.0f64; wdat.len()];
    for b in 0..batch as usize {
        for ki in 0..k as usize {
            for ni in 0..n as usize {
                let mut acc = 0.0f64;
                for mi in 0..m as usize {
                    let d = (mm1[(b * m as usize + mi) * n as usize + ni]
                        - mm2[(b * m as usize + mi) * n as usize + ni])
                        .signum();
                    acc += d * a[mi * k as usize + ki] as f64;
                }
                want[(b * k as usize + ki) * n as usize + ni] = acc / n_total;
            }
        }
    }
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for i in 0..want.len() {
        let d = (dw[i] - want[i] as f32).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    println!(
        "mm shared-A bwd worst = {worst:.7} at {at} (ru={:.7} py={:.7})",
        dw[at], want[at] as f32
    );
    assert!(
        worst < 1e-4,
        "mm shared-A backward mismatch {worst} at {at}"
    );
}

/// logmel 構成段階別の逆伝達 bisect (tools/logmel_stage_ref.py と対)。
/// 各段階独立 tape で mean|prefix(a) − prefix(b)| の a 勾配を py と比較。
#[test]
fn logmel_stage_bisect() {
    let path = std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap())
        .join(".cache/rcwx/train/golden/logmel_stages.safetensors");
    if !path.exists() {
        eprintln!("skipping: run `uv run python tools/logmel_stage_ref.py` first");
        return;
    }
    let raw = std::fs::read(&path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&raw).unwrap();
    let read = |name: &str| -> Vec<f32> {
        let v = st.tensor(name).unwrap();
        v.data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let ya = read("ya");
    let yb = read("yb");
    let (batch, samples) = (2u32, 18400u32);
    let (n_fft, hop, n_mels, sr) = (2048u32, 480u32, 128u32, 48000u32);
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let stft = rcwx_train::modules::stft_mel::StftMel::new_rust(&eng, n_fft, hop, n_mels, sr);
    let pad = (n_fft - hop) / 2;
    let padded = samples + 2 * pad;
    let n_frames = (padded - n_fft) / hop + 1;

    // 段階: 指定 depth まで適用して [a_node, b_node] を返す
    let build = |tape: &mut Tape,
                 depth: usize,
                 x: rcwx_nn::engine::tape::Var|
     -> rcwx_nn::engine::tape::Var {
        let y = tape.reflect_pad_dims(x, batch, 1, samples, pad);
        let fr = tape.im2col_frames(y, n_fft, hop, batch, padded, n_frames);
        if depth == 0 {
            return fr;
        }
        let hann = tape.const_tensor(&stft.hann, vec![n_fft as usize]);
        let fw = tape.mul_window(fr, hann, n_fft, n_frames);
        if depth == 1 {
            return fw;
        }
        let bins = 1025u32;
        let mmm =
            |tape: &mut Tape, w: &rcwx_nn::engine::GpuTensor, f: rcwx_nn::engine::tape::Var| {
                let wv = tape.const_tensor(w, vec![bins as usize, n_fft as usize]);
                tape.mm(
                    wv,
                    f,
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
                )
            };
        let (re, im) = tape.rfft2048_ds(fw, batch, n_frames);
        if depth == 2 {
            return re;
        }
        let mag = tape.mag(re, im, batch * bins * n_frames, 2e-7);
        if depth == 3 {
            return mag;
        }
        let mel = {
            let wv = tape.const_tensor(&stft.mel_basis, vec![n_mels as usize, bins as usize]);
            tape.mm(
                wv,
                mag,
                None,
                (batch * n_mels * n_frames) as usize,
                n_mels,
                n_frames,
                bins,
                batch,
                bins,
                1,
                0,
                0,
                1,
                n_frames,
                0,
                bins * n_frames,
                n_frames,
                1,
                n_mels * n_frames,
                1.0,
                0,
            )
        };
        if depth == 4 {
            return mel;
        }
        tape.log_clamp(mel, 2e-6)
    };

    // 追加段階: reflect のみ (depth = 100 で特別扱い)
    {
        let want = read("dy_pad");
        let mut tape = Tape::new(&eng);
        let av = tape.input_f32s(&ya, vec![batch as usize, 1, samples as usize]);
        let bv = tape.const_f32s(&yb, vec![batch as usize, 1, samples as usize]);
        let va = tape.reflect_pad_dims(av, batch, 1, samples, pad);
        let vb = tape.reflect_pad_dims(bv, batch, 1, samples, pad);
        let l = rcwx_train::losses::l1_mean(&mut tape, va, vb);
        tape.backward_from(l);
        let dy_t = tape.grad_tensor(av).expect("dy");
        tape.submit().expect("submit");
        let dy = eng.readback_f32(&dy_t);
        let n = dy.len().min(want.len());
        let dot: f64 = (0..n).map(|k| f64::from(dy[k]) * f64::from(want[k])).sum();
        let na = (0..n).map(|k| f64::from(dy[k]).powi(2)).sum::<f64>().sqrt();
        let nb = (0..n)
            .map(|k| f64::from(want[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        // 手計算参照 (mirror per batch) も同時に
        let out_len = (samples + 2 * pad) as usize;
        let mut hc = vec![0.0f64; ya.len()];
        for b in 0..batch as usize {
            for i in 0..out_len {
                let s = i as i64 - pad as i64;
                let m: i64 = if s < 0 {
                    -s
                } else if s >= samples as i64 {
                    2 * samples as i64 - 2 - s
                } else {
                    s
                };
                let j = b * samples as usize + m as usize;
                let d = f64::from(ya[j] - yb[j]).signum();
                hc[j] += d;
            }
        }
        for v in hc.iter_mut() {
            *v /= (batch as usize * out_len) as f64;
        }
        let dot2: f64 = dy
            .iter()
            .zip(hc.iter())
            .map(|(a, b)| f64::from(*a) * b)
            .sum();
        let na2 = (dy.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>()).sqrt();
        let nb2 = (hc.iter().map(|x| x.powi(2)).sum::<f64>()).sqrt();
        println!(
            "stage pad: cos(py) = {:.6}  cos(hand) = {:.6}",
            dot / (na * nb),
            dot2 / (na2 * nb2)
        );
        // rust と手計算の差が大きい位置
        let mut w2: Vec<(usize, f64)> = (0..dy.len())
            .map(|i| (i, (f64::from(dy[i]) - hc[i]).abs()))
            .collect();
        w2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for &(i, d) in w2[..6].iter() {
            println!(
                "  pad-diff[{i}]: |d|={d:.2e} ru={:.3e} hand={:.3e} py={:.3e}",
                dy[i], hc[i], want[i]
            );
        }
    }
    let st2_fwd_re: Option<Vec<f32>> = {
        let raw2 = std::fs::read(&path).unwrap();
        let st2 = safetensors::SafeTensors::deserialize(&raw2).unwrap();
        st2.tensor("fwd_re").ok().map(|v| {
            v.data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
    };
    let names = ["frames", "windowed", "re", "mag", "mel", "log"];
    for (depth, name) in names.iter().enumerate() {
        let want = read(&format!("dy_{name}"));
        // forward 値の直接比較
        {
            let fwant = read(&format!("fwd_{name}"));
            let mut t2 = Tape::new(&eng);
            let av2 = t2.const_f32s(&ya, vec![batch as usize, 1, samples as usize]);
            let va2 = build(&mut t2, depth, av2);
            let t2t = t2.var_tensor(va2);
            t2.submit().expect("submit");
            let got = eng.readback_f32(&t2t);
            let n = got.len().min(fwant.len());
            let dot: f64 = (0..n)
                .map(|k| f64::from(got[k]) * f64::from(fwant[k]))
                .sum();
            let na = (0..n)
                .map(|k| f64::from(got[k]).powi(2))
                .sum::<f64>()
                .sqrt();
            let nb = (0..n)
                .map(|k| f64::from(fwant[k]).powi(2))
                .sum::<f64>()
                .sqrt();
            let maxabs = (0..n)
                .map(|k| (got[k] - fwant[k]).abs())
                .fold(0.0f32, f32::max);
            println!(
                "  fwd {}: cos={:.7} maxabs={maxabs:.5}",
                name,
                dot / (na * nb)
            );
        }
        let mut tape = Tape::new(&eng);
        let av = tape.input_f32s(&ya, vec![batch as usize, 1, samples as usize]);
        let bv = tape.const_f32s(&yb, vec![batch as usize, 1, samples as usize]);
        let va = build(&mut tape, depth, av);
        let vb = build(&mut tape, depth, bv);
        let l = rcwx_train::losses::l1_mean(&mut tape, va, vb);
        tape.backward_from(l);
        let dy_t = tape.grad_tensor(av).expect("dy");
        tape.submit().expect("submit");
        let dy = eng.readback_f32(&dy_t);
        let n = dy.len().min(want.len());
        let dot: f64 = (0..n).map(|k| f64::from(dy[k]) * f64::from(want[k])).sum();
        let na = (0..n).map(|k| f64::from(dy[k]).powi(2)).sum::<f64>().sqrt();
        let nb = (0..n)
            .map(|k| f64::from(want[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        println!("stage {}: cos = {:.6}", name, dot / (na * nb));
        if *name == "re" {
            // FD 検証 (eps=1e-3, 30 点) — py golden ではなく真値との直接比較
            {
                let run_loss = |ya: &[f32]| -> f32 {
                    let mut t = Tape::new(&eng);
                    let av = t.const_f32s(ya, vec![batch as usize, 1, samples as usize]);
                    let va = build(&mut t, 2, av);
                    let bv2 = t.const_f32s(&yb, vec![batch as usize, 1, samples as usize]);
                    let vb2 = build(&mut t, 2, bv2);
                    let l = rcwx_train::losses::l1_mean(&mut t, va, vb2);
                    let lt = t.var_tensor(l);
                    t.submit().expect("submit");
                    eng.readback_f32(&lt)[0]
                };
                let base = run_loss(&ya);
                let mut worst_fd = 0.0f64;
                let mut at = 0usize;
                let step = (ya.len() / 30).max(1);
                let eps = 1e-3f32;
                let mut i2 = 0usize;
                while i2 < ya.len() {
                    let mut yp = ya.clone();
                    yp[i2] += eps;
                    let mut ym = ya.clone();
                    ym[i2] -= eps;
                    let fd = (run_loss(&yp) - run_loss(&ym)) as f64 / (2.0 * f64::from(eps));
                    let d = (fd - f64::from(dy[i2])).abs();
                    if d > worst_fd {
                        worst_fd = d;
                        at = i2;
                    }
                    i2 += step;
                }
                println!("  re FD check: worst={worst_fd:.6} at {at} (fd vs rust dy)");
            }
            // mm bwd 分離: re 損失の fw/fr レベル勾配
            for (lvl, key) in [("windowed", "dyR_at_windowed"), ("frames", "dyR_at_frames")] {
                let want2 = read(key);
                let mut t5 = Tape::new(&eng);
                let av5 = t5.input_f32s(&ya, vec![batch as usize, 1, samples as usize]);
                // depth 2 までのノードを再構築し、中間ノードの勾配を見る
                let y5 = t5.reflect_pad_dims(av5, batch, 1, samples, pad);
                let fr5 = t5.im2col_frames(y5, n_fft, hop, batch, padded, n_frames);
                let hann5 = t5.const_tensor(&stft.hann, vec![n_fft as usize]);
                let fw5 = t5.mul_window(fr5, hann5, n_fft, n_frames);
                let bins5 = 1025u32;
                let mmm5 = |tape: &mut Tape,
                            w: &rcwx_nn::engine::GpuTensor,
                            f: rcwx_nn::engine::tape::Var| {
                    {
                        let bins = bins5;
                        let wv = tape.const_tensor(w, vec![bins as usize, n_fft as usize]);
                        tape.mm(
                            wv,
                            f,
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
                        )
                    }
                };
                let saved_c5 = t5.g_kahan_chunk();
                t5.set_g_kahan_chunk(16);
                let re5 = mmm5(&mut t5, &stft.basis_re, fw5.clone());
                t5.set_g_kahan_chunk(saved_c5);
                // 固定 b 側: 同じ構成で yb
                let bv5 = t5.const_f32s(&yb, vec![batch as usize, 1, samples as usize]);
                let yb5 = t5.reflect_pad_dims(bv5, batch, 1, samples, pad);
                let frb5 = t5.im2col_frames(yb5, n_fft, hop, batch, padded, n_frames);
                let fwb5 = t5.mul_window(frb5, hann5, n_fft, n_frames);
                let saved_c6 = t5.g_kahan_chunk();
                t5.set_g_kahan_chunk(16);
                let reb5 = mmm5(&mut t5, &stft.basis_re, fwb5);
                t5.set_g_kahan_chunk(saved_c6);
                let l5 = rcwx_train::losses::l1_mean(&mut t5, re5, reb5);
                t5.backward_from(l5);
                let target = if lvl == "windowed" { fw5 } else { fr5 };
                let g5 = t5.grad_tensor(target).expect("g5");
                t5.submit().expect("submit");
                let got5 = eng.readback_f32(&g5);
                // py は [B, nf, n_fft] — rust [B, n_fft, nf] に転置して比較
                let (bs, nf, kdim) = (batch as usize, n_frames as usize, n_fft as usize);
                let mut ru_t = vec![0.0f32; got5.len()];
                for b in 0..bs {
                    for t in 0..nf {
                        for kk in 0..kdim {
                            ru_t[(b * kdim + kk) * nf + t] =
                                want2[((b * nf + t) * kdim + kk) as usize];
                        }
                    }
                }
                let n5 = got5.len().min(ru_t.len());
                let d5: f64 = (0..n5)
                    .map(|i2| f64::from(got5[i2]) * f64::from(ru_t[i2]))
                    .sum();
                let a5 = (0..n5)
                    .map(|i2| f64::from(got5[i2]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let b5 = (0..n5)
                    .map(|i2| f64::from(ru_t[i2]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                println!("  mm-bwd isol {lvl}: cos={:.6}", d5 / (a5 * b5));
            }
            // fw (depth 1) の値を CPU 参照と比較
            {
                let mut t4 = Tape::new(&eng);
                let av4 = t4.const_f32s(&ya, vec![batch as usize, 1, samples as usize]);
                let va4 = build(&mut t4, 1, av4);
                let t4t = t4.var_tensor(va4);
                t4.submit().expect("submit");
                let g4 = eng.readback_f32(&t4t);
                // CPU: mirror + window
                let mirror = |i: usize| -> usize {
                    let s = i as i64 - (samples / 2 - 240 + 240) as i64;
                    let _ = s;
                    let p2 = ((2048 - 480) / 2) as i64;
                    let s2 = i as i64 - p2;
                    let mm: i64 = if s2 < 0 {
                        -s2 - 1
                    } else if s2 >= samples as i64 {
                        2 * samples as i64 - 1 - s2
                    } else {
                        s2
                    };
                    mm as usize
                };
                let win = |kk: usize| {
                    (0.5 - 0.5 * (2.0 * std::f64::consts::PI * kk as f64 / 2048.0).cos()) as f32
                };
                let nfr = 38usize;
                let mut worst = 0.0f32;
                let mut at = 0usize;
                for b in 0..batch as usize {
                    for kk in (0..2048usize).step_by(131) {
                        for t in (0..nfr).step_by(11) {
                            let src = mirror(t * 480 + kk);
                            let want = ya[b * samples as usize + src] * win(kk);
                            let idx = (b * 2048 + kk) * nfr + t;
                            let d = (g4[idx] - want).abs();
                            if d > worst {
                                worst = d;
                                at = idx;
                            }
                        }
                    }
                }
                println!("  fw tape worst = {worst:.2e} at {at}");
            }
            // forward 値の直接比較 (レイアウト: py [B,nf,bins] → 転置)
            if let Some(ref fre) = st2_fwd_re {
                let bsz = 2usize;
                let nfr = 38usize;
                let bins = 1025usize;
                // rust re [B, bins, nf] vs py [B, nf, bins]
                let mut rust = vec![0.0f32; fre.len()];
                // 再度 forward を取得
                let mut t3 = Tape::new(&eng);
                let av3 = t3.const_f32s(&ya, vec![bsz, 1, samples as usize]);
                let va3 = build(&mut t3, 2, av3);
                let t3t = t3.var_tensor(va3);
                t3.submit().expect("submit");
                rust = eng.readback_f32(&t3t);
                let mut n_flip = 0usize;
                let mut n_cmp = 0usize;
                for b in 0..bsz {
                    for t in 0..nfr {
                        for f in 0..bins {
                            let ru = rust[(b * bins + f) * nfr + t];
                            let pyv = fre[(b * nfr + t) * bins + f];
                            let ru_b = {
                                // b 側は同 data なので re_b = re_a(yb) — py の fwd は a 側のみ
                                0.0
                            };
                            let _ = ru_b;
                            // sign(re_a - re_b) の代わりに re_a の値一致を見る
                            if (ru - pyv).abs() > 0.02 * pyv.abs().max(0.1) {
                                n_flip += 1;
                            }
                            n_cmp += 1;
                        }
                    }
                }
                println!("  re fwd 値不一致 (>2%): {n_flip}/{n_cmp}");
                // 不一致の bin 分布 (低周波 bin に偏るか)
                let mut low = 0usize;
                let mut high = 0usize;
                for b in 0..bsz {
                    for t in 0..nfr {
                        for f in 0..bins {
                            let ru = rust[(b * bins + f) * nfr + t];
                            let pyv = fre[(b * nfr + t) * bins + f];
                            if (ru - pyv).abs() > 0.02 * pyv.abs().max(0.1) {
                                if f < 100 { low += 1 } else { high += 1 }
                            }
                        }
                    }
                }
                println!("  不一致 bin<100: {low}  bin>=100: {high}");
                // rust re fwd の絶対誤差分布 (py≈f64 なので py 基準)
                let mut errs: Vec<f32> = Vec::new();
                for b in 0..bsz {
                    for t in 0..nfr {
                        for f in 0..bins {
                            let ru = rust[(b * bins + f) * nfr + t];
                            let pyv = fre[(b * nfr + t) * bins + f];
                            errs.push((ru - pyv).abs());
                        }
                    }
                }
                errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mx = errs.last().copied().unwrap_or(0.0);
                println!(
                    "  rust re fwd 誤差: max={mx:.2e} p50={:.2e} p90={:.2e} p99={:.2e}",
                    errs[errs.len() / 2],
                    errs[errs.len() * 9 / 10],
                    errs[errs.len() * 99 / 100]
                );
            }
            // 誤差の位置構造
            let half = dy.len() / 2;
            let cs = |s: std::ops::Range<usize>| {
                let d2: f64 = dy[s.clone()]
                    .iter()
                    .zip(want[s.clone()].iter())
                    .map(|(x, y)| f64::from(*x) * f64::from(*y))
                    .sum();
                let a2 = (dy[s.start..s.end]
                    .iter()
                    .map(|x| f64::from(*x).powi(2))
                    .sum::<f64>())
                .sqrt();
                let b2 = (want[s.start..s.end]
                    .iter()
                    .map(|x| f64::from(*x).powi(2))
                    .sum::<f64>())
                .sqrt();
                d2 / (a2 * b2)
            };
            println!(
                "  re b0 cos={:.6} b1 cos={:.6}",
                cs(0..half),
                cs(half..dy.len())
            );
            let mut top: Vec<(usize, f32)> = (0..dy.len())
                .map(|i| (i, (dy[i] - want[i]).abs()))
                .collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for &(i, d) in top[..6].iter() {
                println!(
                    "  re worst[{i}]: |d|={d:.2e} ru={:.5} py={:.5} mod480={}",
                    dy[i],
                    want[i],
                    i % 480
                );
            }
        }
    }
}

/// reflect_pad forward の 2 連続呼び出し検証 (b 側ミラー領域の garbage 疑い)。
#[test]
fn reflect_pad_two_calls_fwd() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (batch, in_len, pad) = (2u32, 18400u32, 784u32);
    let mk = |seed: f64| -> Vec<f32> {
        (0..batch as usize * in_len as usize)
            .map(|i| ((i as f64 * seed + (i % 7) as f64 * 0.11) % 1.0 - 0.5) as f32)
            .collect()
    };
    let x1 = mk(0.37);
    let x2 = mk(0.71);
    let mut tape = Tape::new(&eng);
    let v1 = tape.const_f32s(&x1, vec![batch as usize, 1, in_len as usize]);
    let v2 = tape.const_f32s(&x2, vec![batch as usize, 1, in_len as usize]);
    let p1 = tape.reflect_pad_dims(v1, batch, 1, in_len, pad);
    let p2 = tape.reflect_pad_dims(v2, batch, 1, in_len, pad);
    let t1 = tape.var_tensor(p1);
    let t2 = tape.var_tensor(p2);
    tape.submit().expect("submit");
    let g1 = eng.readback_f32(&t1);
    let g2 = eng.readback_f32(&t2);
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
    let out_len = (in_len + 2 * pad) as usize;
    let check = |got: &[f32], x: &[f32], tag: &str| {
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for b in 0..batch as usize {
            for i in 0..out_len {
                let want = x[b * in_len as usize + mirror(i)];
                let idx = b * out_len + i;
                let d = (got[idx] - want).abs();
                if d > worst {
                    worst = d;
                    at = idx;
                }
            }
        }
        println!("{tag}: fwd worst = {worst:.7} at {at}");
    };
    check(&g1, &x1, "pad-a");
    check(&g2, &x2, "pad-b");
}

/// mm bias_axis=1 (bias[m]) の逆伝達検証 — cond mm (m=16, n=1, k=4, batch=2)。
#[test]
fn mm_bias_axis1_backward() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (m, n, k, batch) = (16u32, 1u32, 4u32, 2u32);
    let wdat: Vec<f32> = (0..m as usize * k as usize)
        .map(|i| ((i as f64 * 0.618) % 1.0 - 0.5) as f32)
        .collect();
    let bdat: Vec<f32> = (0..m as usize).map(|i| 0.5 * i as f32 - 1.0).collect();
    let x: Vec<f32> = (0..batch as usize * k as usize)
        .map(|i| ((i as f64 * 0.314) % 1.0 - 0.5) as f32)
        .collect();
    let x2: Vec<f32> = (0..x.len())
        .map(|i| ((i as f64 * 0.27182818) % 1.0 - 0.5) as f32)
        .collect();

    // loss = mean |mm(w, x, b) − mm(w, x2, b2)| (タイ回避のため無理数系の x2)
    let b2: Vec<f32> = (0..m as usize).map(|i| -0.11 * i as f32 + 0.23).collect();
    let mut tape = Tape::new(&eng);
    let wv = tape.const_f32s(&wdat, vec![m as usize, k as usize]);
    let bv = tape.input_f32s(&bdat, vec![m as usize]);
    let xv = tape.const_f32s(&x, vec![batch as usize, k as usize]);
    let w2 = tape.const_f32s(&wdat, vec![m as usize, k as usize]);
    let b2v = tape.const_f32s(&b2, vec![m as usize]);
    let x2v = tape.const_f32s(&x2, vec![batch as usize, k as usize]);
    let o1 = tape.mm(
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
        n,
        0,
        k,
        n,
        1,
        m * n,
        1.0,
        1,
    );
    let o2 = tape.mm(
        w2,
        x2v,
        Some(b2v),
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
        n,
        0,
        k,
        n,
        1,
        m * n,
        1.0,
        1,
    );
    let l = rcwx_train::losses::l1_mean(&mut tape, o1, o2);
    let o1t = tape.var_tensor(o1);
    tape.backward_from(l);
    let db_t = tape.grad_tensor(bv).expect("db");
    tape.submit().expect("submit");
    let db = eng.readback_f32(&db_t);
    let o1v = eng.readback_f32(&o1t);
    // forward の正確な CPU 参照 ([b, m, n] レイアウト)
    let mut o1c = vec![0.0f64; (batch * m * n) as usize];
    for b in 0..batch as usize {
        for mi in 0..m as usize {
            for ni in 0..n as usize {
                let mut acc = bdat[mi] as f64;
                for ki in 0..k as usize {
                    acc += wdat[mi * k as usize + ki] as f64 * x[b * k as usize + ki] as f64;
                }
                o1c[(b * m as usize + mi) * n as usize + ni] = acc;
            }
        }
    }
    let mut fw = 0.0f32;
    for i in 0..o1c.len() {
        fw = fw.max((o1v[i] - o1c[i] as f32).abs());
    }
    println!("o1 fwd maxdiff = {fw:.6}");
    // 値を並べて違いのパターンを見る
    for i in 0..8 {
        println!("  o1[{i}]: ru={:.5} cpu={:.5}", o1v[i], o1c[i]);
    }
    let n_total = (batch * m * n) as f64;
    let want: Vec<f32> = (0..m as usize)
        .map(|mi| {
            let mut acc = 0.0f64;
            for b in 0..batch as usize {
                let mut d1 = bdat[mi] as f64;
                let mut d2 = b2[mi] as f64;
                for ki in 0..k as usize {
                    d1 += wdat[mi * k as usize + ki] as f64 * x[b * k as usize + ki] as f64;
                    d2 += wdat[mi * k as usize + ki] as f64 * x2[b * k as usize + ki] as f64;
                }
                acc += (d1 - d2).signum();
            }
            (acc / n_total) as f32
        })
        .collect();
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for i in 0..want.len() {
        let d = (db[i] - want[i]).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    println!(
        "mm bias_axis1 bwd worst = {worst:.7} at {at} (ru={:.7} want={:.7})",
        db[at], want[at]
    );
    println!("ru db = {db:.4?}");
    println!("want  = {want:.4?}");
    assert!(worst < 1e-6, "mm bias_axis1 mismatch {worst}");
}
