//! conv2d fwd / dX / dW を CPU 手計算参照 (zero-pad) と突き合わせる。

use rcwx_nn::engine::tape::Tape;
use rcwx_nn::engine::{Backend, Engine};

#[test]
fn conv2d_probe_discp_geom() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    // disc_p と同一形状: kh=5 kw=1 sh=3 sw=1 ph=2 pw=0, in=2 out=8, batch=2, w=2
    let (c_out, c_in, kh, kw, sh, sw, ph, pw, h, w) =
        (8u32, 2u32, 5u32, 1u32, 3u32, 1u32, 2u32, 0u32, 100u32, 2u32);
    let batch = 2u32;
    let oh = (h + 2 * ph - kh) / sh + 1;
    let ow = w;
    let x0: Vec<f32> = (0..batch as usize * c_in as usize * h as usize * w as usize)
        .map(|i| ((i as f32 * 0.37) % 1.0) - 0.5)
        .collect();
    let w0: Vec<f32> = (0..c_out as usize * c_in as usize * kh as usize * kw as usize)
        .map(|i| ((i as f32 * 0.61) % 1.0) - 0.5)
        .collect();
    let b0: Vec<f32> = vec![0.0; c_out as usize];

    let mut tape = Tape::new(&eng);
    let xv = tape.input_f32s(&x0, vec![x0.len()]);
    let wv = tape.input_f32s(&w0, vec![w0.len()]);
    let bv = tape.input_f32s(&b0, vec![b0.len()]);
    let out = tape.conv2d(
        xv.clone(),
        wv.clone(),
        Some(bv.clone()),
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
    let ot = tape.var_tensor(out);
    tape.submit().expect("submit");
    let got = eng.readback_f32(&ot);

    // CPU 参照
    let mut ref_out = vec![0.0f32; (batch * c_out * oh * ow) as usize];
    for b in 0..batch as usize {
        for o in 0..c_out as usize {
            for oy in 0..oh as usize {
                for ox in 0..ow as usize {
                    let mut acc = b0[o];
                    for c in 0..c_in as usize {
                        for j in 0..kh as usize {
                            for i in 0..kw as usize {
                                let wy = oy * sh as usize + j;
                                let wx = ox + i;
                                if wy < ph as usize + h as usize && wy >= ph as usize {
                                    let wv = w0[((o * c_in as usize + c) * kh as usize + j)
                                        * kw as usize
                                        + i];
                                    let xv = x0[((b * c_in as usize + c) * h as usize + wy
                                        - ph as usize)
                                        * w as usize
                                        + wx];
                                    acc += wv * xv;
                                }
                            }
                        }
                    }
                    ref_out
                        [(b * c_out as usize + o) * (oh * ow) as usize + oy * ow as usize + ox] =
                        acc;
                }
            }
        }
    }
    let worst = got
        .iter()
        .zip(ref_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("conv2d(disc_p geom) fwd maxdiff = {worst:.6}");
    assert!(worst < 1e-4, "conv2d disc_p geom mismatch {worst}");
}

#[test]
fn conv2d_probe_discp_scale() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    // fmap0 と同一: in=1 out=32 h=9200 w=2 kh=5 sh=3 ph=2 batch=2
    let (c_out, c_in, kh, kw, sh, sw, ph, pw, h, w) = (
        32u32, 1u32, 5u32, 1u32, 3u32, 1u32, 2u32, 0u32, 9200u32, 2u32,
    );
    let batch = 2u32;
    let oh = (h + 2 * ph - kh) / sh + 1;
    let x0: Vec<f32> = (0..batch as usize * c_in as usize * h as usize * w as usize)
        .map(|i| ((i as f32 * 0.3700001) % 1.0) - 0.5)
        .collect();
    let w0: Vec<f32> = (0..c_out as usize * c_in as usize * kh as usize * kw as usize)
        .map(|i| ((i as f32 * 0.6100001) % 1.0) - 0.5)
        .collect();
    let b0: Vec<f32> = vec![0.0; c_out as usize];
    let mut tape = Tape::new(&eng);
    let xv = tape.input_f32s(&x0, vec![x0.len()]);
    let wv = tape.const_f32s(&w0, vec![w0.len()]);
    let bv = tape.const_f32s(&b0, vec![b0.len()]);
    let out = tape.conv2d(
        xv.clone(),
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
    let ot = tape.var_tensor(out);
    tape.backward_from(out);
    let dx_t = tape.grad_tensor(xv).expect("dx");
    tape.submit().expect("submit");
    let got = eng.readback_f32(&ot);
    let dx = eng.readback_f32(&dx_t);
    // dX のスポット FD (2 点)
    let eps = 1e-3f32;
    let fwd_loss = |xa: &[f32]| -> f32 {
        let mut tp = Tape::new(&eng);
        let xvv = tp.const_f32s(xa, vec![xa.len()]);
        let wvv = tp.const_f32s(&w0, vec![w0.len()]);
        let bvv = tp.const_f32s(&b0, vec![b0.len()]);
        let o = tp.conv2d(
            xvv,
            wvv,
            Some(bvv),
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
        let s = tp.sum_all(o);
        let st = tp.var_tensor(s);
        tp.submit().expect("submit");
        eng.readback_f32(&st)[0]
    };
    for xi in [0usize, 5000, 18399] {
        let mut xp = x0.clone();
        xp[xi] += eps;
        let fp = fwd_loss(&xp);
        xp[xi] -= 2.0 * eps;
        let fm = fwd_loss(&xp);
        let fd = (fp - fm) / (2.0 * eps);
        println!("  dX[{xi}]: fd={fd:.5} analytic={:.5}", dx[xi]);
    }
    // CPU 参照 (rust 側で stride 込みの直接計算 — 一部サンプルのみ)
    let flat = |b: usize, c: usize, y: usize, x: usize| -> f32 {
        x0[((b * c_in as usize + c) * h as usize + y) * w as usize + x]
    };
    let wgt = |o: usize, c: usize, j: usize| -> f32 {
        w0[((o * c_in as usize + c) * kh as usize + j) * kw as usize]
    };
    let mut worst = 0.0f32;
    for (b, o, oy, ox) in [
        (0usize, 0usize, 0usize, 0usize),
        (0, 5, 100, 1),
        (1, 17, 2000, 0),
        (1, 31, 3066, 1),
    ] {
        let mut acc = 0.0;
        for j in 0..kh as usize {
            let wy = oy * sh as usize + j;
            if wy >= ph as usize && wy < ph as usize + h as usize {
                acc += wgt(o, 0, j) * flat(b, 0, wy - ph as usize, ox);
            }
        }
        let idx = (b * c_out as usize + o) * (oh as usize * w as usize) + oy * w as usize + ox;
        println!(
            "  spot (b={b},o={o},oy={oy},ox={ox}): rust={:.6} cpu={:.6}",
            got[idx], acc
        );
        worst = worst.max((got[idx] - acc).abs());
    }
    println!("conv2d fmap0-scale spot maxdiff = {worst:.6} (oh={oh})");
    assert!(worst < 1e-4, "fmap0 scale mismatch {worst}");
}

#[test]
fn conv2d_probe_ref() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (c_out, c_in, kh, kw, sh, sw, ph, pw, h, w) =
        (3u32, 2u32, 3u32, 1u32, 2u32, 1u32, 1u32, 0u32, 6u32, 3u32);
    let batch = 1u32;
    let oh = (h + 2 * ph).saturating_sub(kh) / sh + 1;
    let ow = (w + 2 * pw).saturating_sub(kw) / sw + 1;
    let x0: Vec<f32> = (0..batch as usize * c_in as usize * h as usize * w as usize)
        .map(|i| ((i as f32 * 0.37) % 1.0) - 0.5)
        .collect();
    let w0: Vec<f32> = (0..c_out as usize * c_in as usize * kh as usize * kw as usize)
        .map(|i| ((i as f32 * 0.61) % 1.0) - 0.5)
        .collect();

    let b0: Vec<f32> = (0..c_out as usize).map(|i| 0.1 * i as f32 - 0.1).collect();
    let fwd = |xa: &[f32], wa: &[f32], ba: &[f32]| -> Vec<f32> {
        let mut tape = Tape::new(&eng);
        let xv = tape.const_f32s(xa, vec![xa.len()]);
        let wv = tape.const_f32s(wa, vec![wa.len()]);
        let bv = tape.const_f32s(ba, vec![ba.len()]);
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
        let ot = tape.var_tensor(out);
        tape.submit().expect("submit");
        eng.readback_f32(&ot)
    };
    let got = fwd(&x0, &w0, &b0);

    // zero-pad した入力 (h×w → hp×w)
    let hp = h as usize + 2 * ph as usize;
    let mut pad_in = vec![0.0f32; c_in as usize * hp * w as usize];
    for c in 0..c_in as usize {
        for y in 0..h as usize {
            for x in 0..w as usize {
                pad_in[(c * hp + y + ph as usize) * w as usize + x] =
                    x0[(c * h as usize + y) * w as usize + x];
            }
        }
    }
    let mut ref_out = vec![0.0f32; (batch * c_out * oh * ow) as usize];
    let mut ref_dx = vec![0.0f32; x0.len()];
    let mut ref_dw = vec![0.0f32; w0.len()];
    for o in 0..c_out as usize {
        for oy in 0..oh as usize {
            for ox in 0..ow as usize {
                let mut acc = 0.0;
                for c in 0..c_in as usize {
                    for j in 0..kh as usize {
                        for i in 0..kw as usize {
                            let wy = oy * sh as usize + j;
                            let wx = ox * sw as usize + i;
                            let wv =
                                w0[((o * c_in as usize + c) * kh as usize + j) * kw as usize + i];
                            let xv = pad_in[(c * hp + wy) * w as usize + wx];
                            acc += xv * wv;
                            let _ = b0;
                            // dX (dy=1): pad 前の座標に落とす
                            if wy >= ph as usize && wy < h as usize + ph as usize {
                                ref_dx[(c * h as usize + wy - ph as usize) * w as usize + wx] += wv;
                            }
                            ref_dw
                                [((o * c_in as usize + c) * kh as usize + j) * kw as usize + i] +=
                                xv;
                        }
                    }
                }
                ref_out[o * (oh * ow) as usize + oy * ow as usize + ox] = acc + b0[o];
            }
        }
    }
    let worst = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let fwd_w = worst(&got, &ref_out);

    let mut tape = Tape::new(&eng);
    let xv = tape.input_f32s(&x0, vec![x0.len()]);
    let wv = tape.input_f32s(&w0, vec![w0.len()]);
    let bv = tape.input_f32s(&b0, vec![b0.len()]);
    let out = tape.conv2d(
        xv.clone(),
        wv.clone(),
        Some(bv.clone()),
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
    let db_t = tape.grad_tensor(bv).expect("db");
    tape.submit().expect("submit");
    let dx = eng.readback_f32(&dx_t);
    let dw = eng.readback_f32(&dw_t);
    let db = eng.readback_f32(&db_t);
    // db 参照: Σ over (b,) oy, ox of dy=1 → oh·ow per channel
    let ref_db: Vec<f32> = (0..c_out as usize).map(|_| (oh * ow) as f32).collect();
    println!("db = {:?} (want {:?})", db, ref_db);

    println!(
        "fwd maxdiff={fwd_w:.6} dX maxdiff={} dW maxdiff={}",
        worst(&dx, &ref_dx),
        worst(&dw, &ref_dw)
    );
    assert!(fwd_w < 1e-4, "fwd {fwd_w}");
    assert!(worst(&dx, &ref_dx) < 1e-4, "dX");
    assert!(worst(&dw, &ref_dw) < 1e-4, "dW");
}
