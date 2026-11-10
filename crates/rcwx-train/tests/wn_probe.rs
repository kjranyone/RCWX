//! adopt_wn + conv1d_grouped の FD gradcheck (fold 逆伝達の値検証)。
//! g=1 は回帰テスト (合格)。g=2 は既知バグ (grouped dW レイアウト崩れ —
//! AGENTS.md 学習セクション参照) のため ignore の再現ケースとして残す。

use rcwx_nn::engine::tape::{ParamStore, Tape};
use rcwx_nn::engine::{Backend, Engine};

#[test]
fn wn_grad_probe_g1() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    // WN: [out=4, in*k=6] (in=2, k=3, g=2 → cin_g=1? conv1d_grouped g=2, cin=2, cout=4)
    let v_d: Vec<f32> = (0..24)
        .map(|i| ((i * 7) % 13) as f32 / 13.0 - 0.5)
        .collect();
    let g_d: Vec<f32> = vec![1.1, 0.9, 1.3, 0.7];

    let x_d: Vec<f32> = (0..2 * 2 * 9).map(|i| 0.01 * i as f32 - 0.09).collect();
    let b_d: Vec<f32> = vec![0.1, -0.1, 0.2, -0.2];

    let loss = |v: &[f32], g: &[f32]| -> f32 {
        let mut store = ParamStore::new();
        store.add_weight_norm(&eng, "tst", v, g, vec![4, 6]);
        let mut tape = Tape::new(&eng);
        let vv = tape.adopt_param(&store, store.wns[0].v_idx);
        let gv = tape.adopt_param(&store, store.wns[0].g_idx);
        let w = tape.adopt_wn(&store, 0, vv, gv);
        let x = tape.const_f32s(&x_d, vec![x_d.len()]);
        let bv = tape.const_f32s(&b_d, vec![4]);
        let h = tape.conv1d_grouped(x, w, Some(bv), 4, 2, 3, 1, 0, 1, 2, 9, 7);
        let s = tape.sum_all(h);
        let st = tape.var_tensor(s);
        tape.submit().expect("submit");
        eng.readback_f32(&st)[0]
    };

    // 解析
    let mut store = ParamStore::new();
    store.add_weight_norm(&eng, "tst", &v_d, &g_d, vec![4, 6]);
    let mut tape = Tape::new(&eng);
    let vv = tape.adopt_param(&store, store.wns[0].v_idx);
    let gv = tape.adopt_param(&store, store.wns[0].g_idx);
    let w = tape.adopt_wn(&store, 0, vv, gv);
    let x = tape.const_f32s(&x_d, vec![x_d.len()]);
    let bv = tape.const_f32s(&b_d, vec![4]);
    let h = tape.conv1d_grouped(x, w, Some(bv), 4, 2, 3, 1, 0, 1, 2, 9, 7);
    tape.backward_from(h);
    let dv_t = tape.grad_tensor(vv).expect("dv");
    let dg_t = tape.grad_tensor(gv).expect("dg");
    let dw_t = tape.grad_tensor(w).expect("dw");
    tape.submit().expect("submit");
    let dwv = eng.readback_f32(&dw_t);
    println!("  dw(folded) = {:?}", &dwv[..6.min(dwv.len())]);
    let dv = eng.readback_f32(&dv_t);
    let dg = eng.readback_f32(&dg_t);

    let eps = 1e-3f32;
    let mut worst_v = 0.0f32;
    for i in [0usize, 5, 12, 23] {
        let mut vp = v_d.clone();
        vp[i] += eps;
        let fp = loss(&vp, &g_d);
        vp[i] -= 2.0 * eps;
        let fm = loss(&vp, &g_d);
        let fd = (fp - fm) / (2.0 * eps);
        let d = (fd - dv[i]).abs() / fd.abs().max(1e-3);
        println!("  dv[{i}]: fd={fd:.5} analytic={:.5}", dv[i]);
        worst_v = worst_v.max(d);
    }
    let mut worst_g = 0.0f32;
    for i in 0..4 {
        let mut gp = g_d.clone();
        gp[i] += eps;
        let fp = loss(&v_d, &gp);
        gp[i] -= 2.0 * eps;
        let fm = loss(&v_d, &gp);
        let fd = (fp - fm) / (2.0 * eps);
        let d = (fd - dg[i]).abs() / fd.abs().max(1e-3);
        println!("  dg[{i}]: fd={fd:.5} analytic={:.5}", dg[i]);
        worst_g = worst_g.max(d);
    }
    println!("WN FD: dv worst={worst_v:.5} dg worst={worst_g:.5}");
    assert!(worst_v < 2e-2, "WN dv FD mismatch");
    assert!(worst_g < 2e-2, "WN dg FD mismatch");
}

/// WN fold × grouped conv (g≥2, batch≥2) の FD 回帰。
/// 旧バグ: t_im2col が cin_g を入力バッチ幅に使う + fwd matmul の col
/// バッチストライドが 0 (b≥1 が b0 の col を再利用)。
#[test]
fn wn_grad_probe_g2() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let v_d: Vec<f32> = (0..24)
        .map(|i| ((i * 7) % 13) as f32 / 13.0 - 0.5)
        .collect();
    let g_d: Vec<f32> = vec![1.1, 0.9, 1.3, 0.7];
    let x_d: Vec<f32> = (0..2 * 4 * 9).map(|i| 0.01 * i as f32 - 0.09).collect();
    let b_d: Vec<f32> = vec![0.1, -0.1, 0.2, -0.2];
    let loss = |v: &[f32], g: &[f32]| -> f32 {
        let mut store = ParamStore::new();
        store.add_weight_norm(&eng, "tst", v, g, vec![4, 6]);
        let mut tape = Tape::new(&eng);
        let vv = tape.adopt_param(&store, store.wns[0].v_idx);
        let gv = tape.adopt_param(&store, store.wns[0].g_idx);
        let w = tape.adopt_wn(&store, 0, vv, gv);
        let x = tape.const_f32s(&x_d, vec![x_d.len()]);
        let bv = tape.const_f32s(&b_d, vec![4]);
        let h = tape.conv1d_grouped(x, w, Some(bv), 4, 4, 3, 1, 0, 2, 2, 9, 7);
        let s = tape.sum_all(h);
        let st = tape.var_tensor(s);
        tape.submit().expect("submit");
        eng.readback_f32(&st)[0]
    };
    let mut store = ParamStore::new();
    store.add_weight_norm(&eng, "tst", &v_d, &g_d, vec![4, 6]);
    let mut tape = Tape::new(&eng);
    let vv = tape.adopt_param(&store, store.wns[0].v_idx);
    let gv = tape.adopt_param(&store, store.wns[0].g_idx);
    let w = tape.adopt_wn(&store, 0, vv, gv);
    let x = tape.const_f32s(&x_d, vec![x_d.len()]);
    let bv = tape.const_f32s(&b_d, vec![4]);
    let h = tape.conv1d_grouped(x, w, Some(bv), 4, 4, 3, 1, 0, 2, 2, 9, 7);
    tape.backward_from(h);
    let dv_t = tape.grad_tensor(vv).expect("dv");
    tape.submit().expect("submit");
    let dv = eng.readback_f32(&dv_t);
    let eps = 1e-3f32;
    let mut worst = 0.0f32;
    for i in [0usize, 5, 12, 23] {
        let mut vp = v_d.clone();
        vp[i] += eps;
        let fp = loss(&vp, &g_d);
        vp[i] -= 2.0 * eps;
        let fm = loss(&vp, &g_d);
        let fd = (fp - fm) / (2.0 * eps);
        worst = worst.max((fd - dv[i]).abs() / fd.abs().max(1e-3));
    }
    assert!(worst < 2e-2, "WN×grouped dv FD mismatch {worst:.5}");
}

/// grouped conv (g=2, batch=2) の fwd (h) / dW / dX を CPU 参照と全要素比較。
/// バッチストライド系回帰の全数検証。
#[test]
fn wn_g2_full_reference() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let v_d: Vec<f32> = (0..24)
        .map(|i| ((i * 7) % 13) as f32 / 13.0 - 0.5)
        .collect();
    let g_d: Vec<f32> = vec![1.1, 0.9, 1.3, 0.7];
    let x_d: Vec<f32> = (0..2 * 4 * 9).map(|i| 0.01 * i as f32 - 0.09).collect();
    let b_d: Vec<f32> = vec![0.1, -0.1, 0.2, -0.2];
    let (c_out, c_in, k, s, g, t, olen) = (4u32, 4u32, 3u32, 1u32, 2u32, 9u32, 7u32);
    let batch = 2u32;
    let (cin_g, cout_g) = (c_in / g, c_out / g);

    let mut store = ParamStore::new();
    store.add_weight_norm(&eng, "tst", &v_d, &g_d, vec![4, 6]);
    let mut tape = Tape::new(&eng);
    let vv = tape.adopt_param(&store, store.wns[0].v_idx);
    let gv = tape.adopt_param(&store, store.wns[0].g_idx);
    let w = tape.adopt_wn(&store, 0, vv, gv);
    let x = tape.input_f32s(&x_d, vec![x_d.len()]);
    let bv = tape.const_f32s(&b_d, vec![4]);
    let h = tape.conv1d_grouped(x, w, Some(bv), c_out, c_in, k, s, 0, g, batch, t, olen);
    // loss = Σ h → dw[oc, (c_loc, j)] = Σ_b Σ_tt x[b, gg·cin_g + c_loc, tt + j]
    // forward 比較: h を先に readback (backward 前に var_tensor 確保)
    let h_t = tape.var_tensor(h);
    tape.backward_from(h);
    let dw_t = tape.grad_tensor(w).expect("dw");
    let dx_t = tape.grad_tensor(x).expect("dx");
    tape.submit().expect("submit");
    let dw = eng.readback_f32(&dw_t);
    let dx = eng.readback_f32(&dx_t);
    let h_got = eng.readback_f32(&h_t);
    // CPU 参照 forward
    let fold = |oc: usize| -> Vec<f32> {
        let row = &v_d[oc * 6..oc * 6 + 6];
        let n = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        row.iter().map(|x| x * g_d[oc] / n).collect::<Vec<f32>>()
    };
    let mut ref_h = vec![0.0f32; (batch * c_out * olen) as usize];
    for b in 0..batch {
        for gg in 0..g {
            for oc in 0..cout_g {
                let wf = fold((gg * cout_g + oc) as usize);
                for tt in 0..olen {
                    let mut acc = 0.0f32;
                    for c_loc in 0..cin_g {
                        for j in 0..k {
                            acc += wf[(c_loc * k + j) as usize]
                                * x_d[((b * c_in + gg * cin_g + c_loc) * t + tt + j) as usize];
                        }
                    }
                    ref_h[((b * c_out + gg * cout_g + oc) * olen + tt) as usize] =
                        acc + b_d[(gg * cout_g + oc) as usize];
                }
            }
        }
    }
    let hw = h_got
        .iter()
        .zip(ref_h.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("h fwd maxdiff = {hw:.6}");

    // CPU 参照 (dy=1)
    let mut ref_dw = vec![0.0f32; 24];
    for gg in 0..g {
        for oc in 0..cout_g {
            for c_loc in 0..cin_g {
                for j in 0..k {
                    let mut acc = 0.0;
                    for b in 0..batch {
                        for tt in 0..olen {
                            acc += x_d[((b * c_in + gg * cin_g + c_loc) * t + tt + j) as usize];
                        }
                    }
                    ref_dw[(((gg * cout_g + oc) * cin_g + c_loc) * k + j) as usize] = acc;
                }
            }
        }
    }
    let mut ref_dx = vec![0.0f32; x_d.len()];
    for b in 0..batch {
        for c in 0..c_in {
            for i in 0..t {
                let gg = c / cin_g;
                let c_loc = c % cin_g;
                let oc0 = gg * cout_g;
                let mut acc = 0.0;
                for oc in 0..cout_g {
                    for j in 0..k {
                        let tt = i as i32 - j as i32;
                        if tt >= 0 && (tt as u32) < olen {
                            acc += v_d_numerator(
                                (oc0 + oc) as usize,
                                c_loc as usize,
                                j as usize,
                                &v_d,
                                &g_d,
                            );
                        }
                    }
                }
                ref_dx[((b * c_in + c) * t + i) as usize] = acc;
            }
        }
    }
    let w = dw
        .iter()
        .zip(ref_dw.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let xw = dx
        .iter()
        .zip(ref_dx.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("dw maxdiff={w:.6} dx maxdiff={xw:.6}");
    assert!(hw < 1e-4, "grouped fwd maxdiff {hw}");
    assert!(w < 1e-4, "grouped dW maxdiff {w}");
    assert!(xw < 1e-4, "grouped dX maxdiff {xw}");
}

fn v_d_numerator(oc: usize, c_loc: usize, j: usize, v_d: &[f32], g_d: &[f32]) -> f32 {
    // w[oc, c_loc·k + j] = v[oc·6 + c_loc·3 + j] · g[oc] / ||v_row||
    let row = &v_d[oc * 6..oc * 6 + 6];
    let n = row.iter().map(|x| x * x).sum::<f32>().sqrt();
    v_d[oc * 6 + c_loc * 3 + j] * g_d[oc] / n
}
