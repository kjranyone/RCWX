//! tape autograd の有限差分 gradcheck + AdamW 収束 + uniform slot 回収 smoke
//! (Issue #8 M-T0/M-T1 ゲート)。実 GPU 必要 (Vulkan)。

use rcwx_nn::engine::tape::{ParamStore, Tape, Var};
use rcwx_nn::engine::{Backend, Engine, GpuTensor};

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let v = (*state >> 33) as f64 / (1u64 << 33) as f64;
    ((v - 0.5) * 2.0) as f32
}

fn rand_vec(n: usize, state: &mut u64) -> Vec<f32> {
    (0..n).map(|_| lcg(state)).collect()
}

/// 有限差分 gradcheck:
/// 1) build で forward + backward_from(out) (loss = Σout) を submit し解析勾配
/// 2) build を再度実行して forward のみの Plan を作り、各検査変数を ±eps 摂動して
///    loss を再計算し中心差分と比較する。
fn run_fd(
    eng: &Engine,
    name: &str,
    build: impl Fn(&mut Tape<'_>, &ParamStore) -> (Var, Vec<(Var, Vec<f32>)>),
    init: impl FnOnce(&mut ParamStore),
) {
    let mut store = ParamStore::new();
    init(&mut store);

    // pass 1: 解析勾配
    let mut tape = Tape::new(eng);
    let (out1, check1) = build(&mut tape, &store);
    tape.backward_from(out1);
    // 入力変数の勾配は backward で挿入されるので、ここで収集する
    let grad_ts: Vec<Option<GpuTensor>> =
        check1.iter().map(|(v, _)| tape.grad_tensor(*v)).collect();
    tape.submit().expect("submit");
    let analytic: Vec<Vec<f32>> = grad_ts
        .iter()
        .map(|g| eng.readback_f32(g.as_ref().expect("grad missing")))
        .collect();

    // pass 2: FD 用 forward のみ Plan
    let mut tape2 = Tape::new(eng);
    let (out2, check2) = build(&mut tape2, &store);
    let out_t = tape2.var_tensor(out2);
    let perturb_ts: Vec<GpuTensor> = check2.iter().map(|(v, _)| tape2.var_tensor(*v)).collect();
    let plan = tape2.into_plan().expect("plan");

    let eval = |buf: &GpuTensor, data: &[f32]| -> f64 {
        let staging = eng.tensor_from_f32(data, vec![data.len()]);
        eng.replay(&plan, Some((&staging, buf)));
        let out = eng.readback_f32(&out_t);
        out.iter().map(|&v| v as f64).sum()
    };

    let eps = 1e-2f64;
    let tol = 3e-2f64;
    let mut worst = 0.0f64;
    let mut failures = 0usize;
    for (vi, (_var, data)) in check2.iter().enumerate() {
        let mut d = data.clone();
        let grad = &analytic[vi];
        assert_eq!(grad.len(), d.len(), "{name}: var {vi} length");
        for k in 0..d.len() {
            let orig = d[k] as f64;
            d[k] = (orig + eps) as f32;
            let f_plus = eval(&perturb_ts[vi], &d);
            d[k] = (orig - eps) as f32;
            let f_minus = eval(&perturb_ts[vi], &d);
            d[k] = orig as f32;
            let _ = eval(&perturb_ts[vi], &d); // 元の値に戻す
            let fd = (f_plus - f_minus) / (2.0 * eps);
            let ad = grad[k] as f64;
            let err = (fd - ad).abs() / (1.0 + fd.abs() + ad.abs());
            worst = worst.max(err);
            if err > tol {
                failures += 1;
                if failures <= 5 {
                    eprintln!("{name}: var {vi}[{k}] fd={fd:.6} ad={ad:.6} relerr={err:.4}");
                }
            }
        }
        let staging = eng.tensor_from_f32(&d, vec![d.len()]);
        eng.replay(&plan, Some((&staging, &perturb_ts[vi])));
    }
    println!("{name}: worst relerr = {worst:.5}");
    assert_eq!(failures, 0, "{name}: {failures} grad mismatches");
}

#[test]
fn gradcheck_linear_gelu_tanh() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    // out = tanh( linear2( gelu( linear1(x) ) ) )
    let build = move |tape: &mut Tape<'_>, store: &ParamStore| -> (Var, Vec<(Var, Vec<f32>)>) {
        let (m, k, n) = (6u32, 5u32, 4u32);
        let mut st = 42u64;
        let x_d = rand_vec((m * k) as usize, &mut st);
        let _skip = rand_vec((n * k) as usize, &mut st); // w1 (store 内と同シード列)
        let _ = rand_vec(n as usize, &mut st); // b1
        let _ = rand_vec((n * n) as usize, &mut st); // w2
        let _ = rand_vec(n as usize, &mut st); // b2
        let x = tape.input_f32s(&x_d, vec![m as usize, k as usize]);
        let w1 = tape.adopt_param(store, 0);
        let b1 = tape.adopt_param(store, 1);
        let w2 = tape.adopt_param(store, 2);
        let b2 = tape.adopt_param(store, 3);
        let h = tape.linear(x, w1, Some(b1), m, k, n);
        let h = tape.gelu(h);
        let h = tape.linear(h, w2, Some(b2), m, n, n);
        let out = tape.tanh(h);
        // 検査変数のデータ (store 構築時と同じシード列で再生成)
        let mut st2 = 42u64;
        let _ = rand_vec((m * k) as usize, &mut st2);
        let w1_d = rand_vec((n * k) as usize, &mut st2);
        let b1_d = rand_vec(n as usize, &mut st2);
        let w2_d = rand_vec((n * n) as usize, &mut st2);
        let b2_d = rand_vec(n as usize, &mut st2);
        (
            out,
            vec![(x, x_d), (w1, w1_d), (b1, b1_d), (w2, w2_d), (b2, b2_d)],
        )
    };
    run_fd(&eng, "linear_gelu_tanh", build, |s| {
        let mut st = 42u64;
        let _ = rand_vec(30, &mut st); // x
        s.add(&eng, "w1", &rand_vec(20, &mut st), vec![4, 5]);
        s.add(&eng, "b1", &rand_vec(4, &mut st), vec![4]);
        s.add(&eng, "w2", &rand_vec(16, &mut st), vec![4, 4]);
        s.add(&eng, "b2", &rand_vec(4, &mut st), vec![4]);
    });
}

#[test]
fn gradcheck_softmax_ln_mul_add() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let build = move |tape: &mut Tape<'_>, store: &ParamStore| -> (Var, Vec<(Var, Vec<f32>)>) {
        let (rows, cols) = (4u32, 8u32);
        let mut st = 7u64;
        let x_d = rand_vec((rows * cols) as usize, &mut st);
        let gamma_d = rand_vec(cols as usize, &mut st);
        let beta_d = rand_vec(cols as usize, &mut st);
        let c_d = rand_vec((rows * cols) as usize, &mut st);
        let d_d = rand_vec((rows * cols) as usize, &mut st);
        let x = tape.input_f32s(&x_d, vec![rows as usize, cols as usize]);
        let gamma = tape.adopt_param(store, 0);
        let beta = tape.adopt_param(store, 1);
        let c = tape.const_f32s(&c_d, vec![rows as usize, cols as usize]);
        let d = tape.const_f32s(&d_d, vec![rows as usize, cols as usize]);
        let h = tape.softmax(x, rows, cols);
        let h = tape.layernorm(h, gamma, beta, rows, cols, 1e-5);
        let h = tape.mul2(h, c);
        let out = tape.add(h, d, false);
        (out, vec![(x, x_d), (gamma, gamma_d), (beta, beta_d)])
    };
    run_fd(&eng, "softmax_ln_mul_add", build, |s| {
        let mut st = 7u64;
        let _ = rand_vec(32, &mut st); // x
        s.add(&eng, "gamma", &rand_vec(8, &mut st), vec![8]);
        s.add(&eng, "beta", &rand_vec(8, &mut st), vec![8]);
    });
}

#[test]
fn gradcheck_weight_norm() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    // out = gelu( linear(x, w=wn(v,g), b) )
    let build = move |tape: &mut Tape<'_>, store: &ParamStore| -> (Var, Vec<(Var, Vec<f32>)>) {
        let (m, k, n) = (5u32, 6u32, 4u32);
        let mut st = 99u64;
        let x_d = rand_vec((m * k) as usize, &mut st);
        let v_d = rand_vec((n * k) as usize, &mut st);
        let b_d = rand_vec(n as usize, &mut st);
        let pv = tape.adopt_all(store);
        let x = tape.input_f32s(&x_d, vec![m as usize, k as usize]);
        let (v, g, b) = (pv[0], pv[1], pv[2]);
        let w = tape.adopt_wn(store, 0, v, g);
        let h = tape.linear(x, w, Some(b), m, k, n);
        let out = tape.gelu(h);
        (
            out,
            vec![(x, x_d), (v, v_d), (g, vec![1.7f32; 4]), (b, b_d)],
        )
    };
    run_fd(&eng, "weight_norm", build, |s| {
        let mut st = 99u64;
        let _ = rand_vec(30, &mut st); // x
        let v = rand_vec(24, &mut st);
        s.add_weight_norm(&eng, "lin", &v, &[1.7f32; 4], vec![4, 6]);
        s.add(&eng, "b", &rand_vec(4, &mut st), vec![4]);
    });
}

#[test]
fn adamw_converges() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let mut store = ParamStore::new();
    store.adam.lr = 0.1;
    let mut st = 5u64;
    let n = 16usize;
    let x_d = rand_vec(n, &mut st);
    let target_d: Vec<f32> = x_d.iter().map(|v| v * 2.0).collect();
    store.add(&eng, "w", &rand_vec(n, &mut st), vec![n as usize]);

    let mut losses = Vec::new();
    for step in 0..80 {
        let mut tape = Tape::new(&eng);
        let pv = tape.adopt_all(&store);
        tape.zero_grads(&store);
        let x = tape.const_f32s(&x_d, vec![n as usize]);
        let y = tape.const_f32s(&target_d, vec![n as usize]);
        let prod = tape.mul2(pv[0], x);
        let resid = tape.add(prod, y, true);
        let sq = tape.mul2(resid, resid);
        let loss = tape.sum_all(sq);
        tape.backward_from(sq);
        tape.adamw_step(&mut store, None);
        let loss_t = tape.var_tensor(loss);
        tape.submit().expect("submit");
        if step % 10 == 0 || step == 79 {
            losses.push(eng.readback_f32(&loss_t)[0]);
        }
    }
    println!("adamw losses: {losses:?}");
    assert!(
        losses.last().copied().unwrap_or(1.0) < 1e-3 * losses[0].max(1e-6),
        "loss not decreased enough: {losses:?}"
    );
}

/// conv1d の FD gradcheck。stride/dilation/pad/batch を変えた 2 ケース。
fn conv1d_case(eng: &Engine, name: &str, stride: u32, dil: u32, pad: u32) {
    let (batch, in_c, out_c, kernel, in_len) = (2u32, 3u32, 4u32, 3u32, 9u32);
    let out_len = (in_len + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
    let build = move |tape: &mut Tape<'_>, store: &ParamStore| -> (Var, Vec<(Var, Vec<f32>)>) {
        let mut st = 1234u64;
        let x_d = rand_vec((batch * in_c * in_len) as usize, &mut st);
        let w_d = rand_vec((out_c * in_c * kernel) as usize, &mut st);
        let b_d = rand_vec(out_c as usize, &mut st);
        let x = tape.input_f32s(&x_d, vec![batch as usize, in_c as usize, in_len as usize]);
        let w = tape.adopt_param(store, 0);
        let b = tape.adopt_param(store, 1);
        let h = tape.conv1d(
            x,
            w,
            Some(b),
            out_c,
            in_c,
            kernel,
            stride,
            dil,
            pad,
            batch,
            in_len,
            out_len,
        );
        let out = tape.gelu(h);
        (out, vec![(x, x_d), (w, w_d), (b, b_d)])
    };
    run_fd(eng, name, build, |s| {
        let mut st = 1234u64;
        let _ = rand_vec((batch * in_c * in_len) as usize, &mut st);
        s.add(
            &eng,
            "w",
            &rand_vec((out_c * in_c * kernel) as usize, &mut st),
            vec![(out_c * in_c * kernel) as usize],
        );
        s.add(
            &eng,
            "b",
            &rand_vec(out_c as usize, &mut st),
            vec![out_c as usize],
        );
    });
}

#[test]
fn gradcheck_conv1d_unit() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    conv1d_case(&eng, "conv1d_s1", 1, 1, 1);
    conv1d_case(&eng, "conv1d_s2_d2", 2, 2, 2);
}

#[test]
fn gradcheck_conv_transpose1d() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (batch, in_c, out_c, kernel, stride, dil, pad, in_len) =
        (2u32, 3u32, 2u32, 4u32, 3u32, 1u32, 1u32, 7u32);
    let out_len = (in_len - 1) * stride + dil * (kernel - 1) + 1 - 2 * pad;
    let build = move |tape: &mut Tape<'_>, store: &ParamStore| -> (Var, Vec<(Var, Vec<f32>)>) {
        let mut st = 777u64;
        let x_d = rand_vec((batch * in_c * in_len) as usize, &mut st);
        let w_d = rand_vec((in_c * out_c * kernel) as usize, &mut st);
        let b_d = rand_vec(out_c as usize, &mut st);
        let x = tape.input_f32s(&x_d, vec![batch as usize, in_c as usize, in_len as usize]);
        let w = tape.adopt_param(store, 0);
        let b = tape.adopt_param(store, 1);
        let h = tape.conv_transpose1d(
            x,
            w,
            Some(b),
            out_c,
            in_c,
            kernel,
            stride,
            dil,
            pad,
            batch,
            in_len,
            out_len,
        );
        let out = tape.tanh(h);
        (out, vec![(x, x_d), (w, w_d), (b, b_d)])
    };
    run_fd(&eng, "conv_transpose1d", build, |s| {
        let mut st = 777u64;
        let _ = rand_vec((batch * in_c * in_len) as usize, &mut st);
        s.add(
            &eng,
            "w",
            &rand_vec((in_c * out_c * kernel) as usize, &mut st),
            vec![(in_c * out_c * kernel) as usize],
        );
        s.add(
            &eng,
            "b",
            &rand_vec(out_c as usize, &mut st),
            vec![out_c as usize],
        );
    });
}

/// conv → lrelu → convT の合成 (dec 側の基本骨格)。
#[test]
fn gradcheck_conv_chain() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (batch, c0, c1, c2) = (2u32, 3u32, 5u32, 2u32);
    let t0 = 11u32;
    let k = 3u32;
    let t1 = t0 + 2; // conv k3 pad1
    let stride = 2u32;
    let t2 = (t1 - 1) * stride + k; // convT k3 stride2
    let build = move |tape: &mut Tape<'_>, store: &ParamStore| -> (Var, Vec<(Var, Vec<f32>)>) {
        let mut st = 55u64;
        let x_d = rand_vec((batch * c0 * t0) as usize, &mut st);
        let w1_d = rand_vec((c1 * c0 * k) as usize, &mut st);
        let b1_d = rand_vec(c1 as usize, &mut st);
        let w2_d = rand_vec((c1 * c2 * k) as usize, &mut st);
        let b2_d = rand_vec(c2 as usize, &mut st);
        let x = tape.input_f32s(&x_d, vec![batch as usize, c0 as usize, t0 as usize]);
        let w1 = tape.adopt_param(store, 0);
        let b1 = tape.adopt_param(store, 1);
        let w2 = tape.adopt_param(store, 2);
        let b2 = tape.adopt_param(store, 3);
        let h = tape.conv1d(x, w1, Some(b1), c1, c0, k, 1, 1, 1, batch, t0, t1);
        let h = tape.lrelu(h, 0.1);
        let out = tape.conv_transpose1d(h, w2, Some(b2), c2, c1, k, stride, 1, 0, batch, t1, t2);
        (
            out,
            vec![(x, x_d), (w1, w1_d), (b1, b1_d), (w2, w2_d), (b2, b2_d)],
        )
    };
    run_fd(&eng, "conv_chain", build, |s| {
        let mut st = 55u64;
        let _ = rand_vec((batch * c0 * t0) as usize, &mut st);
        s.add(
            &eng,
            "w1",
            &rand_vec((c1 * c0 * k) as usize, &mut st),
            vec![(c1 * c0 * k) as usize],
        );
        s.add(
            &eng,
            "b1",
            &rand_vec(c1 as usize, &mut st),
            vec![c1 as usize],
        );
        s.add(
            &eng,
            "w2",
            &rand_vec((c1 * c2 * k) as usize, &mut st),
            vec![(c1 * c2 * k) as usize],
        );
        s.add(
            &eng,
            "b2",
            &rand_vec(c2 as usize, &mut st),
            vec![c2 as usize],
        );
    });
}

#[test]
fn uniform_slot_recycle() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    // 恒久 Plan を 1 本作り、その後大量の使い捨て tape submit を回す。
    // 回収なしでは 16384 slot 上限で panic する。
    let mut store = ParamStore::new();
    store.add(&eng, "w", &vec![0.5f32; 16], vec![16]);
    let mut tape = Tape::new(&eng);
    let pv = tape.adopt_all(&store);
    let x = tape.const_f32s(&vec![1.0f32; 16], vec![16]);
    let out = tape.mul2(pv[0], x);
    let out_t = tape.var_tensor(out);
    let plan = tape.into_plan().expect("plan");
    eng.replay(&plan, None);
    assert!((eng.readback_f32(&out_t)[0] - 0.5).abs() < 1e-6);

    for i in 0..4000 {
        let mut t = Tape::new(&eng);
        let pv = t.adopt_all(&store);
        let x = t.const_f32s(&vec![i as f32 * 0.001; 16], vec![16]);
        let a = t.mul2(pv[0], x);
        let b = t.gelu(a);
        let c = t.tanh(b);
        let _ = t.sum_all(c);
        t.submit().expect("submit");
    }
    // 恒久 plan は影響を受けていない
    eng.replay(&plan, None);
    assert!((eng.readback_f32(&out_t)[0] - 0.5).abs() < 1e-6);
}
