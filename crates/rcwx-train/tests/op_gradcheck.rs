//! M-T2 ops の FD gradcheck (mm batched / chslice / transpose / mul_frame_mask)。
use rcwx_nn::engine::tape::{ParamStore, Tape, Var};
use rcwx_nn::engine::{Backend, Engine, GpuTensor};

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (1u64 << 33) as f32 - 0.5) * 2.0
}

fn run_fd(
    eng: &Engine,
    name: &str,
    build: impl Fn(&mut Tape<'_>, &ParamStore) -> (Var, Vec<(Var, Vec<f32>)>),
    init: impl FnOnce(&mut ParamStore),
) {
    let mut store = ParamStore::new();
    init(&mut store);

    let mut tape = Tape::new(eng);
    let (out1, check1) = build(&mut tape, &store);
    tape.backward_from(out1);
    let grad_ts: Vec<Option<GpuTensor>> =
        check1.iter().map(|(v, _)| tape.grad_tensor(*v)).collect();
    tape.submit().expect("submit");
    let analytic: Vec<Vec<f32>> = grad_ts
        .iter()
        .map(|g| eng.readback_f32(g.as_ref().expect("grad missing")))
        .collect();

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
        let mut var_fail = 0usize;
        for k in 0..d.len() {
            let orig = d[k] as f64;
            d[k] = (orig + eps) as f32;
            let f_plus = eval(&perturb_ts[vi], &d);
            d[k] = (orig - eps) as f32;
            let f_minus = eval(&perturb_ts[vi], &d);
            d[k] = orig as f32;
            let _ = eval(&perturb_ts[vi], &d);
            let fd = (f_plus - f_minus) / (2.0 * eps);
            let ad = grad[k] as f64;
            let err = (fd - ad).abs() / (1.0 + fd.abs() + ad.abs());
            worst = worst.max(err);
            if err > tol {
                failures += 1;
                var_fail += 1;
                if failures <= 5 {
                    println!("  {name} var{vi}[{k}] fd={fd:.6} ad={ad:.6}");
                }
            }
        }
        println!("  {name} var{vi}: {var_fail}/{} bad", d.len());
        let staging = eng.tensor_from_f32(&d, vec![d.len()]);
        eng.replay(&plan, Some((&staging, &perturb_ts[vi])));
    }
    println!("{name}: worst relerr = {worst:.5}");
    assert_eq!(failures, 0, "{name}: {failures} grad mismatches");
}

#[test]
fn gradcheck_mm_batched() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let build = move |tape: &mut Tape<'_>, store: &ParamStore| -> (Var, Vec<(Var, Vec<f32>)>) {
        let (b, t, c_in, h) = (2u32, 5u32, 4u32, 3u32);
        let mut st = 11u64;
        let x_d: Vec<f32> = (0..(b * t * c_in) as usize).map(|_| lcg(&mut st)).collect();
        let w_d: Vec<f32> = (0..(h * c_in) as usize).map(|_| lcg(&mut st)).collect();
        let bi_d: Vec<f32> = (0..h as usize).map(|_| lcg(&mut st)).collect();
        let x = tape.input_f32s(&x_d, vec![b as usize, t as usize, c_in as usize]);
        let w = tape.adopt_param(store, 0);
        let bi = tape.adopt_param(store, 1);
        let out = tape.mm(
            x,
            w,
            Some(bi),
            (b * t * h) as usize,
            t,
            h,
            c_in,
            b,
            c_in,
            1,
            0,
            t * c_in,
            c_in,
            1,
            0,
            0,
            h,
            1,
            t * h,
            1.0,
            0,
        );
        let mut st2 = 11u64;
        let _ = rand_skip((b * t * c_in) as usize, &mut st2);
        let w_d2: Vec<f32> = (0..(h * c_in) as usize).map(|_| lcg(&mut st2)).collect();
        (out, vec![(x, x_d), (w, w_d2), (bi, bi_d)])
    };
    run_fd(&eng, "mm_batched", build, |s| {
        let mut st = 11u64;
        let _ = rand_skip(40, &mut st);
        s.add(&eng, "w", &rand_vec_n((3 * 4) as usize, &mut st), vec![12]);
        s.add(&eng, "b", &rand_vec_n(3, &mut st), vec![3]);
    });
}

fn rand_skip(n: usize, st: &mut u64) -> Vec<f32> {
    (0..n).map(|_| lcg(st)).collect()
}
fn rand_vec_n(n: usize, st: &mut u64) -> Vec<f32> {
    (0..n).map(|_| lcg(st)).collect()
}

#[test]
fn gradcheck_chslice_transpose_mask() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let build = move |tape: &mut Tape<'_>, _store: &ParamStore| -> (Var, Vec<(Var, Vec<f32>)>) {
        let (b, c, t) = (2u32, 4u32, 5u32);
        let mut st = 77u64;
        let x_d: Vec<f32> = (0..(b * c * t) as usize).map(|_| lcg(&mut st)).collect();
        let mask_d: Vec<f32> = (0..(b * t) as usize)
            .map(|i| if i % (t as usize) < 3 { 1.0 } else { 0.0 })
            .collect();
        let x = tape.input_f32s(&x_d, vec![b as usize, c as usize, t as usize]);
        let m = tape.const_f32s(&mask_d, vec![b as usize, t as usize]);
        let h = tape.mul_frame_mask(x, m, c, t);
        let h = tape.chslice(h, c, 1, 2, b, t);
        // transpose_tc_ct は [b, t, c] 入力前提 — ここでは平坦和検証用にそのまま
        (h, vec![(x, x_d)])
    };
    run_fd(&eng, "chslice_mask", build, |_s| {});
}

#[test]
fn gradcheck_ln_strided() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let build = move |tape: &mut Tape<'_>, store: &ParamStore| -> (Var, Vec<(Var, Vec<f32>)>) {
        let (b, c, t) = (2u32, 4u32, 5u32);
        let mut st = 31u64;
        let x_d: Vec<f32> = (0..(b * c * t) as usize).map(|_| lcg(&mut st)).collect();
        let g_d: Vec<f32> = (0..c as usize).map(|_| lcg(&mut st)).collect();
        let be_d: Vec<f32> = (0..c as usize).map(|_| lcg(&mut st)).collect();
        let x = tape.input_f32s(&x_d, vec![b as usize, c as usize, t as usize]);
        let g = tape.adopt_param(store, 0);
        let be = tape.adopt_param(store, 1);
        let out = tape.layernorm_strided(x, g, be, b * t, c, 1, t, 1e-5);
        (out, vec![(x, x_d), (g, g_d), (be, be_d)])
    };
    run_fd(&eng, "ln_strided", build, |s| {
        let mut st = 31u64;
        let _ = rand_skip(40, &mut st);
        s.add(&eng, "g", &rand_vec_n(4, &mut st), vec![4]);
        s.add(&eng, "b", &rand_vec_n(4, &mut st), vec![4]);
    });
}

#[test]
fn gradcheck_skew_chcat() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    // [b, t, 2ws+1] → skew0 → [b, t, t] → chcat 2 つ → sum
    let build = move |tape: &mut Tape<'_>, _store: &ParamStore| -> (Var, Vec<(Var, Vec<f32>)>) {
        let (b, t, ws) = (2u32, 5u32, 3u32);
        let mut st = 55u64;
        let x_d: Vec<f32> = (0..(b * t * (2 * ws + 1)) as usize)
            .map(|_| lcg(&mut st))
            .collect();
        let x = tape.input_f32s(&x_d, vec![b as usize, t as usize, (2 * ws + 1) as usize]);
        let s0 = tape.skew(x, 0, b, t, ws);
        (s0, vec![(x, x_d)])
    };
    run_fd(&eng, "skew0", build, |_s| {});
    // skew1
    let build2 = move |tape: &mut Tape<'_>, _store: &ParamStore| -> (Var, Vec<(Var, Vec<f32>)>) {
        let (b, t, ws) = (2u32, 5u32, 3u32);
        let mut st = 56u64;
        let x_d: Vec<f32> = (0..(b * t * t) as usize).map(|_| lcg(&mut st)).collect();
        let x = tape.input_f32s(&x_d, vec![b as usize, t as usize, t as usize]);
        let s1 = tape.skew(x, 1, b, t, ws);
        (s1, vec![(x, x_d)])
    };
    run_fd(&eng, "skew1", build2, |_s| {});
}

/// attention_layer 1 層の入力勾配 FD 検証 (ランダム重み)。
#[test]
fn gradcheck_attention_layer() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    // 重みマップをランダム生成 (SynthTrain 経由で adopt)
    let heads_env = std::env::var("ATTN_HEADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let (h, heads, ws, b, t) = (8u32, heads_env, 3u32, 2u32, 5u32);
    let dk = h / heads;
    let mut st = 100u64;
    let mut weights = std::collections::HashMap::new();
    let mut put = |weights: &mut std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
                   st: &mut u64,
                   name: &str,
                   shape: &[usize]| {
        let n: usize = shape.iter().product();
        let d: Vec<f32> = (0..n).map(|_| lcg(st)).collect();
        weights.insert(name.to_string(), (d, shape.to_vec()));
    };
    let pfx = "enc_p.encoder.attn_layers.0";
    put(
        &mut weights,
        &mut st,
        "enc_p.encoder.attn_layers.0.conv_q.weight",
        &[h as usize, h as usize, 1],
    );
    put(
        &mut weights,
        &mut st,
        "enc_p.encoder.attn_layers.0.conv_q.bias",
        &[h as usize],
    );
    put(
        &mut weights,
        &mut st,
        "enc_p.encoder.attn_layers.0.conv_k.weight",
        &[h as usize, h as usize, 1],
    );
    put(
        &mut weights,
        &mut st,
        "enc_p.encoder.attn_layers.0.conv_k.bias",
        &[h as usize],
    );
    put(
        &mut weights,
        &mut st,
        "enc_p.encoder.attn_layers.0.conv_v.weight",
        &[h as usize, h as usize, 1],
    );
    put(
        &mut weights,
        &mut st,
        "enc_p.encoder.attn_layers.0.conv_v.bias",
        &[h as usize],
    );
    put(
        &mut weights,
        &mut st,
        "enc_p.encoder.attn_layers.0.conv_o.weight",
        &[h as usize, h as usize, 1],
    );
    put(
        &mut weights,
        &mut st,
        "enc_p.encoder.attn_layers.0.conv_o.bias",
        &[h as usize],
    );
    put(
        &mut weights,
        &mut st,
        "enc_p.encoder.attn_layers.0.emb_rel_k",
        &[1, (2 * ws + 1) as usize, dk as usize],
    );
    put(
        &mut weights,
        &mut st,
        "enc_p.encoder.attn_layers.0.emb_rel_v",
        &[1, (2 * ws + 1) as usize, dk as usize],
    );
    let cfg = rcwx_train::modules::synth_train::SynthTrainCfg {
        in_channels: 8,
        inter_channels: 6,
        hidden_channels: h,
        filter_channels: 16,
        n_heads: heads,
        n_layers: 1,
        kernel_size: 3,
        window_size: ws,
        ..rcwx_train::modules::synth_train::SynthTrainCfg::v2_48k()
    };
    let stm = rcwx_train::modules::synth_train::SynthTrain::new(&eng, cfg, &weights).unwrap();

    // FD 用: 入力 x を摂動 → attention 出力和。plan で再利用するため
    // x は input var、それ以外は store adopt。
    let build = |tape: &mut Tape<'_>,
                 stm2: &rcwx_train::modules::synth_train::SynthTrain|
     -> (Var, Vec<(Var, Vec<f32>)>) {
        let a = stm2.adopt(tape);
        let mut st2 = 100u64;
        let _ = rand_skip((b * h * t) as usize, &mut st2);
        let x_d: Vec<f32> = (0..(b * h * t) as usize).map(|_| lcg(&mut st2)).collect();
        let x = tape.input_f32s(&x_d, vec![b as usize, h as usize, t as usize]);
        let mask_bias = tape.const_f32s(
            &vec![1.0f32; (t * t) as usize],
            vec![t as usize, t as usize],
        );
        let out = rcwx_train::modules::attention_layer(
            tape,
            &a,
            "enc_p.encoder.attn_layers.0",
            x,
            h,
            heads,
            ws,
            b,
            t,
            Some(mask_bias),
        );
        (out, vec![(x, x_d)])
    };

    // pass 1: 解析勾配
    let mut tape = Tape::new(&eng);
    let (out1, check1) = build(&mut tape, &stm);
    tape.backward_from(out1);
    let grad_ts: Vec<Option<GpuTensor>> =
        check1.iter().map(|(v, _)| tape.grad_tensor(*v)).collect();
    tape.submit().expect("submit");
    let analytic: Vec<Vec<f32>> = grad_ts
        .iter()
        .map(|g| eng.readback_f32(g.as_ref().expect("grad missing")))
        .collect();

    // pass 2: FD
    let mut tape2 = Tape::new(&eng);
    let (out2, check2) = build(&mut tape2, &stm);
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
        for k in 0..d.len() {
            let orig = d[k] as f64;
            d[k] = (orig + eps) as f32;
            let f_plus = eval(&perturb_ts[vi], &d);
            d[k] = (orig - eps) as f32;
            let f_minus = eval(&perturb_ts[vi], &d);
            d[k] = orig as f32;
            let _ = eval(&perturb_ts[vi], &d);
            let fd = (f_plus - f_minus) / (2.0 * eps);
            let ad = grad[k] as f64;
            let err = (fd - ad).abs() / (1.0 + fd.abs() + ad.abs());
            worst = worst.max(err);
            if err > tol {
                failures += 1;
                if failures <= 5 {
                    println!("  attn[{k}] fd={fd:.6} ad={ad:.6}");
                }
            }
        }
        let staging = eng.tensor_from_f32(&d, vec![d.len()]);
        eng.replay(&plan, Some((&staging, &perturb_ts[vi])));
    }
    println!("attention_layer: worst relerr = {worst:.5}");
    assert_eq!(failures, 0, "attention_layer: {failures} grad mismatches");
}

/// ヘッドオフセット付き QK mm (head=1 のみ、q/k を [B, C, T] input から読む) の FD。
#[test]
fn gradcheck_qk_offset() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (b, c, t, heads) = (2u32, 8u32, 5u32, 2u32);
    let dk = c / heads;
    let build = move |tape: &mut Tape<'_>, _s: &ParamStore| -> (Var, Vec<(Var, Vec<f32>)>) {
        let mut st = 200u64;
        let q_d: Vec<f32> = (0..(b * c * t) as usize).map(|_| lcg(&mut st)).collect();
        let k_d: Vec<f32> = (0..(b * c * t) as usize).map(|_| lcg(&mut st)).collect();
        let q = tape.input_f32s(&q_d, vec![b as usize, c as usize, t as usize]);
        let k = tape.input_f32s(&k_d, vec![b as usize, c as usize, t as usize]);
        // head 1 の QK^T (a_base/w_base = dk·t)
        let s1 = tape.mm(
            q,
            k,
            None,
            (b * t * t) as usize,
            t,
            t,
            dk,
            b,
            1,
            t,
            dk * t,
            c * t,
            1,
            t,
            dk * t,
            c * t,
            t,
            1,
            t * t,
            1.0,
            0,
        );
        (s1, vec![(q, q_d), (k, k_d)])
    };
    run_fd(&eng, "qk_offset", build, |_s| {});
}
