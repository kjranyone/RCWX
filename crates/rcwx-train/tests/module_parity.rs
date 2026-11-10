//! enc_p (TextEncoder) の PyTorch golden パリティ (Issue #8 M-T2 ゲート)。
//! 事前生成: uv run python tools/train_ref.py

use std::collections::HashMap;

use rcwx_nn::engine::tape::Tape;
use rcwx_nn::engine::{Backend, Engine};
use rcwx_train::modules::synth_train::{SynthTrain, SynthTrainCfg, enc_p};

fn golden_path() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap())
        .join(".cache/rcwx/train/golden/encp_golden.safetensors")
}

fn load_golden(path: &std::path::Path, prefix: &str) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let raw = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&raw).unwrap();
    let mut out = HashMap::new();
    for name in st.names() {
        let name: &str = name;
        if let Some(short) = name.strip_prefix(prefix) {
            let v = st.tensor(name).unwrap();
            let n: usize = v.shape().iter().product();
            let data = v.data();
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(floats.len(), n);
            out.insert(short.to_string(), (floats, v.shape().to_vec()));
        }
    }
    out
}

struct Fixture {
    cfg: SynthTrainCfg,
    batch: u32,
    t_len: u32,
    phone: Vec<f32>,
    onehot: Vec<f32>,
    mask: Vec<f32>,
    bias: Vec<f32>,
    want_out: HashMap<String, (Vec<f32>, Vec<usize>)>,
    want_grads: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

fn fixture(
    inputs: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    want_out: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    want_grads: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Fixture {
    let cfg = SynthTrainCfg {
        in_channels: 8,
        inter_channels: 6,
        hidden_channels: 16,
        filter_channels: 32,
        n_heads: 2,
        n_layers: 2,
        kernel_size: 3,
        window_size: 10,
        ..SynthTrainCfg::v2_48k()
    };
    let (batch, t_len, length) = (2u32, 13u32, 10u32);
    let (phone_d, _) = &inputs["phone"];
    let (pitch_d, _) = &inputs["pitch"];
    let mut onehot = vec![0.0f32; (batch * t_len * 256) as usize];
    for i in 0..(batch * t_len) as usize {
        onehot[i * 256 + (pitch_d[i].round().max(0.0) as usize).min(255)] = 1.0;
    }
    let mask: Vec<f32> = (0..batch * t_len)
        .map(|i| if i % t_len < length { 1.0 } else { 0.0 })
        .collect();
    let mut bias = vec![1.0f32; (t_len * t_len) as usize];
    for i in 0..t_len {
        for j in 0..t_len {
            if i >= length || j >= length {
                bias[(i * t_len + j) as usize] = 0.0;
            }
        }
    }
    Fixture {
        cfg,
        batch,
        t_len,
        phone: phone_d.clone(),
        onehot,
        mask,
        bias,
        want_out: want_out.clone(),
        want_grads: want_grads.clone(),
    }
}

/// x0 パイプライン + conv_q の検証 (STAGE 環境変数で段階有効化)。
fn debug_stages(eng: &Engine, st: &SynthTrain, f: &Fixture) {
    let stage: u32 = std::env::var("STAGE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let mut tape = Tape::new(eng);
    let a = st.adopt(&mut tape);
    let h = f.cfg.hidden_channels;
    let phone = tape.const_f32s(&f.phone, vec![f.batch as usize, f.t_len as usize, 8]);
    let onehot = tape.const_f32s(&f.onehot, vec![(f.batch * f.t_len) as usize, 256]);
    let mask = tape.const_f32s(&f.mask, vec![f.batch as usize, f.t_len as usize]);
    let bias = tape.const_f32s(&f.bias, vec![f.t_len as usize, f.t_len as usize]);
    let bt = f.batch * f.t_len;
    let w = a.vec("enc_p.emb_phone.weight");
    let xp = tape.mm(
        phone,
        w,
        a.bias("enc_p.emb_phone"),
        (bt * h) as usize,
        f.t_len,
        h,
        8,
        f.batch,
        8,
        1,
        0,
        f.t_len * 8,
        8,
        1,
        0,
        0,
        h,
        1,
        f.t_len * h,
        1.0,
    );
    let xp_t = tape.var_tensor(xp);
    let xp_lin = tape.linear(
        phone,
        a.vec("enc_p.emb_phone.weight"),
        a.bias("enc_p.emb_phone"),
        bt,
        8,
        h,
    );
    let mut last = xp;
    if stage >= 2 {
        let emb = a.vec("enc_p.emb_pitch.weight");
        let xq = tape.mm(
            onehot,
            emb,
            None,
            (bt * h) as usize,
            f.t_len,
            h,
            256,
            f.batch,
            256,
            1,
            0,
            f.t_len * 256,
            1,
            h,
            0,
            0,
            h,
            1,
            f.t_len * h,
            1.0,
        );
        last = xq;
        if stage >= 3 {
            let xsum = tape.add(xp, xq, false);
            let xs = tape.pw(xsum, 7, (h as f32).sqrt());
            last = tape.lrelu(xs, 0.1);
            if stage >= 4 {
                last = tape.transpose_tc_ct(last, f.t_len, h);
                if stage >= 5 {
                    last = tape.mul_frame_mask(last, mask, h, f.t_len);
                }
            }
        }
    }
    if stage == 10 {
        last = tape.transpose_tc_ct(xp, f.t_len, h);
    }
    if stage == 13 {
        // xp を tape.linear (matmul_ex 直呼び) で再計算
        last = tape.transpose_tc_ct(xp_lin, f.t_len, h);
    }
    if stage == 11 {
        // xp を golden 定数で差し替え (mm を除外)
        let (xpw, _) = &f.want_out["xp"];
        let xpc = tape.const_f32s(xpw, vec![f.batch as usize, f.t_len as usize, h as usize]);
        last = tape.transpose_tc_ct(xpc, f.t_len, h);
    }
    let xm = last;
    let q0 = if stage >= 4 || stage == 10 || stage == 11 || stage == 13 {
        let v = rcwx_train::modules::conv1x1(
            &mut tape,
            &a,
            "enc_p.encoder.attn_layers.0.conv_q",
            xm,
            h,
            h,
            f.batch,
            f.t_len,
        );
        Some(tape.var_tensor(v))
    } else {
        None
    };
    // attention 部分段階: qk0 (head0) と rel0
    let dk = h / f.cfg.n_heads;
    let ws = f.cfg.window_size;
    let q0v = rcwx_train::modules::conv1x1(
        &mut tape,
        &a,
        "enc_p.encoder.attn_layers.0.conv_q",
        xm,
        h,
        h,
        f.batch,
        f.t_len,
    );
    let k0v = rcwx_train::modules::conv1x1(
        &mut tape,
        &a,
        "enc_p.encoder.attn_layers.0.conv_k",
        xm,
        h,
        h,
        f.batch,
        f.t_len,
    );
    let scale = 1.0f32 / (dk as f32).sqrt();
    let qk_h0 = tape.mm(
        q0v,
        k0v,
        None,
        (f.batch * f.t_len * f.t_len) as usize,
        f.t_len,
        f.t_len,
        dk,
        f.batch,
        1,
        f.t_len,
        0,
        h * f.t_len,
        1,
        f.t_len,
        0,
        h * f.t_len,
        f.t_len,
        1,
        f.t_len * f.t_len,
        scale,
    );
    let qk_t = tape.var_tensor(qk_h0);
    let relk = a.vec("enc_p.encoder.attn_layers.0.emb_rel_k");
    let m_rel = tape.mm(
        q0v,
        relk,
        None,
        (f.batch * f.t_len * (2 * ws + 1)) as usize,
        f.t_len,
        2 * ws + 1,
        dk,
        f.batch,
        1,
        f.t_len,
        0,
        h * f.t_len,
        dk,
        1,
        0,
        0,
        2 * ws + 1,
        1,
        f.t_len * (2 * ws + 1),
        scale,
    );
    let rel_bias0 = tape.skew(m_rel, 0, f.batch, f.t_len, ws);
    let rel_t = tape.var_tensor(rel_bias0);
    // p0 (softmax 後)
    let scores_h0 = tape.add(qk_h0, rel_bias0, false);
    let scores_h0 = tape.mask_fill2d(scores_h0, bias, f.t_len, -1e4);
    let p_h0 = tape.softmax(scores_h0, f.batch * f.t_len, f.t_len);
    let p_t = tape.var_tensor(p_h0);
    let att = if std::env::var("SKIP_ATTN").is_err() {
        let v = rcwx_train::modules::attention_layer(
            &mut tape,
            &a,
            "enc_p.encoder.attn_layers.0",
            xm,
            h,
            f.cfg.n_heads,
            f.cfg.window_size,
            f.batch,
            f.t_len,
            Some(bias),
        );
        Some(tape.var_tensor(v))
    } else {
        None
    };
    tape.submit().expect("submit");
    let got = eng.readback_f32(&xp_t);
    let (want, _) = &f.want_out["xp"];
    println!(
        "dbg xp (stage {stage}): cos={:.6}",
        rcwx_train::cosine_pub(&got, want)
    );
    if let Some(q0_t) = &q0 {
        let got = eng.readback_f32(q0_t);
        let (want_gold, _) = &f.want_out["q0"];
        let mut want = want_gold.clone();
        if stage == 10 || stage == 11 || stage == 13 {
            // 期待値: conv_q(transpose(xp_golden))
            let (xpw, _) = &f.want_out["xp"];
            let we = st
                .store
                .entries
                .iter()
                .find(|e| e.name == "enc_p.encoder.attn_layers.0.conv_q.weight")
                .unwrap();
            let be = st
                .store
                .entries
                .iter()
                .find(|e| e.name == "enc_p.encoder.attn_layers.0.conv_q.bias")
                .unwrap();
            let wbuf = eng.readback_f32(&we.value);
            let bbuf = eng.readback_f32(&be.value);
            let bt = (f.batch * f.t_len) as usize;
            want = vec![0.0f32; (f.batch * h * f.t_len) as usize];
            for b in 0..f.batch as usize {
                for o in 0..h as usize {
                    for t in 0..f.t_len as usize {
                        let mut acc = bbuf[o];
                        for c in 0..h as usize {
                            // transpose(xp): xm[b, c, t] = xp[b, t, c]
                            acc += wbuf[o * h as usize + c]
                                * xpw[(b * f.t_len as usize + t) * h as usize + c];
                        }
                        want[(b * h as usize + o) * f.t_len as usize + t] = acc;
                    }
                }
            }
            let _ = bt;
        }
        println!(
            "dbg q0 (stage {stage}): cos={:.6}",
            rcwx_train::cosine_pub(&got, &want)
        );
        if let Some(att_t) = &att {
            let got = eng.readback_f32(att_t);
            let (want, _) = &f.want_out["att0"];
            println!(
                "dbg att0 (stage {stage}): cos={:.6}",
                rcwx_train::cosine_pub(&got, want)
            );
        }
        {
            // golden [b, heads, t, t] から head0 を抽出して比較
            let tt = (f.t_len * f.t_len) as usize;
            let heads = f.cfg.n_heads as usize;
            let head0 = |v: &[f32]| -> Vec<f32> {
                let mut out = Vec::new();
                for b in 0..f.batch as usize {
                    let base = (b * heads) * tt;
                    out.extend_from_slice(&v[base..base + tt]);
                }
                out
            };
            let got = eng.readback_f32(&qk_t);
            let want = head0(&f.want_out["qk0"].0);
            println!("dbg qk0: cos={:.6}", rcwx_train::cosine_pub(&got, &want));
            let got = eng.readback_f32(&rel_t);
            let want = head0(&f.want_out["rel0"].0);
            println!("dbg rel0: cos={:.6}", rcwx_train::cosine_pub(&got, &want));
            let got = eng.readback_f32(&p_t);
            let want = head0(&f.want_out["p0"].0);
            println!("dbg p0:  cos={:.6}", rcwx_train::cosine_pub(&got, &want));
            for (i, j) in [(0u32, 0u32), (5, 5), (10, 0), (11, 0), (12, 12)] {
                let base = (i * 13 + j) as usize;
                println!("  p0[{i},{j}] got={:.6} want={:.6}", got[base], want[base]);
            }
        }
        let bad_b0 = got
            .iter()
            .zip(want.iter())
            .take(208)
            .filter(|(g, w)| (**g - **w).abs() > 0.01)
            .count();
        let bad_b1 = got
            .iter()
            .zip(want.iter())
            .skip(208)
            .filter(|(g, w)| (**g - **w).abs() > 0.01)
            .count();
        println!("  bad: b0={bad_b0}/208 b1={bad_b1}/208");
        let be = st
            .store
            .entries
            .iter()
            .find(|e| e.name == "enc_p.encoder.attn_layers.0.conv_q.bias")
            .unwrap();
        let bbuf = eng.readback_f32(&be.value);
        println!("  bias[3]={:.5} got(1,3,5)={:.5}", bbuf[3], got[252]);
    }
}

fn h_probe() -> u32 {
    16
}

#[test]
fn encp_parity() {
    let path = golden_path();
    if !path.exists() {
        eprintln!("skipping: run `uv run python tools/train_ref.py` first");
        return;
    }
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let weights = load_golden(&path, "w.");
    let inputs = load_golden(&path, "in.");
    let want_out = load_golden(&path, "out.");
    let want_grads = load_golden(&path, "g.");
    let dbg_out = load_golden(&path, "dbg.");
    let mut want_all = want_out.clone();
    for (k, v) in dbg_out {
        want_all.insert(k, v);
    }
    let want_out = want_all;

    let f = fixture(&inputs, &want_out, &want_grads);
    let st = SynthTrain::new(&eng, f.cfg.clone(), &weights).expect("synth");

    debug_stages(&eng, &st, &f);

    // 隔離検証: golden x0 を直接入力に conv_q を計算
    {
        let mut tape = Tape::new(&eng);
        let a = st.adopt(&mut tape);
        let (x0w, _) = &f.want_out["x0"];
        let x0 = tape.const_f32s(
            x0w,
            vec![f.batch as usize, h_probe() as usize, f.t_len as usize],
        );
        let q0 = rcwx_train::modules::conv1x1(
            &mut tape,
            &a,
            "enc_p.encoder.attn_layers.0.conv_q",
            x0,
            h_probe(),
            h_probe(),
            f.batch,
            f.t_len,
        );
        let q0_t = tape.var_tensor(q0);
        tape.submit().expect("submit");
        let got = eng.readback_f32(&q0_t);
        let (want, _) = &f.want_out["q0"];
        println!("dbg q0_iso: cos={:.6}", rcwx_train::cosine_pub(&got, want));
    }

    // 本体: enc_p 全体 + 勾配
    let mut tape = Tape::new(&eng);
    let a = st.adopt(&mut tape);
    let phone = tape.const_f32s(&f.phone, vec![f.batch as usize, f.t_len as usize, 8]);
    let onehot = tape.const_f32s(&f.onehot, vec![(f.batch * f.t_len) as usize, 256]);
    let mask = tape.const_f32s(&f.mask, vec![f.batch as usize, f.t_len as usize]);
    let bias = tape.const_f32s(&f.bias, vec![f.t_len as usize, f.t_len as usize]);
    let (m_p, logs_p) = enc_p(
        &mut tape, &a, &f.cfg, phone, onehot, mask, bias, f.batch, f.t_len,
    );
    let m_t = tape.var_tensor(m_p);
    let l_t = tape.var_tensor(logs_p);
    tape.backward_from(m_p);
    tape.backward_from(logs_p);
    tape.submit().expect("submit");

    let got_m = eng.readback_f32(&m_t);
    let got_l = eng.readback_f32(&l_t);
    let (want_m, _) = &f.want_out["m_p"];
    let (want_l, _) = &f.want_out["logs_p"];
    let cm = rcwx_train::cosine_pub(&got_m, want_m);
    let cl = rcwx_train::cosine_pub(&got_l, want_l);
    println!("encp fwd: m_p cos={cm:.6} logs_p cos={cl:.6}");
    assert!(cm >= 0.999, "m_p parity {cm}");
    assert!(cl >= 0.999, "logs_p parity {cl}");

    let checks = [
        "emb_phone.weight",
        "emb_pitch.weight",
        "encoder.attn_layers.0.conv_q.weight",
        "encoder.attn_layers.0.conv_o.weight",
        "encoder.attn_layers.0.emb_rel_k",
        "encoder.attn_layers.0.emb_rel_v",
        "encoder.norm_layers_1.0.gamma",
        "encoder.norm_layers_1.0.beta",
        "encoder.ffn_layers.0.conv_1.weight",
        "encoder.ffn_layers.0.conv_2.weight",
        "proj.weight",
        "proj.bias",
    ];
    for name in checks {
        let Some((want_g, _)) = f.want_grads.get(&format!("enc_p.{name}")) else {
            eprintln!("grad golden missing: {name}");
            continue;
        };
        let Some(entry) = st
            .store
            .entries
            .iter()
            .find(|e| e.name == format!("enc_p.{name}"))
        else {
            panic!("param not found: enc_p.{name}");
        };
        let got_g = eng.readback_f32(&entry.grad);
        let cg = rcwx_train::cosine_pub(&got_g, want_g);
        println!("encp grad {name}: cos={cg:.6}");
        assert!(cg >= 0.999, "grad parity {name}: {cg}");
    }
}
