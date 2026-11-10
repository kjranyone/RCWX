//! enc_q + flow + decoder (GeneratorNSF) の PyTorch golden パリティ (Issue #8 M-T2)。
//! 事前生成: uv run python tools/train_ref2.py

use std::collections::HashMap;

use rcwx_nn::engine::tape::Tape;
use rcwx_nn::engine::{Backend, Engine};
use rcwx_train::modules::synth_train::{SynthTrain, SynthTrainCfg, decoder, enc_q, flow_forward};

fn golden_path() -> std::path::PathBuf {
    let name = if std::env::var("RCWX_MP2_LONG").is_ok() {
        "generator_golden_t2310.safetensors"
    } else {
        "generator_golden.safetensors"
    };
    std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap())
        .join(".cache/rcwx/train/golden")
        .join(name)
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
            let floats: Vec<f32> = v
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(floats.len(), n);
            out.insert(short.to_string(), (floats, v.shape().to_vec()));
        }
    }
    out
}

/// train_ref2.py と同一の tiny 設定。
fn cfg() -> SynthTrainCfg {
    SynthTrainCfg {
        in_channels: 8,
        inter_channels: 6,
        hidden_channels: 8,
        filter_channels: 16,
        n_heads: 2,
        n_layers: 1,
        kernel_size: 3,
        window_size: 3,
        resblock_kernel_sizes: vec![3, 7],
        resblock_dilation_sizes: vec![vec![1, 3], vec![1, 3]],
        upsample_rates: vec![4, 2],
        upsample_kernel_sizes: vec![8, 4],
        upsample_initial_channel: 16,
        gin_channels: 4,
        n_speakers: 1,
        sr: 48000,
        n_flows: 2,
        flow_wn_layers: 2,
        flow_kernel: 5,
        encq_wn_layers: 2,
        encq_kernel: 5,
    }
}

#[test]
fn generator_parity() {
    let long_mode = std::env::var("RCWX_MP2_LONG").is_ok();
    let path = golden_path();
    if !path.exists() {
        eprintln!("skipping: run `uv run python tools/train_ref2.py` first");
        return;
    }
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (batch, t_len, spec_bins) = if long_mode {
        (2u32, 2310u32, 16u32)
    } else {
        (2u32, 11u32, 16u32)
    };
    let c = cfg();

    let weights = load_golden(&path, "w.");
    let inputs = load_golden(&path, "in.");
    let want_out = load_golden(&path, "out.");
    let want_grads = load_golden(&path, "g.");

    let st = SynthTrain::new(&eng, c.clone(), &weights).expect("synth");
    let mut tape = Tape::new(&eng);
    let a = st.adopt(&mut tape);

    let (spec_d, _) = &inputs["spec"];
    let (mask_d, _) = &inputs["mask"];
    let (g_d, _) = &inputs["g"];
    let (randn_d, _) = &inputs["randn"];
    let (har_d, _) = &inputs["har"];

    let spec = tape.const_f32s(
        spec_d,
        vec![batch as usize, spec_bins as usize, t_len as usize],
    );
    let mask = tape.const_f32s(mask_d, vec![batch as usize, t_len as usize]);
    let g = tape.const_f32s(g_d, vec![(batch * c.gin_channels) as usize]);
    let randn = tape.const_f32s(
        randn_d,
        vec![batch as usize, c.inter_channels as usize, t_len as usize],
    );
    let har = tape.const_f32s(har_d, vec![(batch * t_len * c.upp()) as usize]);

    // enc_q 段階検証: pre
    {
        let mut tape = Tape::new(&eng);
        let a = st.adopt(&mut tape);
        let spec = tape.const_f32s(
            &inputs["spec"].0,
            vec![batch as usize, spec_bins as usize, t_len as usize],
        );
        let mask = tape.const_f32s(&inputs["mask"].0, vec![batch as usize, t_len as usize]);
        let g = tape.const_f32s(&inputs["g"].0, vec![(batch * c.gin_channels) as usize]);
        let h = c.hidden_channels;
        let pre = rcwx_train::modules::conv1x1(
            &mut tape,
            &a,
            "enc_q.pre",
            spec,
            h,
            spec_bins,
            batch,
            t_len,
        );
        let pre = tape.mul_frame_mask(pre, mask, h, t_len);
        let pre_t = tape.var_tensor(pre);
        // cond
        let cond = tape.mm(
            a.w("enc_q.enc.cond_layer"),
            g,
            a.bias("enc_q.enc.cond_layer"),
            (batch * 2 * h * c.encq_wn_layers) as usize,
            2 * h * c.encq_wn_layers,
            1,
            c.gin_channels,
            batch,
            c.gin_channels,
            1,
            0,
            0,
            1,
            1,
            0,
            c.gin_channels,
            1,
            1,
            2 * h * c.encq_wn_layers,
            1.0,
            1,
        );
        let cond_t = tape.var_tensor(cond);
        // wn
        let wn = rcwx_train::modules::wn(
            &mut tape,
            &a,
            "enc_q.enc",
            pre,
            h,
            c.encq_kernel,
            c.encq_wn_layers,
            batch,
            t_len,
            Some(g),
            Some(mask),
        );
        let wn_t = tape.var_tensor(wn);
        tape.submit().expect("submit");
        let dbg = load_golden(&path, "dbg.");
        for (name, t, key) in [
            ("pre", pre_t, "pre"),
            ("cond", cond_t, "cond"),
            ("wn", wn_t, "wn"),
        ] {
            let got = eng.readback_f32(&t);
            let want = dbg[key].0.clone();
            let cos = rcwx_train::cosine_pub(&got, &want);
            println!(
                "encq stage {name}: cos={cos:.6} (len {} vs {})",
                got.len(),
                want.len()
            );
            if key == "cond" && cos < 0.999 {
                // CPU 検算
                let wq = st
                    .store
                    .entries
                    .iter()
                    .find(|e| e.name == "enc_q.enc.cond_layer.weight_v")
                    .unwrap();
                let gq = st
                    .store
                    .entries
                    .iter()
                    .find(|e| e.name == "enc_q.enc.cond_layer.weight_g")
                    .unwrap();
                let vb = eng.readback_f32(&wq.value);
                let gb = eng.readback_f32(&gq.value);
                let bq = st
                    .store
                    .entries
                    .iter()
                    .find(|e| e.name == "enc_q.enc.cond_layer.bias")
                    .unwrap();
                let bb = eng.readback_f32(&bq.value);
                let gin = c.gin_channels as usize;
                let rows = (2 * h * c.encq_wn_layers) as usize;
                let g_in = &inputs["g"].0;
                let mut cpu = vec![0.0f32; batch as usize * rows];
                for b in 0..batch as usize {
                    for r in 0..rows {
                        let n = (0..gin)
                            .map(|j| vb[r * gin + j] * vb[r * gin + j])
                            .sum::<f32>()
                            .sqrt();
                        let mut acc = bb[r];
                        for j in 0..gin {
                            acc += (vb[r * gin + j] * gb[r] / n) * g_in[b * gin + j];
                        }
                        cpu[b * rows + r] = acc;
                    }
                }
                println!(
                    "  cond cpu cos={:.6} cpu[0..4]={:?} want[0..4]={:?}",
                    rcwx_train::cosine_pub(&cpu, &want),
                    &cpu[0..4],
                    &want[0..4]
                );
            }
        }
    }

    // flow 段階検証 (別 tape)
    {
        let mut tape = Tape::new(&eng);
        let a = st.adopt(&mut tape);
        let mask = tape.const_f32s(&inputs["mask"].0, vec![batch as usize, t_len as usize]);
        let g = tape.const_f32s(&inputs["g"].0, vec![(batch * c.gin_channels) as usize]);
        let z_in = tape.const_f32s(
            &want_out["z"].0,
            vec![batch as usize, c.inter_channels as usize, t_len as usize],
        );
        let dbg = load_golden(&path, "dbg.");
        let mut cur = z_in;
        let mut stage_t: Vec<(String, rcwx_nn::engine::GpuTensor)> = Vec::new();
        for i in 0..c.n_flows {
            let inter = c.inter_channels;
            let half = inter / 2;
            let h = c.hidden_channels;
            let fp = format!("flow.flows.{}", i * 2);
            let x0 = tape.chslice(cur, inter, 0, half, batch, t_len);
            let x1 = tape.chslice(cur, inter, half, half, batch, t_len);
            let hh = rcwx_train::modules::conv1x1(
                &mut tape,
                &a,
                &format!("{fp}.pre"),
                x0,
                h,
                half,
                batch,
                t_len,
            );
            let hh = tape.mul_frame_mask(hh, mask, h, t_len);
            let hh = rcwx_train::modules::wn(
                &mut tape,
                &a,
                &format!("{fp}.enc"),
                hh,
                h,
                c.flow_kernel,
                c.flow_wn_layers,
                batch,
                t_len,
                Some(g),
                Some(mask),
            );
            let m = rcwx_train::modules::conv1x1(
                &mut tape,
                &a,
                &format!("{fp}.post"),
                hh,
                half,
                h,
                batch,
                t_len,
            );
            let m = tape.mul_frame_mask(m, mask, half, t_len);
            let x1n = tape.add(x1, m, false);
            let x1n = tape.mul_frame_mask(x1n, mask, half, t_len);
            let cat = tape.chcat(x0, x1n, half, half, batch, t_len);
            stage_t.push((format!("cat{i}"), tape.var_tensor(cat)));
            cur = tape.flip_ch(cat, batch, inter, t_len);
            stage_t.push((format!("flip{i}"), tape.var_tensor(cur)));
        }
        let cur_t = tape.var_tensor(cur);
        tape.submit().expect("submit");
        for (name, t) in &stage_t {
            let got = eng.readback_f32(t);
            // cat{i} は flow{2i}、flip{i} は flow{2i+1} に対応
            let idx: usize = if name.starts_with("cat") {
                name[3..].parse::<usize>().unwrap() * 2
            } else {
                name[4..].parse::<usize>().unwrap() * 2 + 1
            };
            let want = dbg[&format!("flow{idx}")].0.clone();
            println!(
                "flow stage {name}: cos={:.6}",
                rcwx_train::cosine_pub(&got, &want)
            );
        }
        let got = eng.readback_f32(&cur_t);
        let want = dbg[&format!("flow{}", c.n_flows * 2 - 1)].0.clone();
        println!("flow final: cos={:.6}", rcwx_train::cosine_pub(&got, &want));
        // 各段階
        for i in 0..(c.n_flows * 2) as usize {
            let want_i = dbg[&format!("flow{i}")].0.clone();
            let _ = want_i;
        }
    }

    // decoder 段階検証 (別 tape)
    {
        let mut tape = Tape::new(&eng);
        let a = st.adopt(&mut tape);
        let g = tape.const_f32s(&inputs["g"].0, vec![(batch * c.gin_channels) as usize]);
        let har = tape.const_f32s(&inputs["har"].0, vec![(batch * t_len * c.upp()) as usize]);
        let z_in = tape.const_f32s(
            &want_out["z"].0,
            vec![batch as usize, c.inter_channels as usize, t_len as usize],
        );
        let dbg = load_golden(&path, "dbg.");
        let stages = rcwx_train::modules::synth_train::decoder_stages(
            &mut tape, &a, &c, z_in, har, g, batch, t_len,
        );
        let stage_ts: Vec<(String, rcwx_nn::engine::GpuTensor)> = stages
            .iter()
            .map(|(n, v)| (n.clone(), tape.var_tensor(*v)))
            .collect();
        tape.submit().expect("submit");
        for (name, t) in &stage_ts {
            let Some((want, _)) = dbg.get(name) else {
                continue;
            };
            let got = eng.readback_f32(t);
            let cos = rcwx_train::cosine_pub(&got, want);
            println!("dec stage {name}: cos={cos:.6}",);
            if name == "d_lrelu_final" && cos >= 0.9999 {
                // CPU: conv_post + tanh
                let we = st
                    .store
                    .entries
                    .iter()
                    .find(|e| e.name == "dec.conv_post.weight")
                    .unwrap();
                let wb = eng.readback_f32(&we.value);
                let (oc, ic, kk, pp) = (1usize, 4, 7, 3usize);
                let tt = want.len() / (batch as usize * ic);
                let mut cpu = vec![0.0f32; batch as usize * oc * tt];
                for b in 0..batch as usize {
                    for o in 0..oc {
                        for p in 0..tt {
                            let mut acc = 0.0f32;
                            for c in 0..ic {
                                for j in 0..kk {
                                    let pos = p as i64 - pp as i64 + j as i64;
                                    if pos >= 0 && (pos as usize) < tt {
                                        acc += wb[o * ic * kk + c * kk + j]
                                            * want[b * ic * tt + c * tt + pos as usize];
                                    }
                                }
                            }
                            cpu[b * oc * tt + o * tt + p] = acc.tanh();
                        }
                    }
                }
                let yh = &want_out["y_hat"].0;
                println!(
                    "  conv_post cpu vs y_hat: cos={:.6} cpu[0..3]={:?} want[0..3]={:?}",
                    rcwx_train::cosine_pub(&cpu, yh),
                    &cpu[0..3],
                    &yh[0..3]
                );
            }
        }
    }

    let (z, m_q, logs_q) = enc_q(
        &mut tape, &a, &c, spec, spec_bins, g, mask, randn, batch, t_len,
    );
    let z_p = flow_forward(&mut tape, &a, &c, z, g, mask, batch, t_len);
    let y_hat = decoder(&mut tape, &a, &c, z, har, g, batch, t_len);

    let z_t = tape.var_tensor(z);
    let zp_t = tape.var_tensor(z_p);
    let mq_t = tape.var_tensor(m_q);
    let lq_t = tape.var_tensor(logs_q);
    let yh_t = tape.var_tensor(y_hat);

    // loss = Σy_hat + Σz_p + Σm_q + Σlogs_q (golden と同一)
    tape.backward_from_all(&[y_hat, z_p, m_q, logs_q]);
    tape.submit().expect("submit");

    // 全出力の cos を先に算出
    let mut all_cos: Vec<(String, f64)> = Vec::new();
    for (name, got_t, want_key) in [
        ("m_q", &mq_t, "m_q"),
        ("logs_q", &lq_t, "logs_q"),
        ("z", &z_t, "z"),
        ("z_p", &zp_t, "z_p"),
        ("y_hat", &yh_t, "y_hat"),
    ] {
        let got = eng.readback_f32(got_t);
        let (want, _) = &want_out[want_key];
        let cos = rcwx_train::cosine_pub(&got, want);
        all_cos.push((name.to_string(), cos));
        println!(
            "gen fwd {name}: cos={cos:.6} (len {} vs {})",
            got.len(),
            want.len()
        );
        if cos < 0.999 {
            let mut worst = (0usize, 0.0f32);
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                let d = (g - w).abs();
                if d > worst.1 {
                    worst = (i, d);
                }
            }
            let inter = c.inter_channels as usize;
            let t_len_u = t_len as usize;
            let (ch, b, t) = (
                worst.0 / (batch as usize * t_len_u),
                (worst.0 / t_len_u) % batch as usize,
                worst.0 % t_len_u,
            );
            println!(
                "  worst i={} (b={b},ch={ch},t={t}) got={:.5} want={:.5}",
                worst.0, got[worst.0], want[worst.0]
            );
        }
    }
    for (name, cos) in &all_cos {
        assert!(*cos >= 0.999, "{name} parity {cos}");
    }

    // 勾配比較 (代表的)
    let checks = [
        "enc_q.pre.weight",
        "enc_q.enc.cond_layer.weight",
        "enc_q.enc.in_layers.0.weight",
        "enc_q.enc.res_skip_layers.0.weight",
        "enc_q.proj.weight",
        "flow.flows.0.pre.weight",
        "flow.flows.0.enc.in_layers.1.weight",
        "flow.flows.0.post.weight",
        "flow.flows.2.enc.cond_layer.weight",
        "dec.conv_pre.weight",
        "dec.cond.weight",
        "dec.ups.0.weight",
        "dec.ups.1.weight",
        "dec.noise_convs.0.weight",
        "dec.noise_convs.1.weight",
        "dec.resblocks.0.convs1.0.weight",
        "dec.resblocks.0.convs2.1.weight",
        "dec.resblocks.2.convs1.1.weight",
        "dec.conv_post.weight",
    ];
    for name in checks {
        let Some((want_g, _)) = want_grads.get(name) else {
            eprintln!("grad golden missing: {name}");
            continue;
        };
        let Some(entry) = st.store.entries.iter().find(|e| e.name == name) else {
            panic!("param not found: {name}");
        };
        let got_g = eng.readback_f32(&entry.grad);
        let nz_g = want_g.iter().filter(|v| v.abs() > 1e-9).count();
        let cos = rcwx_train::cosine_pub(&got_g, want_g);
        println!("gen grad {name}: cos={cos:.6} (want_nz={nz_g})",);
        if nz_g > 0 {
            assert!(cos >= 0.999, "grad parity {name}: {cos}");
        }
    }
}
