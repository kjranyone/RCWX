//! M-T4 smoke: 合成データで数ステップ学習し mel 損失が低下するか確認。
//! 使い方: cargo run -p rcwx-train --release --bin train_smoke

use anyhow::Result;
use rcwx_train::modules::synth_train::{SynthTrainCfg, Weights};
use rcwx_train::train_loop::{Batch, StepInputs, Trainer};

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (1u64 << 33) as f32 - 0.5) * 2.0
}

fn tiny_cfg() -> SynthTrainCfg {
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

fn put(gw: &mut Weights, st: &mut u64, name: &str, n: usize, shape: Vec<usize>) {
    // RVC の init_weights (normal 0.01) 相当に小さく (tanh 飽和回避)
    let d: Vec<f32> = (0..n).map(|_| lcg(st) * 0.05).collect();
    gw.insert(name.to_string(), (d, shape));
}

fn gen_g_weights() -> Weights {
    let mut gw = Weights::new();
    let mut gs = 12345u64;
    put(&mut gw, &mut gs, "emb_g.weight", 4, vec![1, 4]);
    put(
        &mut gw,
        &mut gs,
        "enc_p.emb_phone.weight",
        8 * 8,
        vec![8, 8],
    );
    put(
        &mut gw,
        &mut gs,
        "enc_p.emb_pitch.weight",
        256 * 8,
        vec![256, 8],
    );
    for l in 0..1 {
        for c in ["conv_q", "conv_k", "conv_v", "conv_o"] {
            put(
                &mut gw,
                &mut gs,
                &format!("enc_p.encoder.attn_layers.{l}.{c}.weight"),
                64,
                vec![8, 8, 1],
            );
            put(
                &mut gw,
                &mut gs,
                &format!("enc_p.encoder.attn_layers.{l}.{c}.bias"),
                8,
                vec![8],
            );
        }
        put(
            &mut gw,
            &mut gs,
            &format!("enc_p.encoder.attn_layers.{l}.emb_rel_k"),
            7 * 4,
            vec![1, 7, 4],
        );
        put(
            &mut gw,
            &mut gs,
            &format!("enc_p.encoder.attn_layers.{l}.emb_rel_v"),
            7 * 4,
            vec![1, 7, 4],
        );
        for nl in ["norm_layers_1", "norm_layers_2"] {
            put(
                &mut gw,
                &mut gs,
                &format!("enc_p.encoder.{nl}.{l}.gamma"),
                8,
                vec![8],
            );
            put(
                &mut gw,
                &mut gs,
                &format!("enc_p.encoder.{nl}.{l}.beta"),
                8,
                vec![8],
            );
        }
        for (ci, c) in ["conv_1", "conv_2"].iter().enumerate() {
            let (oc, ic) = if ci == 0 { (16, 8) } else { (8, 16) };
            put(
                &mut gw,
                &mut gs,
                &format!("enc_p.encoder.ffn_layers.{l}.{c}.weight"),
                oc * ic * 3,
                vec![oc, ic, 3],
            );
            put(
                &mut gw,
                &mut gs,
                &format!("enc_p.encoder.ffn_layers.{l}.{c}.bias"),
                oc,
                vec![oc],
            );
        }
    }
    put(
        &mut gw,
        &mut gs,
        "enc_p.proj.weight",
        12 * 8,
        vec![12, 8, 1],
    );
    put(&mut gw, &mut gs, "enc_p.proj.bias", 12, vec![12]);
    put(&mut gw, &mut gs, "enc_q.pre.weight", 8 * 16, vec![8, 16, 1]);
    put(&mut gw, &mut gs, "enc_q.pre.bias", 8, vec![8]);
    put(
        &mut gw,
        &mut gs,
        "enc_q.enc.cond_layer.weight_v",
        8 * 4,
        vec![8, 4, 1],
    );
    put(
        &mut gw,
        &mut gs,
        "enc_q.enc.cond_layer.weight_g",
        8,
        vec![8, 1, 1],
    );
    put(&mut gw, &mut gs, "enc_q.enc.cond_layer.bias", 8, vec![8]);
    for l in 0..2 {
        put(
            &mut gw,
            &mut gs,
            &format!("enc_q.enc.in_layers.{l}.weight_v"),
            16 * 8 * 5,
            vec![16, 8, 5],
        );
        put(
            &mut gw,
            &mut gs,
            &format!("enc_q.enc.in_layers.{l}.weight_g"),
            16,
            vec![16, 1, 1],
        );
        put(
            &mut gw,
            &mut gs,
            &format!("enc_q.enc.in_layers.{l}.bias"),
            16,
            vec![16],
        );
        let rs = if l < 1 { 16 } else { 8 };
        put(
            &mut gw,
            &mut gs,
            &format!("enc_q.enc.res_skip_layers.{l}.weight_v"),
            rs * 8,
            vec![rs, 8, 1],
        );
        put(
            &mut gw,
            &mut gs,
            &format!("enc_q.enc.res_skip_layers.{l}.weight_g"),
            rs,
            vec![rs, 1, 1],
        );
        put(
            &mut gw,
            &mut gs,
            &format!("enc_q.enc.res_skip_layers.{l}.bias"),
            rs,
            vec![rs],
        );
    }
    put(
        &mut gw,
        &mut gs,
        "enc_q.proj.weight",
        12 * 8,
        vec![12, 8, 1],
    );
    put(&mut gw, &mut gs, "enc_q.proj.bias", 12, vec![12]);
    for f in [0usize, 2] {
        put(
            &mut gw,
            &mut gs,
            &format!("flow.flows.{f}.pre.weight"),
            8 * 3,
            vec![8, 3, 1],
        );
        put(
            &mut gw,
            &mut gs,
            &format!("flow.flows.{f}.pre.bias"),
            8,
            vec![8],
        );
        put(
            &mut gw,
            &mut gs,
            &format!("flow.flows.{f}.enc.cond_layer.weight_v"),
            8 * 4,
            vec![8, 4, 1],
        );
        put(
            &mut gw,
            &mut gs,
            &format!("flow.flows.{f}.enc.cond_layer.weight_g"),
            8,
            vec![8, 1, 1],
        );
        put(
            &mut gw,
            &mut gs,
            &format!("flow.flows.{f}.enc.cond_layer.bias"),
            8,
            vec![8],
        );
        for l in 0..2 {
            put(
                &mut gw,
                &mut gs,
                &format!("flow.flows.{f}.enc.in_layers.{l}.weight_v"),
                16 * 8 * 5,
                vec![16, 8, 5],
            );
            put(
                &mut gw,
                &mut gs,
                &format!("flow.flows.{f}.enc.in_layers.{l}.weight_g"),
                16,
                vec![16, 1, 1],
            );
            put(
                &mut gw,
                &mut gs,
                &format!("flow.flows.{f}.enc.in_layers.{l}.bias"),
                16,
                vec![16],
            );
            let rs = if l < 1 { 16 } else { 8 };
            put(
                &mut gw,
                &mut gs,
                &format!("flow.flows.{f}.enc.res_skip_layers.{l}.weight_v"),
                rs * 8,
                vec![rs, 8, 1],
            );
            put(
                &mut gw,
                &mut gs,
                &format!("flow.flows.{f}.enc.res_skip_layers.{l}.weight_g"),
                rs,
                vec![rs, 1, 1],
            );
            put(
                &mut gw,
                &mut gs,
                &format!("flow.flows.{f}.enc.res_skip_layers.{l}.bias"),
                rs,
                vec![rs],
            );
        }
        put(
            &mut gw,
            &mut gs,
            &format!("flow.flows.{f}.post.weight"),
            3 * 8,
            vec![3, 8, 1],
        );
        put(
            &mut gw,
            &mut gs,
            &format!("flow.flows.{f}.post.bias"),
            3,
            vec![3],
        );
    }
    put(
        &mut gw,
        &mut gs,
        "dec.conv_pre.weight",
        16 * 6 * 7,
        vec![16, 6, 7],
    );
    put(&mut gw, &mut gs, "dec.conv_pre.bias", 16, vec![16]);
    put(&mut gw, &mut gs, "dec.cond.weight", 16 * 4, vec![16, 4, 1]);
    put(&mut gw, &mut gs, "dec.cond.bias", 16, vec![16]);
    let chans = [8usize, 4];
    let cin_s = [16usize, 8];
    let ks = [8usize, 4];
    for i in 0..2 {
        put(
            &mut gw,
            &mut gs,
            &format!("dec.ups.{i}.weight_v"),
            cin_s[i] * chans[i] * ks[i],
            vec![cin_s[i], chans[i], ks[i]],
        );
        put(
            &mut gw,
            &mut gs,
            &format!("dec.ups.{i}.weight_g"),
            cin_s[i],
            vec![cin_s[i], 1, 1],
        );
        put(
            &mut gw,
            &mut gs,
            &format!("dec.ups.{i}.bias"),
            chans[i],
            vec![chans[i]],
        );
        put(
            &mut gw,
            &mut gs,
            &format!("dec.noise_convs.{i}.weight"),
            chans[i] * [4, 1][i],
            vec![chans[i], 1, [4, 1][i]],
        );
        put(
            &mut gw,
            &mut gs,
            &format!("dec.noise_convs.{i}.bias"),
            chans[i],
            vec![chans[i]],
        );
        for j in 0..2 {
            let k = [3usize, 7][j];
            for c in ["convs1", "convs2"] {
                for d in 0..2 {
                    let kd = if c == "convs1" { k } else { 3 };
                    put(
                        &mut gw,
                        &mut gs,
                        &format!("dec.resblocks.{}.{}.{}.weight_v", i * 2 + j, c, d),
                        chans[i] * chans[i] * kd,
                        vec![chans[i], chans[i], kd],
                    );
                    put(
                        &mut gw,
                        &mut gs,
                        &format!("dec.resblocks.{}.{}.{}.weight_g", i * 2 + j, c, d),
                        chans[i],
                        vec![chans[i], 1, 1],
                    );
                    put(
                        &mut gw,
                        &mut gs,
                        &format!("dec.resblocks.{}.{}.{}.bias", i * 2 + j, c, d),
                        chans[i],
                        vec![chans[i]],
                    );
                }
            }
        }
    }
    put(
        &mut gw,
        &mut gs,
        "dec.conv_post.weight",
        4 * 7,
        vec![1, 4, 7],
    );
    gw
}

fn synth_target(t: usize, sr: u32, seed: u64) -> Vec<f32> {
    let mut st = seed;
    let f0 = 140.0 + 60.0 * lcg(&mut st);
    let mut out = Vec::with_capacity(t);
    for i in 0..t {
        let phase = 2.0 * std::f32::consts::PI * f0 * i as f32 / sr as f32;
        out.push(0.5 * phase.sin() + 0.05 * lcg(&mut st));
    }
    out
}

fn main() -> Result<()> {
    let eng = rcwx_nn::engine::Engine::new(rcwx_nn::engine::Backend::Vulkan).expect("engine");
    let cfg = tiny_cfg();
    let gw = gen_g_weights();
    // D は空 (smoke では GAN 損失なし — mel+KL のみ)
    let dw = rcwx_train::modules::mpd::MpdWeights::new();
    let mut trainer = Trainer::new(&eng, cfg, &gw, &dw)?;

    let (batch, t_len, seg_t) = (2usize, 11, 11);
    let seg_samples = seg_t * 8;
    let mut st = 999u64;
    let phone: Vec<f32> = (0..batch * t_len * 8).map(|_| lcg(&mut st)).collect();
    let pitch: Vec<u32> = (0..batch * t_len)
        .map(|_| 100 + (lcg(&mut st) * 100.0) as u32)
        .collect();
    let spec_bins = 16;
    let spec: Vec<f32> = (0..batch * spec_bins * t_len)
        .map(|_| lcg(&mut st) * 0.5)
        .collect();
    let wave1 = synth_target(seg_samples, 48000, 777);
    let wave: Vec<f32> = (0..batch).flat_map(|_| wave1.iter().copied()).collect();
    let har: Vec<f32> = (0..batch * seg_samples)
        .map(|_| lcg(&mut st) * 0.1)
        .collect();
    let randn: Vec<f32> = (0..batch * 6 * t_len).map(|_| lcg(&mut st)).collect();
    let start: Vec<u32> = vec![0; batch];
    let b = Batch {
        phone,
        pitch,
        spec,
        wave: wave.clone(),
        lengths: vec![t_len as u32; batch],
        t_len: t_len as u32,
        seg_t: seg_t as u32,
        spec_bins: spec_bins as u32,
    };
    let inp = StepInputs {
        randn: &randn,
        har: &har,
        start_frames: &start,
    };

    let mut first_mel = f32::NAN;
    for it in 0..40 {
        let l = trainer.step(&b, &inp)?;
        if it == 0 {
            first_mel = l.mel;
        }
        if it % 5 == 0 || it == 39 {
            println!(
                "it {it}: disc={:.4} gen={:.4} fm={:.4} mel={:.4} kl={:.4}",
                l.disc, l.gen_adv, l.fm, l.mel, l.kl
            );
        }
    }
    println!("mel: first={first_mel:.4}");
    std::fs::create_dir_all("target/smoke")?;
    rcwx_train::data::preprocess::save_f32(
        std::path::Path::new("target/smoke/loss.safetensors"),
        &[first_mel],
    )?;
    Ok(())
}
