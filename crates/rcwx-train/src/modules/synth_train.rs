//! 学習モード Synthesizer (enc_p + enc_q + flow 順方向 + GeneratorNSF) の
//! tape 実装。重み名は PyTorch state_dict キーと同一 (weight_norm は _v/_g 保持)。

use std::collections::HashMap;

use rcwx_nn::engine::Engine;
use rcwx_nn::engine::tape::{ParamStore, Tape, Var};

use crate::modules::{Adopted, attention_layer, conv1d, conv1x1, ffn, wn};

#[derive(Clone)]
pub struct SynthTrainCfg {
    pub in_channels: u32,
    pub inter_channels: u32,
    pub hidden_channels: u32,
    pub filter_channels: u32,
    pub n_heads: u32,
    pub n_layers: u32,
    pub kernel_size: u32,
    pub window_size: u32,
    pub resblock_kernel_sizes: Vec<u32>,
    pub resblock_dilation_sizes: Vec<Vec<u32>>,
    pub upsample_rates: Vec<u32>,
    pub upsample_kernel_sizes: Vec<u32>,
    pub upsample_initial_channel: u32,
    pub gin_channels: u32,
    pub n_speakers: u32,
    pub sr: u32,
    /// flow の coupling 数 (RVC v2: 4)
    pub n_flows: u32,
    /// flow WN の層数 (RVC: 3) と kernel (5)
    pub flow_wn_layers: u32,
    pub flow_kernel: u32,
    /// enc_q WN の層数 (RVC: 16) と kernel (5)
    pub encq_wn_layers: u32,
    pub encq_kernel: u32,
}

impl SynthTrainCfg {
    /// RVC v2 48k 既定値。
    pub fn v2_48k() -> Self {
        Self {
            in_channels: 768,
            inter_channels: 192,
            hidden_channels: 192,
            filter_channels: 768,
            n_heads: 2,
            n_layers: 6,
            kernel_size: 3,
            window_size: 10,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5]; 3],
            upsample_rates: vec![12, 10, 2, 2],
            upsample_kernel_sizes: vec![24, 20, 4, 4],
            upsample_initial_channel: 512,
            gin_channels: 256,
            n_speakers: 1,
            sr: 48000,
            n_flows: 4,
            flow_wn_layers: 3,
            flow_kernel: 5,
            encq_wn_layers: 16,
            encq_kernel: 5,
        }
    }

    pub fn upp(&self) -> u32 {
        self.upsample_rates.iter().product()
    }
}

#[derive(Clone, Copy)]
pub enum ConvRef {
    Plain(usize),
    Wn(usize),
}

pub type Weights = HashMap<String, (Vec<f32>, Vec<usize>)>;

pub struct SynthTrain {
    pub cfg: SynthTrainCfg,
    pub store: ParamStore,
    pub convs: HashMap<String, ConvRef>,
    pub vecs: HashMap<String, usize>,
}

impl SynthTrain {
    pub fn new(eng: &Engine, cfg: SynthTrainCfg, w: &Weights) -> anyhow::Result<Self> {
        let mut store = ParamStore::new();
        let mut convs = HashMap::new();
        let mut vecs = HashMap::new();
        let keys: Vec<String> = w.keys().cloned().collect();
        for k in keys {
            // WN 畳み込み: .weight_v を起点に登録 (plain .weight キーは存在しない)
            if k.ends_with(".weight_v") {
                let prefix = k.trim_end_matches(".weight_v").to_string();
                if store
                    .wns
                    .iter()
                    .any(|wn| store.entries[wn.v_idx].name == format!("{prefix}.weight_v"))
                {
                    continue;
                }
                let (v, vs) = &w[&k];
                let g = w
                    .get(&format!("{prefix}.weight_g"))
                    .map(|(gv, _)| gv.as_slice())
                    .unwrap_or(&[1.0]);
                let idx = store.add_weight_norm(eng, &prefix, v, g, vs.clone());
                convs.insert(prefix, ConvRef::Wn(idx));
                continue;
            }
            if k.ends_with(".weight_g") {
                continue;
            }
            let (d, s) = &w[&k];
            if k.ends_with(".weight") {
                let prefix = k.trim_end_matches(".weight").to_string();
                if s.len() <= 2 || k.contains("emb_rel") {
                    // Linear / embedding の重み (1 次元)
                    vecs.insert(k.clone(), store.add(eng, &k, d, s.clone()));
                    continue;
                }
                if let Some((v, vs)) = w.get(&format!("{prefix}.weight_v")) {
                    let g = w
                        .get(&format!("{prefix}.weight_g"))
                        .map(|(gv, _)| gv.as_slice())
                        .unwrap_or(&[1.0]);
                    let idx = store.add_weight_norm(eng, &prefix, v, g, vs.clone());
                    convs.insert(prefix, ConvRef::Wn(idx));
                } else {
                    convs.insert(prefix, ConvRef::Plain(store.add(eng, &k, d, s.clone())));
                }
            } else if k.ends_with(".bias") {
                vecs.insert(k.clone(), store.add(eng, &k, d, s.clone()));
            } else {
                // bare Parameter (LayerNorm gamma/beta 等)
                vecs.insert(k.clone(), store.add(eng, &k, d, s.clone()));
            }
        }
        Ok(Self {
            cfg,
            store,
            convs,
            vecs,
        })
    }

    /// 毎ステップ冒頭: 全パラメータ adopt + WN fold。
    pub fn adopt(&self, tape: &mut Tape<'_>) -> Adopted {
        let mut a = Adopted::default();
        for (prefix, r) in &self.convs {
            match r {
                ConvRef::Plain(i) => {
                    let v = a.adopt_entry(tape, &self.store, *i);
                    a.p.insert(format!("{prefix}.w"), v);
                }
                ConvRef::Wn(i) => {
                    let wn = &self.store.wns[*i];
                    let v = a.adopt_entry(tape, &self.store, wn.v_idx);
                    let g = a.adopt_entry(tape, &self.store, wn.g_idx);
                    let w = tape.adopt_wn(&self.store, *i, v, g);
                    a.p.insert(format!("{prefix}.w"), w);
                }
            }
        }
        for (name, i) in &self.vecs {
            let v = a.adopt_entry(tape, &self.store, *i);
            a.p.insert(name.clone(), v);
        }
        a
    }
}

/// TextEncoder (enc_p)。
#[allow(clippy::too_many_arguments)]
pub fn enc_p(
    tape: &mut Tape<'_>,
    a: &Adopted,
    cfg: &SynthTrainCfg,
    phone: Var,
    pitch_onehot: Var,
    mask: Var,
    attn_bias: Var,
    batch: u32,
    t_len: u32,
) -> EncPOut {
    let h = cfg.hidden_channels;
    let bt = batch * t_len;
    // 埋め込みは [B, T, in] × W[in, h] → [B, T, h] (batch=B・m=T で shape 整合)
    let x_emb = {
        let w = a.vec("enc_p.emb_phone.weight");
        tape.mm(
            phone,
            w,
            a.bias("enc_p.emb_phone"),
            (bt * h) as usize,
            t_len,
            h,
            cfg.in_channels,
            batch,
            cfg.in_channels,
            1,
            0,
            t_len * cfg.in_channels,
            cfg.in_channels,
            1,
            0,
            0,
            h,
            1,
            t_len * h,
            1.0,
            0,
        )
    };
    let xp = {
        let emb = a.vec("enc_p.emb_pitch.weight");
        tape.mm(
            pitch_onehot,
            emb,
            None,
            (bt * h) as usize,
            t_len,
            h,
            256,
            batch,
            256,
            1,
            0,
            t_len * 256,
            1,
            h,
            0,
            0,
            h,
            1,
            t_len * h,
            1.0,
            0,
        )
    };
    let xsum = tape.add(x_emb, xp, false);
    let xs = tape.pw(xsum, 7, (h as f32).sqrt());
    let xa = tape.lrelu(xs, 0.1);
    let xt_dbg = tape.transpose_tc_ct(xa, t_len, h);
    let mut x = tape.mul_frame_mask(xt_dbg, mask, h, t_len);
    let x0 = x;
    for i in 0..cfg.n_layers {
        let y = attention_layer(
            tape,
            a,
            &format!("enc_p.encoder.attn_layers.{i}"),
            x,
            h,
            cfg.n_heads,
            cfg.window_size,
            batch,
            t_len,
            Some(attn_bias),
        );
        let x2 = tape.add(x, y, false);
        let x2 = tape.layernorm_strided(
            x2,
            a.vec(&format!("enc_p.encoder.norm_layers_1.{i}.gamma")),
            a.vec(&format!("enc_p.encoder.norm_layers_1.{i}.beta")),
            batch * t_len,
            h,
            1,
            t_len,
            1e-5,
        );
        let y = ffn(
            tape,
            a,
            &format!("enc_p.encoder.ffn_layers.{i}"),
            x2,
            h,
            cfg.filter_channels,
            cfg.kernel_size,
            batch,
            t_len,
            Some(mask),
        );
        let x3 = tape.add(x2, y, false);
        x = tape.layernorm_strided(
            x3,
            a.vec(&format!("enc_p.encoder.norm_layers_2.{i}.gamma")),
            a.vec(&format!("enc_p.encoder.norm_layers_2.{i}.beta")),
            batch * t_len,
            h,
            1,
            t_len,
            1e-5,
        );
    }
    x = tape.mul_frame_mask(x, mask, h, t_len);
    let stats = conv1x1(
        tape,
        a,
        "enc_p.proj",
        x,
        2 * cfg.inter_channels,
        h,
        batch,
        t_len,
    );
    let stats = tape.mul_frame_mask(stats, mask, 2 * cfg.inter_channels, t_len);
    let m_p = tape.chslice(
        stats,
        2 * cfg.inter_channels,
        0,
        cfg.inter_channels,
        batch,
        t_len,
    );
    let logs_p = tape.chslice(
        stats,
        2 * cfg.inter_channels,
        cfg.inter_channels,
        cfg.inter_channels,
        batch,
        t_len,
    );
    EncPOut {
        m_p,
        logs_p,
        x0,
        xt: xt_dbg,
        xs,
        xsum,
        x_emb,
        xp,
    }
}

/// enc_p の出力とデバッグ用中間変数。
pub struct EncPOut {
    pub m_p: Var,
    pub logs_p: Var,
    pub x0: Var,
    pub xt: Var,
    pub xs: Var,
    pub xsum: Var,
    pub x_emb: Var,
    pub xp: Var,
}

/// PosteriorEncoder (enc_q)。戻り値 (z, m_q, logs_q)。
#[allow(clippy::too_many_arguments)]
pub fn enc_q(
    tape: &mut Tape<'_>,
    a: &Adopted,
    cfg: &SynthTrainCfg,
    spec: Var,
    spec_bins: u32,
    g: Var,
    mask: Var,
    randn: Var,
    batch: u32,
    t_len: u32,
) -> (Var, Var, Var) {
    let h = cfg.hidden_channels;
    let x = conv1x1(tape, a, "enc_q.pre", spec, h, spec_bins, batch, t_len);
    let x = tape.mul_frame_mask(x, mask, h, t_len);
    let x = wn(
        tape,
        a,
        "enc_q.enc",
        x,
        h,
        cfg.encq_kernel,
        cfg.encq_wn_layers,
        batch,
        t_len,
        Some(g),
        Some(mask),
    );
    let stats = conv1x1(
        tape,
        a,
        "enc_q.proj",
        x,
        2 * cfg.inter_channels,
        h,
        batch,
        t_len,
    );
    let stats = tape.mul_frame_mask(stats, mask, 2 * cfg.inter_channels, t_len);
    let m_q = tape.chslice(
        stats,
        2 * cfg.inter_channels,
        0,
        cfg.inter_channels,
        batch,
        t_len,
    );
    let logs_q = tape.chslice(
        stats,
        2 * cfg.inter_channels,
        cfg.inter_channels,
        cfg.inter_channels,
        batch,
        t_len,
    );
    let e = tape.exp(logs_q);
    let ne = tape.mul2(randn, e);
    let z = tape.add(m_q, ne, false);
    let z = tape.mul_frame_mask(z, mask, cfg.inter_channels, t_len);
    (z, m_q, logs_q)
}

/// Flow 順方向 (学習): mean_only coupling + Flip ×4。
#[allow(clippy::too_many_arguments)]
pub fn flow_forward(
    tape: &mut Tape<'_>,
    a: &Adopted,
    cfg: &SynthTrainCfg,
    z: Var,
    g: Var,
    mask: Var,
    batch: u32,
    t_len: u32,
) -> Var {
    let inter = cfg.inter_channels;
    let half = inter / 2;
    let h = cfg.hidden_channels;
    let mut x = z;
    for i in 0..cfg.n_flows {
        let fp = format!("flow.flows.{}", i * 2);
        let x0 = tape.chslice(x, inter, 0, half, batch, t_len);
        let x1 = tape.chslice(x, inter, half, half, batch, t_len);
        let hh = conv1x1(tape, a, &format!("{fp}.pre"), x0, h, half, batch, t_len);
        let hh = tape.mul_frame_mask(hh, mask, h, t_len);
        let hh = wn(
            tape,
            a,
            &format!("{fp}.enc"),
            hh,
            h,
            cfg.flow_kernel,
            cfg.flow_wn_layers,
            batch,
            t_len,
            Some(g),
            Some(mask),
        );
        let m = conv1x1(tape, a, &format!("{fp}.post"), hh, half, h, batch, t_len);
        let m = tape.mul_frame_mask(m, mask, half, t_len);
        let x1n = tape.add(x1, m, false);
        let x1n = tape.mul_frame_mask(x1n, mask, half, t_len);
        let cat = tape.chcat(x0, x1n, half, half, batch, t_len);
        x = tape.flip_ch(cat, batch, inter, t_len);
    }
    x
}

/// GeneratorNSF (decoder)。z_slice [B, inter, seg_t] → y_hat [B·seg_samples]。
#[allow(clippy::too_many_arguments)]
pub fn decoder(
    tape: &mut Tape<'_>,
    a: &Adopted,
    cfg: &SynthTrainCfg,
    z: Var,
    har: Var,
    g: Var,
    batch: u32,
    seg_t: u32,
) -> Var {
    let inter = cfg.inter_channels;
    let mut t = seg_t;
    let mut x = conv1d(
        tape,
        a,
        "dec.conv_pre",
        z,
        cfg.upsample_initial_channel,
        inter,
        7,
        1,
        1,
        3,
        batch,
        t,
    );
    {
        let gin = tape.shape(a.w("dec.cond"))[1] as u32;
        let gcond = tape.mm(
            a.w("dec.cond"),
            g,
            a.bias("dec.cond"),
            (batch * cfg.upsample_initial_channel) as usize,
            cfg.upsample_initial_channel,
            1,
            gin,
            batch,
            gin,
            1,
            0,
            0,
            1,
            1,
            0,
            gin,
            1,
            1,
            cfg.upsample_initial_channel,
            1.0,
            1,
        );
        x = tape.add_bcast_t(x, gcond, cfg.upsample_initial_channel, t);
    }
    let n_up = cfg.upsample_rates.len() as u32;
    let upp = cfg.upp();
    let mut c_in = cfg.upsample_initial_channel;
    for i in 0..n_up {
        let u = cfg.upsample_rates[i as usize];
        let k = cfg.upsample_kernel_sizes[i as usize];
        let c_cur = cfg.upsample_initial_channel / (1 << (i + 1));
        let t_out = (t - 1) * u + k - 2 * ((k - u) / 2);
        x = tape.lrelu(x, 0.1);
        let w = a.w(&format!("dec.ups.{i}"));
        let b = a.bias(&format!("dec.ups.{i}"));
        x = tape.conv_transpose1d(x, w, b, c_cur, c_in, k, u, 1, (k - u) / 2, batch, t, t_out);
        let har_v = if i + 1 < n_up {
            let stride_f0: u32 = cfg.upsample_rates[(i + 1) as usize..].iter().product();
            let nk = 2 * stride_f0;
            let t_in = seg_t * upp;
            let t_n = (t_in + 2 * (stride_f0 / 2)).saturating_sub(nk) / stride_f0 + 1;
            let hw = a.w(&format!("dec.noise_convs.{i}"));
            let hb = a.bias(&format!("dec.noise_convs.{i}"));
            tape.conv1d(
                har,
                hw,
                hb,
                c_cur,
                1,
                nk,
                stride_f0,
                1,
                stride_f0 / 2,
                batch,
                t_in,
                t_n,
            )
        } else {
            let hw = a.w(&format!("dec.noise_convs.{i}"));
            let hb = a.bias(&format!("dec.noise_convs.{i}"));
            let t_in = seg_t * upp;
            tape.conv1d(har, hw, hb, c_cur, 1, 1, 1, 1, 0, batch, t_in, t_in)
        };
        x = tape.add(x, har_v, false);
        let n_k = cfg.resblock_kernel_sizes.len() as u32;
        let mut xs: Option<Var> = None;
        for j in 0..n_k {
            let rb = resblock1(
                tape,
                a,
                &format!("dec.resblocks.{}", i * n_k + j),
                x,
                c_cur,
                cfg.resblock_kernel_sizes[j as usize],
                &cfg.resblock_dilation_sizes[j as usize],
                batch,
                t_out,
            );
            xs = Some(match xs {
                Some(s) => tape.add(s, rb, false),
                None => rb,
            });
        }
        x = tape.pw(xs.expect("resblocks"), 7, 1.0 / n_k as f32);
        c_in = c_cur;
        t = t_out;
    }
    // RVC の quirk: conv_post 直前の leaky_relu はデフォルト slope 0.01
    let x = tape.lrelu(x, 0.01);
    let x = conv1d(tape, a, "dec.conv_post", x, 1, c_in, 7, 1, 1, 3, batch, t);
    tape.tanh(x)
}

#[allow(clippy::too_many_arguments)]
fn resblock1(
    tape: &mut Tape<'_>,
    a: &Adopted,
    prefix: &str,
    x: Var,
    ch: u32,
    kernel: u32,
    dilations: &[u32],
    batch: u32,
    t_len: u32,
) -> Var {
    let mut x = x;
    for (li, &d) in dilations.iter().enumerate() {
        let pad1 = d * (kernel - 1) / 2;
        let xt = tape.lrelu(x, 0.1);
        let xt = conv1d(
            tape,
            a,
            &format!("{prefix}.convs1.{li}"),
            xt,
            ch,
            ch,
            kernel,
            1,
            d,
            pad1,
            batch,
            t_len,
        );
        let xt = tape.lrelu(xt, 0.1);
        let xt = conv1d(
            tape,
            a,
            &format!("{prefix}.convs2.{li}"),
            xt,
            ch,
            ch,
            kernel,
            1,
            1,
            (kernel - 1) / 2,
            batch,
            t_len,
        );
        x = tape.add(x, xt, false);
    }
    x
}

/// decoder の段階出力 (検証用)。decoder() と同一計算で中間 Var を返す。
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn decoder_stages(
    tape: &mut Tape<'_>,
    a: &Adopted,
    cfg: &SynthTrainCfg,
    z: Var,
    har: Var,
    g: Var,
    batch: u32,
    seg_t: u32,
) -> Vec<(String, Var)> {
    let inter = cfg.inter_channels;
    let mut t = seg_t;
    let mut x = conv1d(
        tape,
        a,
        "dec.conv_pre",
        z,
        cfg.upsample_initial_channel,
        inter,
        7,
        1,
        1,
        3,
        batch,
        t,
    );
    let mut out: Vec<(String, Var)> = vec![("d_pre".into(), x)];
    {
        let gin = tape.shape(a.w("dec.cond"))[1] as u32;
        let gcond = tape.mm(
            a.w("dec.cond"),
            g,
            a.bias("dec.cond"),
            (batch * cfg.upsample_initial_channel) as usize,
            cfg.upsample_initial_channel,
            1,
            gin,
            batch,
            gin,
            1,
            0,
            0,
            1,
            1,
            0,
            gin,
            1,
            1,
            cfg.upsample_initial_channel,
            1.0,
            1,
        );
        x = tape.add_bcast_t(x, gcond, cfg.upsample_initial_channel, t);
    }
    out.push(("d_cond".into(), x));
    let n_up = cfg.upsample_rates.len() as u32;
    let upp = cfg.upp();
    let mut c_in = cfg.upsample_initial_channel;
    for i in 0..n_up {
        let u = cfg.upsample_rates[i as usize];
        let k = cfg.upsample_kernel_sizes[i as usize];
        let c_cur = cfg.upsample_initial_channel / (1 << (i + 1));
        let t_out = (t - 1) * u + k - 2 * ((k - u) / 2);
        x = tape.lrelu(x, 0.1);
        let w = a.w(&format!("dec.ups.{i}"));
        let b = a.bias(&format!("dec.ups.{i}"));
        x = tape.conv_transpose1d(x, w, b, c_cur, c_in, k, u, 1, (k - u) / 2, batch, t, t_out);
        out.push((format!("d_up{i}"), x));
        let har_v = if i + 1 < n_up {
            let stride_f0: u32 = cfg.upsample_rates[(i + 1) as usize..].iter().product();
            let nk = 2 * stride_f0;
            let t_in = seg_t * upp;
            let t_n = (t_in + 2 * (stride_f0 / 2)).saturating_sub(nk) / stride_f0 + 1;
            let hw = a.w(&format!("dec.noise_convs.{i}"));
            let hb = a.bias(&format!("dec.noise_convs.{i}"));
            tape.conv1d(
                har,
                hw,
                hb,
                c_cur,
                1,
                nk,
                stride_f0,
                1,
                stride_f0 / 2,
                batch,
                t_in,
                t_n,
            )
        } else {
            let hw = a.w(&format!("dec.noise_convs.{i}"));
            let hb = a.bias(&format!("dec.noise_convs.{i}"));
            let t_in = seg_t * upp;
            tape.conv1d(har, hw, hb, c_cur, 1, 1, 1, 1, 0, batch, t_in, t_in)
        };
        x = tape.add(x, har_v, false);
        out.push((format!("d_noise{i}"), x));
        let n_k = cfg.resblock_kernel_sizes.len() as u32;
        let mut xs: Option<Var> = None;
        for j in 0..n_k {
            let rb = resblock1(
                tape,
                a,
                &format!("dec.resblocks.{}", i * n_k + j),
                x,
                c_cur,
                cfg.resblock_kernel_sizes[j as usize],
                &cfg.resblock_dilation_sizes[j as usize],
                batch,
                t_out,
            );
            xs = Some(match xs {
                Some(s) => tape.add(s, rb, false),
                None => rb,
            });
        }
        x = tape.pw(xs.expect("resblocks"), 7, 1.0 / n_k as f32);
        out.push((format!("d_rb{i}"), x));
        c_in = c_cur;
        t = t_out;
    }
    // RVC の quirk: conv_post 直前の leaky_relu はデフォルト slope 0.01
    let x = tape.lrelu(x, 0.01);
    out.push(("d_lrelu_final".into(), x));
    let x = conv1d(tape, a, "dec.conv_post", x, 1, c_in, 7, 1, 1, 3, batch, t);
    out.push(("d_post".into(), tape.tanh(x)));
    out
}
