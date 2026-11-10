//! 学習モジュール共通部品: adopt 済み変数マップ + conv/WN/attention ビルダ。

pub mod mpd;
pub mod stft_mel;
pub mod synth_train;

use std::collections::HashMap;

use rcwx_nn::engine::tape::{ParamStore, Tape, Var};

/// adopt 済みパラメータ変数マップ。
/// conv 系は "<prefix>.w" (WN は fold 済み) / "<prefix>.bias"。
#[derive(Default)]
pub struct Adopted {
    pub p: HashMap<String, Var>,
}

impl Adopted {
    /// entry を adopt し (adopt 済みなら再利用) Var を返す。
    pub fn adopt_entry(&mut self, tape: &mut Tape<'_>, store: &ParamStore, idx: usize) -> Var {
        let key = format!("__entry{idx}");
        if let Some(v) = self.p.get(&key) {
            return *v;
        }
        let v = tape.adopt_param(store, idx);
        self.p.insert(key, v);
        v
    }
    pub fn w(&self, prefix: &str) -> Var {
        self.p[&format!("{prefix}.w")]
    }
    pub fn bias(&self, prefix: &str) -> Option<Var> {
        self.p.get(&format!("{prefix}.bias")).copied()
    }
    pub fn vec(&self, name: &str) -> Var {
        self.p
            .get(name)
            .copied()
            .unwrap_or_else(|| panic!("vec key missing: {name} (have: {:?})", {
                let mut ks: Vec<&String> = self.p.keys().collect();
                ks.sort();
                ks
            }))
    }
}

/// conv1d ラッパ。w は [out_c, in_c·k]。same padding 想定は呼び出し側で pad 指定。
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    tape: &mut Tape<'_>,
    a: &Adopted,
    prefix: &str,
    x: Var,
    out_c: u32,
    in_c: u32,
    kernel: u32,
    stride: u32,
    dilation: u32,
    pad_l: u32,
    batch: u32,
    in_len: u32,
) -> Var {
    let out_len = (in_len + 2 * pad_l).saturating_sub(dilation * (kernel - 1) + 1) / stride + 1;
    let w = a.w(prefix);
    let b = a.bias(prefix);
    tape.conv1d(
        x, w, b, out_c, in_c, kernel, stride, dilation, pad_l, batch, in_len, out_len,
    )
}

/// 1x1 conv (mm ベース)。x [B, in_c, T] → [B, out_c, T]。
pub fn conv1x1(
    tape: &mut Tape<'_>,
    a: &Adopted,
    prefix: &str,
    x: Var,
    out_c: u32,
    in_c: u32,
    batch: u32,
    t_len: u32,
) -> Var {
    let w = a.w(prefix);
    let out_len = (batch * out_c * t_len) as usize;
    let out = tape.mm(
        w, x, None, out_len, out_c, t_len, in_c, batch, in_c, 1, 0, 0, 1, t_len, 0, in_c * t_len,
        t_len, 1, out_c * t_len, 1.0,
    );
    if let Some(b) = a.bias(prefix) {
        tape.add_ch_bias(out, b, out_c, t_len)
    } else {
        out
    }
}

/// WN (WaveNet, dilation_rate=1)。x [B, h, T] → [B, h, T]。
/// g は [B·gin] 平坦 (emb 行)。
#[allow(clippy::too_many_arguments)]
pub fn wn(
    tape: &mut Tape<'_>,
    a: &Adopted,
    prefix: &str,
    x: Var,
    h: u32,
    kernel: u32,
    n_layers: u32,
    batch: u32,
    t_len: u32,
    g: Option<Var>,
    mask: Option<Var>,
) -> Var {
    // cond_layer: g [B·gin] → [B·(2h·n_layers)] (時間方向 broadcast 値)
    let g_full = g.map(|g| {
        let gin = tape.shape(a.w(&format!("{prefix}.cond_layer")))[1] as u32;
        let out_len = (batch * 2 * h * n_layers) as usize;
        tape.mm(
            a.w(&format!("{prefix}.cond_layer")),
            g,
            a.bias(&format!("{prefix}.cond_layer")),
            out_len,
            2 * h * n_layers,
            1,
            gin,
            batch,
            gin,
            1,
            0,
            0,
            1,
            gin,
            0,
            gin,
            2 * h * n_layers,
            1,
            0,
            1.0,
        )
    });
    let mut x_cur = x;
    let mut output: Option<Var> = None;
    for i in 0..n_layers {
        let pad = (kernel - 1) / 2;
        let lin = conv1d(
            tape,
            a,
            &format!("{prefix}.in_layers.{i}"),
            x_cur,
            2 * h,
            h,
            kernel,
            1,
            1,
            pad,
            batch,
            t_len,
        );
        let lin = if let Some(gf) = &g_full {
            // g_l = gf のチャネル [i·2h, (i+1)·2h) を [B·2h] に切り出し
            let gl = tape.chslice(*gf, 2 * h * n_layers, i * 2 * h, 2 * h, batch, 1);
            tape.add_bcast_t(lin, gl, 2 * h, t_len)
        } else {
            lin
        };
        let acts = tape.gating(lin, batch, h, t_len);
        let rs_out_ch = if i < n_layers - 1 { 2 * h } else { h };
        let rs = conv1x1(
            tape,
            a,
            &format!("{prefix}.res_skip_layers.{i}"),
            acts,
            rs_out_ch,
            h,
            batch,
            t_len,
        );
        if i < n_layers - 1 {
            let res = tape.chslice(rs, rs_out_ch, 0, h, batch, t_len);
            let skip = tape.chslice(rs, rs_out_ch, h, h, batch, t_len);
            x_cur = tape.add(x_cur, res, false);
            if let Some(m) = mask {
                x_cur = tape.mul_frame_mask(x_cur, m, h, t_len);
            }
            output = Some(match output {
                Some(o) => tape.add(o, skip, false),
                None => skip,
            });
        } else {
            output = Some(match output {
                Some(o) => tape.add(o, rs, false),
                None => rs,
            });
        }
    }
    let out = output.expect("wn: no layers");
    if let Some(m) = mask {
        tape.mul_frame_mask(out, m, h, t_len)
    } else {
        out
    }
}

/// rel-pos 付き self-attention 1 層 (分解実装・heads_share=true)。
/// x [B, C, T] → [B, C, T]。
#[allow(clippy::too_many_arguments)]
pub fn attention_layer(
    tape: &mut Tape<'_>,
    a: &Adopted,
    prefix: &str,
    x: Var,
    c: u32,
    n_heads: u32,
    ws: u32,
    batch: u32,
    t_len: u32,
    mask_bias: Option<Var>, // [T, T] 加算バイアス (無効位置 -1e4)。勾配なし。
) -> Var {
    let dk = c / n_heads;
    let scale = 1.0 / (dk as f32).sqrt();
    let q = conv1x1(tape, a, &format!("{prefix}.conv_q"), x, c, c, batch, t_len);
    let k = conv1x1(tape, a, &format!("{prefix}.conv_k"), x, c, c, batch, t_len);
    let v = conv1x1(tape, a, &format!("{prefix}.conv_v"), x, c, c, batch, t_len);
    let relk = a.vec(&format!("{prefix}.emb_rel_k"));
    let relv = a.vec(&format!("{prefix}.emb_rel_v"));
    let mut attn: Option<Var> = None;
    for h in 0..n_heads {
        // scores_h [B, t, t] = q_h · k_hᵀ · scale
        let scores_h = tape.mm(
            q,
            k,
            None,
            (batch * t_len * t_len) as usize,
            t_len,
            t_len,
            dk,
            batch,
            1,
            t_len,
            h * dk * t_len,
            c * t_len,
            1,
            t_len,
            h * dk * t_len,
            c * t_len,
            t_len,
            1,
            t_len * t_len,
            scale,
        );
        // rel bias: M = q_h · relkᵀ · scale [B, t, 2ws+1] → skew → [B, t, t]
        let m_rel = tape.mm(
            q,
            relk,
            None,
            (batch * t_len * (2 * ws + 1)) as usize,
            t_len,
            2 * ws + 1,
            dk,
            batch,
            1,
            t_len,
            h * dk * t_len,
            c * t_len,
            dk,
            1,
            0,
            0,
            2 * ws + 1,
            1,
            t_len * (2 * ws + 1),
            scale,
        );
        let rel_bias = tape.skew(m_rel, 0, batch, t_len, ws);
        let scores_h = tape.add(scores_h, rel_bias, false);
        let scores_h = if let Some(mb) = mask_bias {
            tape.mask_fill2d(scores_h, mb, t_len, -1e4)
        } else {
            scores_h
        };
        let p = tape.softmax(scores_h, batch * t_len, t_len);
        // out_h [B, dk, t] = p · v_h (m=t, n=dk → 配置 [B, dk, t])
        let out_h = tape.mm(
            p,
            v,
            None,
            (batch * dk * t_len) as usize,
            t_len,
            dk,
            t_len,
            batch,
            t_len,
            1,
            0,
            t_len * t_len,
            t_len,
            1,
            h * dk * t_len,
            c * t_len,
            1,
            t_len,
            dk * t_len,
            1.0,
        );
        // rel values: P2 = skew1(p) [B, t, 2ws+1] → · relv [B, dk, t]
        let p2 = tape.skew(p, 1, batch, t_len, ws);
        let rel_out = tape.mm(
            p2,
            relv,
            None,
            (batch * dk * t_len) as usize,
            t_len,
            dk,
            2 * ws + 1,
            batch,
            2 * ws + 1,
            1,
            0,
            t_len * (2 * ws + 1),
            1,
            dk,
            0,
            0,
            1,
            t_len,
            dk * t_len,
            1.0,
        );
        let out_h = tape.add(out_h, rel_out, false);
        attn = Some(match attn {
            Some(prev) => tape.chcat(prev, out_h, h * dk, dk, batch, t_len),
            None => out_h,
        });
    }
    let attn = attn.expect("attention heads");
    conv1x1(tape, a, &format!("{prefix}.conv_o"), attn, c, c, batch, t_len)
}

/// FFN (relu)。x [B, C, T] → [B, C, T]。
pub fn ffn(
    tape: &mut Tape<'_>,
    a: &Adopted,
    prefix: &str,
    x: Var,
    c: u32,
    filter: u32,
    kernel: u32,
    batch: u32,
    t_len: u32,
    mask: Option<Var>,
) -> Var {
    let pad = (kernel - 1) / 2;
    let h = conv1d(tape, a, &format!("{prefix}.conv_1"), x, filter, c, kernel, 1, 1, pad, batch, t_len);
    let h = tape.relu(h);
    let h = conv1d(tape, a, &format!("{prefix}.conv_2"), h, c, filter, kernel, 1, 1, pad, batch, t_len);
    if let Some(m) = mask {
        tape.mul_frame_mask(h, m, c, t_len)
    } else {
        h
    }
}
