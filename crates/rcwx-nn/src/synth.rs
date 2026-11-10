//! RVC Synthesizer (TextEncoder768 + Flow + GeneratorNSF) の独自 wgpu エンジン実装。
//!
//! 重み: tools/convert_synth.py 出力の synth.safetensors (weight_norm 解決済み・enc_q 除外)
//! と synth.json。レイアウトは [C, T] 中心。SineGen legacy は純 Rust (CPU, fp32, 励振ノイズ 0)。

use anyhow::{Context, Result};

use crate::engine::{Engine, GpuTensor, Graph};

pub struct SynthConfig {
    pub inter_channels: u32,   // 192
    pub hidden_channels: u32,  // 192
    pub filter_channels: u32,  // 768
    pub n_heads: u32,          // 2
    pub n_layers: u32,         // 6
    pub kernel_size: u32,      // 3
    pub upsample_rates: Vec<u32>,
    pub upsample_initial_channel: u32, // 512
    pub upsample_kernel_sizes: Vec<u32>,
    pub resblock_kernel_sizes: Vec<u32>,
    pub gin_channels: u32,     // 256
    pub sr: u32,
    pub n_speakers: u32,
}

pub struct Synth {
    pub cfg: SynthConfig,
    w: std::collections::HashMap<String, GpuTensor>,
    /// None でなければ infer 中間を保持する (段階検証)。
    pub debug_keep: std::cell::RefCell<Option<Vec<(&'static str, GpuTensor)>>>,
    pub emb_g_row0: Vec<f32>,
    pub emb_pitch: Vec<f32>, // [256 * hidden]
    pub l_linear_w: f32,
    pub l_linear_b: f32,
}

pub fn default_synth_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("USERPROFILE").expect("USERPROFILE"))
        .join(".cache/rcwx/models/rust")
}

impl Synth {
    pub fn load(eng: &Engine, dir: &std::path::Path) -> Result<Self> {
        let raw = std::fs::read(dir.join("synth.safetensors"))?;
        let st = safetensors::SafeTensors::deserialize(&raw)?;
        let mut w = std::collections::HashMap::new();
        let mut cpu: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
        for name in st.names() {
            let name: &str = name;
            let v = st.tensor(name)?;
            let shape = v.shape().to_vec();
            let n: usize = shape.iter().product();
            let mut out = Vec::with_capacity(n);
            for c in v.data().chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            if matches!(
                name,
                "emb_g.weight" | "enc_p.emb_pitch.weight" | "dec.m_source.l_linear.weight" | "dec.m_source.l_linear.bias"
            ) {
                cpu.insert(name.to_string(), out.clone());
            }
            w.insert(name.to_string(), eng.tensor_from_f32(&out, shape));
        }
        let cfg_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("synth.json"))?)?;
        let gu = |k: &str, d: u64| cfg_json[k].as_u64().unwrap_or(d) as u32;
        let gvec = |k: &str| -> Vec<u32> {
            cfg_json[k]
                .as_array()
                .and_then(|a| Some(a.iter().map(|v| v.as_u64().unwrap() as u32).collect()))
                .unwrap_or_default()
        };
        let cfg = SynthConfig {
            inter_channels: gu("inter_channels", 192),
            hidden_channels: gu("hidden_channels", 192),
            filter_channels: gu("filter_channels", 768),
            n_heads: gu("n_heads", 2),
            n_layers: gu("n_layers", 6),
            kernel_size: gu("kernel_size", 3),
            upsample_rates: gvec("upsample_rates"),
            upsample_initial_channel: gu("upsample_initial_channel", 512),
            upsample_kernel_sizes: gvec("upsample_kernel_sizes"),
            resblock_kernel_sizes: gvec("resblock_kernel_sizes"),
            gin_channels: gu("gin_channels", 256),
            sr: gu("sr", 40000),
            n_speakers: gu("n_speakers", 1),
        };
        let emb_g = cpu.get("emb_g.weight").context("emb_g")?;
        let gin = cfg.gin_channels as usize;
        Ok(Self {
            cfg,
            debug_keep: std::cell::RefCell::new(None),
            emb_g_row0: emb_g[..gin].to_vec(),
            emb_pitch: cpu.get("enc_p.emb_pitch.weight").context("emb_pitch")?.clone(),
            l_linear_w: cpu.get("dec.m_source.l_linear.weight").unwrap()[0],
            l_linear_b: cpu.get("dec.m_source.l_linear.bias").unwrap()[0],
            w,
        })
    }

    fn g(&self, name: &str) -> &GpuTensor {
        self.w.get(name).unwrap_or_else(|| panic!("weight {name}"))
    }

    /// デバッグ/外部検証用。
    pub fn wget(&self, name: &str) -> &GpuTensor {
        self.g(name)
    }

    /// infer 中間 keep を有効化 (呼び出し後に取り出し)。
    pub fn enable_debug(&self) {
        *self.debug_keep.borrow_mut() = Some(Vec::new());
    }

    pub fn take_debug(&self) -> Vec<(&'static str, GpuTensor)> {
        self.debug_keep.borrow_mut().take().unwrap_or_default()
    }

    /// 1×1 conv: x [cin, T] (base 指定可) → out [cout, T]。
    fn conv1x1(
        &self,
        g: &mut Graph<'_>,
        x: &GpuTensor,
        x_base: u32,
        wname: &str,
        out: &GpuTensor,
        cin: u32,
        cout: u32,
        t: u32,
        scale: f32,
    ) {
        g.matmul_ex(
            self.g(&format!("{wname}.weight")), x,
            Some(self.g(&format!("{wname}.bias"))), out,
            cout, t, cin, 1,
            cin, 1, 0, 0,
            1, t, x_base, 0,
            t, 1, 0, 0,
            scale, 1,
        );
    }

    /// 公開版 (検証 bin 用)。
    pub fn conv_k_pub(
        &self,
        g: &mut crate::engine::Graph<'_>,
        x: &crate::engine::GpuTensor,
        wname: &str,
        out: &crate::engine::GpuTensor,
        cin: u32,
        cout: u32,
        k: u32,
        dil: u32,
        t: u32,
    ) {
        self.conv_k(g, x, 0, wname, out, cin, cout, k, dil, t);
    }

    /// k 番 conv (same padding) [C, T] → [C', T]。im2col + matmul。
    /// 大規模 T では 128MB バッファ上限を超えるため直接 conv1d カーネルへ切替。
    fn conv_k(
        &self,
        g: &mut Graph<'_>,
        x: &GpuTensor,
        x_base: u32,
        wname: &str,
        out: &GpuTensor,
        cin: u32,
        cout: u32,
        k: u32,
        dil: u32,
        t: u32,
    ) {
        if x_base == 0 && (cin * k * t) as usize * 4 > 96_000_000 {
            g.conv1d(
                x, self.g(&format!("{wname}.weight")),
                Some(self.g(&format!("{wname}.bias"))), out,
                cout, cin, k, 1, dil, (k - 1) * dil / 2, t, t, 1,
            );
            return;
        }
        let ic = g.engine_tensor_uninit((cin * k * t) as usize);
        g.im2col_off(x, x_base, &ic, cin, k, 1, dil, (k - 1) * dil / 2, t, t);
        g.matmul_ex(
            self.g(&format!("{wname}.weight")), &ic,
            Some(self.g(&format!("{wname}.bias"))), out,
            cout, t, cin * k, 1,
            cin * k, 1, 0, 0,
            1, t, 0, 0,
            t, 1, 0, 0,
            1.0, 1,
        );
    }

    /// SineGen legacy (純 Rust, fp32, noise=0) → har [T*upp]
    pub fn sinegen(&self, f0: &[f32], t: u32) -> Vec<f32> {
        let upp = self.cfg.sr / 100;
        let l = t as usize;
        let n = l * upp as usize;
        let sr = self.cfg.sr as f32;
        let rad: Vec<f32> = f0.iter().map(|f| (f / sr).rem_euclid(1.0)).collect();
        let mut cum = vec![0.0f32; l];
        let mut s = 0.0f32;
        for i in 0..l {
            s += rad[i];
            cum[i] = s * upp as f32;
        }
        let tou: Vec<f32> = {
            let ln = l;
            let on = n;
            let mut o = vec![0.0f32; on];
            if on == 1 {
                o[0] = cum[0];
            } else {
                let scale = (ln - 1) as f32 / (on - 1) as f32;
                for i in 0..on {
                    let pos = i as f32 * scale;
                    let i0 = pos.floor() as usize;
                    let i1 = (i0 + 1).min(ln - 1);
                    let fr = pos - i0 as f32;
                    o[i] = cum[i0] + (cum[i1] - cum[i0]) * fr;
                }
            }
            o
        };
        let mut phase = 0.0f32;
        let two_pi = 2.0 * std::f32::consts::PI;
        let mut out = vec![0.0f32; n];
        let mut prev_rem = 0.0f32;
        for i in 0..n {
            let rem = tou[i].rem_euclid(1.0);
            if i > 0 && rem < prev_rem {
                phase -= 1.0;
            }
            prev_rem = rem;
            phase += rad[(i / upp as usize).min(l - 1)];
            let uv = if f0[(i / upp as usize).min(l - 1)] > 0.0 { 1.0 } else { 0.0 };
            let sine = (phase * two_pi).sin() * 0.1 * uv;
            out[i] = (self.l_linear_w * sine + self.l_linear_b).tanh();
        }
        out
    }

    /// sinegen のフレーム範囲版: [frame_start, frame_start+n) フレーム分の样本のみ
    /// 生成 (位相・補間はフル系列と同一の値)。
    pub fn sinegen_range(&self, f0: &[f32], frame_start: usize, n_frames: usize) -> Vec<f32> {
        let upp = self.cfg.sr as usize / 100;
        let l = f0.len();
        let n_all = l * upp;
        let s = frame_start * upp;
        let e = (frame_start + n_frames) * upp;
        let sr = self.cfg.sr as f32;
        let rad: Vec<f32> = f0.iter().map(|f| (*f / sr).rem_euclid(1.0)).collect();
        let mut cum = vec![0.0f32; l];
        let mut acc = 0.0f32;
        for i in 0..l {
            acc += rad[i];
            cum[i] = acc * upp as f32;
        }
        // フル系列と同じ tou 補間を大域位置 i_g に適用
        let tou = |i_g: usize| -> f32 {
            if n_all == 1 {
                return cum[0];
            }
            let scale = (l - 1) as f32 / (n_all - 1) as f32;
            let pos = i_g as f32 * scale;
            let i0 = pos.floor() as usize;
            let i1 = (i0 + 1).min(l - 1);
            let fr = pos - i0 as f32;
            cum[i0] + (cum[i1] - cum[i0]) * fr
        };
        let two_pi = 2.0 * std::f32::consts::PI;
        let mut out = vec![0.0f32; e - s];
        let mut phase = 0.0f32;
        let mut prev_rem = 0.0f32;
        for i_g in 0..e {
            let rem = tou(i_g).rem_euclid(1.0);
            if i_g > 0 && rem < prev_rem {
                phase -= 1.0;
            }
            prev_rem = rem;
            let f_idx = (i_g / upp).min(l - 1);
            phase += rad[f_idx];
            if i_g >= s {
                let uv = if f0[f_idx] > 0.0 { 1.0 } else { 0.0 };
                let sine = (phase * two_pi).sin() * 0.1 * uv;
                out[i_g - s] = (self.l_linear_w * sine + self.l_linear_b).tanh();
            }
        }
        out
    }

    /// feats [T, 768] (GPU) + pitch idx (CPU) + pitchf Hz (CPU) → 音声 [T * zc]。
    /// noise_scale=0 決定論 (z = m)。
    pub fn infer(
        &self,
        eng: &Engine,
        feats: &GpuTensor,
        pitch_idx: &[u32],
        pitchf: &[f32],
        t: u32,
    ) -> GpuTensor {
        let z = self.encode_raw(eng, feats, pitch_idx, t);
        self.decode_window(eng, &z, pitchf, 0, t)
    }

    /// TextEncoder + Flow。z [inter, T] を返す。
    fn encode_raw(
        &self,
        eng: &Engine,
        feats: &GpuTensor,
        pitch_idx: &[u32],
        t: u32,
    ) -> GpuTensor {
        let cfg = &self.cfg;
        let h = cfg.hidden_channels;
        let heads = cfg.n_heads;
        let dk = h / heads;
        let ws: u32 = 10;
        let filt = cfg.filter_channels;
        let nl = cfg.n_layers;
        let inter = cfg.inter_channels;
        let mut g = eng.graph();

        // ---- TextEncoder ----
        // x[ch, t] = (Σ_k feats[t,k]·W_phone[ch,k])·√h + emb_pitch[pitch[t]][ch]
        let mut pitch_emb_ct = vec![0.0f32; (h * t) as usize];
        for r in 0..t as usize {
            let idx = pitch_idx[r].clamp(0, 255) as usize;
            for c in 0..h as usize {
                pitch_emb_ct[c * t as usize + r] =
                    self.emb_pitch[idx * h as usize + c] * (h as f32).sqrt();
            }
        }
        let pitch_emb = eng.tensor_from_f32(&pitch_emb_ct, vec![]);
        let x0 = eng.tensor_uninit((h * t) as usize);
        g.matmul_ex(
            self.g("enc_p.emb_phone.weight"), feats,
            Some(self.g("enc_p.emb_phone.bias")), &x0,
            h, t, 768, 1,
            768, 1, 0, 0,
            768, 1, 0, 0,
            t, 1, 0, 0,
            (h as f32).sqrt(), 1,
        );
        let mut x = eng.tensor_uninit((h * t) as usize);
        g.add(&x0, &pitch_emb, &x, h * t);
        g.lrelu(&x, h * t);
        if let Some(k) = self.debug_keep.borrow_mut().as_mut() {
            k.push(("x_in", x.clone_handle()));
        }

        // Encoder 6 層
        for l in 0..nl {
            let p = format!("enc_p.encoder.attn_layers.{l}");
            let q = eng.tensor_uninit((h * t) as usize);
            let k = eng.tensor_uninit((h * t) as usize);
            let v = eng.tensor_uninit((h * t) as usize);
            self.conv1x1(&mut g, &x, 0, &format!("{p}.conv_q"), &q, h, h, t, 1.0);
            self.conv1x1(&mut g, &x, 0, &format!("{p}.conv_k"), &k, h, h, t, 1.0);
            self.conv1x1(&mut g, &x, 0, &format!("{p}.conv_v"), &v, h, h, t, 1.0);
            // scores[heads, T, T] = (q·k^T)/√dk + rel
            if l == 0 {
                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                    kk.push(("q0", q.clone_handle()));
                    kk.push(("k0", k.clone_handle()));
                    kk.push(("v0", v.clone_handle()));
                }
            }
            let scores = eng.tensor_uninit((heads * t * t) as usize);
            g.matmul(
                &q, &k, None, &scores,
                t, t, dk, heads,
                1, t, 0, dk * t,
                1, t, 0, dk * t,
                t, 1, 0, t * t,
                1.0 / (dk as f32).sqrt(),
            );
            g.rel_attn_scores(&q, self.g(&format!("{p}.emb_rel_k")), &scores, t, dk, ws, heads, 1.0 / (dk as f32).sqrt());

            if l == 0 {
                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                    kk.push(("scores0", scores.clone_handle()));
                }
            }
            g.softmax(&scores, heads * t, t);
            // ctx = p·v + p·rel_v
            let ctx = eng.tensor_uninit((h * t) as usize);
            g.matmul(
                &scores, &v, None, &ctx,
                t, dk, t, heads,
                t, 1, 0, t * t,
                t, 1, 0, dk * t,
                1, t, 0, dk * t,   // C: channel-major [ch=h*96+d, i] — ch*T + i
                1.0,
            );
            if l == 0 {
                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                    kk.push(("p_attn0", scores.clone_handle()));
                    kk.push(("ctx0", ctx.clone_handle()));
                }
            }
            g.rel_attn_values(&scores, self.g(&format!("{p}.emb_rel_v")), &ctx, t, dk, ws, heads);
            let attn = eng.tensor_uninit((h * t) as usize);
            self.conv1x1(&mut g, &ctx, 0, &format!("{p}.conv_o"), &attn, h, h, t, 1.0);
            if l == 0 {
                if let Some(k) = self.debug_keep.borrow_mut().as_mut() {
                    k.push(("attn0", attn.clone_handle()));
                }
            }
            let xr = eng.tensor_uninit((h * t) as usize);
            g.add(&x, &attn, &xr, h * t);
            let xn = eng.tensor_uninit((h * t) as usize);
            g.layer_norm(&xr, self.g(&format!("enc_p.encoder.norm_layers_1.{l}.gamma")),
                self.g(&format!("enc_p.encoder.norm_layers_1.{l}.beta")), &xn,
                t, h, 1, t, 1, t, 1e-5);
            x = xn;

            if l == 0 {
                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                    kk.push(("norm1_0", x.clone_handle()));
                }
            }
            // FFN: conv k3 → gelu_approx → conv k3 → 残差 → LN
            let f = format!("enc_p.encoder.ffn_layers.{l}");
            let ff1 = eng.tensor_uninit((filt * t) as usize);
            self.conv_k(&mut g, &x, 0, &format!("{f}.conv_1"), &ff1, h, filt, cfg.kernel_size, 1, t);
            g.pointwise(&ff1, None, filt * t, 7, 0.0, 0, 0, 0); // relu (RCWX Encoder FFN)
            if l == 0 {
                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                    kk.push(("ffn0_gelu_mid", ff1.clone_handle()));
                }
            }
            let ff2 = eng.tensor_uninit((h * t) as usize);
            self.conv_k(&mut g, &ff1, 0, &format!("{f}.conv_2"), &ff2, filt, h, cfg.kernel_size, 1, t);
            if l == 0 {
                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                    kk.push(("ffn0", ff2.clone_handle()));
                }
            }
            let xr2 = eng.tensor_uninit((h * t) as usize);
            g.add(&x, &ff2, &xr2, h * t);
            let xn2 = eng.tensor_uninit((h * t) as usize);
            g.layer_norm(&xr2, self.g(&format!("enc_p.encoder.norm_layers_2.{l}.gamma")),
                self.g(&format!("enc_p.encoder.norm_layers_2.{l}.beta")), &xn2,
                t, h, 1, t, 1, t, 1e-5);
            x = xn2;
        }
        // proj → stats [384, T]
        let stats = eng.tensor_uninit((inter * 2 * t) as usize);
        self.conv1x1(&mut g, &x, 0, "enc_p.proj", &stats, h, inter * 2, t, 1.0);
        if let Some(k) = self.debug_keep.borrow_mut().as_mut() {
            k.push(("enc_out", x.clone_handle()));
            k.push(("proj_out", stats.clone_handle()));
        }

        // ---- Flow (reverse, mean_only, z = m = stats[0:192]) ----
        // g [gin, T] (全列同一) を構築
        let mut gbuf = vec![0.0f32; (cfg.gin_channels * t) as usize];
        for c in 0..cfg.gin_channels as usize {
            for r in 0..t as usize {
                gbuf[c * t as usize + r] = self.emb_g_row0[c];
            }
        }
        let gt = eng.tensor_from_f32(&gbuf, vec![]);

        let n_flows = 4u32;
        let mut z = stats; // m = rows [0:192] — そのまま flow 入力
        for i in (0..n_flows).rev() {
            {
                let zf = eng.tensor_uninit((inter * t) as usize);
                g.flip_ch(&z, &zf, t, inter);
                z = zf;
            }
            let fp = format!("flow.flows.{}", i * 2);
            let x0_base = 0u32;       // z rows [0:96]
            let x1_base = 96 * t;     // z rows [96:192]
            // pre(x0) [192, T]
            let hbuf = eng.tensor_uninit((h * t) as usize);
            self.conv1x1(&mut g, &z, x0_base, &format!("{fp}.pre"), &hbuf, inter / 2, h, t, 1.0);
            let hbuf_keep = hbuf.clone_handle();
            // cond(g) [3*2h, T]
            let cond = eng.tensor_uninit((h * 6) as usize * t as usize);
            self.conv1x1(&mut g, &gt, 0, &format!("{fp}.enc.cond_layer"), &cond, cfg.gin_channels, h * 6, t, 1.0);
            let mut skip: Option<GpuTensor> = None;
            let mut hh = hbuf;
            for l in 0..3u32 {
                let lin = eng.tensor_uninit((h * 2 * t) as usize);
                self.conv_k(&mut g, &hh, 0, &format!("{fp}.enc.in_layers.{l}"), &lin, h, h * 2, 5, 1, t);
                // acts = gating(lin + cond[l*2h : l*2h+2h])
                let gated = eng.tensor_uninit((h * t) as usize);
                let with_cond = eng.tensor_uninit((h * 2 * t) as usize);
                g.add_off(&lin, &cond, &with_cond, h * 2 * t, 0, l * 2 * h * t, false);
                if i == 3 && l == 0 {
                    if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                        kk.push(("wn6_in0", lin.clone_handle()));
                        kk.push(("wn6_cond", cond.clone_handle()));
                    }
                }
                g.gating(&with_cond, t, h);
                // gating は in-place (前半) — with_cond の前半が acts。独立バッファに act 部分だけは使えないので
                // res_skip は with_cond の前半 [0 : h*t] を入力とする (w_base 指定)
                let rs_out_ch = if l < 2 { h * 2 } else { h };
                let rs = eng.tensor_uninit((rs_out_ch * t) as usize);
                if i == 3 && l == 0 {
                    if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                        kk.push(("wn6_acts0", with_cond.clone_handle()));
                    }
                }
                g.matmul_ex(
                    self.g(&format!("{fp}.enc.res_skip_layers.{l}.weight")), &with_cond,
                    Some(self.g(&format!("{fp}.enc.res_skip_layers.{l}.bias"))), &rs,
                    rs_out_ch, t, h, 1,
                    h, 1, 0, 0,
                    1, t, 0, 0,
                    t, 1, 0, 0,
                    1.0, 1,
                );
                let _ = gated;
                if i == 3 && l == 0 {
                    if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                        kk.push(("wn6_rs0", rs.clone_handle()));
                    }
                }
                let mut hn_opt: Option<GpuTensor> = None;
                let mut skipn_opt: Option<GpuTensor> = None;
                if l < 2 {
                    // h = h + res (rs rows [0:h]), skip += rs rows [h:2h]
                    let hn = eng.tensor_uninit((h * t) as usize);
                    g.add_off(&hh, &rs, &hn, h * t, 0, 0, false);
                    let skipn = eng.tensor_uninit((h * t) as usize);
                    match &skip {
                        None => {
                            let zeros = eng.tensor_from_f32(&vec![0.0f32; (h * t) as usize], vec![]);
                            g.add_off(&rs, &zeros, &skipn, h * t, h * t, 0, false);
                        }
                        Some(sk) => g.add_off(sk, &rs, &skipn, h * t, 0, h * t, false),
                    }
                    if i == 3 {
                        if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                            kk.push((Box::leak(format!("wn6_h{l}").into_boxed_str()), hn.clone_handle()));
                            kk.push((Box::leak(format!("wn6_skip{l}").into_boxed_str()), skipn.clone_handle()));
                        }
                    }
                    hh = hn;
                    skip = Some(skipn);
                } else {
                    let skipn = eng.tensor_uninit((h * t) as usize);
                    match &skip {
                        None => skip = Some(rs),
                        Some(sk) => {
                            g.add_off(sk, &rs, &skipn, h * t, 0, 0, false);
                            if i == 3 {
                                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                                    kk.push((Box::leak(format!("wn6_skip{l}").into_boxed_str()), skipn.clone_handle()));
                                }
                            }
                            skip = Some(skipn);
                        }
                    }
                    let _ = &mut hn_opt;
                    let _ = &mut skipn_opt;
                }
            }
            let m = eng.tensor_uninit((inter / 2 * t) as usize);
            let skip_t = skip.unwrap();
            self.conv1x1(&mut g, &skip_t, 0, &format!("{fp}.post"), &m, h, inter / 2, t, 1.0);
            if i == 3 {
                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                    kk.push(("flow6_wn", skip_t.clone_handle()));
                    kk.push(("flow6_post", m.clone_handle()));
                }
            }
            if let Some(k) = self.debug_keep.borrow_mut().as_mut() {
                k.push((Box::leak(format!("flow{}_pre", i * 2).into_boxed_str()), hbuf_keep.clone_handle()));
            }
            // zn = [x0 (copy), x1 - m]
            let zn = eng.tensor_uninit((inter * t) as usize);
            let zeros = eng.tensor_from_f32(&vec![0.0f32; (inter / 2 * t) as usize], vec![]);
            g.add_off3(&z, &zeros, &zn, inter / 2 * t, 0, 0, 0, false);          // zn[0:96t] = x0 + 0
            g.add_off3(&z, &m, &zn, inter / 2 * t, x1_base, 0, x1_base, true); // zn[96t:] = x1-m
            if let Some(k) = self.debug_keep.borrow_mut().as_mut() {
                k.push((Box::leak(format!("flow{}_out", i * 2).into_boxed_str()), zn.clone_handle()));
            }
            z = zn;
        }
        g.submit().expect("submit");
        z
    }

    /// デコーダ窓生成: z [inter, n] + pitchf フル系列 → 音声 [n * zc]。
    /// frame_start は sinegen 大域位相整合用 (オフライン全体では 0)。
    pub fn decode_window(
        &self,
        eng: &Engine,
        z: &GpuTensor,
        pitchf: &[f32],
        frame_start: u32,
        n: u32,
    ) -> GpuTensor {
        let cfg = &self.cfg;
        let h = cfg.hidden_channels;
        let inter = cfg.inter_channels;
        let t = n;
        let mut g = eng.graph();
        // cond(g) 行列
        let mut gbuf = vec![0.0f32; (cfg.gin_channels * t) as usize];
        for c in 0..cfg.gin_channels as usize {
            for r in 0..t as usize {
                gbuf[c * t as usize + r] = self.emb_g_row0[c];
            }
        }
        let gt = eng.tensor_from_f32(&gbuf, vec![]);

        // ---- Decoder ----
        let ups = &cfg.upsample_rates;
        let zc: u32 = ups.iter().product();
        let chs: Vec<u32> = (0..=ups.len())
            .map(|i| cfg.upsample_initial_channel / (1u32 << i))
            .collect();
        // conv_pre + cond(g)
        let mut dx = eng.tensor_uninit((chs[0] * t) as usize);
        self.conv_k(&mut g, z, 0, "dec.conv_pre", &dx, inter, chs[0], 7, 1, t);
        let gcond = eng.tensor_uninit((chs[0] * t) as usize);
        self.conv1x1(&mut g, &gt, 0, "dec.cond", &gcond, cfg.gin_channels, chs[0], t, 1.0);
        if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
            kk.push(("dec_conv_pre", dx.clone_handle()));
        }
        let dx2 = eng.tensor_uninit((chs[0] * t) as usize);
        g.add(&dx, &gcond, &dx2, chs[0] * t);
        dx = dx2;

        // har (CPU, フル系列と同一の位相) → GPU
        let har = self.sinegen_range(pitchf, frame_start as usize, n as usize);
        let har_t = eng.tensor_from_f32(&har, vec![]);
        if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
            kk.push(("har", har_t.clone_handle()));
        }

        let mut len = t;
        for i in 0..ups.len() {
            let (u, kk) = (ups[i], cfg.upsample_kernel_sizes[i]);
            let (cin, cout) = (chs[i], chs[i + 1]);
            // lrelu → upsample
            g.lrelu(&dx, cin * len);
            let up = eng.tensor_uninit((cout * len * u) as usize);
            g.conv_transpose1d(
                &dx, self.g(&format!("dec.ups.{i}.weight")),
                Some(self.g(&format!("dec.ups.{i}.bias"))), &up,
                cout, cin, kk, u, (kk - u) / 2, len, len * u,
            );
            // noise_conv on har
            let ns = eng.tensor_uninit((cout * len * u) as usize);
            let stride_f0: u32 = ups[i + 1..].iter().product::<u32>().max(1);
            let nk = if stride_f0 > 1 { stride_f0 * 2 } else { 1 };
            if nk == 1 {
                g.conv1d(&har_t, self.g(&format!("dec.noise_convs.{i}.weight")),
                    Some(self.g(&format!("dec.noise_convs.{i}.bias"))), &ns,
                    cout, 1, 1, 1, 1, 0, har.len() as u32, len * u, 1);
            } else {
                g.conv1d(&har_t, self.g(&format!("dec.noise_convs.{i}.weight")),
                    Some(self.g(&format!("dec.noise_convs.{i}.bias"))), &ns,
                    cout, 1, nk, stride_f0, 1, stride_f0 / 2, har.len() as u32, len * u, 1);
            }
            let summed = eng.tensor_uninit((cout * len * u) as usize);
            g.add(&up, &ns, &summed, cout * len * u);
            if i == 0 {
                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                    kk.push(("ups0", up.clone_handle()));
                    kk.push(("noise0", ns.clone_handle()));
                }
            }
            // resblocks ×3 → 平均
            let mut acc: Option<GpuTensor> = None;
            for j in 0..3usize {
                let rb = format!("dec.resblocks.{}.convs", i * 3 + j);
                let k_j = cfg.resblock_kernel_sizes[j as usize];
                let mut bx = eng.tensor_uninit((cout * len * u) as usize);
                let zeros = eng.tensor_from_f32(&vec![0.0f32; (cout * len * u) as usize], vec![]);
                g.add_off(&summed, &zeros, &bx, cout * len * u, 0, 0, false); // copy
                for m in 0..3u32 {
                    let dil = [1u32, 3, 5][m as usize];
                    let xt = eng.tensor_uninit((cout * len * u) as usize);
                    // lrelu → convs1[m] → lrelu → convs2[m] → 残差加算
                    let lr = eng.tensor_uninit((cout * len * u) as usize);
                    g.pointwise(&bx, Some(&lr), cout * len * u, 1, 0.1, 0, 0, 0);

                    let c1 = eng.tensor_uninit((cout * len * u) as usize);
                    self.conv_k(&mut g, &lr, 0, &format!("{rb}1.{m}"), &c1, cout, cout, k_j, dil, len * u);
                    let lr2 = eng.tensor_uninit((cout * len * u) as usize);
                    g.pointwise(&c1, Some(&lr2), cout * len * u, 1, 0.1, 0, 0, 0);
                    let c2 = eng.tensor_uninit((cout * len * u) as usize);
                    self.conv_k(&mut g, &lr2, 0, &format!("{rb}2.{m}"), &c2, cout, cout, k_j, 1, len * u);
                    g.add_off(&bx, &c2, &xt, cout * len * u, 0, 0, false);
                    bx = xt;
                }
                if i == 0 && j == 0 {
                    if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                        kk.push(("rb0", bx.clone_handle()));
                    }
                }
                acc = Some(match acc {
                    None => bx,
                    Some(a) => {
                        let s = eng.tensor_uninit((cout * len * u) as usize);
                        g.add_off(&a, &bx, &s, cout * len * u, 0, 0, false);
                        s
                    }
                });
            }
            let avg = eng.tensor_uninit((cout * len * u) as usize);
            g.pointwise(&acc.unwrap(), Some(&avg), cout * len * u, 4, 1.0 / 3.0, 0, 0, 0);
            dx = avg;
            len *= u;
        }
        // 最終: lrelu → conv_post (bias 無し) → tanh
        g.lrelu(&dx, chs[ups.len()] * len);
        let audio = eng.tensor_uninit(len as usize);
        self.conv_k_nobias(&mut g, &dx, "dec.conv_post.weight", &audio, chs[ups.len()], 1, 7, 1, len);
        g.tanh_inplace(&audio, len);

        g.submit().expect("submit");
        audio
    }

    /// bias 無し k7 conv (conv_post)。
    fn conv_k_nobias(
        &self,
        g: &mut Graph<'_>,
        x: &GpuTensor,
        wname: &str,
        out: &GpuTensor,
        cin: u32,
        cout: u32,
        k: u32,
        dil: u32,
        t: u32,
    ) {
        if (cin * k * t) as usize * 4 > 96_000_000 {
            g.conv1d(
                x, self.g(wname), None, out,
                cout, cin, k, 1, dil, (k - 1) * dil / 2, t, t, 1,
            );
            return;
        }
        let ic = g.engine_tensor_uninit((cin * k * t) as usize);
        g.im2col_off(x, 0, &ic, cin, k, 1, dil, (k - 1) * dil / 2, t, t);
        g.matmul(
            self.g(wname), &ic, None, out,
            cout, t, cin * k, 1,
            cin * k, 1, 0, 0,
            1, t, 0, 0,
            t, 1, 0, 0,
            1.0,
        );
    }

    /// 長尺オフライン用チャンク分割生成。
    /// TextEncoder/Flow は全 T で一度だけ実行し、デコーダを
    /// `chunk` フレーム窓 (左右 margin フレームの受容野マージン付き) に
    /// 分割する。窓内の中間出力はフルバッチ演算と同一 (受容野外は零埋め)。
    /// 戻り値: 音声 [t * zc] (CPU)。
    pub fn infer_chunked(
        &self,
        eng: &Engine,
        feats: &GpuTensor,
        pitch_idx: &[u32],
        pitchf: &[f32],
        t: u32,
        chunk: u32,
        margin: u32,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let ups = &cfg.upsample_rates;
        let zc: u32 = ups.iter().product();
        let inter = cfg.inter_channels;
        let t = t as usize;
        let chunk = chunk as usize;
        let margin = margin as usize;

        let z = self.encode_raw(eng, feats, pitch_idx, t as u32);
        let z_cpu = eng.readback_f32(&z); // [inter, T] — 小さい

        let mut out = vec![0.0f32; t * zc as usize];
        let mut start = 0usize;
        while start < t {
            let end = (start + chunk).min(t);
            let w_start = start.saturating_sub(margin);
            let w_end = (end + margin).min(t);
            let n = (w_end - w_start) as u32;
            let z_win: Vec<f32> = (0..inter as usize)
                .flat_map(|c| z_cpu[c * t + w_start..c * t + w_end].to_vec())
                .collect();
            let z_win_t = eng.tensor_from_f32(&z_win, vec![]);
            let audio = self.decode_window(eng, &z_win_t, pitchf, w_start as u32, n);
            let samples = eng.readback_f32(&audio);
            let s_off = start - w_start;
            let take = end - start;
            let row = zc as usize * take;
            out[start * zc as usize..start * zc as usize + row]
                .copy_from_slice(&samples[s_off * zc as usize..s_off * zc as usize + row]);
            start = end;
        }
        out
    }
}
