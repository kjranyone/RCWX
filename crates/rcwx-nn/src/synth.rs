//! RVC Synthesizer (TextEncoder + Flow + GeneratorNSF) の独自 wgpu エンジン実装。
//! RVC v1 (256-dim 特徴・n_layers=3・kernel 5) / v2 (768-dim・n_layers=6・kernel 3) 両対応。
//!
//! 重み: tools/convert_synth.py 出力の synth.safetensors (weight_norm 解決済み・enc_q 除外)
//! と synth.json。レイアウトは [C, T] 中心。SineGen legacy は純 Rust (CPU, fp32, 励振ノイズ 0)。

use anyhow::{Context, Result};

use crate::engine::{Engine, GpuTensor, Graph};

pub struct SynthConfig {
    pub version: u32,     // 1 (256-dim) / 2 (768-dim)
    pub in_channels: u32, // emb_phone 入力次元: 256 (v1) / 768 (v2) = HuBERT 特徴次元
    /// make_test_model.py 由来のランダム重みモデル (パリティ gate が 0.995 に緩和される)
    pub random_test: bool,
    pub inter_channels: u32,  // 192
    pub hidden_channels: u32, // 192
    pub filter_channels: u32, // 768
    pub n_heads: u32,         // 2
    pub n_layers: u32,        // v2: 6 / v1: 3
    pub kernel_size: u32,     // v2: 3 / v1: 5
    pub upsample_rates: Vec<u32>,
    pub upsample_initial_channel: u32, // 512
    pub upsample_kernel_sizes: Vec<u32>,
    pub resblock_kernel_sizes: Vec<u32>,
    pub gin_channels: u32, // 256
    pub sr: u32,
    pub n_speakers: u32,
}

/// ストリーミング sinegen の持ち越し位相状態。
#[derive(Default, Clone)]
pub struct SineState {
    phase: f32,
    prev_rem: f32,
    fresh: bool,
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
    crate::rust_models_dir()
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
                "emb_g.weight"
                    | "enc_p.emb_pitch.weight"
                    | "dec.m_source.l_linear.weight"
                    | "dec.m_source.l_linear.bias"
            ) {
                cpu.insert(name.to_string(), out.clone());
            }
            w.insert(name.to_string(), eng.tensor_from_f32(&out, shape));
        }
        // 先に synth.json を解釈 (QKV fusion の層数・次元が version 依存のため)
        let cfg_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("synth.json"))?)?;
        let gu = |k: &str, d: u64| cfg_json[k].as_u64().unwrap_or(d) as u32;
        let gvec = |k: &str| -> Vec<u32> {
            cfg_json[k]
                .as_array()
                .and_then(|a| Some(a.iter().map(|v| v.as_u64().unwrap() as u32).collect()))
                .unwrap_or_default()
        };
        let version = gu("version", 2);
        let cfg = SynthConfig {
            version,
            in_channels: if version == 1 { 256 } else { 768 },
            random_test: cfg_json["random_test"].as_bool().unwrap_or(false),
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
        // TextEncoder QKV fusion: conv_q/k/v (hidden x hidden) を連結して 1 matmul 化
        // (層数は v1 で 3・v2 で 6)
        {
            let hd = cfg.hidden_channels as usize;
            let nl = cfg.n_layers;
            let names: Vec<String> = (0..nl)
                .flat_map(|l| {
                    [
                        format!("enc_p.encoder.attn_layers.{l}.conv_q.weight"),
                        format!("enc_p.encoder.attn_layers.{l}.conv_k.weight"),
                        format!("enc_p.encoder.attn_layers.{l}.conv_v.weight"),
                        format!("enc_p.encoder.attn_layers.{l}.conv_q.bias"),
                        format!("enc_p.encoder.attn_layers.{l}.conv_k.bias"),
                        format!("enc_p.encoder.attn_layers.{l}.conv_v.bias"),
                    ]
                })
                .collect();
            let data: std::collections::HashMap<String, Vec<f32>> = names
                .iter()
                .map(|n| {
                    let t = w.get(n).unwrap_or_else(|| panic!("weight {n} (synth.json の n_layers/version と重みの層数が不一致の可能性)"));
                    (n.clone(), eng.readback_f32(t))
                })
                .collect();
            for l in 0..nl {
                let get = |k: &str| -> Vec<f32> { data.get(k).expect(k).clone() };
                let qw = get(&format!("enc_p.encoder.attn_layers.{l}.conv_q.weight"));
                let kw = get(&format!("enc_p.encoder.attn_layers.{l}.conv_k.weight"));
                let vw = get(&format!("enc_p.encoder.attn_layers.{l}.conv_v.weight"));
                let mut qkv_w = Vec::with_capacity(hd * 3 * hd);
                qkv_w.extend_from_slice(&qw);
                qkv_w.extend_from_slice(&kw);
                qkv_w.extend_from_slice(&vw);
                w.insert(
                    format!("enc_p.encoder.attn_layers.{l}.qkv.weight"),
                    eng.tensor_from_f32(&qkv_w, vec![]),
                );
                let qb = get(&format!("enc_p.encoder.attn_layers.{l}.conv_q.bias"));
                let kb = get(&format!("enc_p.encoder.attn_layers.{l}.conv_k.bias"));
                let vb = get(&format!("enc_p.encoder.attn_layers.{l}.conv_v.bias"));
                let mut qkv_b = Vec::with_capacity(hd * 3);
                qkv_b.extend_from_slice(&qb);
                qkv_b.extend_from_slice(&kb);
                qkv_b.extend_from_slice(&vb);
                w.insert(
                    format!("enc_p.encoder.attn_layers.{l}.qkv.bias"),
                    eng.tensor_from_f32(&qkv_b, vec![]),
                );
            }
        }
        let emb_g = cpu.get("emb_g.weight").context("emb_g")?;
        let gin = cfg.gin_channels as usize;
        Ok(Self {
            cfg,
            debug_keep: std::cell::RefCell::new(None),
            emb_g_row0: emb_g[..gin].to_vec(),
            emb_pitch: cpu
                .get("enc_p.emb_pitch.weight")
                .context("emb_pitch")?
                .clone(),
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
            self.g(&format!("{wname}.weight")),
            x,
            Some(self.g(&format!("{wname}.bias"))),
            out,
            cout,
            t,
            cin,
            1,
            cin,
            1,
            0,
            0,
            1,
            t,
            x_base,
            0,
            t,
            1,
            0,
            0,
            scale,
            1,
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

    /// k 番 conv (same padding) [C, T] → [C', T]。
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
        self.conv_k_post(g, x, x_base, wname, out, cin, cout, k, dil, t, 0)
    }

    /// conv + 残差加算 fuse: out = conv(x) + bias + residual。
    /// im2col 経路は matmul_tiled_res、gemm 経路は conv 後 add_off。
    #[allow(clippy::too_many_arguments)]
    fn conv_k_res(
        &self,
        g: &mut Graph<'_>,
        x: &GpuTensor,
        residual: &GpuTensor,
        wname: &str,
        out: &GpuTensor,
        cin: u32,
        cout: u32,
        k: u32,
        dil: u32,
        t: u32,
    ) {
        let w = self.g(&format!("{wname}.weight"));
        let b = self.g(&format!("{wname}.bias"));
        if std::env::var("RCWX_CONV").ok().as_deref() == Some("gemm") {
            g.matmul_conv(
                w,
                x,
                Some(b),
                out,
                cout,
                t,
                cin * k,
                k,
                dil,
                (k - 1) * dil / 2,
                t,
                0,
            );
            g.add_off(out, residual, out, cout * t, 0, 0, false);
            return;
        }
        if (cin * k * t) as usize * 4 > 96_000_000 {
            g.conv1d(
                x,
                w,
                Some(b),
                out,
                cout,
                cin,
                k,
                1,
                dil,
                (k - 1) * dil / 2,
                t,
                t,
                1,
            );
            g.add_off(out, residual, out, cout * t, 0, 0, false);
            return;
        }
        let ic = g.engine_tensor_uninit((cin * k * t) as usize);
        g.im2col_off(x, 0, &ic, cin, k, 1, dil, (k - 1) * dil / 2, t, t);
        g.matmul_ex_res(
            w,
            &ic,
            Some(b),
            residual,
            out,
            cout,
            t,
            cin * k,
            cin * k,
            1,
            1,
            t,
            t,
            1,
        );
    }

    /// conv_k + 畳み込み直後の活性化 fuse (post: 0=none 1=lrelu)。
    /// 既定は implicit-GEMM (im2col なし)。RCWX_CONV=im2col で従来経路。
    fn conv_k_post(
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
        post: u32,
    ) {
        // 30s 持続 A/B: im2col+matmul_tiled 43.8ms vs implicit-GEMM 49.0ms → im2col 既定。
        // RCWX_CONV=gemm で implicit-GEMM、かつ lrelu fuse は gemm のみ有効。
        let use_gemm = x_base == 0 && std::env::var("RCWX_CONV").ok().as_deref() == Some("gemm");
        if use_gemm {
            g.matmul_conv(
                self.g(&format!("{wname}.weight")),
                x,
                Some(self.g(&format!("{wname}.bias"))),
                out,
                cout,
                t,
                cin * k,
                k,
                dil,
                (k - 1) * dil / 2,
                t,
                post,
            );
            return;
        }
        if x_base == 0 && (cin * k * t) as usize * 4 > 96_000_000 {
            g.conv1d(
                x,
                self.g(&format!("{wname}.weight")),
                Some(self.g(&format!("{wname}.bias"))),
                out,
                cout,
                cin,
                k,
                1,
                dil,
                (k - 1) * dil / 2,
                t,
                t,
                1,
            );
            if post == 1 {
                g.pointwise(out, None, cout * t, 1, 0.1, 0, 0, 0);
            }
            return;
        }
        let ic = g.engine_tensor_uninit((cin * k * t) as usize);
        g.im2col_off(x, x_base, &ic, cin, k, 1, dil, (k - 1) * dil / 2, t, t);
        if post == 0 {
            g.matmul_ex(
                self.g(&format!("{wname}.weight")),
                &ic,
                Some(self.g(&format!("{wname}.bias"))),
                out,
                cout,
                t,
                cin * k,
                1,
                cin * k,
                1,
                0,
                0,
                1,
                t,
                0,
                0,
                t,
                1,
                0,
                0,
                1.0,
                1,
            );
        } else {
            g.matmul_ex_x(
                self.g(&format!("{wname}.weight")),
                &ic,
                Some(self.g(&format!("{wname}.bias"))),
                None,
                out,
                cout,
                t,
                cin * k,
                cin * k,
                1,
                1,
                t,
                t,
                1,
                1.0,
                post,
            );
        }
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
            let uv = if f0[(i / upp as usize).min(l - 1)] > 0.0 {
                1.0
            } else {
                0.0
            };
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

    /// ストリーミング用 sinegen: 位相を SineState で持ち越す。
    /// 窓はローカルの cum/tou 補間で計算し、wrap 補正は持ち越し位相に適用する。
    pub fn sinegen_carry(&self, f0_win: &[f32], state: &mut SineState) -> Vec<f32> {
        let upp = self.cfg.sr as usize / 100;
        let l = f0_win.len();
        let n = l * upp;
        let sr = self.cfg.sr as f32;
        let rad: Vec<f32> = f0_win.iter().map(|f| (*f / sr).rem_euclid(1.0)).collect();
        let mut cum = vec![0.0f32; l];
        let mut acc = 0.0f32;
        for i in 0..l {
            acc += rad[i];
            cum[i] = acc * upp as f32;
        }
        let tou = |i: usize| -> f32 {
            if n == 1 {
                return cum[0];
            }
            let scale = (l - 1) as f32 / (n - 1) as f32;
            let pos = i as f32 * scale;
            let i0 = pos.floor() as usize;
            let i1 = (i0 + 1).min(l - 1);
            let fr = pos - i0 as f32;
            cum[i0] + (cum[i1] - cum[i0]) * fr
        };
        let two_pi = 2.0 * std::f32::consts::PI;
        let mut out = vec![0.0f32; n];
        let mut phase = state.phase;
        let mut prev_rem = state.prev_rem;
        let mut first = state.fresh;
        for i in 0..n {
            let rem = tou(i).rem_euclid(1.0);
            if !first && rem < prev_rem {
                phase -= 1.0;
            }
            first = false;
            prev_rem = rem;
            let f_idx = (i / upp).min(l - 1);
            phase += rad[f_idx];
            let uv = if f0_win[f_idx] > 0.0 { 1.0 } else { 0.0 };
            let sine = (phase * two_pi).sin() * 0.1 * uv;
            out[i] = (self.l_linear_w * sine + self.l_linear_b).tanh();
        }
        state.phase = phase;
        state.prev_rem = prev_rem;
        state.fresh = false;
        out
    }

    /// feats [T, in_channels] (GPU) + pitch idx (CPU) + pitchf Hz (CPU) → 音声 [T * zc]。
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
    pub fn encode_raw(
        &self,
        eng: &Engine,
        feats: &GpuTensor,
        pitch_idx: &[u32],
        t: u32,
    ) -> GpuTensor {
        let pitch_emb = eng.tensor_from_f32(&self.pitch_embedding(pitch_idx, t), vec![]);
        let mut g = eng.graph();
        let z = self.encode_graph(eng, &mut g, feats, &pitch_emb, t);
        g.submit().expect("submit");
        z
    }

    /// pitch 埋め込み行 [h, t] (ch-major) を CPU で構築。
    pub fn pitch_embedding(&self, pitch_idx: &[u32], t: u32) -> Vec<f32> {
        let h = self.cfg.hidden_channels as usize;
        let mut out = vec![0.0f32; h * t as usize];
        for r in 0..t as usize {
            let idx = pitch_idx.get(r).copied().unwrap_or(0).clamp(0, 255) as usize;
            for c in 0..h {
                out[c * t as usize + r] = self.emb_pitch[idx * h + c] * (h as f32).sqrt();
            }
        }
        out
    }

    /// TextEncoder + Flow を graph に記録 (submit しない・プランキャッシュ用)。
    pub fn encode_graph(
        &self,
        eng: &Engine,
        mut g: &mut Graph<'_>,
        feats: &GpuTensor,
        pitch_emb: &GpuTensor,
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

        // ---- TextEncoder ----
        // x[ch, t] = (Σ_k feats[t,k]·W_phone[ch,k])·√h + emb_pitch[pitch[t]][ch]
        // (feats 次元は v1=256 / v2=768)
        let in_ch = cfg.in_channels;
        let x0 = eng.tensor_uninit((h * t) as usize);
        g.matmul_ex(
            self.g("enc_p.emb_phone.weight"),
            feats,
            Some(self.g("enc_p.emb_phone.bias")),
            &x0,
            h,
            t,
            in_ch,
            1,
            in_ch,
            1,
            0,
            0,
            in_ch,
            1,
            0,
            0,
            t,
            1,
            0,
            0,
            (h as f32).sqrt(),
            1,
        );
        let mut x = eng.tensor_uninit((h * t) as usize);
        g.add(&x0, pitch_emb, &x, h * t);
        g.lrelu(&x, h * t);
        if let Some(k) = self.debug_keep.borrow_mut().as_mut() {
            k.push(("x_in", x.clone_handle()));
        }

        // Encoder (v1: 3 層 / v2: 6 層。QKV は fusion 済み重みで 1 matmul、q/k/v は base offset 参照)
        for l in 0..nl {
            let p = format!("enc_p.encoder.attn_layers.{l}");
            let qkv = eng.tensor_uninit((3 * h * t) as usize);
            g.matmul_ex(
                self.g(&format!("{p}.qkv.weight")),
                &x,
                Some(self.g(&format!("{p}.qkv.bias"))),
                &qkv,
                3 * h,
                t,
                h,
                1,
                h,
                1,
                0,
                0,
                1,
                t,
                0,
                0,
                t,
                1,
                0,
                0,
                1.0,
                1,
            );
            // scores[heads, T, T] = (q·k^T)/√dk + rel (q = qkv[0:h], k = qkv[h:2h])
            let scores = eng.tensor_uninit((heads * t * t) as usize);
            g.matmul(
                &qkv,
                &qkv,
                None,
                &scores,
                t,
                t,
                dk,
                heads,
                1,
                t,
                0,
                dk * t,
                1,
                t,
                h * t,
                dk * t,
                t,
                1,
                0,
                t * t,
                1.0 / (dk as f32).sqrt(),
            );
            g.rel_attn_scores(
                &qkv,
                self.g(&format!("{p}.emb_rel_k")),
                &scores,
                t,
                dk,
                ws,
                heads,
                1.0 / (dk as f32).sqrt(),
            );

            if l == 0 {
                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                    kk.push(("scores0", scores.clone_handle()));
                }
            }
            g.softmax(&scores, heads * t, t);
            // ctx = p·v + p·rel_v (v = qkv[2h:3h])
            let ctx = eng.tensor_uninit((h * t) as usize);
            g.matmul(
                &scores,
                &qkv,
                None,
                &ctx,
                t,
                dk,
                t,
                heads,
                t,
                1,
                0,
                t * t,
                t,
                1,
                2 * h * t,
                dk * t,
                1,
                t,
                0,
                dk * t, // C: channel-major [ch=h*96+d, i] — ch*T + i
                1.0,
            );
            if l == 0 {
                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                    kk.push(("p_attn0", scores.clone_handle()));
                    kk.push(("ctx0", ctx.clone_handle()));
                }
            }
            g.rel_attn_values(
                &scores,
                self.g(&format!("{p}.emb_rel_v")),
                &ctx,
                t,
                dk,
                ws,
                heads,
            );
            let attn = eng.tensor_uninit((h * t) as usize);
            self.conv1x1(&mut g, &ctx, 0, &format!("{p}.conv_o"), &attn, h, h, t, 1.0);
            if l == 0 {
                if let Some(k) = self.debug_keep.borrow_mut().as_mut() {
                    k.push(("attn0", attn.clone_handle()));
                }
            }
            let xn = eng.tensor_uninit((h * t) as usize);
            g.add_layer_norm(
                &x,
                &attn,
                self.g(&format!("enc_p.encoder.norm_layers_1.{l}.gamma")),
                self.g(&format!("enc_p.encoder.norm_layers_1.{l}.beta")),
                &xn,
                t,
                h,
                1,
                t,
                1,
                t,
                1e-5,
            );
            x = xn;

            if l == 0 {
                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                    kk.push(("norm1_0", x.clone_handle()));
                }
            }
            // FFN: conv k3 → gelu_approx → conv k3 → 残差 → LN
            let f = format!("enc_p.encoder.ffn_layers.{l}");
            let ff1 = eng.tensor_uninit((filt * t) as usize);
            // FFN conv1 の relu は matmul 出力に fuse
            self.conv_k_post(
                &mut g,
                &x,
                0,
                &format!("{f}.conv_1"),
                &ff1,
                h,
                filt,
                cfg.kernel_size,
                1,
                t,
                2,
            );
            if l == 0 {
                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                    kk.push(("ffn0_gelu_mid", ff1.clone_handle()));
                }
            }
            let ff2 = eng.tensor_uninit((h * t) as usize);
            self.conv_k(
                &mut g,
                &ff1,
                0,
                &format!("{f}.conv_2"),
                &ff2,
                filt,
                h,
                cfg.kernel_size,
                1,
                t,
            );
            if l == 0 {
                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                    kk.push(("ffn0", ff2.clone_handle()));
                }
            }
            let xn2 = eng.tensor_uninit((h * t) as usize);
            g.add_layer_norm(
                &x,
                &ff2,
                self.g(&format!("enc_p.encoder.norm_layers_2.{l}.gamma")),
                self.g(&format!("enc_p.encoder.norm_layers_2.{l}.beta")),
                &xn2,
                t,
                h,
                1,
                t,
                1,
                t,
                1e-5,
            );
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
            let x0_base = 0u32; // z rows [0:96]
            let x1_base = 96 * t; // z rows [96:192]
            // pre(x0) [192, T]
            let hbuf = eng.tensor_uninit((h * t) as usize);
            self.conv1x1(
                &mut g,
                &z,
                x0_base,
                &format!("{fp}.pre"),
                &hbuf,
                inter / 2,
                h,
                t,
                1.0,
            );
            let hbuf_keep = hbuf.clone_handle();
            // cond(g) [3*2h, T]
            let cond = eng.tensor_uninit((h * 6) as usize * t as usize);
            self.conv1x1(
                &mut g,
                &gt,
                0,
                &format!("{fp}.enc.cond_layer"),
                &cond,
                cfg.gin_channels,
                h * 6,
                t,
                1.0,
            );
            let mut skip: Option<GpuTensor> = None;
            let mut hh = hbuf;
            for l in 0..3u32 {
                let lin = eng.tensor_uninit((h * 2 * t) as usize);
                self.conv_k(
                    &mut g,
                    &hh,
                    0,
                    &format!("{fp}.enc.in_layers.{l}"),
                    &lin,
                    h,
                    h * 2,
                    5,
                    1,
                    t,
                );
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
                    self.g(&format!("{fp}.enc.res_skip_layers.{l}.weight")),
                    &with_cond,
                    Some(self.g(&format!("{fp}.enc.res_skip_layers.{l}.bias"))),
                    &rs,
                    rs_out_ch,
                    t,
                    h,
                    1,
                    h,
                    1,
                    0,
                    0,
                    1,
                    t,
                    0,
                    0,
                    t,
                    1,
                    0,
                    0,
                    1.0,
                    1,
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
                            let zeros =
                                eng.tensor_from_f32(&vec![0.0f32; (h * t) as usize], vec![]);
                            g.add_off(&rs, &zeros, &skipn, h * t, h * t, 0, false);
                        }
                        Some(sk) => g.add_off(sk, &rs, &skipn, h * t, 0, h * t, false),
                    }
                    if i == 3 {
                        if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                            kk.push((
                                Box::leak(format!("wn6_h{l}").into_boxed_str()),
                                hn.clone_handle(),
                            ));
                            kk.push((
                                Box::leak(format!("wn6_skip{l}").into_boxed_str()),
                                skipn.clone_handle(),
                            ));
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
                                    kk.push((
                                        Box::leak(format!("wn6_skip{l}").into_boxed_str()),
                                        skipn.clone_handle(),
                                    ));
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
            self.conv1x1(
                &mut g,
                &skip_t,
                0,
                &format!("{fp}.post"),
                &m,
                h,
                inter / 2,
                t,
                1.0,
            );
            if i == 3 {
                if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
                    kk.push(("flow6_wn", skip_t.clone_handle()));
                    kk.push(("flow6_post", m.clone_handle()));
                }
            }
            if let Some(k) = self.debug_keep.borrow_mut().as_mut() {
                k.push((
                    Box::leak(format!("flow{}_pre", i * 2).into_boxed_str()),
                    hbuf_keep.clone_handle(),
                ));
            }
            // zn = [x0 (copy), x1 - m]
            let zn = eng.tensor_uninit((inter * t) as usize);
            let zeros = eng.tensor_from_f32(&vec![0.0f32; (inter / 2 * t) as usize], vec![]);
            g.add_off3(&z, &zeros, &zn, inter / 2 * t, 0, 0, 0, false); // zn[0:96t] = x0 + 0
            g.add_off3(&z, &m, &zn, inter / 2 * t, x1_base, 0, x1_base, true); // zn[96t:] = x1-m
            if let Some(k) = self.debug_keep.borrow_mut().as_mut() {
                k.push((
                    Box::leak(format!("flow{}_out", i * 2).into_boxed_str()),
                    zn.clone_handle(),
                ));
            }
            z = zn;
        }
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
        let har = self.sinegen_range(pitchf, frame_start as usize, n as usize);
        self.decode_window_har(eng, z, &har, n)
    }

    /// har (窓分・CPU 生成済み) を与えるデコーダ窓生成。ストリーミング用。
    pub fn decode_window_har(&self, eng: &Engine, z: &GpuTensor, har: &[f32], n: u32) -> GpuTensor {
        let har_t = eng.tensor_from_f32(har, vec![]);
        let mut g = eng.graph();
        let audio = self.decode_graph(eng, &mut g, z, &har_t, n, n);
        g.submit().expect("submit");
        audio
    }

    /// decode_graph の任意オフセット版 (チャンク分割生成用)。
    pub fn decode_graph_off(
        &self,
        eng: &Engine,
        g: &mut Graph<'_>,
        z_full: &GpuTensor,
        har_t: &GpuTensor,
        n_in: u32,
        off: u32,
        n: u32,
    ) -> GpuTensor {
        let inter = self.cfg.inter_channels;
        let z_dec = eng.tensor_uninit((inter * n) as usize);
        g.slice_ch(z_full, &z_dec, inter, n_in, off, n);
        self.decode_graph_inner(eng, g, &z_dec, har_t, n)
    }

    /// デコーダを graph に記録 (submit しない・プランキャッシュ用)。
    /// z_full [inter, n_in] (n_in > n なら末尾 n フレームを GPU で抽出) +
    /// har_t [n*zc] → 音声 [n * zc]。
    pub fn decode_graph(
        &self,
        eng: &Engine,
        g: &mut Graph<'_>,
        z_full: &GpuTensor,
        har_t: &GpuTensor,
        n_in: u32,
        n: u32,
    ) -> GpuTensor {
        let inter = self.cfg.inter_channels;
        let z_dec;
        let z: &GpuTensor = if n_in > n {
            z_dec = eng.tensor_uninit((inter * n) as usize);
            g.slice_ch(z_full, &z_dec, inter, n_in, n_in - n, n);
            &z_dec
        } else {
            z_full
        };
        self.decode_graph_inner(eng, g, z, har_t, n)
    }

    fn decode_graph_inner(
        &self,
        eng: &Engine,
        mut g: &mut Graph<'_>,
        z: &GpuTensor,
        har_t: &GpuTensor,
        n: u32,
    ) -> GpuTensor {
        let cfg = &self.cfg;
        let inter = cfg.inter_channels;
        let t = n;
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
        let chs: Vec<u32> = (0..=ups.len())
            .map(|i| cfg.upsample_initial_channel / (1u32 << i))
            .collect();
        // conv_pre + cond(g)
        let mut dx = eng.tensor_uninit((chs[0] * t) as usize);
        self.conv_k(&mut g, z, 0, "dec.conv_pre", &dx, inter, chs[0], 7, 1, t);
        let gcond = eng.tensor_uninit((chs[0] * t) as usize);
        self.conv1x1(
            &mut g,
            &gt,
            0,
            "dec.cond",
            &gcond,
            cfg.gin_channels,
            chs[0],
            t,
            1.0,
        );
        if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
            kk.push(("dec_conv_pre", dx.clone_handle()));
        }
        let dx2 = eng.tensor_uninit((chs[0] * t) as usize);
        g.add(&dx, &gcond, &dx2, chs[0] * t);
        dx = dx2;

        // har → GPU (呼び出し側で用意)
        if let Some(kk) = self.debug_keep.borrow_mut().as_mut() {
            kk.push(("har", har_t.clone_handle()));
        }

        let mut len = t;
        // 計測用: RCWX_DEC_STAGES=k は段 k まで実施し以降をゼロ化 (タイミング研究専用)
        let dec_stages: i32 = std::env::var("RCWX_DEC_STAGES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(99);
        for i in 0..ups.len() {
            let (u, kk) = (ups[i], cfg.upsample_kernel_sizes[i]);
            let (cin, cout) = (chs[i], chs[i + 1]);
            if i as i32 > dec_stages {
                len *= u;
                dx = eng.tensor_zeros((chs[i + 1] * len) as usize, vec![]);
                continue;
            }
            // lrelu → upsample
            g.lrelu(&dx, cin * len);
            let up = eng.tensor_uninit((cout * len * u) as usize);
            g.conv_transpose1d(
                &dx,
                self.g(&format!("dec.ups.{i}.weight")),
                Some(self.g(&format!("dec.ups.{i}.bias"))),
                &up,
                cout,
                cin,
                kk,
                u,
                (kk - u) / 2,
                len,
                len * u,
            );
            // noise_conv on har
            let ns = eng.tensor_uninit((cout * len * u) as usize);
            let stride_f0: u32 = ups[i + 1..].iter().product::<u32>().max(1);
            let nk = if stride_f0 > 1 { stride_f0 * 2 } else { 1 };
            if nk == 1 {
                g.conv1d(
                    &har_t,
                    self.g(&format!("dec.noise_convs.{i}.weight")),
                    Some(self.g(&format!("dec.noise_convs.{i}.bias"))),
                    &ns,
                    cout,
                    1,
                    1,
                    1,
                    1,
                    0,
                    har_t.len as u32,
                    len * u,
                    1,
                );
            } else {
                g.conv1d(
                    &har_t,
                    self.g(&format!("dec.noise_convs.{i}.weight")),
                    Some(self.g(&format!("dec.noise_convs.{i}.bias"))),
                    &ns,
                    cout,
                    1,
                    nk,
                    stride_f0,
                    1,
                    stride_f0 / 2,
                    har_t.len as u32,
                    len * u,
                    1,
                );
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
                // summed は読み取り専用 → 初回コピー不要 (bx は m ループ内で読むだけ)
                let mut bx = summed.clone_handle();
                for m in 0..3u32 {
                    let dil = [1u32, 3, 5][m as usize];
                    // lrelu → convs1[m] → convs2[m] (残差加算 fuse) → bx 更新
                    let lr = eng.tensor_uninit((cout * len * u) as usize);
                    g.pointwise(&bx, Some(&lr), cout * len * u, 1, 0.1, 0, 0, 0);

                    // conv1 の lrelu は GEMM のストアに融合する。別カーネルだと p99 が 20ms を超える。
                    let c1 = eng.tensor_uninit((cout * len * u) as usize);
                    self.conv_k_post(
                        &mut g,
                        &lr,
                        0,
                        &format!("{rb}1.{m}"),
                        &c1,
                        cout,
                        cout,
                        k_j,
                        dil,
                        len * u,
                        1,
                    );
                    let c2 = eng.tensor_uninit((cout * len * u) as usize);
                    self.conv_k_res(
                        &mut g,
                        &c1,
                        &bx,
                        &format!("{rb}2.{m}"),
                        &c2,
                        cout,
                        cout,
                        k_j,
                        1,
                        len * u,
                    );
                    bx = c2;
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
            g.pointwise(
                &acc.unwrap(),
                Some(&avg),
                cout * len * u,
                4,
                1.0 / 3.0,
                0,
                0,
                0,
            );
            dx = avg;
            len *= u;
        }
        // 最終: lrelu → conv_post (bias 無し) → tanh
        g.lrelu(&dx, chs[ups.len()] * len);
        let audio = eng.tensor_uninit(len as usize);
        self.conv_k_nobias(
            g,
            &dx,
            "dec.conv_post.weight",
            &audio,
            chs[ups.len()],
            1,
            7,
            1,
            len,
        );
        g.tanh_inplace(&audio, len);

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
                x,
                self.g(wname),
                None,
                out,
                cout,
                cin,
                k,
                1,
                dil,
                (k - 1) * dil / 2,
                t,
                t,
                1,
            );
            return;
        }
        let ic = g.engine_tensor_uninit((cin * k * t) as usize);
        g.im2col_off(x, 0, &ic, cin, k, 1, dil, (k - 1) * dil / 2, t, t);
        g.matmul(
            self.g(wname),
            &ic,
            None,
            out,
            cout,
            t,
            cin * k,
            1,
            cin * k,
            1,
            0,
            0,
            1,
            t,
            0,
            0,
            t,
            1,
            0,
            0,
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
            // z の末尾抽出は GPU 内で (slice_ch) — 全 z を一度アップロード
            let z_full_t = eng.tensor_from_f32(&z_cpu, vec![]);
            let har = self.sinegen_range(pitchf, w_start, n as usize);
            let har_t = eng.tensor_from_f32(&har, vec![]);
            let mut g = eng.graph();
            let audio =
                self.decode_graph_off(eng, &mut g, &z_full_t, &har_t, t as u32, w_start as u32, n);
            g.submit().expect("submit");
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
