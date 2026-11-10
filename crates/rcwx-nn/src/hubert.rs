//! HuBERT (RVC hubert_base.pt 相当) の独自 wgpu エンジン実装。
//!
//! transformers `HubertModel` (group mode) と同一の計算順序:
//! conv frontend 7段 (padding なし / bias なし / GN は第1段のみ / 各段 exact GELU) →
//! LayerNorm(512)+Linear(512→768) → pos_conv (depthwise k128 groups16, pad (64,63)) +
//! 残差 → encoder LN → 12層 (post-LN) → v2: そのまま / v1: 9層+LN+final_proj。
//!
//! 重みは `tools/convert_weights.py hubert` の出力 (hubert.safetensors)。
//! レイアウト: frontend は [C, L]、transformer は [T, C] (fp_ln で転置書き込み)。

use anyhow::{Context, Result};

use crate::engine::{Engine, GpuTensor};

pub const CONV_KERNELS: [u32; 7] = [10, 3, 3, 3, 3, 2, 2];
pub const CONV_STRIDES: [u32; 7] = [5, 2, 2, 2, 2, 2, 2];

const CONV_CH: u32 = 512;
const H: u32 = 768;
const HEADS: u32 = 12;
const HEAD_DIM: u32 = 64;
const FFN: u32 = 3072;
const POS_KERNEL: u32 = 128;
const POS_GROUPS: u32 = 16;
const LN_EPS: f32 = 1e-5;

pub struct LayerW {
    pub qkv_w: GpuTensor,
    pub qkv_b: GpuTensor,
    pub o_w: GpuTensor,
    pub o_b: GpuTensor,
    pub ln_a_g: GpuTensor,
    pub ln_a_b: GpuTensor,
    pub ff1_w: GpuTensor,
    pub ff1_b: GpuTensor,
    pub ff2_w: GpuTensor,
    pub ff2_b: GpuTensor,
    pub ln_f_g: GpuTensor,
    pub ln_f_b: GpuTensor,
}

pub struct PlanEntry {
    pub plan: crate::engine::Plan,
    pub input: GpuTensor,
    pub out: GpuTensor,
}

pub struct Hubert {
    plans: std::cell::RefCell<std::collections::HashMap<(u32, u32, u32), PlanEntry>>,
    pub convs: Vec<(GpuTensor, Option<GpuTensor>)>,
    pub gn0_g: GpuTensor,
    pub gn0_b: GpuTensor,
    pub fp_ln_g: GpuTensor,
    pub fp_ln_b: GpuTensor,
    pub fp_proj_w: GpuTensor,
    pub fp_proj_b: GpuTensor,
    pub pos_w: GpuTensor,
    pub pos_b: GpuTensor,
    pub enc_ln_g: GpuTensor,
    pub enc_ln_b: GpuTensor,
    pub layers: Vec<LayerW>,
    pub final_proj: Option<(GpuTensor, GpuTensor)>,
}

pub fn default_weights_path() -> std::path::PathBuf {
    crate::rust_models_dir().join("hubert.safetensors")
}

/// RVC モデルバージョン → HuBERT 抽出 (layer, dim)。
/// v1: layer 9 + encoder LN + final_proj → 256-dim / v2: layer 12 (last_hidden) → 768-dim。
pub fn layer_dim_for_version(version: u32) -> (u32, u32) {
    if version == 1 { (9, 256) } else { (12, 768) }
}

impl Hubert {
    pub fn load(eng: &Engine, path: &std::path::Path) -> Result<Self> {
        let raw = std::fs::read(path)?;
        let st = safetensors::SafeTensors::deserialize(&raw)?;
        let gpu = |name: &str| -> Result<GpuTensor> {
            let v = st.tensor(name).with_context(|| format!("tensor {name}"))?;
            let shape = v.shape().to_vec();
            let n: usize = shape.iter().product();
            let mut out = Vec::with_capacity(n);
            for c in v.data().chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Ok(eng.tensor_from_f32(&out, shape))
        };

        let mut convs = Vec::new();
        for i in 0..7 {
            let w = gpu(&format!("conv.{i}.weight"))?;
            let b = if st.tensor(&format!("conv.{i}.bias")).is_ok() {
                Some(gpu(&format!("conv.{i}.bias"))?)
            } else {
                None
            };
            convs.push((w, b));
        }

        let mut layers = Vec::new();
        for i in 0..12 {
            layers.push(LayerW {
                qkv_w: gpu(&format!("layers.{i}.qkv.weight"))?,
                qkv_b: gpu(&format!("layers.{i}.qkv.bias"))?,
                o_w: gpu(&format!("layers.{i}.o.weight"))?,
                o_b: gpu(&format!("layers.{i}.o.bias"))?,
                ln_a_g: gpu(&format!("layers.{i}.ln_a.weight"))?,
                ln_a_b: gpu(&format!("layers.{i}.ln_a.bias"))?,
                ff1_w: gpu(&format!("layers.{i}.ff1.weight"))?,
                ff1_b: gpu(&format!("layers.{i}.ff1.bias"))?,
                ff2_w: gpu(&format!("layers.{i}.ff2.weight"))?,
                ff2_b: gpu(&format!("layers.{i}.ff2.bias"))?,
                ln_f_g: gpu(&format!("layers.{i}.ln_f.weight"))?,
                ln_f_b: gpu(&format!("layers.{i}.ln_f.bias"))?,
            });
        }

        let final_proj = if st.tensor("final_proj.weight").is_ok() {
            Some((gpu("final_proj.weight")?, gpu("final_proj.bias")?))
        } else {
            None
        };

        Ok(Self {
            plans: std::cell::RefCell::new(std::collections::HashMap::new()),
            convs,
            gn0_g: gpu("gn0.weight")?,
            gn0_b: gpu("gn0.bias")?,
            fp_ln_g: gpu("fp_ln.weight")?,
            fp_ln_b: gpu("fp_ln.bias")?,
            fp_proj_w: gpu("fp_proj.weight")?,
            fp_proj_b: gpu("fp_proj.bias")?,
            pos_w: gpu("pos_conv.weight")?,
            pos_b: gpu("pos_conv.bias")?,
            enc_ln_g: gpu("enc_ln.weight")?,
            enc_ln_b: gpu("enc_ln.bias")?,
            layers,
            final_proj,
        })
    }

    /// v2 抽出 (layer 12, 768ch)。wav: [N] (GPU 常駐)。戻り値 [T, 768]。
    pub fn features(&self, eng: &Engine, wav: &GpuTensor, n_samples: u32) -> GpuTensor {
        self.extract(eng, wav, n_samples, 12, 768)
    }

    /// モデルバージョンに対応する抽出 (v1: layer 9 + final_proj → 256 / v2: layer 12 → 768)。
    pub fn features_ver(
        &self,
        eng: &Engine,
        wav: &GpuTensor,
        n_samples: u32,
        version: u32,
    ) -> GpuTensor {
        let (layer, dim) = layer_dim_for_version(version);
        self.extract(eng, wav, n_samples, layer, dim)
    }

    /// キャッシュ済みプランのクローン (input への wav copy と out 読み戻しは
    /// 呼び出し側の統合 submit に載せる)。実行は extract 相当を自前で組み立てる
    /// リアルタイム経路のみが使う。
    pub fn features_plan_clone(
        &self,
        eng: &Engine,
        n_samples: u32,
        version: u32,
    ) -> (GpuTensor, GpuTensor, crate::engine::Plan) {
        let (layer, dim) = layer_dim_for_version(version);
        let mut plans = self.plans.borrow_mut();
        let entry = plans
            .entry((n_samples, layer, dim))
            .or_insert_with(|| self.build_plan(eng, n_samples, layer, dim));
        (
            entry.input.clone_handle(),
            entry.out.clone_handle(),
            entry.plan.clone(),
        )
    }

    /// layer=12 (v2) / 9 (v1), dim=768/256。戻り値 [T, dim]。
    /// 形状ごとに実行プランを構築して再利用 (バッファ・bind group・uniform はキャッシュ)。
    pub fn extract(
        &self,
        eng: &Engine,
        wav: &GpuTensor,
        n_samples: u32,
        layer: u32,
        dim: u32,
    ) -> GpuTensor {
        let mut plans = self.plans.borrow_mut();
        let entry = plans
            .entry((n_samples, layer, dim))
            .or_insert_with(|| self.build_plan(eng, n_samples, layer, dim));
        eng.replay(&entry.plan, Some((wav, &entry.input)));
        entry.out.clone_handle()
    }

    fn build_plan(&self, eng: &Engine, n_samples: u32, layer: u32, dim: u32) -> PlanEntry {
        let n_layers = layer.min(12);
        let mut g = eng.graph();
        let wav = eng.tensor_uninit(n_samples as usize);

        // ---- conv frontend [C, L] ----
        let mut lens: Vec<u32> = Vec::with_capacity(7);
        let mut l = n_samples;
        for i in 0..7 {
            l = (l - CONV_KERNELS[i]) / CONV_STRIDES[i] + 1;
            lens.push(l);
        }
        let t = lens[6];

        // frontend: im2col + matmul (tiled)。W [out_c, in_c*k] × IC [in_c*k, L_out]
        let mut x = eng.tensor_uninit((CONV_CH * lens[0]) as usize);
        {
            let (w0, b0) = &self.convs[0];
            let ic = eng.tensor_uninit((1 * 10 * lens[0]) as usize);
            g.im2col(&wav, &ic, 1, 10, 5, 1, 0, n_samples, lens[0]);
            g.matmul(
                w0,
                &ic,
                b0.as_ref(),
                &x,
                CONV_CH,
                lens[0],
                10,
                1,
                10,
                1,
                0,
                0,
                1,
                lens[0] as u32,
                0,
                0,
                lens[0],
                1,
                0,
                0,
                1.0,
            );
        }
        g.group_norm(&x, &self.gn0_g, &self.gn0_b, CONV_CH, lens[0], LN_EPS);
        g.gelu(&x, CONV_CH * lens[0]);
        for i in 1..7 {
            let (w, b) = &self.convs[i];
            let out = eng.tensor_uninit((CONV_CH * lens[i]) as usize);
            let kk = CONV_KERNELS[i];
            let ic = eng.tensor_uninit((CONV_CH * kk * lens[i]) as usize);
            g.im2col(
                &x,
                &ic,
                CONV_CH,
                kk,
                CONV_STRIDES[i],
                1,
                0,
                lens[i - 1],
                lens[i],
            );
            g.matmul(
                w,
                &ic,
                b.as_ref(),
                &out,
                CONV_CH,
                lens[i],
                CONV_CH * kk,
                1,
                CONV_CH * kk,
                1,
                0,
                0,
                1,
                lens[i],
                0,
                0,
                lens[i],
                1,
                0,
                0,
                1.0,
            );
            g.gelu(&out, CONV_CH * lens[i]);
            x = out;
        }

        // ---- feature projection: LN(512) → Linear(512→768)、[512,T] から [768,T] へ転置書き込み ----
        let x2 = eng.tensor_uninit((t * CONV_CH) as usize);
        g.layer_norm(
            &x,
            &self.fp_ln_g,
            &self.fp_ln_b,
            &x2,
            t,
            CONV_CH,
            1,
            t,
            CONV_CH,
            1,
            LN_EPS,
        );
        let pos_pre = eng.tensor_uninit((t * H) as usize);
        g.matmul(
            &x2,
            &self.fp_proj_w,
            Some(&self.fp_proj_b),
            &pos_pre,
            t,
            H,
            CONV_CH,
            1,
            CONV_CH,
            1,
            0,
            0,
            CONV_CH,
            1,
            0,
            0,
            1,
            t,
            0,
            0,
            1.0,
        );

        // ---- pos_conv (depthwise k128 groups16, zero pad (64,63)) + gelu + 残差 ----
        let pos = eng.tensor_uninit((H * t) as usize);
        g.conv1d(
            &pos_pre,
            &self.pos_w,
            Some(&self.pos_b),
            &pos,
            H,
            H,
            POS_KERNEL,
            1,
            1,
            POS_KERNEL / 2,
            t,
            t,
            POS_GROUPS,
        );
        g.gelu(&pos, H * t);
        let summed = eng.tensor_uninit((H * t) as usize);
        g.add(&pos_pre, &pos, &summed, H * t);

        // ---- encoder LN → [T, 768] ----
        let x = eng.tensor_uninit((t * H) as usize);
        g.layer_norm(
            &summed,
            &self.enc_ln_g,
            &self.enc_ln_b,
            &x,
            t,
            H,
            1,
            t,
            H,
            1,
            LN_EPS,
        );

        // ---- transformer layers (post-LN) ----
        // 作業バッファは全層で 1 回確保して再利用する。従来は層ごとに確保し plan が
        // 握り続けるため t≈1700 で scores 12 層分だけで ~1.7GB が resident 化し、
        // WDDM の residency 圧迫から以降の全ステージが数分の 1 に減速したり長尺 2 窓で
        // device lost する (実測)。同一バッファへの再書き込みは常に読み取り dispatch
        // の後 (キュー内順序) なので層をまたいだ aliasing は安全。x は ping-pong。
        let x_b = eng.tensor_uninit((t * H) as usize);
        let qkv_s = eng.tensor_uninit((t * 3 * H) as usize);
        let ctx_s = eng.tensor_uninit((t * H) as usize);
        let o_s = eng.tensor_uninit((t * H) as usize);
        let x3_s = eng.tensor_uninit((t * H) as usize);
        let ff_s = eng.tensor_uninit((t * FFN) as usize);
        let ff2_s = eng.tensor_uninit((t * H) as usize);
        let scores_s = if eng.attn_flash {
            None
        } else {
            Some(eng.tensor_uninit((HEADS * t * t) as usize))
        };
        let mut parity = false;
        for l in &self.layers[..n_layers as usize] {
            let (x_cur, x_nxt) = if parity { (&x_b, &x) } else { (&x, &x_b) };
            parity = !parity;
            g.matmul(
                x_cur,
                &l.qkv_w,
                Some(&l.qkv_b),
                &qkv_s,
                t,
                3 * H,
                H,
                1,
                H,
                1,
                0,
                0,
                H,
                1,
                0,
                0,
                3 * H,
                1,
                0,
                0,
                1.0,
            );
            if eng.attn_flash {
                // fused attention (flash 形式): T² スコア行列を材料化せず online softmax
                // で直接 ctx[T, 768] に書く。q/k/v は qkv バッファ内の strided 参照。
                g.flash_attn(&qkv_s, &ctx_s, t, 3 * H, HEADS, 0, H, 2 * H, 0.125, H);
            } else {
                // 古典 3-kernel 経路 (T² scores 材料化 — scratch 再利用)
                let scores = scores_s.as_ref().unwrap();
                g.matmul(
                    &qkv_s,
                    &qkv_s,
                    None,
                    scores,
                    t,
                    t,
                    HEAD_DIM,
                    HEADS,
                    3 * H,
                    1,
                    0,
                    HEAD_DIM,
                    3 * H,
                    1,
                    H,
                    HEAD_DIM,
                    t,
                    1,
                    0,
                    t * t,
                    0.125,
                );
                g.softmax(scores, HEADS * t, t);
                g.matmul(
                    scores,
                    &qkv_s,
                    None,
                    &ctx_s,
                    t,
                    HEAD_DIM,
                    t,
                    HEADS,
                    t,
                    1,
                    0,
                    t * t,
                    1,
                    3 * H,
                    2 * H,
                    HEAD_DIM,
                    H,
                    1,
                    0,
                    HEAD_DIM,
                    1.0,
                );
            }
            g.matmul(
                &ctx_s,
                &l.o_w,
                Some(&l.o_b),
                &o_s,
                t,
                H,
                H,
                1,
                H,
                1,
                0,
                0,
                H,
                1,
                0,
                0,
                H,
                1,
                0,
                0,
                1.0,
            );
            g.add_layer_norm(
                x_cur, &o_s, &l.ln_a_g, &l.ln_a_b, &x3_s, t, H, H, 1, H, 1, LN_EPS,
            );

            g.matmul(
                &x3_s,
                &l.ff1_w,
                Some(&l.ff1_b),
                &ff_s,
                t,
                FFN,
                H,
                1,
                H,
                1,
                0,
                0,
                H,
                1,
                0,
                0,
                FFN,
                1,
                0,
                0,
                1.0,
            );
            g.gelu(&ff_s, t * FFN);
            g.matmul(
                &ff_s,
                &l.ff2_w,
                Some(&l.ff2_b),
                &ff2_s,
                t,
                H,
                FFN,
                1,
                FFN,
                1,
                0,
                0,
                FFN,
                1,
                0,
                0,
                H,
                1,
                0,
                0,
                1.0,
            );
            g.add_layer_norm(
                &x3_s, &ff2_s, &l.ln_f_g, &l.ln_f_b, x_nxt, t, H, H, 1, H, 1, LN_EPS,
            );
        }
        // 偶数層は x_b に・奇数層は x に書く → 最終層の出力側 (n_layers 偶数で x)
        let x = if parity { x_b } else { x };

        // v1: layer<12 → encoder LN → final_proj
        if dim == 256 {
            let Some((fw, fb)) = self.final_proj.as_ref() else {
                panic!("dim=256 (v1) requires final_proj weights");
            };
            let xn = eng.tensor_uninit((t * H) as usize);
            g.layer_norm(
                &x,
                &self.enc_ln_g,
                &self.enc_ln_b,
                &xn,
                t,
                H,
                H,
                1,
                H,
                1,
                LN_EPS,
            );
            let out = eng.tensor_uninit((t * 256) as usize);
            g.matmul(
                &xn,
                fw,
                Some(fb),
                &out,
                t,
                256,
                H,
                1,
                H,
                1,
                0,
                0,
                H,
                1,
                0,
                0,
                256,
                1,
                0,
                0,
                1.0,
            );
            let plan = g.into_plan().expect("into_plan");
            return PlanEntry {
                plan,
                input: wav,
                out,
            };
        }
        let plan = g.into_plan().expect("into_plan");
        PlanEntry {
            plan,
            input: wav,
            out: x,
        }
    }
}

impl Hubert {
    /// 段階検証用: extract と同一計算を中間テンソル保持付きで実行し readback。
    pub fn debug_stages(
        &self,
        eng: &Engine,
        wav: &GpuTensor,
        n_samples: u32,
    ) -> Vec<(&'static str, Vec<f32>)> {
        let mut lens: Vec<u32> = Vec::new();
        let mut l = n_samples;
        for i in 0..7 {
            l = (l - CONV_KERNELS[i]) / CONV_STRIDES[i] + 1;
            lens.push(l);
        }
        let t = lens[6];
        let mut keep: Vec<(&'static str, GpuTensor)> = Vec::new();

        let mut g = eng.graph();
        let c0 = eng.tensor_uninit((CONV_CH * lens[0]) as usize);
        let (w0, b0) = &self.convs[0];
        g.conv1d(
            &wav,
            w0,
            b0.as_ref(),
            &c0,
            CONV_CH,
            1,
            CONV_KERNELS[0],
            CONV_STRIDES[0],
            1,
            0,
            n_samples,
            lens[0],
            1,
        );
        g.group_norm(&c0, &self.gn0_g, &self.gn0_b, CONV_CH, lens[0], LN_EPS);
        g.gelu(&c0, CONV_CH * lens[0]);
        let mut cur = c0;
        for i in 1..7 {
            let (w, b) = &self.convs[i];
            let o = eng.tensor_uninit((CONV_CH * lens[i]) as usize);
            g.conv1d(
                &cur,
                w,
                b.as_ref(),
                &o,
                CONV_CH,
                CONV_CH,
                CONV_KERNELS[i],
                CONV_STRIDES[i],
                1,
                0,
                lens[i - 1],
                lens[i],
                1,
            );
            g.gelu(&o, CONV_CH * lens[i]);
            cur = o;
        }
        let x2 = eng.tensor_uninit((t * CONV_CH) as usize);
        g.layer_norm(
            &cur,
            &self.fp_ln_g,
            &self.fp_ln_b,
            &x2,
            t,
            CONV_CH,
            1,
            t,
            CONV_CH,
            1,
            LN_EPS,
        );
        let pos_pre = eng.tensor_uninit((t * H) as usize);
        g.matmul(
            &x2,
            &self.fp_proj_w,
            Some(&self.fp_proj_b),
            &pos_pre,
            t,
            H,
            CONV_CH,
            1,
            CONV_CH,
            1,
            0,
            0,
            CONV_CH,
            1,
            0,
            0,
            1,
            t,
            0,
            0,
            1.0,
        );
        keep.push(("pos_pre", pos_pre.clone_buf()));
        let pos = eng.tensor_uninit((H * t) as usize);
        g.conv1d(
            &pos_pre,
            &self.pos_w,
            Some(&self.pos_b),
            &pos,
            H,
            H,
            POS_KERNEL,
            1,
            1,
            POS_KERNEL / 2,
            t,
            t,
            POS_GROUPS,
        );
        g.gelu(&pos, H * t);
        keep.push(("pos", pos.clone_buf()));
        let summed = eng.tensor_uninit((H * t) as usize);
        g.add(&pos_pre, &pos, &summed, H * t);
        let mut x = eng.tensor_uninit((t * H) as usize);
        g.layer_norm(
            &summed,
            &self.enc_ln_g,
            &self.enc_ln_b,
            &x,
            t,
            H,
            1,
            t,
            H,
            1,
            LN_EPS,
        );
        keep.push(("enc_pre", x.clone_buf()));

        for (li, l) in self.layers.iter().enumerate() {
            let qkv = eng.tensor_uninit((t * 3 * H) as usize);
            if li == 0 {
                keep.push(("L0.qkv", qkv.clone_buf()));
            }
            g.matmul(
                &x,
                &l.qkv_w,
                Some(&l.qkv_b),
                &qkv,
                t,
                3 * H,
                H,
                1,
                H,
                1,
                0,
                0,
                H,
                1,
                0,
                0,
                3 * H,
                1,
                0,
                0,
                1.0,
            );
            let ctx = eng.tensor_uninit((t * H) as usize);
            if eng.attn_flash {
                // fused attention (flash 形式) — features_ver と同一 (T² 材料化なし)
                g.flash_attn(&qkv, &ctx, t, 3 * H, HEADS, 0, H, 2 * H, 0.125, H);
            } else {
                // 古典 3-kernel 経路 (FXC 向き: コンパイルが速く実行も十分速い。
                // T² scores を材料化するため長尺は窓分割が前提)
                let scores = eng.tensor_uninit((HEADS * t * t) as usize);
                g.matmul(
                    &qkv,
                    &qkv,
                    None,
                    &scores,
                    t,
                    t,
                    HEAD_DIM,
                    HEADS,
                    3 * H,
                    1,
                    0,
                    HEAD_DIM,
                    3 * H,
                    1,
                    H,
                    HEAD_DIM,
                    t,
                    1,
                    0,
                    t * t,
                    0.125,
                );
                g.softmax(&scores, HEADS * t, t);
                g.matmul(
                    &scores,
                    &qkv,
                    None,
                    &ctx,
                    t,
                    HEAD_DIM,
                    t,
                    HEADS,
                    t,
                    1,
                    0,
                    t * t,
                    1,
                    3 * H,
                    2 * H,
                    HEAD_DIM,
                    H,
                    1,
                    0,
                    HEAD_DIM,
                    1.0,
                );
            }
            let o_out = eng.tensor_uninit((t * H) as usize);
            g.matmul(
                &ctx,
                &l.o_w,
                Some(&l.o_b),
                &o_out,
                t,
                H,
                H,
                1,
                H,
                1,
                0,
                0,
                H,
                1,
                0,
                0,
                H,
                1,
                0,
                0,
                1.0,
            );
            if li == 0 {
                keep.push(("L0.ctx", ctx.clone_buf()));
                keep.push(("L0.attn_out", o_out.clone_buf()));
            }
            let x3 = eng.tensor_uninit((t * H) as usize);
            g.add_layer_norm(
                &x, &o_out, &l.ln_a_g, &l.ln_a_b, &x3, t, H, H, 1, H, 1, LN_EPS,
            );
            let ff = eng.tensor_uninit((t * FFN) as usize);
            // gelu は matmul 出力に fuse (erf 近似は GELU カーネルと同一式)
            g.matmul_ex_x(
                &x3,
                &l.ff1_w,
                Some(&l.ff1_b),
                None,
                &ff,
                t,
                FFN,
                H,
                H,
                1,
                H,
                1,
                FFN,
                1,
                1.0,
                3,
            );
            if li == 0 {
                keep.push(("L0.ff_hidden", ff.clone_buf()));
            }
            let ff2 = eng.tensor_uninit((t * H) as usize);
            g.matmul(
                &ff,
                &l.ff2_w,
                Some(&l.ff2_b),
                &ff2,
                t,
                H,
                FFN,
                1,
                FFN,
                1,
                0,
                0,
                FFN,
                1,
                0,
                0,
                H,
                1,
                0,
                0,
                1.0,
            );
            let x5 = eng.tensor_uninit((t * H) as usize);
            g.add_layer_norm(
                &x3, &ff2, &l.ln_f_g, &l.ln_f_b, &x5, t, H, H, 1, H, 1, LN_EPS,
            );
            x = x5;
            if li == 0 {
                keep.push(("L0.ff_out", ff2.clone_buf()));
                keep.push(("L0.ln_a", x3.clone_buf()));
            }
            if li < 2 || li == 11 {
                keep.push((
                    Box::leak(format!("layer{li}").into_boxed_str()),
                    x.clone_buf(),
                ));
            }
        }
        g.submit().expect("submit");
        keep.into_iter()
            .map(|(n, b)| (n, eng.readback_f32(&b)))
            .collect()
    }
}

impl GpuTensor {
    /// デバッグ用: 同一内容の別バッファ複製 (readback 用に保持)。
    fn clone_buf(&self) -> GpuTensor {
        GpuTensor {
            buffer: self.buffer.clone(),
            shape: self.shape.clone(),
            len: self.len,
        }
    }
}
