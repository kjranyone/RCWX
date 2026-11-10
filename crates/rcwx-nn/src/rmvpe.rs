//! RMVPE (F0 推論) の独自エンジン実装。
//!
//! 重み: tools/convert_rmvpe.py + BN fold 済み rmvpe_folded.safetensors。
//! - MelSpectrogram: rustfft による STFT (n_fft 1024 / hop 160 / reflect pad 432) — CPU fp32
//! - DeepUnet: WGSL conv2d (encoder/intermediate/decoder)。BN は fold 済み
//! - BiGRU + Linear + Sigmoid: WGSL matmul 逐次 (T=56 程度)
//! - decode (argmax + 9bin 加重平均): CPU
//!
//! レイアウト: [B, C, H(frames), W(mel)] を engine では [C, H, W] 平坦で扱う。

use anyhow::{Context, Result};

use crate::engine::{Engine, GpuTensor, Graph};

const N_MEL: u32 = 128;
const EN_DE_LAYERS: u32 = 5;
const INTER_LAYERS: u32 = 4;
const N_BLOCKS: u32 = 4;

pub struct Rmvpe {
    w: std::collections::HashMap<String, GpuTensor>,
    pub mel_basis: Vec<f32>,   // [128, 513]
    pub hann: Vec<f32>,        // [1024]
    pub cents_mapping: Vec<f32>, // [360]
}

pub fn default_rmvpe_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("USERPROFILE").expect("USERPROFILE"))
        .join(".cache/rcwx/models/rust")
}

impl Rmvpe {
    pub fn load(eng: &Engine, dir: &std::path::Path) -> Result<Self> {
        let raw = std::fs::read(dir.join("rmvpe_folded.safetensors"))?;
        let st = safetensors::SafeTensors::deserialize(&raw)?;
        let mut w = std::collections::HashMap::new();
        let mut cpu = std::collections::HashMap::new();
        for name in st.names() {
            let v = st.tensor(name)?;
            let shape = v.shape().to_vec();
            let n: usize = shape.iter().product();
            let mut out = Vec::with_capacity(n);
            for c in v.data().chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            if matches!(name, "mel_basis" | "hann_window" | "cents_mapping") {
                cpu.insert(name.to_string(), out.clone());
            }
            w.insert(name.to_string(), eng.tensor_from_f32(&out, shape));
        }
        Ok(Self {
            w,
            mel_basis: cpu.get("mel_basis").context("mel_basis")?.clone(),
            hann: cpu.get("hann_window").context("hann")?.clone(),
            cents_mapping: cpu.get("cents_mapping").context("cents")?.clone(),
        })
    }

    fn g(&self, name: &str) -> &GpuTensor {
        self.w.get(name).unwrap_or_else(|| panic!("weight {name}"))
    }

    /// STFT → mel [128, T]。CPU fp32 (rustfft)。
    pub fn mel_spectrogram(&self, audio: &[f32]) -> Vec<f32> {
        let n_fft = 1024usize;
        let hop = 160usize;
        let pad = (n_fft - hop) / 2; // 432
        // reflect pad
        let n = audio.len();
        let mut buf = vec![0.0f32; n + 2 * pad];
        for i in 0..n {
            buf[pad + i] = audio[i];
        }
        for i in 0..pad {
            buf[pad - 1 - i] = audio.get(i + 1).copied().unwrap_or(0.0);
            buf[pad + n + i] = audio.get(n - 2 - i).copied().unwrap_or(0.0);
        }
        let n_frames = 1 + (buf.len().saturating_sub(n_fft)) / hop;
        let mut planner = rustfft::FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let mut mel = vec![0.0f32; N_MEL as usize * n_frames];
        let mut frame = vec![rustfft::num_complex::Complex::<f32>::new(0.0, 0.0); n_fft];
        for f in 0..n_frames {
            for i in 0..n_fft {
                frame[i] = rustfft::num_complex::Complex {
                    re: buf[f * hop + i] * self.hann[i],
                    im: 0.0,
                };
            }
            fft.process(&mut frame);
            for m in 0..N_MEL as usize {
                let mut acc = 0.0f32;
                for (k, coef) in frame.iter().take(513).enumerate() {
                    let mag = (coef.re * coef.re + coef.im * coef.im).sqrt();
                    acc += self.mel_basis[m * 513 + k] * mag;
                }
                mel[m * n_frames + f] = acc.max(1e-5).ln();
            }
        }
        mel
    }

    /// conv2d 3x3 (stride 1, pad 1) を im2col 風の直接カーネルで。
    /// 簡易実装: 各出力 (co, y, x) のスレッドが全入力を走査。
    #[allow(clippy::too_many_arguments)]
    fn conv2d3(
        &self,
        g: &mut Graph<'_>,
        eng: &Engine,
        x: &GpuTensor,
        wname: &str,
        out: &GpuTensor,
        cin: u32,
        cout: u32,
        h: u32,
        w: u32,
        k: u32,
    ) {
        // im2col [cin*k*k, h*w] → matmul [cout, cin*k*k] x ic
        let ic = eng.tensor_uninit((cin * k * k * h * w) as usize);
        g.im2col2d(x, &ic, cin, h, w);
        g.matmul_ex(
            self.g(&format!("{wname}.weight")), &ic,
            Some(self.g(&format!("{wname}.bias"))), out,
            cout, h * w, cin * k * k, 1,
            cin * k * k, 1, 0, 0,
            1, h * w, 0, 0,
            h * w, 1, 0, 0,
            1.0, 1,
        );
    }

    /// ConvBlockRes (folded): relu(conv(x)) ×2 + shortcut
    #[allow(clippy::too_many_arguments)]
    fn conv_block_res(
        &self,
        g: &mut Graph<'_>,
        eng: &Engine,
        x: &GpuTensor,
        pfx: &str,
        out: &GpuTensor,
        cin: u32,
        cout: u32,
        h: u32,
        w: u32,
    ) {
        let t1 = eng.tensor_uninit((cout * h * w) as usize);
        self.conv2d3(g, eng, x, &format!("{pfx}.conv.0"), &t1, cin, cout, h, w, 3);
        g.relu(&t1, cout * h * w);
        let t2 = eng.tensor_uninit((cout * h * w) as usize);
        self.conv2d3(g, eng, &t1, &format!("{pfx}.conv.3"), &t2, cout, cout, h, w, 3);
        g.relu(&t2, cout * h * w);
        // shortcut
        if cin != cout {
            let sc = eng.tensor_uninit((cout * h * w) as usize);
            g.matmul_ex(
                self.g(&format!("{pfx}.shortcut.weight")), x,
                Some(self.g(&format!("{pfx}.shortcut.bias"))), &sc,
                cout, h * w, cin, 1,
                cin, 1, 0, 0,
                1, h * w, 0, 0,
                h * w, 1, 0, 0,
                1.0, 1,
            );
            g.add(&t2, &sc, out, cout * h * w);
        } else {
            g.add(&t2, x, out, cout * h * w);
        }
    }

    /// 平均プール (2,2) [C, H, W] → [C, H/2, W/2]
    fn avg_pool2x2(&self, g: &mut Graph<'_>, x: &GpuTensor, out: &GpuTensor, c: u32, h: u32, w: u32) {
        g.avg_pool2x2(x, out, c, h, w);
    }

    /// ConvTranspose2d (3x3, stride, pad 1, out_pad) — gather カーネル
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose2d(
        &self,
        g: &mut Graph<'_>,
        x: &GpuTensor,
        wname: &str,
        out: &GpuTensor,
        cin: u32,
        cout: u32,
        h_in: u32,
        w_in: u32,
        h_out: u32,
        w_out: u32,
        stride_h: u32,
        stride_w: u32,
        out_pad_h: u32,
        out_pad_w: u32,
    ) {
        g.conv_transpose2d(x, self.g(&format!("{wname}.weight")),
            Some(self.g(&format!("{wname}.bias"))), out,
            cin, cout, h_in, w_in, h_out, w_out, stride_h, stride_w, out_pad_h, out_pad_w);
    }

    pub fn infer(&self, eng: &Engine, audio: &[f32]) -> Vec<f32> {
        // ---- mel (CPU) ----
        let mel = self.mel_spectrogram(audio);
        let frames = (mel.len() / N_MEL as usize) as u32;
        let t = frames;

        // [B,1,frames,128] → engine では [1, frames, 128] 平坦
        let mut g = eng.graph();

        // encoder.bn (fold なし・独立): y = (x-mean)/sqrt(var+eps)*w+b — pointwise mul/add で
        let bn_w = self.w.get("unet.encoder.bn.weight").unwrap();
        let _ = bn_w;

        // 簡易: encoder BN を CPU で適用した mel を作る
        let (w_arr, b_arr, m_arr, v_arr) = {
            let read = |n: &str| -> Vec<f32> {
                eng.readback_f32(self.w.get(n).unwrap())
            };
            (read("unet.encoder.bn.weight"), read("unet.encoder.bn.bias"),
             read("unet.encoder.bn.running_mean"), read("unet.encoder.bn.running_var"))
        };

        // mel [128, T] → x は [1(チャネル), frames, 128]... E2E への入力は
        // mel.transpose → [frames, 128] で channel 1。engine では [1, T, 128]。
        let mut x = vec![0.0f32; (t * N_MEL) as usize];
        for f in 0..t as usize {
            for c in 0..N_MEL as usize {
                let v = mel[c * t as usize + f];
                let bn_c = c; // channel=mel bin? いや BN は channel=1
                let _ = bn_c;
                x[f * N_MEL as usize + c] = v;
            }
        }
        // encoder.bn は channel=1 → 全要素同一変換
        let scale = w_arr[0] / (v_arr[0] + 1e-5).sqrt();
        let shift = b_arr[0] - m_arr[0] * scale;
        for v in x.iter_mut() {
            *v = *v * scale + shift;
        }
        let x_t = eng.tensor_from_f32(&x, vec![]);

        // ---- U-Net ----
        let mut h = t;
        let mut wdim = N_MEL;
        let mut cur = x_t;
        let mut skips: Vec<(GpuTensor, u32, u32, u32)> = Vec::new(); // (buf, c, h, w)
        let mut dbg: Vec<(&'static str, GpuTensor)> = Vec::new();
        let mut ch = 1u32;
        let mut ch_out = 16u32;
        for i in 0..EN_DE_LAYERS {
            let pfx = format!("unet.encoder.layers.{i}");
            // n_blocks=4 の ConvBlockRes
            let mut xx = eng.tensor_uninit((ch_out * h * wdim) as usize);
            self.conv_block_res(&mut g, eng, &cur, &format!("{pfx}.conv.0"), &xx, ch, ch_out, h, wdim);
            for b in 1..N_BLOCKS {
                let nx = eng.tensor_uninit((ch_out * h * wdim) as usize);
                self.conv_block_res(&mut g, eng, &xx, &format!("{pfx}.conv.{b}"), &nx, ch_out, ch_out, h, wdim);
                xx = nx;
            }
            let pooled = eng.tensor_uninit((ch_out * (h / 2) * (wdim / 2)) as usize);
            self.avg_pool2x2(&mut g, &xx, &pooled, ch_out, h, wdim);
            dbg.push((Box::leak(format!("enc{i}").into_boxed_str()), pooled.clone_handle()));
            skips.push((xx.clone_handle(), ch_out, h, wdim));
            cur = pooled;
            ch = ch_out;
            ch_out *= 2;
            h /= 2;
            wdim /= 2;
        }
        // intermediate: 各 layer は n_blocks=4 の ConvBlockRes (conv.0: in→out, conv.1-3: out→out)
        let mut xx = eng.tensor_uninit((512 * h * wdim) as usize);
        self.conv_block_res(&mut g, eng, &cur, "unet.intermediate.layers.0.conv.0", &xx, 256, 512, h, wdim);
        for b in 1..N_BLOCKS {
            let nx = eng.tensor_uninit((512 * h * wdim) as usize);
            self.conv_block_res(&mut g, eng, &xx, &format!("unet.intermediate.layers.0.conv.{b}"), &nx, 512, 512, h, wdim);
            xx = nx;
        }
        for l in 1..INTER_LAYERS {
            for b in 0..N_BLOCKS {
                let nx = eng.tensor_uninit((512 * h * wdim) as usize);
                self.conv_block_res(&mut g, eng, &xx, &format!("unet.intermediate.layers.{l}.conv.{b}"), &nx, 512, 512, h, wdim);
                xx = nx;
            }
        }
        dbg.push(("inter", xx.clone_handle()));
        // decoder
        let mut dch = 512u32;
        for i in 0..EN_DE_LAYERS {
            let (skip, _s_c, s_h, s_w) = &skips[EN_DE_LAYERS as usize - 1 - i as usize];
            let d_out = dch / 2;
            let stride_h = 2u32;
            let stride_w = 2u32;
            let out_pad = (1u32, 1u32);
            // torch 式: out = (in-1)*stride - 2*pad(=1) + k(=3) + output_padding
            let h_ct = (h - 1) * stride_h + 1 + out_pad.0;
            let w_ct = (wdim - 1) * stride_w + 1 + out_pad.1;
            let h_o = *s_h;
            let w_o = *s_w;
            let up_raw = eng.tensor_uninit((d_out * h_ct * w_ct) as usize);
            self.conv_transpose2d(&mut g, &xx, &format!("unet.decoder.layers.{i}.conv1.0"), &up_raw,
                dch, d_out, h, wdim, h_ct, w_ct, stride_h, stride_w, out_pad.0, out_pad.1);
            g.relu(&up_raw, d_out * h_ct * w_ct);
            let up = if h_ct != h_o || w_ct != w_o {
                let padded = eng.tensor_from_f32(&vec![0.0f32; (d_out * h_o * w_o) as usize], vec![]);
                g.pad2d(&up_raw, &padded, d_out, h_ct, w_ct, h_o, w_o);
                padded
            } else {
                up_raw
            };
            dbg.push((Box::leak(format!("dec{i}_up").into_boxed_str()), up.clone_handle()));
            // concat [up, skip] → conv2 ×4
            let cat = eng.tensor_uninit((d_out * 2 * h_o * w_o) as usize);
            g.concat_ch(&up, skip, &cat, d_out, d_out, h_o * w_o);
            if i == 0 {
                dbg.push(("dec0_cat", cat.clone_handle()));
            }
            let mut cc = eng.tensor_uninit((d_out * h_o * w_o) as usize);
            self.conv_block_res(&mut g, eng, &cat, &format!("unet.decoder.layers.{i}.conv2.0"), &cc, d_out * 2, d_out, h_o, w_o);
            if i == 0 {
                dbg.push(("dec0_c2b0", cc.clone_handle()));
            }
            for b in 1..N_BLOCKS {
                let nx = eng.tensor_uninit((d_out * h_o * w_o) as usize);
                self.conv_block_res(&mut g, eng, &cc, &format!("unet.decoder.layers.{i}.conv2.{b}"), &nx, d_out, d_out, h_o, w_o);
                cc = nx;
            }
            xx = cc;
            dbg.push((Box::leak(format!("dec{i}").into_boxed_str()), xx.clone_handle()));
            dch = d_out;
            h = h_o;
            wdim = w_o;
        }
        // cnn 3x3 → [3, T, 128]
        let cnn_out = eng.tensor_uninit((3 * h * wdim) as usize);
        self.conv2d3(&mut g, eng, &xx, "cnn", &cnn_out, 16, 3, h, wdim, 3);
        dbg.push(("cnn_out", cnn_out.clone_handle()));
        // transpose(1,2).flatten: [3, T, 128] → [T, 384]
        let flat = eng.tensor_uninit((t * 384) as usize);
        g.transpose_flatten(&cnn_out, &flat, 3, t, 128);
        // BiGRU (T 逐次) — CPU で実装する方が確実なので readback
        g.submit().expect("submit");
        if std::env::var("RCWX_RMVPE_DUMP").is_ok() {
            for (n, b) in &dbg {
                let v = eng.readback_f32(b);
                let _ = std::fs::write(format!("../../target/golden/rmvpe_{n}_got.bin"), bytemuck::cast_slice(&v));
            }
        }
        let hidden_in = eng.readback_f32(&flat);

        // ---- BiGRU (CPU) ----
        let read = |n: &str| -> Vec<f32> { eng.readback_f32(self.w.get(n).unwrap()) };
        let w_ih_f = read("fc.0.gru.weight_ih_l0");
        let w_hh_f = read("fc.0.gru.weight_hh_l0");
        let b_ih_f = read("fc.0.gru.bias_ih_l0");
        let b_hh_f = read("fc.0.gru.bias_hh_l0");
        let w_ih_b = read("fc.0.gru.weight_ih_l0_reverse");
        let w_hh_b = read("fc.0.gru.weight_hh_l0_reverse");
        let b_ih_b = read("fc.0.gru.bias_ih_l0_reverse");
        let b_hh_b = read("fc.0.gru.bias_hh_l0_reverse");
        let lin_w = read("fc.1.weight");
        let lin_b = read("fc.1.bias");
        let t_n = t as usize;
        let hsize = 256usize;
        // 双方向をまとめて: fwd は上の順、bwd は時刻逆順
        let run_dir = |reverse: bool, w_ih: &[f32], w_hh: &[f32], b_ih: &[f32], b_hh: &[f32]| -> Vec<f32> {
            let mut hst = vec![0.0f64; hsize];
            let mut outs = vec![0.0f32; t_n * hsize];
            let order: Vec<usize> = if reverse { (0..t_n).rev().collect() } else { (0..t_n).collect() };
            let mut g_all = vec![vec![0.0f64; hsize]; 3];
            for &tt in &order {
                let xt = &hidden_in[tt * 384..(tt + 1) * 384];
                for gate in 0..3 {
                    for hh in 0..hsize {
                        let mut acc = (b_ih[gate * hsize + hh] + b_hh[gate * hsize + hh]) as f64;
                        for k in 0..384 {
                            acc += w_ih[(gate * hsize + hh) * 384 + k] as f64 * xt[k] as f64;
                        }
                        for k in 0..hsize {
                            acc += w_hh[(gate * hsize + hh) * hsize + k] as f64 * hst[k];
                        }
                        g_all[gate][hh] = acc;
                    }
                }
                for hh in 0..hsize {
                    let r = 1.0 / (1.0 + (-g_all[0][hh]).exp());
                    let z = 1.0 / (1.0 + (-g_all[1][hh]).exp());
                    // n = tanh(W_in·x + b_in + r * (W_hn·h + b_hn)) — g_all[2] は (W_in·x + b_in + W_hn·h + b_hn)
                    let n_raw = g_all[2][hh]; // = in + hh (r 未適用) → in 部分 = n_raw - hh_part
                    // hh_part を分離再計算
                    let mut hh_part = 0.0f64;
                    for k in 0..hsize {
                        hh_part += w_hh[(2 * hsize + hh) * hsize + k] as f64 * hst[k];
                    }
                    hh_part += b_hh[2 * hsize + hh] as f64;
                    let in_part = n_raw - hh_part;
                    let n = (in_part + r * hh_part).tanh();
                    hst[hh] = (1.0 - z) * n + z * hst[hh];
                }
                let _ = hst;
                outs[tt * hsize..(tt + 1) * hsize].copy_from_slice(&hst.iter().map(|v| *v as f32).collect::<Vec<_>>());
            }
            outs
        };
        let fwd = run_dir(false, &w_ih_f, &w_hh_f, &b_ih_f, &b_hh_f);
        let bwd = run_dir(true, &w_ih_b, &w_hh_b, &b_ih_b, &b_hh_b);

        // linear 512→360 + sigmoid
        let mut hidden = vec![0.0f32; t_n * 360];
        for tt in 0..t_n {
            for o in 0..360usize {
                let mut acc = lin_b[o] as f64;
                for k in 0..hsize {
                    acc += lin_w[o * 512 + k] as f64 * fwd[tt * hsize + k] as f64;
                    acc += lin_w[o * 512 + hsize + k] as f64 * bwd[tt * hsize + k] as f64;
                }
                hidden[tt * 360 + o] = (1.0 / (1.0 + (-acc).exp())) as f32;
            }
        }
        if std::env::var("RCWX_RMVPE_DUMP").is_ok() {
            std::fs::write("../../target/golden/rmvpe_hidden_got.bin", bytemuck::cast_slice(&hidden)).unwrap();
            std::fs::write("../../target/golden/rmvpe_flat_got.bin", bytemuck::cast_slice(&hidden_in)).unwrap();
        }
        self.decode(&hidden, t_n, 0.03)
    }

    pub fn decode(&self, hidden: &[f32], t_n: usize, threshold: f32) -> Vec<f32> {
        let mut f0 = vec![0.0f32; t_n];
        for tt in 0..t_n {
            let frame = &hidden[tt * 360..(tt + 1) * 360];
            let mut center = 0usize;
            let mut max_v = f32::MIN;
            for (i, v) in frame.iter().enumerate() {
                if *v > max_v {
                    max_v = *v;
                    center = i;
                }
            }
            let mut sum_sal = 0.0f64;
            let mut sum_cents = 0.0f64;
            for off in -4i64..=4 {
                let idx = (center as i64 + off).clamp(0, 359) as usize;
                sum_sal += frame[idx] as f64;
                sum_cents += frame[idx] as f64 * self.cents_mapping[idx] as f64;
            }
            let cents = if sum_sal > 1e-8 { sum_cents / sum_sal } else { 0.0 };
            let cents = if max_sal(frame) > threshold { cents } else { 0.0 };
            if cents > 0.0 {
                f0[tt] = 10.0 * 2.0f64.powf(cents / 1200.0) as f32;
            }
        }
        f0
    }
}

fn max_sal(frame: &[f32]) -> f32 {
    frame.iter().cloned().fold(f32::MIN, f32::max)
}
