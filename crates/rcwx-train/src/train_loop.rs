//! 学習ループ (Issue #8 M-T4)。
//! 1 iteration = D step → (D 更新) → D 再 forward → G step → (G 更新) を
//! 単一 Tape = 単一 compute pass + submit に構築する (pass 内 dispatch 順序で
//! 重み更新が後段の forward に反映される)。

use anyhow::Result;

use rcwx_nn::engine::Engine;
use rcwx_nn::engine::tape::{AdamW, Tape};

use crate::losses;
use crate::modules::mpd::{self, Mpd};
use crate::modules::stft_mel::{MultiScaleMel, StftMel};
use crate::modules::synth_train::{SynthTrain, SynthTrainCfg, decoder, enc_p, enc_q, flow_forward};

/// 1 iteration 分のバッチ入力 (host 側)。
pub struct Batch {
    pub phone: Vec<f32>,   // [B, T, 768]
    pub pitch: Vec<u32>,   // [B, T] coarse 1..255
    pub spec: Vec<f32>,    // [B, 1025, T]
    pub wave: Vec<f32>,    // [B, seg_samples] 教師波形セグメント
    pub lengths: Vec<u32>, // [B] 実フレーム長 (= T でバケット pad 前提)
    pub t_len: u32,
    pub seg_t: u32,
    pub spec_bins: u32,
}

/// 乱数・セグメント抽出入力。
pub struct StepInputs<'a> {
    pub randn: &'a [f32],        // [B, inter, T]
    pub har: &'a [f32],          // [B, seg_samples]
    pub start_frames: &'a [u32], // [B]
}

/// 損失の内訳。
#[derive(Default, Clone, Copy)]
pub struct LossValues {
    pub disc: f32,
    pub gen_adv: f32,
    pub fm: f32,
    pub mel: f32,
    pub kl: f32,
}

pub struct Trainer<'e> {
    pub eng: &'e Engine,
    pub cfg: SynthTrainCfg,
    pub synth: SynthTrain,
    pub mpd: Mpd,
    pub stft: StftMel,
    /// --mel-loss multi 時の複数解像度 mel (None なら mono)。
    pub multi_mel: Option<MultiScaleMel>,
    pub opt_g: AdamW,
    pub opt_d: AdamW,
    pub c_mel: f32,
    pub c_kl: f32,
    /// デバッグダンプ用 step カウンタ (step 1 のみダンプ)。
    debug_step: u32,
}

impl<'e> Trainer<'e> {
    pub fn new(
        eng: &'e Engine,
        cfg: SynthTrainCfg,
        g_weights: &crate::modules::synth_train::Weights,
        d_weights: &crate::modules::mpd::MpdWeights,
    ) -> Result<Self> {
        let synth = SynthTrain::new(eng, cfg.clone(), g_weights)?;
        let mpd = Mpd::new(eng, d_weights, &mpd::infer_periods(d_weights))?;
        let stft = StftMel::new_rust(eng, 2048, 480, 128, cfg.sr);
        let mut opt_g = AdamW::new(1e-4);
        opt_g.weight_decay = 0.01;
        let mut opt_d = AdamW::new(1e-4);
        opt_d.weight_decay = 0.01;
        Ok(Self {
            eng,
            cfg,
            synth,
            mpd,
            stft,
            multi_mel: None,
            opt_g,
            opt_d,
            c_mel: 45.0,
            c_kl: 1.0,
            debug_step: 0,
        })
    }

    /// --mel-loss multi: 複数解像度 mel 損失を有効化。
    pub fn with_multi_mel(mut self) -> Self {
        self.multi_mel = Some(MultiScaleMel::new_rust(self.eng, self.cfg.sr));
        self
    }

    /// 1 iteration 実行。
    pub fn step(&mut self, b: &Batch, inp: &StepInputs<'_>) -> Result<LossValues> {
        let eng = self.eng;
        let mut tape = Tape::new(eng);
        let batch = b.lengths.len() as u32;
        let t_len = b.t_len;
        let seg_t = b.seg_t;
        let inter = self.cfg.inter_channels;
        let upp = self.cfg.upp();
        let seg_samples = seg_t * upp;

        // ---- G forward ----
        let sg = self.synth.adopt(&mut tape);
        let phone = tape.const_f32s(
            &b.phone,
            vec![
                batch as usize,
                t_len as usize,
                self.cfg.in_channels as usize,
            ],
        );
        let mut onehot = vec![0.0f32; (batch * t_len * 256) as usize];
        for i in 0..(batch * t_len) as usize {
            let p = (b.pitch[i] as usize).min(255);
            onehot[i * 256 + p] = 1.0;
        }
        let onehot = tape.const_f32s(&onehot, vec![(batch * t_len) as usize, 256]);
        let mask: Vec<f32> = (0..batch * t_len)
            .map(|i| {
                if i % t_len < b.lengths[(i / t_len) as usize].min(t_len) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let mask = tape.const_f32s(&mask, vec![batch as usize, t_len as usize]);
        let spec = tape.const_f32s(
            &b.spec,
            vec![batch as usize, b.spec_bins as usize, t_len as usize],
        );
        let randn = tape.const_f32s(
            inp.randn,
            vec![batch as usize, inter as usize, t_len as usize],
        );
        // attention bias [T, T]: 長さ一様前提 (バケット) の 0/1
        let valid = b
            .lengths
            .iter()
            .map(|l| (*l).min(t_len))
            .min()
            .unwrap_or(t_len);
        let mut bias = vec![1.0f32; (t_len * t_len) as usize];
        for i in 0..t_len {
            for j in 0..t_len {
                if i >= valid || j >= valid {
                    bias[(i * t_len + j) as usize] = 0.0;
                }
            }
        }
        let bias = tape.const_f32s(&bias, vec![t_len as usize, t_len as usize]);
        // g = emb_g 行 0 を [B·gin] へ broadcast (学習可能)
        let g = {
            let idx = self
                .synth
                .store
                .entries
                .iter()
                .position(|e| e.name == "emb_g.weight")
                .context("emb_g.weight missing")?;
            let emb = tape.adopt_param(&self.synth.store, idx);
            let gin = tape.shape(emb)[1] as u32;
            let row = tape.chslice(emb, tape.shape(emb)[0] as u32, 0, gin, 1, 1);
            let ones = tape.const_f32s(&vec![1.0f32; batch as usize], vec![batch as usize]);
            // [B,1]·[1,gin] → [B,gin]
            tape.mm(
                ones,
                row,
                None,
                (batch * gin) as usize,
                batch,
                gin,
                1,
                1,
                1,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                gin,
                1,
                0,
                1.0,
                0,
            )
        };

        let ep = enc_p(
            &mut tape, &sg, &self.cfg, phone, onehot, mask, bias, batch, t_len,
        );
        let (m_p, logs_p) = (ep.m_p, ep.logs_p);
        let (z, m_q_dump, logs_q) = enc_q(
            &mut tape,
            &sg,
            &self.cfg,
            spec,
            b.spec_bins,
            g,
            mask,
            randn,
            batch,
            t_len,
        );
        // z の即時 readback (ダンプ信頼性検証用 — mid-forward flush の影響調査)
        let z_early = if std::env::var("RCWX_DUMP_EARLY").is_ok() && self.debug_step == 1 {
            let zt = tape.var_tensor(z);
            tape.flush();
            let v = eng.readback_f32(&zt);
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            let _ = std::fs::write(
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../target/golden/cli_rust_z_early.bin"
                ),
                bytes,
            );
            true
        } else {
            false
        };
        let _ = z_early;
        let z_p = flow_forward(&mut tape, &sg, &self.cfg, z, g, mask, batch, t_len);
        // G 段階ダンプ (z / z_p / m_q / logs_q) — golden 比較用 (step 1 のみ)
        self.debug_step += 1;
        let dump_now = std::env::var("RCWX_DUMP_GSTAGES").is_ok() && self.debug_step == 1;
        let gstages = if dump_now {
            Some((
                tape.var_tensor(z),
                tape.var_tensor(z_p),
                tape.var_tensor(m_q_dump),
                tape.var_tensor(logs_q),
            ))
        } else {
            None
        };
        let start_f32: Vec<f32> = inp.start_frames.iter().map(|v| *v as f32).collect();
        let start_t = eng.tensor_from_f32(&start_f32, vec![batch as usize]);
        let z_slice = tape.gather_frames(z, &start_t, inter, t_len, seg_t, 1.0);
        let har = tape.const_f32s(inp.har, vec![(batch * seg_samples) as usize]);
        let y_hat = decoder(&mut tape, &sg, &self.cfg, z_slice, har, g, batch, seg_t);
        let zseg_dump = if dump_now {
            Some(tape.var_tensor(z_slice))
        } else {
            None
        };
        let y = tape.const_f32s(&b.wave, vec![(batch * seg_samples) as usize]);

        // ---- D step ----
        let md = self.mpd.adopt(&mut tape);
        let y_det = tape.detach(y_hat);
        let (dr_logits, dr_fmaps) =
            mpd::mpd_forward(&mut tape, &self.mpd, &md, y, batch, seg_samples);
        let (df_logits, _) = mpd::mpd_forward(&mut tape, &self.mpd, &md, y_det, batch, seg_samples);
        let d_loss = if dr_logits.is_empty() {
            tape.const_scalar(0.0)
        } else {
            losses::lsgan_d_loss(&mut tape, &dr_logits, &df_logits)
        };
        let d_loss_t = tape.var_tensor(d_loss);
        tape.zero_grads(&self.mpd.store);
        tape.backward_from(d_loss);
        // D の WN v/g Var を記録 (refold 用)
        // D 更新 (pass 内)
        if std::env::var("RCWX_FREEZE_D").is_err() {
            tape.adamw_step_all(&mut self.opt_d, &mut self.mpd.store);
        }
        // D WN 再 fold (更新後)
        self.refold_mpd_wn(&mut tape);

        // ---- D 再 forward (更新後の重み) → G step ----
        let (dg2_logits, dg2_fmaps) =
            mpd::mpd_forward(&mut tape, &self.mpd, &md, y_hat, batch, seg_samples);
        let no_gan = std::env::var("RCWX_NO_GAN").is_ok();
        let g_adv = if dg2_logits.is_empty() || no_gan {
            tape.const_scalar(0.0)
        } else {
            losses::lsgan_g_loss(&mut tape, &dg2_logits)
        };
        let g_fm = if dr_fmaps.is_empty() || no_gan {
            tape.const_scalar(0.0)
        } else {
            losses::feature_matching(&mut tape, &dr_fmaps, &dg2_fmaps)
        };
        let g_mel = if std::env::var("RCWX_NO_MEL").is_ok() {
            tape.const_scalar(0.0)
        } else if let Some(mm) = &self.multi_mel {
            mm.l1_loss(&mut tape, y, y_hat, batch, seg_samples, self.c_mel)
        } else {
            let y_hat_mel = self.stft.logmel(&mut tape, y_hat, batch, seg_samples);
            let y_mel = self.stft.logmel(&mut tape, y, batch, seg_samples);
            losses::mel_l1_loss(&mut tape, y_mel, y_hat_mel, self.c_mel)
        };
        let n_mask: f32 = b
            .lengths
            .iter()
            .map(|l| (*l).min(t_len) as f32)
            .sum::<f32>()
            * inter as f32;
        let g_kl = if std::env::var("RCWX_NO_KL").is_ok() {
            tape.const_scalar(0.0)
        } else {
            losses::kl_loss(
                &mut tape,
                z_p,
                logs_q,
                m_p,
                logs_p,
                mask,
                (batch * inter * t_len) as u32,
                t_len,
                inter,
                n_mask,
            )
        };
        let mut g_total = tape.add_scalar(g_adv, g_fm);
        g_total = tape.add_scalar(g_total, g_mel);
        g_total = tape.add_scalar(g_total, g_kl);
        let d_loss_t = d_loss_t;
        let g_adv_t = tape.var_tensor(g_adv);
        let g_fm_t = tape.var_tensor(g_fm);
        let g_mel_t = tape.var_tensor(g_mel);
        let g_kl_t = tape.var_tensor(g_kl);
        let g_total_t = tape.var_tensor(g_total);
        // G 勾配をゼロに (D 側は detach なので G へ流れない — 残留加算を防ぐ)
        tape.zero_grads(&self.synth.store);
        let debug_yhat = std::env::var("RCWX_DEBUG_YHAT").is_ok();
        let yh_probe = if debug_yhat {
            Some(tape.var_tensor(y_hat))
        } else {
            None
        };
        // y_hat のバイナリダンプ (golden 比較用)
        let yh_dump = if std::env::var("RCWX_DUMP_YHAT").is_ok() && self.debug_step == 1 {
            Some(tape.var_tensor(y_hat))
        } else {
            None
        };
        tape.backward_from(g_total);
        // G 勾配ダンプ (step 1 のみ)
        let gg_handles: Option<Vec<(String, rcwx_nn::engine::GpuTensor)>> =
            if dump_now && std::env::var("RCWX_DUMP_GGRADS").is_ok() {
                Some(
                    self.synth
                        .store
                        .entries
                        .iter()
                        .map(|e| (e.name.clone(), e.grad.clone_handle()))
                        .collect(),
                )
            } else {
                None
            };
        tape.adamw_step_all(&mut self.opt_g, &mut self.synth.store);
        // 読み出し用の損失バッファは更新前に確保済み — submit
        tape.submit()?;
        if let Some(hs) = &gg_handles {
            let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/golden/");
            std::fs::create_dir_all(format!("{root}ggrads")).ok();
            for (name, t) in hs {
                let v = eng.readback_f32(t);
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                let safe = name.replace('.', "_");
                let _ = std::fs::write(format!("{root}ggrads/{safe}.bin"), bytes);
            }
        }

        if let Some(p) = &yh_probe {
            let v = eng.readback_f32(p);
            eprintln!("  y_hat[0..3] = {:?}", &v[0..3]);
        }
        if let Some(p) = &yh_dump {
            let v = eng.readback_f32(p);
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            let _ = std::fs::write(
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../target/golden/cli_rust_yhat.bin"
                ),
                bytes,
            );
        }
        if let Some((zt, zpt, mqt, lqt)) = &gstages {
            let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/golden/");
            for (name, t) in [
                ("cli_rust_z.bin", zt),
                ("cli_rust_zp.bin", zpt),
                ("cli_rust_mq.bin", mqt),
                ("cli_rust_lq.bin", lqt),
                ("cli_rust_zseg.bin", zseg_dump.as_ref().unwrap()),
            ] {
                let v = eng.readback_f32(t);
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{root}{name}"), bytes);
            }
        }
        let read = |t: &rcwx_nn::engine::GpuTensor| eng.readback_f32(t)[0];
        Ok(LossValues {
            disc: read(&d_loss_t),
            gen_adv: read(&g_adv_t),
            fm: read(&g_fm_t),
            mel: read(&g_mel_t),
            kl: read(&g_kl_t),
        })
        .inspect(|_| {
            let _ = &g_total_t;
        })
    }

    fn refold_mpd_wn(&mut self, _tape: &mut Tape<'_>) {
        // adamw 更新後 (v/g は in-place 更新済み) に WN を再 fold し、
        // 再 forward が更新後の実効重みを使うようにする (upstream RVC と同一)。
        // fold 出力バッファ (wn.w) は採用済み folded Var が共有するため、
        // conv は自動的に新しい重みを見る。t_wn_fold_rows は norm を再計算する。
        // refold は forward 専用なので値テンソルを const で渡す (勾配は
        // 採用時の adopt_wn op が生バッファを読んで処理する)。
        let mut tape = _tape;
        for wi in 0..self.mpd.store.wns.len() {
            let (vi, gi) = {
                let wn = &self.mpd.store.wns[wi];
                (wn.v_idx, wn.g_idx)
            };
            let v = tape.const_tensor(&self.mpd.store.entries[vi].value, vec![]);
            let g = tape.const_tensor(&self.mpd.store.entries[gi].value, vec![]);
            tape.refold_wn(&self.mpd.store, wi, v, g);
        }
    }
}

use anyhow::Context;
