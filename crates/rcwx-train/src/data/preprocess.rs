//! 学習データ前処理 (Issue #8 M-T3)。Applio Automatic 方式:
//! 静音スライス → 3.0s / 0.3s overlap チャンク化 → butter5 48Hz HPF →
//! peak 正規化 (既定 OFF) → 48k wav + 16k wav → RMVPE f0 (補間+メディアン) →
//! HuBERT 768-dim 特徴 (×2 repeat) → linear spec cache (rustfft)。

use anyhow::Result;

use crate::data::slicer::{SlicerCfg, slice_and_chunk};

/// 前処理設定。
#[derive(Clone)]
pub struct PreprocessCfg {
    pub sr: u32,
    /// チャンク長 (秒)
    pub per: f32,
    /// チャンク overlap (秒)
    pub overlap: f32,
    /// butter5 HPF (Hz)。0 で無効
    pub hpf_hz: f32,
    /// peak 正規化 (Applio: x/max*0.675 + 0.25x)
    pub normalize: bool,
    pub out_dir: std::path::PathBuf,
}

impl Default for PreprocessCfg {
    fn default() -> Self {
        Self {
            sr: 48000,
            per: 3.0,
            overlap: 0.3,
            hpf_hz: 48.0,
            normalize: false,
            out_dir: std::path::PathBuf::from("."),
        }
    }
}

/// 1 ファイル分の前処理結果のパス群。
pub struct PreprocessResult {
    pub wav_paths: Vec<std::path::PathBuf>,
    pub f0_paths: Vec<std::path::PathBuf>,
    pub feat_paths: Vec<std::path::PathBuf>,
    pub spec_paths: Vec<std::path::PathBuf>,
}

/// wav をモノラル f32 で読む (int16/24/32 対応。hound 経由)。
pub fn read_wav_mono(path: &std::path::Path) -> Result<(u32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path)?;
    let sr = reader.spec().sample_rate;
    let ch = reader.spec().channels as usize;
    let bits = reader.spec().bits_per_sample;
    let is_float = reader.spec().sample_format == hound::SampleFormat::Float;
    let scale = (1i64 << (bits - 1)) as f32;
    let samples: Vec<f32> = if is_float {
        reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
    } else {
        reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / scale).unwrap_or(0.0))
            .collect()
    };
    let mono = if ch > 1 {
        (0..samples.len() / ch)
            .map(|i| (0..ch).map(|c| samples[i * ch + c]).sum::<f32>() / ch as f32)
            .collect()
    } else {
        samples
    };
    Ok((sr, mono))
}

/// f32 モノラル wav を書く (32-bit float)。
pub fn write_wav_f32(path: &std::path::Path, sr: u32, data: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for v in data {
        w.write_sample(*v)?;
    }
    w.finalize()?;
    Ok(())
}

/// 5 次 Butterworth ハイパス @48Hz (scipy butter(5) 係数。sr で設計を切替)。
/// 48k は学習前処理用 (golden: scipy butter(5, 48/24000, 'high') と maxdiff 1e-5)。
pub fn butter5_hpf(x: &[f32], sr: u32, hz: f32) -> Vec<f32> {
    if hz <= 0.0 || x.len() < 100 {
        return x.to_vec();
    }
    // 48Hz @ 16000 (推論経路 rcwx-cli と同一設計)
    const B16: [f64; 6] = [
        0.9699606451838447,
        -4.849803225919223,
        9.699606451938447,
        -9.699606451938447,
        4.849803225919223,
        -0.9699606451838447,
    ];
    const A16: [f64; 6] = [
        1.0,
        -4.939001819168364,
        9.757863526739543,
        -9.639544849413458,
        4.761506797356209,
        -0.9408236532054606,
    ];
    // 48Hz @ 48000 (scipy butter(5, 48.0/(48000/2), btype='high'))
    const B48: [f64; 6] = [
        0.9898850753910278,
        -4.949425376955139,
        9.898850753910278,
        -9.898850753910278,
        4.949425376955139,
        -0.9898850753910278,
    ];
    const A48: [f64; 6] = [
        1.0,
        -4.979667194990071,
        9.918875338137543,
        -9.878621548779623,
        4.919285868123746,
        -0.9798724624819006,
    ];
    let (b, a) = if sr == 16000 {
        (&B16, &A16)
    } else {
        (&B48, &A48)
    };
    let xs: Vec<f64> = x.iter().map(|v| *v as f64).collect();
    let mut y = vec![0.0f64; xs.len()];
    for i in 0..xs.len() {
        let mut acc = 0.0;
        for (j, b) in b.iter().enumerate() {
            if i >= j {
                acc += b * xs[i - j];
            }
        }
        for j in 1..a.len() {
            if i >= j {
                acc -= a[j] * y[i - j];
            }
        }
        y[i] = acc;
    }
    y.iter().map(|v| *v as f32).collect()
}

/// linear spectrogram (1025 bins, magnitude sqrt(re²+im²+2e-7))。center=False
/// + 両側 reflect pad (n_fft−hop)/2 相当は呼び出し側データ長で完全フレーム。
/// FFT は f64 演算 (torch.stft と rtol 1e-4 の golden のため。
/// f32 Radix4 だと累積誤差 ~4e-4 でゲートを超える)。
pub fn spectrogram(wav: &[f32], n_fft: u32, hop: u32) -> Vec<f32> {
    use rustfft::algorithm::Radix4;
    use rustfft::num_complex::Complex64;
    use rustfft::{Fft, FftDirection};
    let n_fft = n_fft as usize;
    let mut hann = vec![0.0f64; n_fft];
    for k in 0..n_fft {
        hann[k] = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / n_fft as f64).cos();
    }
    let n_frames = if wav.len() >= n_fft {
        (wav.len() - n_fft) / hop as usize + 1
    } else {
        0
    };
    let bins = n_fft / 2 + 1;
    let mut out = vec![0.0f32; n_frames * bins];
    let fft = Radix4::new(n_fft, FftDirection::Forward);
    let mut buf: Vec<Complex64> = vec![rustfft::num_complex::Complex::new(0.0, 0.0); n_fft];
    for f in 0..n_frames {
        let off = f * hop as usize;
        for k in 0..n_fft {
            buf[k].re = wav[off + k] as f64 * hann[k];
            buf[k].im = 0.0;
        }
        let mut scratch = vec![rustfft::num_complex::Complex::new(0.0, 0.0); n_fft];
        fft.process_with_scratch(&mut buf, &mut scratch);
        for b in 0..bins {
            let m = (buf[b].re * buf[b].re + buf[b].im * buf[b].im + 2e-7).sqrt();
            out[f * bins + b] = m as f32;
        }
    }
    out
}

/// 1 wav ファイルを前処理して exp_dir へ .npy 相当 (safetensors) を書く。
#[allow(clippy::too_many_arguments)]
pub fn preprocess_wav(
    path: &std::path::Path,
    cfg: &PreprocessCfg,
    hubert: &rcwx_nn::hubert::Hubert,
    rmvpe: &rcwx_nn::rmvpe::Rmvpe,
    eng: &rcwx_nn::engine::Engine,
    _backend: rcwx_nn::engine::Backend,
) -> Result<PreprocessResult> {
    let (file_sr, mono) = read_wav_mono(path)?;
    // 48k へリサンプル
    let wav48 = if file_sr != cfg.sr {
        rcwx_dsp::resample::resample_poly(&mono, cfg.sr as usize, file_sr as usize)
    } else {
        mono
    };
    // HPF
    let wav48 = butter5_hpf(&wav48, cfg.sr, cfg.hpf_hz);
    // スライス + チャンク
    let slicer_cfg = SlicerCfg {
        threshold_db: -42.0,
        min_length_ms: 1500,
        min_interval_ms: 400,
        hop_ms: 15,
        max_sil_kept_ms: 500,
    };
    let chunks = slice_and_chunk(&wav48, cfg.sr, &slicer_cfg, cfg.per, cfg.overlap);
    if std::env::var("RCWX_PP_DEBUG").is_ok() {
        eprintln!(
            "  [pp] wav48={} peak={:.4} chunks={}",
            wav48.len(),
            wav48.iter().fold(0.0f32, |m, v| m.max(v.abs())),
            chunks.len()
        );
    }
    // 出力ディレクトリ
    let stem = path.file_stem().unwrap().to_string_lossy().to_string();
    let gt_dir = cfg.out_dir.join("0_gt_wavs");
    let k16_dir = cfg.out_dir.join("1_16k_wavs");
    let f0_dir = cfg.out_dir.join("2a_f0");
    let f0nsf_dir = cfg.out_dir.join("2b_f0nsf");
    let feat_dir = cfg.out_dir.join("3_feature768");
    let spec_dir = cfg.out_dir.join("spec_cache");
    for d in [&gt_dir, &k16_dir, &f0_dir, &f0nsf_dir, &feat_dir, &spec_dir] {
        std::fs::create_dir_all(d)?;
    }
    let mut res = PreprocessResult {
        wav_paths: vec![],
        f0_paths: vec![],
        feat_paths: vec![],
        spec_paths: vec![],
    };
    for (ci, chunk) in chunks.iter().enumerate() {
        let peak = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if peak > 2.5 || peak <= 1e-6 {
            continue;
        }
        let norm: Vec<f32> = if cfg.normalize {
            chunk
                .iter()
                .map(|v| v / peak * 0.9 * 0.75 + 0.25 * v)
                .collect()
        } else {
            chunk.clone()
        };
        let name = format!("{stem}_{:04}", ci);
        // 48k wav
        let p48 = gt_dir.join(format!("{name}.wav"));
        write_wav_f32(&p48, cfg.sr, &norm)?;
        // 16k wav
        let wav16 = rcwx_dsp::resample::resample_poly(&norm, 16000, cfg.sr as usize);
        let p16 = k16_dir.join(format!("{name}.wav"));
        write_wav_f32(&p16, 16000, &wav16)?;
        // f0 (RMVPE) — 16k, hop 160
        let n16 = wav16.len();
        let wav_t = eng.tensor_from_f32(&wav16, vec![n16]);
        let f0 = rmvpe.infer(eng, &wav16);
        // 有声内線形補間 + メディアン (3)
        let f0 = fill_and_median(&f0);
        // coarse (mel 量子化)
        let coarse: Vec<f32> = f0
            .iter()
            .map(|v| {
                if *v < 50.0 {
                    1.0
                } else {
                    let mel = 1127.0f32 * (1.0f32 + v / 700.0f32).ln();
                    let mel_min = 1127.0f32 * (1.0f32 + 50.0f32 / 700.0f32).ln();
                    let mel_max = 1127.0f32 * (1.0f32 + 1100.0f32 / 700.0f32).ln();
                    let q = (mel - mel_min) / (mel_max - mel_min) * 254.0f32 + 1.0f32;
                    q.floor().clamp(1.0, 255.0)
                }
            })
            .collect();
        save_f32(&f0_dir.join(format!("{name}.safetensors")), &coarse)?;
        save_f32(&f0nsf_dir.join(format!("{name}.safetensors")), &f0)?;
        // HuBERT 特徴 (v2: layer 12)
        let feats_v = hubert.features(eng, &wav_t, n16 as u32);
        let feats = eng.readback_f32(&feats_v);
        // ×2 repeat (20ms → 10ms grid)
        let t_frames = feats.len() / 768;
        let mut rep = Vec::with_capacity(t_frames * 2 * 768);
        for t in 0..t_frames {
            for _ in 0..2 {
                rep.extend_from_slice(&feats[t * 768..(t + 1) * 768]);
            }
        }
        save_f32(&feat_dir.join(format!("{name}.safetensors")), &rep)?;
        // spec cache (48k, n_fft 2048, hop 480)
        let spec = spectrogram(&norm, 2048, 480);
        save_f32(&spec_dir.join(format!("{name}.safetensors")), &spec)?;
        res.wav_paths.push(p48);
        res.f0_paths
            .push(f0_dir.join(format!("{name}.safetensors")));
        res.feat_paths
            .push(feat_dir.join(format!("{name}.safetensors")));
        res.spec_paths
            .push(spec_dir.join(format!("{name}.safetensors")));
    }
    Ok(res)
}

/// 無声区間の線形補間 + メディアンフィルタ (窓 3)。
fn fill_and_median(f0: &[f32]) -> Vec<f32> {
    let n = f0.len();
    let mut out = vec![0.0f32; n];
    // f0 <= 0 を線形補間 (端は 0)
    let mut last = -1isize;
    let mut i = 0;
    while i < n {
        if f0[i] > 0.0 {
            if last >= 0 && i - last as usize > 1 {
                let l = last as usize;
                let span = (i - l) as f32;
                for j in (l + 1)..i {
                    out[j] = f0[l] + (f0[i] - f0[l]) * (j - l) as f32 / span;
                }
            }
            out[i] = f0[i];
            last = i as isize;
        } else if last < 0 {
            out[i] = 0.0;
        }
        i += 1;
    }
    // メディアン 3
    let mut med = out.clone();
    for i in 1..n.saturating_sub(1) {
        let mut w = [out[i - 1], out[i], out[i + 1]];
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        med[i] = w[1];
    }
    med
}

/// safetensors で f32 ベクトルを保存 (キー "data")。
pub fn save_f32(path: &std::path::Path, data: &[f32]) -> Result<()> {
    // safetensors 形式を手書き (単一テンソル)
    use std::io::Write;
    let n = data.len();
    let header = format!(
        r#"{{"data":{{"dtype":"F32","shape":[{n}],"data_offsets":[0,{}]}}}}"#,
        n * 4
    );
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    let pad = ((8 - (header.len() as u64 + 8) % 8) % 8) as usize;
    let hlen = (header.len() + pad) as u64;
    out.write_all(&hlen.to_le_bytes())?;
    out.write_all(header.as_bytes())?;
    out.write_all(&vec![b' '; pad])?;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    out.write_all(&bytes)?;
    out.flush()?;
    Ok(())
}

/// save_f32 で保存したものを読む。
pub fn load_f32(path: &std::path::Path) -> Result<Vec<f32>> {
    let raw = std::fs::read(path)?;
    let hlen = u64::from_le_bytes(raw[0..8].try_into()?) as usize;
    let data_off = 8 + hlen;
    let bytes = &raw[data_off..];
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub mod pth_loader {
    //! base .pth (f0G48k_v2 相当) のロード。
    use super::*;

    /// 学習用 weight マップとして .pth を読む (weight_v/g はそのまま)。
    pub fn load_pth_weights(
        path: &std::path::Path,
    ) -> Result<crate::modules::synth_train::Weights> {
        let mut f = rcwx_nn::pth::PthFile::open(path)?;
        let top = f.parse()?;
        // base G checkpoint: {"model": state_dict, ...} / deployed: {"weight": ...}
        let sd = top
            .get("model")
            .or_else(|| top.get("weight"))
            .ok_or_else(|| anyhow::anyhow!("no model/weight key in pth"))?;
        let mut out = crate::modules::synth_train::Weights::new();
        for (k, v) in sd.as_dict().unwrap() {
            if let rcwx_nn::pth::Value::Tensor(t) = v {
                let data = f.read_f32(t)?;
                let shape: Vec<usize> = t.size.iter().map(|&d| d as usize).collect();
                let name = k.as_str().unwrap_or("").to_string();
                if name.is_empty() {
                    continue;
                }
                // 判別器のリマップ: discriminators.0 → disc_s, .{i} → disc_p.{i-1}
                let name = if let Some(rest) = name.strip_prefix("discriminators.") {
                    let idx: u32 = rest.split('.').next().unwrap_or("0").parse().unwrap_or(0);
                    let tail = rest.split_once('.').map(|(_, t)| t).unwrap_or("");
                    if idx == 0 {
                        format!("disc_s.{tail}")
                    } else {
                        format!("disc_p.{}.{tail}", idx - 1)
                    }
                } else {
                    name
                };
                out.insert(name, (data, shape));
            }
        }
        Ok(out)
    }
}
