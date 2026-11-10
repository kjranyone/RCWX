//! cpal オーディオ入出力 + 推論スレッド + ドリフト制御。

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use rcwx_dsp::drift::{compute_shed_policy, FloorTracker};

use crate::config::RtConfig;
use crate::engine::RtEngine;
use crate::ring::RingBuffer;

/// 仮想実行: wav を hop 送りでエンジンに通し統計を出す (デバイス不要の検証モード)。
/// `realtime_pacing` = true なら実時間に合わせて sleep する。
pub fn run_virtual(
    backend: rcwx_nn::engine::Backend,
    cfg: RtConfig,
    input_wav: &std::path::Path,
    output_wav: &std::path::Path,
    realtime_pacing: bool,
) -> Result<()> {
    let mut reader = hound::WavReader::open(input_wav)?;
    let sr = reader.spec().sample_rate;
    let ch = reader.spec().channels as usize;
    let samples: Vec<f32> = reader
        .samples::<i32>()
        .map(|s| s.map(|v| v as f32 / 32768.0).unwrap_or(0.0))
        .collect();
    let mono: Vec<f32> = if ch > 1 {
        (0..samples.len() / ch)
            .map(|i| (0..ch).map(|c| samples[i * ch + c]).sum::<f32>() / ch as f32)
            .collect()
    } else {
        samples
    };

    let mut cfg = cfg;
    cfg.mic_sample_rate = sr;
    let hop = cfg.hop_mic();
    if hop == 0 {
        anyhow::bail!("hop size 0");
    }

    let mut engine = RtEngine::new(backend, cfg.clone())?;
    let hop_dur = std::time::Duration::from_secs_f64(cfg.chunk_sec);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: cfg.output_sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_wav, spec)?;
    // ウォームアップ: 最初の 2 バッチ分は時計・出力から除外 (プラン構築)
    let warmup_hops = (engine.cfg.batch_hops * 2) as usize;
    let chunks: Vec<&[f32]> = mono.chunks(hop).filter(|c| c.len() == hop).collect();
    for c in chunks.iter().take(warmup_hops) {
        let _ = engine.process_hop(c);
    }
    let start = std::time::Instant::now();
    let mut n_hops = warmup_hops;
    for chunk in chunks.iter().skip(warmup_hops) {
        let out = engine.process_hop(chunk);
        for v in &out {
            writer.write_sample((v.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        n_hops += 1;
        if realtime_pacing {
            let target = start + hop_dur * (n_hops - warmup_hops) as u32;
            let now = std::time::Instant::now();
            if target > now {
                std::thread::sleep(target - now);
            }
        }
    }
    writer.finalize()?;
    if realtime_pacing {
        let audio_dur = (n_hops - warmup_hops) as f64 * hop as f64 / sr as f64;
        let wall = start.elapsed().as_secs_f64();
        // 初回バッチ分 (prebuffer 相当) の一定遅延は許容
        let batch_period = engine.cfg.batch_hops as f64 * hop as f64 / sr as f64;
        let ok = wall <= audio_dur * 1.02 + batch_period;
        println!(
            "realtime pacing: audio {audio_dur:.2}s / wall {wall:.2}s → {}",
            if ok { "追従OK (underrun 相当ゼロ)" } else { "遅延蓄積" }
        );
    }
    let s = &engine.stats;
    println!(
        "virtual run: {n_hops} hops, inference p50={:.2}ms p99={:.2}ms max={:.2}ms (budget {:.1}ms)",
        s.percentile(50.0),
        s.percentile(99.0),
        s.inference_ms.iter().cloned().fold(0.0f64, f64::max),
        cfg.chunk_sec * 1000.0
    );
    let bad = s
        .inference_ms
        .iter()
        .filter(|ms| **ms > cfg.chunk_sec * 1000.0)
        .count();
    println!("hops over budget: {bad}/{}", s.inference_ms.len());
    for w in &engine.warn {
        eprintln!("warn: {w}");
    }
    Ok(())
}

struct Shared {
    ring: RingBuffer,
    underruns: AtomicU64,
    hops: AtomicU64,
    armed: AtomicBool,
    queue_full: Arc<AtomicU64>,
}

/// 実デバイス実行。stop_flag が立つまで実行。
#[allow(clippy::too_many_arguments)]
pub fn run_realtime(
    backend: rcwx_nn::engine::Backend,
    cfg: RtConfig,
    input_name: Option<&str>,
    output_name: Option<&str>,
    stop_flag: Arc<AtomicBool>,
    on_stats: Option<Box<dyn Fn(&RtConfig, u64, u64, u64, f64, f64, f64) + Send>>,
    input_level: Option<Arc<AtomicU32>>,
) -> Result<()> {
    let host = cpal::default_host();
    let in_device = pick_device(&host, input_name, true)?;
    let out_device = pick_device(&host, output_name, false)?;

    let in_cfg = in_device.default_input_config().context("input config")?;
    let out_cfg = out_device.default_output_config().context("output config")?;
    eprintln!(
        "input: {} @ {}Hz {}ch ({:?})",
        in_device,
        in_cfg.sample_rate(),
        in_cfg.channels(),
        in_cfg.sample_format()
    );
    eprintln!(
        "output: {} @ {}Hz {}ch ({:?})",
        out_device,
        out_cfg.sample_rate(),
        out_cfg.channels(),
        out_cfg.sample_format()
    );

    let mut cfg = cfg;
    cfg.mic_sample_rate = in_cfg.sample_rate();
    cfg.output_sample_rate = out_cfg.sample_rate();
    let hop_mic = cfg.hop_mic();
    let hop_out = (cfg.chunk_sec * cfg.output_sample_rate as f64).round() as usize;
    let max_ring = cfg.output_sample_rate as usize * 2;

    let shared = Arc::new(Shared {
        ring: RingBuffer::new(max_ring),
        underruns: AtomicU64::new(0),
        hops: AtomicU64::new(0),
        armed: AtomicBool::new(false),
        queue_full: Arc::new(AtomicU64::new(0)),
    });

    let engine = Arc::new(std::sync::Mutex::new(RtEngine::new(backend, cfg.clone())?));

    // ---- 入力ストリーム: hop 区切りで推論スレッドへ ----
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(8);
    let in_ch = in_cfg.channels() as usize;
    let select_ch = cfg.input_channel;
    let qf = Arc::clone(&shared.queue_full);
    let acc = Arc::new(std::sync::Mutex::new(Vec::<f32>::with_capacity(hop_mic * 2)));
    let acc_cb = Arc::clone(&acc);
    let input_stream = build_input_stream(&in_device, &in_cfg, move |data: &[f32]| {
        let mono: Vec<f32> = match select_ch {
            Some(c) if c < in_ch => data.iter().skip(c).step_by(in_ch).copied().collect(),
            _ => (0..data.len() / in_ch)
                .map(|i| (0..in_ch).map(|k| data[i * in_ch + k]).sum::<f32>() / in_ch as f32)
                .collect(),
        };
        // 入力レベルメーター (RMS → dB 相当の f32 を bits で共有)
        if let Some(lvl) = input_level.as_ref() {
            let sum: f32 = mono.iter().map(|v| v * v).sum();
            let rms = (sum / mono.len().max(1) as f32).sqrt();
            let peak = mono.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            lvl.store((rms.max(peak * 0.5)).to_bits(), Ordering::Relaxed);
        }
        let mut acc = acc_cb.lock().unwrap();
        acc.extend_from_slice(&mono);
        while acc.len() >= hop_mic {
            let hop: Vec<f32> = acc.drain(..hop_mic).collect();
            if tx.try_send(hop).is_err() {
                qf.fetch_add(1, Ordering::Relaxed);
            }
        }
    })?;
    input_stream.play()?;

    // ---- 推論スレッド ----
    let infer_shared = Arc::clone(&shared);
    let infer_engine = Arc::clone(&engine);
    let infer_stop = Arc::clone(&stop_flag);
    let infer_handle = std::thread::spawn(move || {
        while let Ok(hop) = rx.recv() {
            if infer_stop.load(Ordering::Acquire) {
                break;
            }
            let out = {
                let mut guard = infer_engine.lock().unwrap();
                guard.process_hop(&hop)
            };
            infer_shared.ring.write(&out);
            infer_shared.hops.fetch_add(1, Ordering::Relaxed);
        }
    });

    // ---- 出力ストリーム: prebuffer → drift 制御付き読み出し ----
    let out_shared = Arc::clone(&shared);
    let out_ch = out_cfg.channels() as usize;
    let select_out = cfg.output_channel;
    // バースト書き込み (batch hop 分を一度に) を考慮したバッファ政策:
    // prebuffer/trim はバッチサイズ基準。安全マージン = バッチ 1 個分。
    let batch_out = hop_out * cfg.batch_hops as usize;
    let prebuffer_samples = (hop_out * cfg.prebuffer_chunks as usize).max(batch_out + hop_out);
    let margin = cfg.buffer_margin;
    let out_sr = cfg.output_sample_rate;
    let chunk_sec = cfg.chunk_sec;
    let tracker = FloorTracker::new();
    let tracker = std::cell::RefCell::new(tracker);
    let out_stream = build_output_stream(&out_device, &out_cfg, move |data: &mut [f32]| {
        let s = &*out_shared;
        let n = data.len() / out_ch;
        if !s.armed.load(Ordering::Acquire) {
            if s.ring.len() >= prebuffer_samples {
                s.armed.store(true, Ordering::Release);
            } else {
                data.fill(0.0);
                return;
            }
        }
        let mut mono = vec![0.0f32; n];
        let got = s.ring.read(&mut mono);
        if got < n {
            s.underruns.fetch_add(1, Ordering::Relaxed);
            mono[got..].fill(0.0);
            // アンダーラン → prebuffer 再アーム
            s.armed.store(false, Ordering::Release);
            tracker.borrow_mut().clear();
        }
        // ドリフト: post-read フロアが閾値超えならハードスキップ
        let post = s.ring.len();
        {
            let mut tr = tracker.borrow_mut();
            tr.ensure_window(n.max(1), batch_out as i64);
            tr.record_post_read(post as i64);
            if tr.is_full() {
                let policy = compute_shed_policy(
                    false,
                    batch_out as i64,
                    out_sr,
                    chunk_sec * cfg.batch_hops as f64,
                    margin,
                    1000,
                    5.0,
                    8.0,
                    0.01,
                    0.01,
                );
                if tr.floor_samples > policy.threshold {
                    let drop = tr.floor_samples - policy.target;
                    s.ring.skip(drop as usize);
                    tr.after_skip(policy.target);
                }
            }
        }
        match select_out {
            Some(c) if c < out_ch => {
                for (i, frame) in data.chunks_mut(out_ch).enumerate() {
                    frame.fill(mono[i]);
                }
            }
            _ => {
                for (i, frame) in data.chunks_mut(out_ch).enumerate() {
                    for slot in frame.iter_mut() {
                        *slot = mono[i];
                    }
                }
            }
        }
    })?;
    out_stream.play()?;

    // 統計レポート (1s ごと)
    while !stop_flag.load(Ordering::Acquire) {
        std::thread::sleep(std::time::Duration::from_secs(1));
        let eng = engine.lock().unwrap();
        eprintln!(
            "[RT] hops={} ring={} underruns={} queue_full={} corr={:.2} last={:.1}ms",
            shared.hops.load(Ordering::Relaxed),
            shared.ring.len(),
            shared.underruns.load(Ordering::Relaxed),
            shared.queue_full.load(Ordering::Relaxed),
            eng.stats.feedback_corr,
            eng.stats.last_inference_ms
        );
        if let Some(cb) = on_stats.as_ref() {
            cb(
                &cfg,
                shared.hops.load(Ordering::Relaxed),
                shared.underruns.load(Ordering::Relaxed),
                shared.queue_full.load(Ordering::Relaxed),
                eng.stats.percentile(50.0),
                eng.stats.percentile(99.0),
                eng.stats.last_inference_ms,
            );
        }
    }
    drop(input_stream);
    drop(out_stream);
    let _ = infer_handle.join();
    Ok(())
}

fn pick_device(host: &cpal::Host, name: Option<&str>, input: bool) -> Result<cpal::Device> {
    if let Some(want) = name {
        let devices = if input {
            host.input_devices()?
        } else {
            host.output_devices()?
        };
        for d in devices {
            if d.to_string().contains(want) {
                return Ok(d);
            }
        }
        anyhow::bail!("device matching '{want}' not found");
    }
    if input {
        host.default_input_device().context("no default input device")
    } else {
        host.default_output_device().context("no default output device")
    }
}

fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    mut cb: impl FnMut(&[f32]) + Send + 'static,
) -> Result<cpal::Stream> {
    let err = |e: cpal::Error| eprintln!("input stream error: {e}");
    let conf = config.config();
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            conf.clone(),
            move |data: &[f32], _| cb(data),
            err,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            conf.clone(),
            move |data: &[i16], _| {
                let f: Vec<f32> = data.iter().map(|v| *v as f32 / 32768.0).collect();
                cb(&f);
            },
            err,
            None,
        )?,
        other => anyhow::bail!("unsupported input sample format {other:?}"),
    };
    Ok(stream)
}

fn build_output_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    mut cb: impl FnMut(&mut [f32]) + Send + 'static,
) -> Result<cpal::Stream> {
    let err = |e: cpal::Error| eprintln!("output stream error: {e}");
    let conf = config.config();
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_output_stream(
            conf.clone(),
            move |data: &mut [f32], _| cb(data),
            err,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_output_stream(
            conf,
            move |data: &mut [i16], _| {
                let mut f = vec![0.0f32; data.len()];
                cb(&mut f);
                for (d, s) in data.iter_mut().zip(f) {
                    *d = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                }
            },
            err,
            None,
        )?,
        other => anyhow::bail!("unsupported output sample format {other:?}"),
    };
    Ok(stream)
}
