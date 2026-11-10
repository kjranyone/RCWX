//! cpal オーディオ入出力 + 推論スレッド + ドリフト制御。

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rcwx_dsp::drift::{compute_shed_policy, FloorTracker};

use crate::config::{LiveParams, RtConfig, RtReport};
use crate::engine::RtEngine;
use crate::ring::RingBuffer;

/// 実デバイス実行に GUI から渡すフック。未使用の項目は None。
pub type StatsHook = Box<dyn Fn(&RtConfig, RtReport) + Send>;

#[derive(Default)]
pub struct RtHooks {
    pub on_stats: Option<StatsHook>,
    pub input_level: Option<Arc<AtomicU32>>,
    pub input_peak: Option<Arc<AtomicU32>>,
    pub live: Option<Arc<Mutex<LiveParams>>>,
    pub adapter_name: Option<Arc<Mutex<String>>>,
}

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
    let inter_sleep_ms: u64 = std::env::var("RCWX_INTER_BATCH_SLEEP_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    for chunk in chunks.iter().skip(warmup_hops) {
        let out = engine.process_hop(chunk);
        for v in &out {
            writer.write_sample((v.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        n_hops += 1;
        if inter_sleep_ms > 0 && n_hops.is_multiple_of(engine.cfg.batch_hops as usize) {
            std::thread::sleep(std::time::Duration::from_millis(inter_sleep_ms));
        }
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
    trims: AtomicU64,
    hops: AtomicU64,
    armed: AtomicBool,
    queue_full: Arc<AtomicU64>,
}

/// 実デバイス実行。stop_flag が立つまで実行。
/// `cfg.input_wav` があるときはマイクの代わりにその WAV をループする。
pub fn run_realtime(
    backend: rcwx_nn::engine::Backend,
    cfg: RtConfig,
    input_name: Option<&str>,
    output_name: Option<&str>,
    stop_flag: Arc<AtomicBool>,
    mut hooks: RtHooks,
) -> Result<()> {
    let on_stats = hooks.on_stats.take();
    let input_level = hooks.input_level.take();
    let input_peak = hooks.input_peak.take();
    let live = hooks.live.take();
    let adapter_name = hooks.adapter_name.take();

    let host = open_host(&cfg.host_name);
    let out_device = pick_device(&host, output_name, false)?;
    let out_cfg = out_device.default_output_config().context("output config")?;
    eprintln!(
        "output: {} @ {}Hz {}ch ({:?})",
        out_device,
        out_cfg.sample_rate(),
        out_cfg.channels(),
        out_cfg.sample_format()
    );

    let mut cfg = cfg;
    cfg.output_sample_rate = sample_rate_hz(out_cfg.sample_rate());

    let wav_pcm = if let Some(path) = cfg.input_wav.clone() {
        let (sr, mono) = read_wav_mono(Path::new(&path)).with_context(|| format!("wav input {path}"))?;
        eprintln!("input: wav {path} @ {sr}Hz (loop)");
        cfg.mic_sample_rate = sr;
        Some(mono)
    } else {
        None
    };
    let input_setup = if wav_pcm.is_none() {
        let in_device = pick_device(&host, input_name, true)?;
        let in_cfg = in_device.default_input_config().context("input config")?;
        eprintln!(
            "input: {} @ {}Hz {}ch ({:?})",
            in_device,
            in_cfg.sample_rate(),
            in_cfg.channels(),
            in_cfg.sample_format()
        );
        cfg.mic_sample_rate = sample_rate_hz(in_cfg.sample_rate());
        Some((in_device, in_cfg))
    } else {
        None
    };

    let hop_mic = cfg.hop_mic();
    if hop_mic == 0 {
        anyhow::bail!("hop size 0");
    }
    let hop_out = (cfg.chunk_sec * cfg.output_sample_rate as f64).round() as usize;
    let max_ring = cfg.output_sample_rate as usize * 2;

    let shared = Arc::new(Shared {
        ring: RingBuffer::new(max_ring),
        underruns: AtomicU64::new(0),
        trims: AtomicU64::new(0),
        hops: AtomicU64::new(0),
        armed: AtomicBool::new(false),
        queue_full: Arc::new(AtomicU64::new(0)),
    });

    let mut built = RtEngine::new(backend, cfg.clone())?;
    built.live = live.clone();
    if let Some(slot) = &adapter_name {
        let name = built.adapter_name();
        if let Ok(mut guard) = slot.lock() {
            *guard = name;
        }
    }
    let engine = Arc::new(Mutex::new(built));

    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(8);
    let qf = Arc::clone(&shared.queue_full);

    let (input_stream, wav_handle) = if let Some(mono) = wav_pcm {
        let stop = Arc::clone(&stop_flag);
        let level = input_level.clone();
        let peak = input_peak.clone();
        let qf = Arc::clone(&qf);
        let hop_dur = std::time::Duration::from_secs_f64(cfg.chunk_sec);
        let handle = std::thread::spawn(move || {
            feed_wav(mono, hop_mic, hop_dur, tx, stop, qf, (level, peak));
        });
        (None, Some(handle))
    } else if let Some((in_device, in_cfg)) = input_setup {
        let in_ch = in_cfg.channels() as usize;
        let fallback_ch = cfg.input_channel;
        let live_in = live.clone();
        let level = input_level.clone();
        let peak = input_peak.clone();
        let acc = Arc::new(Mutex::new(Vec::<f32>::with_capacity(hop_mic * 2)));
        let stream = build_input_stream(&in_device, &in_cfg, move |data: &[f32]| {
            let select_ch = live_channel(&live_in, true, fallback_ch);
            let mono = downmix(data, in_ch, select_ch);
            observe_level(&mono, &level, &peak);
            let mut acc = acc.lock().unwrap_or_else(|err| err.into_inner());
            acc.extend_from_slice(&mono);
            while acc.len() >= hop_mic {
                let hop: Vec<f32> = acc.drain(..hop_mic).collect();
                if tx.try_send(hop).is_err() {
                    qf.fetch_add(1, Ordering::Relaxed);
                }
            }
        })?;
        stream.play()?;
        (Some(stream), None)
    } else {
        anyhow::bail!("入力がありません");
    };

    let infer_shared = Arc::clone(&shared);
    let infer_engine = Arc::clone(&engine);
    let infer_stop = Arc::clone(&stop_flag);
    let infer_handle = std::thread::spawn(move || {
        while let Ok(hop) = rx.recv() {
            if infer_stop.load(Ordering::Acquire) {
                break;
            }
            let out = {
                let mut guard = infer_engine.lock().unwrap_or_else(|err| err.into_inner());
                guard.process_hop(&hop)
            };
            infer_shared.ring.write(&out);
            infer_shared.hops.fetch_add(1, Ordering::Relaxed);
        }
    });

    let out_shared = Arc::clone(&shared);
    let out_ch = out_cfg.channels() as usize;
    let fallback_out = cfg.output_channel;
    let live_out = live.clone();
    let batch_out = hop_out * cfg.batch_hops as usize;
    let prebuffer_samples = (hop_out * cfg.prebuffer_chunks as usize).max(batch_out + hop_out);
    let margin = cfg.buffer_margin;
    let out_sr = cfg.output_sample_rate;
    let chunk_sec = cfg.chunk_sec;
    let batch_hops = cfg.batch_hops;
    let tracker = std::cell::RefCell::new(FloorTracker::new());
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
            s.armed.store(false, Ordering::Release);
            tracker.borrow_mut().clear();
        }
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
                    chunk_sec * batch_hops as f64,
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
                    s.trims.fetch_add(1, Ordering::Relaxed);
                    tr.after_skip(policy.target);
                }
            }
        }
        let select_out = live_channel(&live_out, false, fallback_out);
        match select_out {
            Some(c) if c < out_ch => {
                for (i, frame) in data.chunks_mut(out_ch).enumerate() {
                    frame.fill(0.0);
                    if let Some(slot) = frame.get_mut(c) {
                        *slot = mono[i];
                    }
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

    while !stop_flag.load(Ordering::Acquire) {
        std::thread::sleep(std::time::Duration::from_secs(1));
        let eng = engine.lock().unwrap_or_else(|err| err.into_inner());
        eprintln!(
            "[RT] hops={} ring={} underruns={} trims={} queue_full={} corr={:.2} last={:.1}ms",
            shared.hops.load(Ordering::Relaxed),
            shared.ring.len(),
            shared.underruns.load(Ordering::Relaxed),
            shared.trims.load(Ordering::Relaxed),
            shared.queue_full.load(Ordering::Relaxed),
            eng.stats.feedback_corr,
            eng.stats.last_inference_ms
        );
        if let Some(cb) = on_stats.as_ref() {
            cb(
                &cfg,
                RtReport {
                    hops: shared.hops.load(Ordering::Relaxed),
                    underruns: shared.underruns.load(Ordering::Relaxed),
                    queue_full: shared.queue_full.load(Ordering::Relaxed),
                    trims: shared.trims.load(Ordering::Relaxed),
                    p50: eng.stats.percentile(50.0),
                    p95: eng.stats.percentile(95.0),
                    p99: eng.stats.percentile(99.0),
                    last_ms: eng.stats.last_inference_ms,
                },
            );
        }
    }
    drop(input_stream);
    drop(out_stream);
    if let Some(handle) = wav_handle {
        let _ = handle.join();
    }
    let _ = infer_handle.join();
    Ok(())
}

fn feed_wav(
    mono: Vec<f32>,
    hop: usize,
    hop_dur: std::time::Duration,
    tx: std::sync::mpsc::SyncSender<Vec<f32>>,
    stop: Arc<AtomicBool>,
    queue_full: Arc<AtomicU64>,
    meters: (Option<Arc<AtomicU32>>, Option<Arc<AtomicU32>>),
) {
    let (level, peak) = meters;
    if mono.len() < hop {
        return;
    }
    let start = std::time::Instant::now();
    let mut pos = 0usize;
    let mut n = 0u32;
    while !stop.load(Ordering::Acquire) {
        if pos + hop > mono.len() {
            pos = 0;
        }
        let chunk = mono[pos..pos + hop].to_vec();
        pos += hop;
        observe_level(&chunk, &level, &peak);
        if tx.try_send(chunk).is_err() {
            queue_full.fetch_add(1, Ordering::Relaxed);
        }
        n += 1;
        let target = start + hop_dur * n;
        let now = std::time::Instant::now();
        if target > now {
            std::thread::sleep(target - now);
        }
    }
}

fn live_channel(live: &Option<Arc<Mutex<LiveParams>>>, input: bool, fallback: Option<usize>) -> Option<usize> {
    let Some(live) = live else {
        return fallback;
    };
    let params = live.lock().unwrap_or_else(|err| err.into_inner());
    if input {
        params.input_channel
    } else {
        params.output_channel
    }
}

fn downmix(data: &[f32], channels: usize, select: Option<usize>) -> Vec<f32> {
    let channels = channels.max(1);
    match select {
        Some(c) if c < channels => data.iter().skip(c).step_by(channels).copied().collect(),
        _ => (0..data.len() / channels)
            .map(|i| (0..channels).map(|k| data[i * channels + k]).sum::<f32>() / channels as f32)
            .collect(),
    }
}

fn observe_level(mono: &[f32], level: &Option<Arc<AtomicU32>>, peak: &Option<Arc<AtomicU32>>) {
    if mono.is_empty() || (level.is_none() && peak.is_none()) {
        return;
    }
    let mut sum = 0.0f32;
    let mut pk = 0.0f32;
    for v in mono {
        sum += v * v;
        pk = pk.max(v.abs());
    }
    if let Some(level) = level {
        let rms = (sum / mono.len() as f32).sqrt();
        level.store(rms.to_bits(), Ordering::Relaxed);
    }
    if let Some(peak) = peak {
        peak.store(pk.to_bits(), Ordering::Relaxed);
    }
}

fn open_host(name: &str) -> cpal::Host {
    if !name.is_empty() {
        for id in cpal::available_hosts() {
            if id.name().eq_ignore_ascii_case(name)
                && let Ok(host) = cpal::host_from_id(id)
            {
                return host;
            }
        }
    }
    cpal::default_host()
}

fn sample_rate_hz(rate: cpal::SampleRate) -> u32 {
    rate
}

/// 3 秒録音。`stop` が立つとそこまでのサンプルを書く。
pub fn record_seconds(
    host_name: &str,
    device: Option<&str>,
    channel: Option<usize>,
    seconds: f64,
    out_wav: &Path,
    stop: &AtomicBool,
) -> Result<()> {
    let host = open_host(host_name);
    let device = pick_device(&host, device, true)?;
    let config = device.default_input_config().context("input config")?;
    let channels = config.channels() as usize;
    let rate = sample_rate_hz(config.sample_rate());
    let buf = Arc::new(Mutex::new(Vec::<f32>::new()));
    let buf_cb = Arc::clone(&buf);
    let stream = build_input_stream(&device, &config, move |data| {
        let mono = downmix(data, channels, channel);
        if let Ok(mut guard) = buf_cb.lock() {
            guard.extend_from_slice(&mono);
        }
    })?;
    stream.play()?;
    let target = (seconds * rate as f64).round() as usize;
    let start = std::time::Instant::now();
    while !stop.load(Ordering::Acquire) {
        let n = buf.lock().map(|guard| guard.len()).unwrap_or(0);
        if n >= target || start.elapsed() > std::time::Duration::from_secs_f64(seconds + 1.0) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    drop(stream);
    let samples = buf.lock().unwrap_or_else(|err| err.into_inner());
    if samples.is_empty() {
        anyhow::bail!("録音データがありません");
    }
    let take = samples.len().min(target.max(1));
    write_wav_mono(out_wav, rate, &samples[..take])
}

/// WAV を出力デバイスへ再生する。デバイスのレートが違えばリサンプルする。
pub fn play_wav(host_name: &str, device: Option<&str>, path: &Path, stop: &AtomicBool) -> Result<()> {
    let (file_rate, mono) = read_wav_mono(path)?;
    let host = open_host(host_name);
    let device = pick_device(&host, device, false)?;
    let config = device.default_output_config().context("output config")?;
    let out_rate = sample_rate_hz(config.sample_rate());
    let samples = if out_rate == file_rate {
        mono
    } else {
        let mut resampler = rcwx_dsp::resample::StatefulResampler::new(file_rate, out_rate);
        resampler.resample_chunk(&mono)
    };
    if samples.is_empty() {
        anyhow::bail!("再生するサンプルがありません");
    }
    let channels = config.channels() as usize;
    let samples = Arc::new(samples);
    let pos = Arc::new(AtomicUsize::new(0));
    let pos_cb = Arc::clone(&pos);
    let audio = Arc::clone(&samples);
    let stream = build_output_stream(&device, &config, move |data| {
        let n = data.len() / channels.max(1);
        let start = pos_cb.fetch_add(n, Ordering::Relaxed);
        for (frame_i, frame) in data.chunks_mut(channels.max(1)).enumerate() {
            let sample = audio.get(start + frame_i).copied().unwrap_or(0.0);
            for slot in frame.iter_mut() {
                *slot = sample;
            }
        }
    })?;
    stream.play()?;
    while !stop.load(Ordering::Acquire) && pos.load(Ordering::Relaxed) < samples.len() {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    drop(stream);
    Ok(())
}

fn read_wav_mono(path: &Path) -> Result<(u32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|sample| sample.unwrap_or(0.0)).collect(),
        hound::SampleFormat::Int => {
            let shift = spec.bits_per_sample.saturating_sub(1).min(31);
            let denom = (1u32 << shift) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.unwrap_or(0) as f32 / denom)
                .collect()
        }
    };
    let mono = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok((spec.sample_rate, mono))
}

fn write_wav_mono(path: &Path, sample_rate: u32, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    writer.finalize()?;
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
            conf,
            move |data: &[f32], _| cb(data),
            err,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            conf,
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
            conf,
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
