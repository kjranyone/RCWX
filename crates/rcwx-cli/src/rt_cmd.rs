//! `rcwx-cli rt` — リアルタイムエンジンのCLIフロント。

use anyhow::{Context, Result, bail};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub fn cmd_rt(args: &[String]) -> Result<()> {
    let mut input: Option<String> = None;
    let mut output = "rt_out.wav".to_string();
    let mut chunk = 0.06f64;
    let mut pitch = 0.0f64;
    let mut f0_method = "swiftf0".to_string();
    let mut denoise = "spectral".to_string();
    let mut noise_gate = false;
    let mut index_rate = 0.0f32;
    let mut index_k: Option<usize> = None;
    let mut formant = 0.0f64;
    let mut protect = 0.0f64;
    let mut rms_mix = 1.0f64;
    let mut output_denoise = false;
    let mut moe = 0.0f64;
    let mut realtime_pacing = false;
    let mut batch_hops = 1u32;
    let mut dec_overlap = 3u32;
    let mut dev_in: Option<String> = None;
    let mut dev_out: Option<String> = None;
    let mut seconds: Option<u64> = None;
    let mut backend = "dx12".to_string();
    let mut model: Option<String> = None;
    let mut wav_in: Option<String> = None;
    let mut out_gain = 0.0f64;
    let mut crossfade = 0.012f64;
    let mut margin = 0.25f64;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--virtual" => input = Some(it.next().context("--virtual <input.wav>")?.clone()),
            "--model" | "-m" => model = Some(it.next().context("--model <voice.pth>")?.clone()),
            "-o" | "--output" => output = it.next().context("-o")?.clone(),
            "--chunk" => chunk = it.next().context("--chunk")?.parse()?,
            "--pitch" => pitch = it.next().context("--pitch")?.parse()?,
            "--formant" => formant = it.next().context("--formant <semitones>")?.parse()?,
            "--protect" => protect = it.next().context("--protect <0..0.5>")?.parse()?,
            "--rms-mix" => rms_mix = it.next().context("--rms-mix <0..1>")?.parse()?,
            "--output-denoise" => output_denoise = true,
            "--f0-method" => f0_method = it.next().context("--f0-method")?.clone(),
            "--denoise" => denoise = it.next().context("--denoise")?.clone(),
            "--gate" => noise_gate = true,
            "--index-rate" => index_rate = it.next().context("--index-rate")?.parse()?,
            "--index-k" => index_k = Some(it.next().context("--index-k")?.parse()?),
            "--moe" => moe = it.next().context("--moe")?.parse()?,
            "--realtime" => realtime_pacing = true,
            "--batch" => batch_hops = it.next().context("--batch")?.parse()?,
            "--dec-overlap" => dec_overlap = it.next().context("--dec-overlap")?.parse()?,
            "--in" => dev_in = Some(it.next().context("--in")?.clone()),
            "--out" => dev_out = Some(it.next().context("--out")?.clone()),
            "--seconds" => seconds = Some(it.next().context("--seconds")?.parse()?),
            "--backend" => backend = it.next().context("--backend")?.clone(),
            "--wav-in" => wav_in = Some(it.next().context("--wav-in <wav>")?.clone()),
            "--out-gain" => out_gain = it.next().context("--out-gain <db>")?.parse()?,
            "--crossfade" => crossfade = it.next().context("--crossfade <sec>")?.parse()?,
            "--margin" => margin = it.next().context("--margin <f64>")?.parse()?,
            other => bail!("unknown rt arg: {other}"),
        }
    }

    let backend = match backend.as_str() {
        "vulkan" => rcwx_nn::engine::Backend::Vulkan,
        _ => rcwx_nn::engine::Backend::Dx12,
    };
    // 既定: 低遅延 chunk は 4 hop バッチ (20ms hop で全 hop 予算内を実測)
    let batch_hops = if args.iter().any(|a| a == "--batch") {
        batch_hops
    } else if chunk < 0.05 {
        4
    } else {
        1
    };
    let cfg = rcwx_rt::RtConfig {
        chunk_sec: chunk,
        batch_hops,
        decoder_overlap_frames: dec_overlap,
        crossfade_sec: crossfade,
        buffer_margin: margin,
        output_gain_db: out_gain,
        input_wav: wav_in.clone(),
        pitch,
        formant_shift: formant,
        protect,
        rms_mix_rate: rms_mix,
        output_denoise,
        f0_method,
        denoise,
        noise_gate,
        index_rate,
        index_k: index_k.unwrap_or(8),
        moe_boost: moe,
        model_path: model,
        ..Default::default()
    };

    if let Some(wav) = input {
        rcwx_rt::audio::run_virtual(
            backend,
            cfg,
            std::path::Path::new(&wav),
            std::path::Path::new(&output),
            realtime_pacing,
        )?;
        return Ok(());
    }

    // 物理デバイス実行
    let stop = Arc::new(AtomicBool::new(false));
    // --seconds N: N 秒後に自動停止 (動作確認用)。既定は Enter/Ctrl-C まで。
    if let Some(sec) = seconds {
        let timer_stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(sec));
            timer_stop.store(true, Ordering::Release);
        });
        println!("realtime: {sec} 秒で自動停止");
    } else {
        println!("realtime: Enter または Ctrl-C で終了");
        let stop_kb = Arc::clone(&stop);
        ctrlc_handler(move || {
            stop_kb.store(true, Ordering::Release);
        })?;
    }
    rcwx_rt::audio::run_realtime(
        backend,
        cfg,
        dev_in.as_deref(),
        dev_out.as_deref(),
        stop,
        rcwx_rt::audio::RtHooks::default(),
    )
}

/// Ctrl-C ハンドラ (windows コンソール)。
fn ctrlc_handler(f: impl Fn() + Send + 'static) -> Result<()> {
    // 簡易: 専用スレッドで stdin の EOF/Enter を監視 + OS 割り込みは既定終了に任せる
    std::thread::spawn(move || {
        let _ = std::io::Read::read(&mut std::io::stdin(), &mut [0u8; 1]);
        f();
    });
    Ok(())
}
