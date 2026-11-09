//! HuBERT 実重みベンチ (独自 wgpu エンジン)。Issue #6 の報告形式: p50/p95/p99/p99.9/max。

use rcwx_nn::engine::{Backend, Engine};
use rcwx_nn::hubert::{default_weights_path, Hubert};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut backend = "vulkan".to_string();
    let mut iters = 200usize;
    let mut warmup = 50usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--backend" => backend = args.next().unwrap(),
            "--iters" => iters = args.next().unwrap().parse()?,
            "--warmup" => warmup = args.next().unwrap().parse()?,
            other => panic!("unknown arg {other}"),
        }
    }
    let b = match backend.as_str() {
        "vulkan" => Backend::Vulkan,
        "dx12" => Backend::Dx12,
        _ => Backend::Auto,
    };
    let eng = Engine::new(b)?;
    println!(
        "# adapter: {} ({:?}) backend={backend} warmup={warmup} iters={iters}",
        eng.adapter_info.name, eng.adapter_info.backend
    );
    let model = Hubert::load(&eng, &default_weights_path())?;

    let golden = std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap())
        .join(".cache/rcwx/models/rust/hubert_golden.safetensors");
    for (label, n) in [("aggr", 10880u32), ("norm", 17920u32)] {
        let (wav, _) = rcwx_nn::read_safetensor_f32(&golden, &format!("{label}.wav"))?;
        let wav_t = eng.tensor_from_f32(&wav, vec![n as usize]);

        for _ in 0..warmup {
            drop(model.features(&eng, &wav_t, n));
        }
        eng.queue.submit([eng
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
            .finish()]); // flush

        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let out = model.features(&eng, &wav_t, n);
            let got = eng.readback_f32(&out); // map 完了まで待つ = hop 完了
            samples.push(t0.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&got);
        }
        samples.sort_by(|a, b| a.total_cmp(b));
        let pick = |q: f64| samples[((samples.len() as f64 - 1.0) * q).round() as usize];
        println!(
            "{label} (N={n}): p50 {:+.3}  p95 {:+.3}  p99 {:+.3}  p99.9 {:+.3}  max {:+.3} ms",
            pick(0.50),
            pick(0.95),
            pick(0.99),
            pick(0.999),
            *samples.last().unwrap()
        );
    }
    Ok(())
}
