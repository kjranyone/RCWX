//! rcwx-cli サブコマンド: info / devices / download / logs。

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait};
use std::io::Read as _;

pub fn cmd_info(args: &[String]) -> Result<()> {
    let model = match args.first() {
        Some(m) if !m.starts_with('-') => m.clone(),
        _ => bail!("usage: rcwx-cli info <model.pth>"),
    };
    println!("Loading: {model}");
    let info = crate::pth_info::read_pth_info(std::path::Path::new(&model))
        .with_context(|| format!("read {model}"))?;

    // detect_model_type と同じ判定
    let version = if let Some(v) = &info.version {
        if v.contains('2') {
            2
        } else if v.contains('1') {
            1
        } else {
            2
        }
    } else if let Some(in_ch) = info.enc_pre_in {
        if in_ch == 768 { 2 } else { 1 }
    } else if info.config.len() > 18 {
        2
    } else {
        1
    };
    let has_f0 = info.f0.map(|f| f != 0).unwrap_or(info.has_nsf);

    println!("\nModel Information:");
    println!(
        "  Version: RVC v{version} ({}-dim features)",
        if version == 2 { "768" } else { "256" }
    );
    println!(
        "  F0 Support: {}",
        if has_f0 {
            "Yes (NSF decoder)"
        } else {
            "No (standard decoder)"
        }
    );
    println!(
        "  Speaker Embedding: {} speakers",
        info.emb_g_dim.unwrap_or(1)
    );
    if info.config.len() > 17 {
        println!("  Sample Rate: {}Hz", info.config[17] as i64);
    }
    Ok(())
}

pub fn cmd_devices() -> Result<()> {
    println!("=== Compute Devices ===");
    match rcwx_nn::engine::Engine::new(rcwx_nn::engine::Backend::Dx12) {
        Ok(eng) => println!("  [o] XPU:0 - {} (Dx12)", eng.adapter_info.name),
        Err(e) => println!("  [x] XPU:0 - unavailable ({e})"),
    }
    println!("  [o] CPU:0 - host");

    println!("\n=== Audio Input Devices ===");
    let host = cpal::default_host();
    let list_devices = |inputs: bool| -> Vec<(String, u32, u32)> {
        let mut out = Vec::new();
        let devices = if inputs {
            host.input_devices()
        } else {
            host.output_devices()
        };
        if let Ok(devices) = devices {
            for d in devices {
                let name = d.to_string();
                let (sr, ch) = d
                    .default_output_config()
                    .or_else(|_| d.default_input_config())
                    .map(|c| (c.sample_rate(), c.channels() as u32))
                    .unwrap_or((0, 0));
                out.push((name, sr, ch));
            }
        }
        out
    };
    for (i, (name, sr, ch)) in list_devices(true).into_iter().enumerate() {
        println!("  [{i}] {name} ({sr}Hz, {ch}ch)");
    }
    println!("\n=== Audio Output Devices ===");
    for (i, (name, sr, ch)) in list_devices(false).into_iter().enumerate() {
        println!("  [{i}] {name} ({sr}Hz, {ch}ch)");
    }
    println!("\n=== Audio Hosts ===");
    if let Some(hosts) = cpal::available_hosts().iter().next() {
        let _ = hosts;
    }
    for h in cpal::available_hosts() {
        let mark = if h == host.id() { "*" } else { " " };
        println!("  {mark} {h:?}");
    }
    Ok(())
}

const HUBERT_URL: &str =
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt";
const RMVPE_URL: &str = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt";
/// swift-f0 は 0.1.2 の ONNX 図形式 (STFT ノード + onnx::Conv 初期化子) に
/// 変換器が依存するためバージョンを固定する (0.3.x は STFT が分解された別形式)。
const SWIFTF0_PYPI: &str = "https://pypi.org/pypi/swift-f0/0.1.2/json";

fn models_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap_or_default())
        .join(".cache/rcwx/models")
}

/// PyPI JSON API から swift-f0 の wheel URL を解決。
fn swiftf0_wheel_url() -> Result<String> {
    let resp = ureq::get(SWIFTF0_PYPI).call().context("pypi api")?;
    let json: serde_json::Value =
        serde_json::from_reader(resp.into_body().into_reader()).context("pypi json")?;
    let url = json["urls"]
        .as_array()
        .and_then(|urls| {
            urls.iter()
                .filter(|u| u["packagetype"].as_str() == Some("bdist_wheel"))
                .find_map(|u| u["url"].as_str().map(|s| s.to_string()))
        })
        .context("swift-f0 wheel url not found")?;
    Ok(url)
}

/// SwiftF0: wheel をダウンロードして zip から model.onnx を抽出。
fn fetch_swiftf0_onnx(cache_dir: &std::path::Path, force: bool) -> Result<std::path::PathBuf> {
    let onnx = cache_dir.join("swiftf0/model.onnx");
    if onnx.exists() && !force {
        return Ok(onnx);
    }
    println!("  Downloading swift-f0 wheel...");
    let url = swiftf0_wheel_url()?;
    let wheel = cache_dir.join("swiftf0/swift_f0.whl");
    if let Some(parent) = wheel.parent() {
        std::fs::create_dir_all(parent)?;
    }
    download_file(&url, &wheel)?;
    println!("  Extracting model.onnx from wheel...");
    let file = std::fs::File::open(&wheel)?;
    let mut zip = zip::ZipArchive::new(file)?;
    let mut entry = zip
        .by_name("swift_f0/model.onnx")
        .context("model.onnx not found in swift-f0 wheel")?;
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut entry, &mut buf)?;
    drop(entry);
    std::fs::write(&onnx, &buf)?;
    let _ = std::fs::remove_file(&wheel);
    Ok(onnx)
}

pub fn cmd_download(args: &[String]) -> Result<()> {
    let force = args.iter().any(|a| a == "--force");
    let dir = models_dir();
    let rust_dir = rcwx_nn::rust_models_dir();
    println!("Models directory: {}", dir.display());
    let targets = [
        ("hubert", dir.join("hubert/hubert_base.pt"), HUBERT_URL),
        ("rmvpe", dir.join("rmvpe/rmvpe.pt"), RMVPE_URL),
    ];
    println!("\nCurrent status:");
    for (name, path, _) in &targets {
        let symbol = if path.exists() { "o" } else { "x" };
        println!("  [{symbol}] {name}");
    }
    let rust_done = [
        "hubert.safetensors",
        "rmvpe_folded.safetensors",
        "swiftf0.safetensors",
    ]
    .iter()
    .all(|w| rust_dir.join(w).exists());
    if rust_done && !force {
        println!("\nAll weights already prepared. Use --force to re-download.");
        return Ok(());
    }
    println!("\nDownloading source models...");
    for (name, path, url) in &targets {
        if path.exists() && !force {
            println!("  {name} already exists.");
            continue;
        }
        println!("  Downloading {name}...");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        download_file(url, path).with_context(|| format!("download {name} from {url}"))?;
        println!(
            "  {name} done. ({} bytes)",
            path.metadata().map(|m| m.len()).unwrap_or(0)
        );
    }
    let onnx = fetch_swiftf0_onnx(&dir, force)?;

    // ---- Rust 推論用重みへの変換 (Python 不要・バイナリ内蔵) ----
    println!("\nConverting weights for Rust inference...");
    if force || !rust_dir.join("hubert.safetensors").exists() {
        rcwx_nn::convert::convert_hubert(&dir.join("hubert/hubert_base.pt"), &rust_dir)
            .context("convert hubert")?;
    }
    if force || !rust_dir.join("rmvpe_folded.safetensors").exists() {
        rcwx_nn::convert::convert_rmvpe(&dir.join("rmvpe/rmvpe.pt"), &rust_dir)
            .context("convert rmvpe")?;
    }
    if force || !rust_dir.join("swiftf0.safetensors").exists() {
        rcwx_nn::convert::convert_swiftf0_onnx(&onnx, &rust_dir).context("convert swiftf0")?;
    }

    println!("\nAll weights ready in {}", rust_dir.display());
    println!("Synth は RVC .pth から実行時に直接ロードされます (変換不要)。");
    Ok(())
}

fn download_file(url: &str, dest: &std::path::Path) -> Result<()> {
    let resp = ureq::get(url).call().context("http request")?;
    let len = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    let mut reader = resp.into_body().into_reader();
    let tmp = dest.with_extension("part");
    let mut file = std::fs::File::create(&tmp)?;
    let mut buf = [0u8; 256 * 1024];
    let mut total = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buf[..n])?;
        total += n as u64;
        if let Some(len) = len {
            eprint!(
                "\r    {total}/{} bytes ({:.0}%)",
                len,
                100.0 * total as f64 / len as f64
            );
        }
    }
    if len.is_some() {
        eprintln!();
    }
    drop(file);
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

pub fn cmd_logs(args: &[String]) -> Result<()> {
    let log_dir = std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap_or_default())
        .join(".config/rcwx/logs");
    if !log_dir.exists() {
        println!("No log files found.");
        return Ok(());
    }
    let mut files: Vec<_> = std::fs::read_dir(&log_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.starts_with("rcwx_") && name.ends_with(".log")
        })
        .collect();
    files
        .sort_by_key(|e| std::cmp::Reverse(e.metadata().map(|m| m.modified().ok()).ok().flatten()));

    if files.is_empty() {
        println!("No log files found.");
        return Ok(());
    }
    if args.iter().any(|a| a == "--open") {
        let latest = &files[0].path();
        println!("Opening: {}", latest.display());
        std::process::Command::new("cmd")
            .args(["/C", "start", "", &latest.to_string_lossy()])
            .spawn()?;
        return Ok(());
    }
    if let Some(pos) = args.iter().position(|a| a == "--tail") {
        let n: usize = args.get(pos + 1).and_then(|v| v.parse().ok()).unwrap_or(50);
        let latest = &files[0].path();
        println!(
            "=== {} (last {n} lines) ===\n",
            latest.file_name().unwrap().to_string_lossy()
        );
        let content = std::fs::read_to_string(latest)?;
        for line in content
            .lines()
            .rev()
            .take(n)
            .collect::<Vec<_>>()
            .iter()
            .rev()
        {
            println!("{line}");
        }
        return Ok(());
    }
    println!("Log directory: {}\n", log_dir.display());
    println!("Recent log files:");
    for f in files.iter().take(10) {
        let size = f.metadata().map(|m| m.len()).unwrap_or(0);
        println!("  {}  ({} bytes)", f.file_name().to_string_lossy(), size);
    }
    println!("\nUse 'logs --open' to open the latest log");
    println!("Use 'logs --tail 50' to show last 50 lines");
    Ok(())
}
