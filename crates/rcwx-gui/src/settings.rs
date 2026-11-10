//! GUI 設定。`~/.config/rcwx/config.json` と互換。
//! 実行時にエンジンへ渡す項目だけを持つ。

use rcwx_rt::RtConfig;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct GuiSettings {
    pub model: String,
    pub input_device: String,
    pub output_device: String,
    /// cpal ホスト名。空なら既定。
    pub host: String,
    /// 入力ch: "auto"(平均) / "0" / "1"
    pub input_channel: String,
    /// 出力ch: "all" / "0" / "1"
    pub output_channel: String,
    pub chunk_sec: f64,
    pub aggressive: bool,
    pub pitch: f64,
    pub use_f0: bool,
    pub f0_method: String,
    pub moe: f64,
    pub use_index: bool,
    pub index_rate: f64,
    pub index_k: f64,
    /// none / spectral
    pub denoise: String,
    pub denoise_strength: f64,
    pub noise_gate: bool,
    /// 0低 / 1中 / 2高
    pub noise_gate_sensitivity: u8,
    pub postprocess: bool,
    pub treble_boost_db: f64,
    pub treble_cutoff_hz: f64,
    pub limiter_threshold_db: f64,
    pub limiter_release_ms: f64,
    pub normalizer_enabled: bool,
    pub normalizer_target_rms: f64,
    pub input_gain_db: f64,
    pub output_gain_db: f64,
    /// off / strict / expand / energy
    pub voice_gate_mode: String,
    pub energy_threshold: f64,
    pub octave_suppress: bool,
    pub f0_slew_limit: bool,
    pub f0_slew_max_step: f64,
    /// auto / dx12 / vulkan
    pub backend: String,
    /// cuda や cpu など、製品が起動しない保存値。選択を変えるまで書き戻さない。
    pub preserve_device: Option<String>,
    pub rvc_models_dir: String,
    pub use_wav_input: bool,
    pub wav_input: String,
}

impl Default for GuiSettings {
    fn default() -> Self {
        Self {
            model: String::new(),
            input_device: String::new(),
            output_device: String::new(),
            host: String::new(),
            input_channel: "auto".into(),
            output_channel: "all".into(),
            chunk_sec: 0.06,
            aggressive: true,
            pitch: 0.0,
            use_f0: true,
            f0_method: "swiftf0".into(),
            moe: 0.0,
            use_index: false,
            index_rate: 0.15,
            index_k: 4.0,
            denoise: "none".into(),
            denoise_strength: 1.0,
            noise_gate: false,
            noise_gate_sensitivity: 1,
            postprocess: true,
            treble_boost_db: 4.0,
            treble_cutoff_hz: 2800.0,
            limiter_threshold_db: -1.0,
            limiter_release_ms: 80.0,
            normalizer_enabled: true,
            normalizer_target_rms: 0.1,
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            voice_gate_mode: "off".into(),
            energy_threshold: 0.2,
            octave_suppress: true,
            f0_slew_limit: true,
            f0_slew_max_step: 3.6,
            backend: "auto".into(),
            preserve_device: None,
            rvc_models_dir: String::new(),
            use_wav_input: false,
            wav_input: String::new(),
        }
    }
}

#[derive(Default, Clone)]
pub struct LiveStats {
    pub hops: u64,
    pub underruns: u64,
    pub queue_full: u64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub last_ms: f64,
    pub trims: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelEntry {
    pub label: String,
    pub path: String,
}

pub fn cfg_path() -> PathBuf {
    PathBuf::from(std::env::var("USERPROFILE").unwrap_or_default()).join(".config/rcwx/config.json")
}

enum CfgSource {
    Missing,
    Value(serde_json::Value),
    /// パースできないファイルは上書きしない。
    Corrupt,
}

fn read_cfg_source() -> CfgSource {
    match std::fs::read_to_string(cfg_path()) {
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => CfgSource::Missing,
        Err(_) => CfgSource::Corrupt,
        Ok(text) if text.trim().is_empty() => CfgSource::Missing,
        Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(value) if value.is_object() => CfgSource::Value(value),
            _ => CfgSource::Corrupt,
        },
    }
}

pub fn read_cfg_json() -> serde_json::Value {
    match read_cfg_source() {
        CfgSource::Value(value) => value,
        _ => serde_json::json!({}),
    }
}

pub fn load_settings() -> GuiSettings {
    settings_from_value(&read_cfg_json())
}

fn settings_from_value(v: &serde_json::Value) -> GuiSettings {
    let mut s = GuiSettings::default();
    if let Some(m) = v["last_model_path"].as_str() {
        s.model = m.to_string();
    }
    let a = &v["audio"];
    if let Some(x) = v["rvc_models_dir"].as_str() {
        s.rvc_models_dir = x.into();
    }
    match v["device"].as_str() {
        Some("xpu" | "dx12") => s.backend = "dx12".into(),
        Some("vulkan") => s.backend = "vulkan".into(),
        Some("auto") | None => s.backend = "auto".into(),
        Some(other) => {
            s.backend = "auto".into();
            s.preserve_device = Some(other.into());
        }
    }
    if let Some(x) = a["input_device_name"].as_str() {
        s.input_device = x.into();
    }
    if let Some(x) = a["output_device_name"].as_str() {
        s.output_device = x.into();
    }
    if let Some(x) = a["input_hostapi_filter"].as_str() {
        s.host = x.into();
    }
    if let Some(x) = a["output_channel_selection"].as_str() {
        s.output_channel = match x {
            "0" | "left" => "0".into(),
            "1" | "right" => "1".into(),
            _ => "all".into(),
        };
    }
    if let Some(x) = a["input_channel_selection"].as_str() {
        s.input_channel = match x {
            "left" => "0".into(),
            "right" => "1".into(),
            "average" => "auto".into(),
            other => other.into(),
        };
    }
    if let Some(x) = a["chunk_sec"].as_f64() {
        s.chunk_sec = x;
    }
    if let Some(x) = a["latency_mode"].as_str() {
        s.aggressive = x == "aggressive";
    }
    if let Some(x) = a["input_gain_db"].as_f64() {
        s.input_gain_db = x;
    }
    if let Some(x) = a["output_gain_db"].as_f64() {
        s.output_gain_db = x;
    }
    let i = &v["inference"];
    if let Some(x) = i["f0_method"].as_str()
        && matches!(x, "rmvpe" | "swiftf0" | "none")
    {
        s.f0_method = x.into();
    }
    if s.f0_method == "none" {
        s.use_f0 = false;
    }
    if let Some(x) = i["use_f0"].as_bool() {
        s.use_f0 = x;
    }
    if let Some(x) = i["voice_gate_mode"].as_str()
        && matches!(x, "off" | "strict" | "expand" | "energy")
    {
        s.voice_gate_mode = x.into();
    }
    if let Some(x) = i["energy_threshold"].as_f64() {
        s.energy_threshold = x;
    }
    if let Some(x) = i["enable_octave_flip_suppress"].as_bool() {
        s.octave_suppress = x;
    }
    if let Some(x) = i["enable_f0_slew_limit"].as_bool() {
        s.f0_slew_limit = x;
    }
    if let Some(x) = i["f0_slew_max_step_st"].as_f64() {
        s.f0_slew_max_step = x.clamp(1.0, 6.0);
    }
    if let Some(x) = i["moe_boost"].as_f64() {
        s.moe = x;
    }
    let has_use_index = i["use_index"].is_boolean();
    if let Some(x) = i["use_index"].as_bool() {
        s.use_index = x;
    }
    if let Some(x) = i["index_ratio"].as_f64()
        && x > 0.0
    {
        s.index_rate = x;
        // 比率だけの古い設定は使用中とみなす。use_index があるときはそちらが優先。
        if !has_use_index {
            s.use_index = true;
        }
    }
    if let Some(x) = i["index_k"].as_f64() {
        s.index_k = x;
    } else if let Some(x) = i["index_k"].as_u64() {
        s.index_k = x as f64;
    }
    if let Some(x) = i["noise_gate_enabled"].as_bool() {
        s.noise_gate = x;
    }
    if let Some(x) = i["noise_gate_sensitivity"].as_str() {
        s.noise_gate_sensitivity = match x {
            "low" => 0,
            "high" => 2,
            _ => 1,
        };
    }
    let d = &i["denoise"];
    // エンジンが実行できるのは spectral だけ。enabled が明示されていれば method より優先する。
    if let Some(method) = d["method"].as_str() {
        s.denoise = if method == "none" {
            "none".into()
        } else {
            "spectral".into()
        };
    }
    if let Some(enabled) = d["enabled"].as_bool() {
        if !enabled {
            s.denoise = "none".into();
        } else if s.denoise == "none" {
            s.denoise = "spectral".into();
        }
    }
    if let Some(x) = d["strength"].as_f64() {
        s.denoise_strength = x;
    }
    let pp = &i["postprocess"];
    if let Some(x) = pp["enabled"].as_bool() {
        s.postprocess = x;
    }
    if let Some(x) = pp["treble_boost_db"].as_f64() {
        s.treble_boost_db = x;
    }
    if let Some(x) = pp["treble_cutoff_hz"].as_f64() {
        s.treble_cutoff_hz = x;
    }
    if let Some(x) = pp["limiter_threshold_db"].as_f64() {
        s.limiter_threshold_db = x;
    }
    if let Some(x) = pp["limiter_release_ms"].as_f64() {
        s.limiter_release_ms = x;
    }
    if let Some(x) = pp["normalizer_enabled"].as_bool() {
        s.normalizer_enabled = x;
    }
    if let Some(x) = pp["normalizer_target_rms"].as_f64() {
        s.normalizer_target_rms = x;
    }
    s
}

pub fn save_settings(s: &GuiSettings) {
    let path = cfg_path();
    let mut v = match read_cfg_source() {
        CfgSource::Corrupt => return,
        CfgSource::Missing => serde_json::json!({}),
        CfgSource::Value(value) => value,
    };
    v["last_model_path"] = serde_json::json!(s.model);
    if !s.rvc_models_dir.is_empty() {
        v["rvc_models_dir"] = serde_json::json!(s.rvc_models_dir);
    }
    if let Some(raw) = &s.preserve_device {
        v["device"] = serde_json::json!(raw);
    } else {
        v["device"] = serde_json::json!(match s.backend.as_str() {
            "dx12" => "xpu",
            "vulkan" => "vulkan",
            _ => "auto",
        });
    }
    v["audio"]["input_device_name"] = serde_json::json!(s.input_device);
    v["audio"]["output_device_name"] = serde_json::json!(s.output_device);
    v["audio"]["input_hostapi_filter"] = serde_json::json!(s.host);
    v["audio"]["output_hostapi_filter"] = serde_json::json!(s.host);
    v["audio"]["input_channel_selection"] = serde_json::json!(match s.input_channel.as_str() {
        "0" => "left",
        "1" => "right",
        _ => "auto",
    });
    v["audio"]["output_channel_selection"] = serde_json::json!(match s.output_channel.as_str() {
        "0" => "0",
        "1" => "1",
        _ => "auto",
    });
    v["audio"]["chunk_sec"] = serde_json::json!(s.chunk_sec);
    v["audio"]["latency_mode"] =
        serde_json::json!(if s.aggressive { "aggressive" } else { "normal" });
    v["audio"]["input_gain_db"] = serde_json::json!(s.input_gain_db);
    v["audio"]["output_gain_db"] = serde_json::json!(s.output_gain_db);
    v["inference"]["use_f0"] = serde_json::json!(s.use_f0);
    v["inference"]["f0_method"] = serde_json::json!(s.f0_method);
    v["inference"]["moe_boost"] = serde_json::json!(s.moe);
    v["inference"]["use_index"] = serde_json::json!(s.use_index);
    v["inference"]["index_ratio"] = serde_json::json!(s.index_rate);
    v["inference"]["index_k"] = serde_json::json!(s.index_k as u32);
    v["inference"]["denoise"]["enabled"] = serde_json::json!(s.denoise != "none");
    v["inference"]["denoise"]["method"] = serde_json::json!(s.denoise);
    v["inference"]["denoise"]["strength"] = serde_json::json!(s.denoise_strength);
    v["inference"]["voice_gate_mode"] = serde_json::json!(s.voice_gate_mode);
    v["inference"]["energy_threshold"] = serde_json::json!(s.energy_threshold);
    v["inference"]["enable_octave_flip_suppress"] = serde_json::json!(s.octave_suppress);
    v["inference"]["enable_f0_slew_limit"] = serde_json::json!(s.f0_slew_limit);
    v["inference"]["f0_slew_max_step_st"] = serde_json::json!(s.f0_slew_max_step);
    v["inference"]["noise_gate_enabled"] = serde_json::json!(s.noise_gate);
    v["inference"]["noise_gate_sensitivity"] = serde_json::json!(match s.noise_gate_sensitivity {
        0 => "low",
        2 => "high",
        _ => "mid",
    });
    v["inference"]["postprocess"]["enabled"] = serde_json::json!(s.postprocess);
    v["inference"]["postprocess"]["treble_boost_db"] = serde_json::json!(s.treble_boost_db);
    v["inference"]["postprocess"]["treble_cutoff_hz"] = serde_json::json!(s.treble_cutoff_hz);
    v["inference"]["postprocess"]["limiter_threshold_db"] =
        serde_json::json!(s.limiter_threshold_db);
    v["inference"]["postprocess"]["limiter_release_ms"] = serde_json::json!(s.limiter_release_ms);
    v["inference"]["postprocess"]["normalizer_enabled"] = serde_json::json!(s.normalizer_enabled);
    v["inference"]["postprocess"]["normalizer_target_rms"] =
        serde_json::json!(s.normalizer_target_rms);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, serde_json::to_string_pretty(&v).unwrap_or_default());
}

fn is_pth(path: &Path) -> bool {
    path.extension()
        .and_then(|x| x.to_str())
        .is_some_and(|x| x.eq_ignore_ascii_case("pth"))
}

fn collect_pths(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_pths(&path, out);
        } else if is_pth(&path) {
            out.push(path);
        }
    }
}

/// サブフォルダ名を表示名にする。直下のファイルはステム。同名はステムを足す。
pub fn label_models(root: &Path, paths: &[PathBuf], current: &str) -> Vec<ModelEntry> {
    let mut used = HashMap::<String, ()>::new();
    let mut out = Vec::new();
    for path in paths {
        let mut label = if path.parent() == Some(root) {
            file_stem(path)
        } else {
            path.parent()
                .and_then(|dir| dir.file_name())
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| file_stem(path))
        };
        if used.contains_key(&label) {
            label = format!("{label} ({})", file_stem(path));
        }
        used.insert(label.clone(), ());
        out.push(ModelEntry {
            label,
            path: path.to_string_lossy().into_owned(),
        });
    }
    if !current.is_empty() && !out.iter().any(|m| m.path == current) {
        out.insert(
            0,
            ModelEntry {
                label: file_stem(Path::new(current)),
                path: current.to_string(),
            },
        );
    }
    out
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "model".into())
}

pub fn models_root(dir: &str) -> PathBuf {
    if dir.is_empty() {
        PathBuf::from("model")
    } else {
        PathBuf::from(dir)
    }
}

pub fn scan_models(root: &Path, current: &str) -> Vec<ModelEntry> {
    let mut paths = Vec::new();
    collect_pths(root, &mut paths);
    label_models(root, &paths, current)
}

pub fn sidecar_index(model: &str) -> bool {
    let Some(dir) = Path::new(model).parent() else {
        return false;
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .path()
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("index"))
    })
}

#[derive(Clone)]
pub struct BundleInfo {
    pub dir: String,
    pub hubert: bool,
    pub synth: bool,
    pub ivf: bool,
}

pub fn bundle_info() -> BundleInfo {
    let dir = rcwx_nn::rmvpe::default_rmvpe_dir();
    BundleInfo {
        dir: dir.display().to_string(),
        hubert: dir.join("hubert.safetensors").is_file(),
        synth: dir.join("synth.safetensors").is_file(),
        ivf: dir.join("ivf.safetensors").is_file() && dir.join("ivf.json").is_file(),
    }
}

/// ピークを -6 dBFS に寄せる入力ゲイン。無音は None。
pub fn recommended_gain_db(peak: f32) -> Option<f64> {
    if peak <= 1e-6 {
        return None;
    }
    let peak_db = 20.0 * f64::from(peak).log10();
    if peak_db <= -60.0 {
        return None;
    }
    Some((-6.0 - peak_db).clamp(-30.0, 24.0))
}

/// 保存名が実デバイス文字列と完全一致しなければ、部分一致へ寄せる。無ければ空（既定）。
pub fn resolve_device(stored: &str, devices: &[String]) -> String {
    if stored.is_empty() {
        return String::new();
    }
    if let Some(exact) = devices.iter().find(|d| d.as_str() == stored) {
        return exact.clone();
    }
    devices
        .iter()
        .find(|d| d.contains(stored) || stored.contains(d.as_str()))
        .cloned()
        .unwrap_or_default()
}

pub fn host_names() -> Vec<String> {
    let mut names: Vec<String> = cpal::available_hosts()
        .into_iter()
        .map(|id| id.name().to_string())
        .collect();
    names.sort();
    names
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

pub fn scan_devices(host_name: &str) -> (Vec<String>, Vec<String>) {
    use cpal::traits::HostTrait;
    let host = open_host(host_name);
    // Display 文字列はエンジンの pick_device (`to_string().contains`) と揃える。
    let ins = host
        .input_devices()
        .map(|d| d.map(|x| x.to_string()).collect())
        .unwrap_or_default();
    let outs = host
        .output_devices()
        .map(|d| d.map(|x| x.to_string()).collect())
        .unwrap_or_default();
    (ins, outs)
}

pub fn device_rate(host_name: &str, device: &str, input: bool) -> Option<u32> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = open_host(host_name);
    let found = if device.is_empty() {
        if input {
            host.default_input_device()
        } else {
            host.default_output_device()
        }
    } else {
        let list = if input {
            host.input_devices().ok()?
        } else {
            host.output_devices().ok()?
        };
        list.into_iter()
            .find(|item| item.to_string().contains(device))
    }?;
    let config = if input {
        found.default_input_config().ok()?
    } else {
        found.default_output_config().ok()?
    };
    Some(config.sample_rate())
}

impl GuiSettings {
    /// (最小 ms, 最大 ms)。Aggressive + swiftf0/none は 20–100、RMVPE は 320–600、ほかは 40–600。
    pub fn chunk_bounds_ms(&self) -> (f64, f64) {
        let method = if self.use_f0 {
            self.f0_method.as_str()
        } else {
            "none"
        };
        if self.aggressive && (method == "swiftf0" || method == "none") {
            (20.0, 100.0)
        } else if method == "rmvpe" {
            (320.0, 600.0)
        } else {
            (40.0, 600.0)
        }
    }

    pub fn snapped_chunk_sec(&self) -> f64 {
        let (lo, hi) = self.chunk_bounds_ms();
        let ms = (self.chunk_sec * 1000.0 / 20.0).round() * 20.0;
        ms.clamp(lo, hi) / 1000.0
    }

    pub fn f0_choice(&self) -> &str {
        if self.use_f0 {
            self.f0_method.as_str()
        } else {
            "none"
        }
    }

    pub fn to_rt_config(&self) -> RtConfig {
        let chunk_sec = self.snapped_chunk_sec();
        RtConfig {
            chunk_sec,
            pitch: self.pitch,
            f0_method: if self.use_f0 {
                self.f0_method.clone()
            } else {
                "none".into()
            },
            denoise: self.denoise.clone(),
            denoise_strength: self.denoise_strength,
            noise_gate: self.noise_gate,
            noise_gate_sensitivity: self.noise_gate_sensitivity,
            index_rate: if self.use_index {
                self.index_rate as f32
            } else {
                0.0
            },
            index_k: self.index_k.max(1.0) as usize,
            moe_boost: self.moe,
            input_gain_db: self.input_gain_db,
            output_gain_db: self.output_gain_db,
            postprocess: self.postprocess,
            treble_boost_db: self.treble_boost_db,
            treble_cutoff_hz: self.treble_cutoff_hz,
            limiter_threshold_db: self.limiter_threshold_db,
            limiter_release_ms: self.limiter_release_ms,
            normalizer_enabled: self.normalizer_enabled,
            normalizer_target_rms: self.normalizer_target_rms,
            input_channel: match self.input_channel.as_str() {
                "0" => Some(0),
                "1" => Some(1),
                _ => None,
            },
            output_channel: match self.output_channel.as_str() {
                "0" => Some(0),
                "1" => Some(1),
                _ => None,
            },
            voice_gate_mode: self.voice_gate_mode.clone(),
            energy_threshold: self.energy_threshold,
            octave_suppress: self.octave_suppress,
            f0_slew_limit: self.f0_slew_limit,
            f0_slew_max_step: self.f0_slew_max_step,
            input_wav: if self.use_wav_input && !self.wav_input.is_empty() {
                Some(self.wav_input.clone())
            } else {
                None
            },
            host_name: self.host.clone(),
            buffer_margin: if self.aggressive { 0.1 } else { 0.25 },
            prebuffer_chunks: if self.aggressive { 2 } else { 1 },
            batch_hops: if chunk_sec < 0.05 { 4 } else { 1 },
            ..Default::default()
        }
    }

    pub fn latency_line(&self) -> String {
        let chunk_sec = self.snapped_chunk_sec();
        let ov_ms = (chunk_sec * 1000.0).clamp(60.0, 300.0);
        let cf_ms = (ov_ms * 0.1).clamp(10.0, 20.0);
        let pre = if self.aggressive { 2 } else { 1 };
        let batch = if chunk_sec < 0.05 { 4 } else { 1 };
        format!(
            "自動設定: オーバーラップ {ov_ms:.0}ms / クロスフェード {cf_ms:.0}ms / プリバッファ {pre} / SOLA探索 10ms / 推論バッチ {batch} hop"
        )
    }

    pub fn estimate_ms(&self) -> f64 {
        let chunk_sec = self.snapped_chunk_sec();
        let batch = if chunk_sec < 0.05 { 4.0 } else { 1.0 };
        batch * chunk_sec * 1000.0 + 35.0
    }

    pub fn to_live(&self) -> rcwx_rt::LiveParams {
        rcwx_rt::LiveParams::from_config(&self.to_rt_config())
    }
}

pub fn pick_folder(title: &str) -> Option<String> {
    let script = format!(
        "Add-Type -AssemblyName System.Windows.Forms; \
         $d = New-Object System.Windows.Forms.FolderBrowserDialog; \
         $d.Description = '{title}'; \
         if ($d.ShowDialog() -eq 'OK') {{ Write-Output $d.SelectedPath }}"
    );
    dialog_output(&script)
}

pub fn pick_save(title: &str, file_name: &str) -> Option<String> {
    let script = format!(
        "Add-Type -AssemblyName System.Windows.Forms; \
         $d = New-Object System.Windows.Forms.SaveFileDialog; \
         $d.Title = '{title}'; \
         $d.Filter = 'WAV files (*.wav)|*.wav'; \
         $d.FileName = '{file_name}'; \
         if ($d.ShowDialog() -eq 'OK') {{ Write-Output $d.FileName }}"
    );
    dialog_output(&script)
}

pub fn pick_file(title: &str, filter: &str) -> Option<String> {
    let script = format!(
        "Add-Type -AssemblyName System.Windows.Forms; \
         $d = New-Object System.Windows.Forms.OpenFileDialog; \
         $d.Title = '{title}'; \
         $d.Filter = '{filter}'; \
         if ($d.ShowDialog() -eq 'OK') {{ Write-Output $d.FileName }}"
    );
    dialog_output(&script)
}

fn dialog_output(script: &str) -> Option<String> {
    let mut cmd = std::process::Command::new("powershell");
    cmd.args(["-NoProfile", "-STA", "-Command", script]);
    hide_window(&mut cmd);
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

pub fn hide_window(cmd: &mut std::process::Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
}

pub fn cli_path() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| {
            p.parent()
                .map(|d| d.join("rcwx-cli.exe").to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "rcwx-cli.exe".into())
}

pub fn convert_output_path(input: &str) -> String {
    Path::new(input)
        .with_extension("rust_converted.wav")
        .to_string_lossy()
        .into_owned()
}

/// オフライン変換。成功時は出力パス、失敗時は短いメッセージ。
pub fn convert_wav(input: &str, cfg: &RtConfig) -> Result<String, String> {
    if input.trim().is_empty() || !Path::new(input).exists() {
        return Err("ファイルがありません".into());
    }
    let out = convert_output_path(input);
    let cli = cli_path();
    let mut cmd = std::process::Command::new(&cli);
    cmd.arg("run")
        .arg(input)
        .arg("-o")
        .arg(&out)
        .arg("--pitch")
        .arg(cfg.pitch.to_string())
        .arg("--f0-method")
        .arg(&cfg.f0_method)
        .arg("--moe")
        .arg(cfg.moe_boost.to_string())
        .arg("--index-rate")
        .arg(cfg.index_rate.to_string())
        .arg("--index-k")
        .arg(cfg.index_k.to_string());
    hide_window(&mut cmd);
    match cmd.output() {
        Ok(o) if o.status.success() => Ok(out),
        Ok(o) => {
            let msg = String::from_utf8_lossy(&o.stderr);
            Err(format!(
                "失敗: {}",
                msg.chars().take(160).collect::<String>()
            ))
        }
        Err(e) => Err(format!("起動失敗: {e} (rcwx-cli.exe を確認)")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_use_parent_folder_and_disambiguate() {
        let root = Path::new("model");
        let paths = vec![
            PathBuf::from("model/kurumi/kurumi.pth"),
            PathBuf::from("model/kurumi/kurumi.pth"),
            PathBuf::from("model/voice.pth"),
        ];
        let labels = label_models(root, &paths, "");
        assert_eq!(labels[0].label, "kurumi");
        assert_eq!(labels[1].label, "kurumi (kurumi)");
        assert_eq!(labels[2].label, "voice");
    }

    #[test]
    fn aggressive_chunk_floors_at_20ms_and_snaps() {
        let mut s = GuiSettings {
            chunk_sec: 0.027,
            aggressive: true,
            ..GuiSettings::default()
        };
        assert!((s.snapped_chunk_sec() - 0.02).abs() < 1e-9);
        s.aggressive = false;
        s.chunk_sec = 0.02;
        assert!((s.snapped_chunk_sec() - 0.04).abs() < 1e-9);
        s.aggressive = true;
        s.chunk_sec = 0.2;
        assert!((s.snapped_chunk_sec() - 0.1).abs() < 1e-9);
        s.f0_method = "rmvpe".into();
        s.chunk_sec = 0.1;
        assert!((s.snapped_chunk_sec() - 0.32).abs() < 1e-9);
        s.aggressive = false;
        s.f0_method = "swiftf0".into();
        s.chunk_sec = 0.7;
        assert!((s.snapped_chunk_sec() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn cuda_device_stays_auto_until_the_user_picks_a_backend() {
        let s = settings_from_value(&serde_json::json!({ "device": "cuda" }));
        assert_eq!(s.backend, "auto");
        assert_eq!(s.preserve_device.as_deref(), Some("cuda"));
        let xpu = settings_from_value(&serde_json::json!({ "device": "xpu" }));
        assert_eq!(xpu.backend, "dx12");
        assert!(xpu.preserve_device.is_none());
    }

    #[test]
    fn voice_gate_and_f0_limits_load() {
        let s = settings_from_value(&serde_json::json!({
            "inference": {
                "voice_gate_mode": "energy",
                "energy_threshold": 0.08,
                "enable_octave_flip_suppress": false,
                "enable_f0_slew_limit": true,
                "f0_slew_max_step_st": 2.5
            }
        }));
        assert_eq!(s.voice_gate_mode, "energy");
        assert!((s.energy_threshold - 0.08).abs() < 1e-9);
        assert!(!s.octave_suppress);
        assert!(s.f0_slew_limit);
        assert!((s.f0_slew_max_step - 2.5).abs() < 1e-9);
    }

    #[test]
    fn recommended_gain_targets_minus_six() {
        let peak = 10f32.powf(-12.0 / 20.0);
        let gain = recommended_gain_db(peak).unwrap();
        assert!((gain - 6.0).abs() < 0.05);
        assert!(recommended_gain_db(0.0).is_none());
    }

    #[test]
    fn missing_device_falls_back_to_default() {
        let devices = vec!["Mic (Realtek)".into()];
        assert_eq!(resolve_device("Realtek", &devices), "Mic (Realtek)");
        assert_eq!(resolve_device("ASIO missing", &devices), "");
    }

    #[test]
    fn explicit_off_flags_survive_stored_ratios() {
        let v = serde_json::json!({
            "inference": {
                "use_index": false,
                "index_ratio": 0.15,
                "denoise": { "enabled": false, "method": "ml", "strength": 1.4 }
            }
        });
        let s = settings_from_value(&v);
        assert!(!s.use_index);
        assert!((s.index_rate - 0.15).abs() < 1e-9);
        assert_eq!(s.denoise, "none");
        assert!((s.denoise_strength - 1.4).abs() < 1e-9);
    }

    #[test]
    fn missing_use_index_follows_ratio_and_enabled_denoise_is_spectral() {
        let v = serde_json::json!({
            "inference": {
                "index_ratio": 0.2,
                "denoise": { "enabled": true, "method": "gtcrn" }
            }
        });
        let s = settings_from_value(&v);
        assert!(s.use_index);
        assert!((s.index_rate - 0.2).abs() < 1e-9);
        assert_eq!(s.denoise, "spectral");
    }
}
