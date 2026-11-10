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
    /// 入力ch: "auto"(平均) / "0" / "1"
    pub input_channel: String,
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
}

impl Default for GuiSettings {
    fn default() -> Self {
        Self {
            model: String::new(),
            input_device: String::new(),
            output_device: String::new(),
            input_channel: "auto".into(),
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
        }
    }
}

#[derive(Default, Clone)]
pub struct LiveStats {
    pub hops: u64,
    pub underruns: u64,
    pub queue_full: u64,
    pub p50: f64,
    pub p99: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelEntry {
    pub label: String,
    pub path: String,
}

pub fn cfg_path() -> PathBuf {
    PathBuf::from(std::env::var("USERPROFILE").unwrap_or_default()).join(".config/rcwx/config.json")
}

pub fn read_cfg_json() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(cfg_path()).unwrap_or_default())
        .unwrap_or_else(|_| serde_json::json!({}))
}

pub fn load_settings() -> GuiSettings {
    let mut s = GuiSettings::default();
    let v = read_cfg_json();
    if let Some(m) = v["last_model_path"].as_str() {
        s.model = m.to_string();
    }
    if let Some(a) = v["audio"].as_object() {
        if let Some(x) = a["input_device_name"].as_str() {
            s.input_device = x.into();
        }
        if let Some(x) = a["output_device_name"].as_str() {
            s.output_device = x.into();
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
    }
    if let Some(i) = v["inference"].as_object() {
        if let Some(x) = i["f0_method"].as_str() {
            if x == "rmvpe" || x == "swiftf0" {
                s.f0_method = x.into();
            }
        }
        if let Some(x) = i["use_f0"].as_bool() {
            s.use_f0 = x;
        }
        if let Some(x) = i["moe_boost"].as_f64() {
            s.moe = x;
        }
        if let Some(x) = i["use_index"].as_bool() {
            s.use_index = x;
        }
        if let Some(x) = i["index_ratio"].as_f64() {
            if x > 0.0 {
                s.index_rate = x;
                s.use_index = true;
            }
        }
        if let Some(x) = i["index_k"].as_f64() {
            s.index_k = x;
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
        if let Some(d) = i["denoise"].as_object() {
            if let Some(x) = d["enabled"].as_bool() {
                s.denoise = if x { "spectral".into() } else { "none".into() };
            }
            if let Some(x) = d["method"].as_str() {
                if x != "none" {
                    s.denoise = "spectral".into();
                }
            }
            if let Some(x) = d["strength"].as_f64() {
                s.denoise_strength = x;
            }
        }
        if let Some(pp) = i["postprocess"].as_object() {
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
        }
    }
    s
}

pub fn save_settings(s: &GuiSettings) {
    let mut v = read_cfg_json();
    if v.as_object().map(|o| o.is_empty()).unwrap_or(true) {
        return;
    }
    v["last_model_path"] = serde_json::json!(s.model);
    v["audio"]["input_device_name"] = serde_json::json!(s.input_device);
    v["audio"]["output_device_name"] = serde_json::json!(s.output_device);
    v["audio"]["input_channel_selection"] = serde_json::json!(match s.input_channel.as_str() {
        "0" => "left",
        "1" => "right",
        _ => "auto",
    });
    v["audio"]["chunk_sec"] = serde_json::json!(s.chunk_sec);
    v["audio"]["latency_mode"] = serde_json::json!(if s.aggressive { "aggressive" } else { "normal" });
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
    v["inference"]["noise_gate_enabled"] = serde_json::json!(s.noise_gate);
    v["inference"]["noise_gate_sensitivity"] = serde_json::json!(match s.noise_gate_sensitivity {
        0 => "low",
        2 => "high",
        _ => "mid",
    });
    v["inference"]["postprocess"]["enabled"] = serde_json::json!(s.postprocess);
    v["inference"]["postprocess"]["treble_boost_db"] = serde_json::json!(s.treble_boost_db);
    v["inference"]["postprocess"]["treble_cutoff_hz"] = serde_json::json!(s.treble_cutoff_hz);
    v["inference"]["postprocess"]["limiter_threshold_db"] = serde_json::json!(s.limiter_threshold_db);
    v["inference"]["postprocess"]["limiter_release_ms"] = serde_json::json!(s.limiter_release_ms);
    v["inference"]["postprocess"]["normalizer_enabled"] = serde_json::json!(s.normalizer_enabled);
    v["inference"]["postprocess"]["normalizer_target_rms"] = serde_json::json!(s.normalizer_target_rms);
    let _ = std::fs::write(cfg_path(), serde_json::to_string_pretty(&v).unwrap_or_default());
}

pub fn models_dir() -> PathBuf {
    let v = read_cfg_json();
    if let Some(dir) = v["rvc_models_dir"].as_str() {
        if !dir.is_empty() {
            return dir.into();
        }
    }
    PathBuf::from("model")
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

pub fn scan_models(current: &str) -> Vec<ModelEntry> {
    let root = models_dir();
    let mut paths = Vec::new();
    collect_pths(&root, &mut paths);
    label_models(&root, &paths, current)
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

pub fn scan_devices() -> (Vec<String>, Vec<String>) {
    use cpal::traits::HostTrait;
    let host = cpal::default_host();
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

impl GuiSettings {
    pub fn chunk_floor_ms(&self) -> f64 {
        if self.aggressive { 20.0 } else { 40.0 }
    }

    pub fn snapped_chunk_sec(&self) -> f64 {
        let ms = (self.chunk_sec * 1000.0 / 20.0).round() * 20.0;
        ms.clamp(self.chunk_floor_ms(), 300.0) / 1000.0
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
            index_rate: if self.use_index { self.index_rate as f32 } else { 0.0 },
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
}

pub fn pick_file(title: &str, filter: &str) -> Option<String> {
    let script = format!(
        "Add-Type -AssemblyName System.Windows.Forms; \
         $d = New-Object System.Windows.Forms.OpenFileDialog; \
         $d.Title = '{title}'; \
         $d.Filter = '{filter}'; \
         if ($d.ShowDialog() -eq 'OK') {{ Write-Output $d.FileName }}"
    );
    let mut cmd = std::process::Command::new("powershell");
    cmd.args(["-NoProfile", "-STA", "-Command", &script]);
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
            Err(format!("失敗: {}", msg.chars().take(160).collect::<String>()))
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
    }

    #[test]
    fn missing_device_falls_back_to_default() {
        let devices = vec!["Mic (Realtek)".into()];
        assert_eq!(resolve_device("Realtek", &devices), "Mic (Realtek)");
        assert_eq!(resolve_device("ASIO missing", &devices), "");
    }
}
