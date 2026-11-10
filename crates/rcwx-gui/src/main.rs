//! rcwx-gui — egui リアルタイムボイスチェンジャー GUI (M7)。
//!
//! 旧 Python GUI (rcwx/gui) の全セクションを復元:
//! モデル/オーディオ(ch選択・ゲイン)/レイテンシ(自動設定+推定)/Pitch&F0(なし対応)/
//! Index/ノイズキャンセリング(強度)/入力ノイズゲート(感度)/
//! ポスト処理(treble・limiter・normalizer)/入力レベルメーター/統計/WAV変換テスト。
//! 設定は ~/.config/rcwx/config.json と互換。

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct GuiSettings {
    model: String,
    input_device: String,
    output_device: String,
    /// 入力ch: "auto"(平均) / "0" / "1"
    input_channel: String,
    chunk_sec: f64,
    aggressive: bool,
    pitch: f64,
    use_f0: bool,
    f0_method: String,
    moe: f64,
    use_index: bool,
    index_rate: f64,
    index_k: f64,
    denoise: String, // none / spectral
    denoise_strength: f64,
    noise_gate: bool,
    noise_gate_sensitivity: u8, // 0低/1中/2高
    postprocess: bool,
    treble_boost_db: f64,
    treble_cutoff_hz: f64,
    limiter_threshold_db: f64,
    limiter_release_ms: f64,
    normalizer_enabled: bool,
    normalizer_target_rms: f64,
    input_gain_db: f64,
    output_gain_db: f64,
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
struct LiveStats {
    hops: u64,
    underruns: u64,
    queue_full: u64,
    p50: f64,
    p99: f64,
}

struct App {
    s: GuiSettings,
    models: Vec<String>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    running: Arc<AtomicBool>,
    stats: Arc<Mutex<LiveStats>>,
    input_level: Arc<AtomicU32>,
    thread_err: Arc<Mutex<Option<String>>>,
    done_flag: Arc<AtomicBool>,
    error: Option<String>,
    test_input: String,
    test_state: Arc<Mutex<String>>,
}

fn cfg_path() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap_or_default())
        .join(".config/rcwx/config.json")
}

fn read_cfg_json() -> serde_json::Value {
    serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(cfg_path()).unwrap_or_default(),
    )
    .unwrap_or_else(|_| serde_json::json!({}))
}

fn load_settings() -> GuiSettings {
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
                // Rust 実装は spectral (gtcrn/auto 相当の劣化なし)
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

fn save_settings(s: &GuiSettings) {
    let mut v = read_cfg_json();
    if v.as_object().map(|o| o.is_empty()).unwrap_or(true) {
        return; // 既存 config がない場合は書き込まない
    }
    v["last_model_path"] = serde_json::json!(s.model);
    v["audio"]["input_device_name"] = serde_json::json!(s.input_device);
    v["audio"]["output_device_name"] = serde_json::json!(s.output_device);
    v["audio"]["input_channel_selection"] =
        serde_json::json!(match s.input_channel.as_str() {
            "0" => "left",
            "1" => "right",
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
    let _ = std::fs::write(cfg_path(), serde_json::to_string_pretty(&v).unwrap_or_default());
}

fn models_dir() -> std::path::PathBuf {
    let v = read_cfg_json();
    if let Some(dir) = v["rvc_models_dir"].as_str() {
        return dir.into();
    }
    std::path::PathBuf::from("model")
}

fn scan_models(current: &str) -> Vec<String> {
    let dir = models_dir();
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().map(|x| x == "pth").unwrap_or(false) {
                out.push(p.to_string_lossy().into_owned());
            }
        }
    }
    out.sort();
    if !current.is_empty() && !out.iter().any(|m| m == current) {
        out.insert(0, current.to_string());
    }
    out
}

fn scan_devices() -> (Vec<String>, Vec<String>) {
    use cpal::traits::HostTrait;
    let host = cpal::default_host();
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

impl App {
    fn new() -> Self {
        let mut s = load_settings();
        let models = scan_models(&s.model);
        let (inputs, outputs) = scan_devices();
        // 実在しないデバイス名 (旧 ASIO 名等) は既定に戻す
        if !s.input_device.is_empty() && !inputs.iter().any(|d| d.contains(&s.input_device)) {
            s.input_device.clear();
        }
        if !s.output_device.is_empty() && !outputs.iter().any(|d| d.contains(&s.output_device)) {
            s.output_device.clear();
        }
        Self {
            s,
            models,
            inputs,
            outputs,
            running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(Mutex::new(LiveStats::default())),
            input_level: Arc::new(AtomicU32::new(0)),
            thread_err: Arc::new(Mutex::new(None)),
            done_flag: Arc::new(AtomicBool::new(false)),
            error: None,
            test_input: String::new(),
            test_state: Arc::new(Mutex::new(String::new())),
        }
    }

    fn build_rt_config(&self) -> rcwx_rt::RtConfig {
        let s = &self.s;
        rcwx_rt::RtConfig {
            chunk_sec: s.chunk_sec,
            pitch: s.pitch,
            f0_method: if s.use_f0 { s.f0_method.clone() } else { "none".into() },
            denoise: s.denoise.clone(),
            denoise_strength: s.denoise_strength,
            noise_gate: s.noise_gate,
            noise_gate_sensitivity: s.noise_gate_sensitivity,
            index_rate: if s.use_index { s.index_rate as f32 } else { 0.0 },
            index_k: s.index_k.max(1.0) as usize,
            moe_boost: s.moe,
            input_gain_db: s.input_gain_db,
            output_gain_db: s.output_gain_db,
            postprocess: s.postprocess,
            treble_boost_db: s.treble_boost_db,
            treble_cutoff_hz: s.treble_cutoff_hz,
            limiter_threshold_db: s.limiter_threshold_db,
            limiter_release_ms: s.limiter_release_ms,
            normalizer_enabled: s.normalizer_enabled,
            normalizer_target_rms: s.normalizer_target_rms,
            input_channel: match s.input_channel.as_str() {
                "0" => Some(0),
                "1" => Some(1),
                _ => None,
            },
            buffer_margin: if s.aggressive { 0.1 } else { 0.25 },
            prebuffer_chunks: if s.aggressive { 2 } else { 1 },
            batch_hops: if s.chunk_sec < 0.05 { 4 } else { 1 },
            ..Default::default()
        }
    }

    fn start(&mut self) {
        if self.running.load(Ordering::Acquire) {
            return;
        }
        let cfg = self.build_rt_config();
        let in_dev = if self.s.input_device.is_empty() { None } else { Some(self.s.input_device.clone()) };
        let out_dev = if self.s.output_device.is_empty() { None } else { Some(self.s.output_device.clone()) };
        let stop = Arc::clone(&self.running);
        stop.store(true, Ordering::Release);
        let stats = Arc::clone(&self.stats);
        let level = Arc::clone(&self.input_level);
        let stats_cb: Box<dyn Fn(&rcwx_rt::RtConfig, u64, u64, u64, f64, f64, f64) + Send> =
            Box::new(move |_, hops, underruns, qfull, p50, p99, _last| {
                *stats.lock().unwrap() = LiveStats {
                    hops,
                    underruns,
                    queue_full: qfull,
                    p50,
                    p99,
                };
            });
        let handle_err = Arc::clone(&self.thread_err);
        let done = Arc::clone(&self.done_flag);
        *handle_err.lock().unwrap() = None;
        done.store(false, Ordering::Release);
        let h2 = Arc::clone(&handle_err);
        std::thread::spawn(move || {
            let result = rcwx_rt::audio::run_realtime(
                rcwx_nn::engine::Backend::Dx12,
                cfg,
                in_dev.as_deref(),
                out_dev.as_deref(),
                stop,
                Some(stats_cb),
                Some(level),
            );
            if let Err(e) = result {
                *h2.lock().unwrap() = Some(format!("{e:#}"));
            }
            done.store(true, Ordering::Release);
        });
        self.error = None;
    }

    fn stop(&mut self) {
        self.running.store(false, Ordering::Release);
    }

    fn run_file_test(&mut self) {
        let input = self.test_input.trim().to_string();
        if input.is_empty() || !std::path::Path::new(&input).exists() {
            *self.test_state.lock().unwrap() = "ファイルがありません".into();
            return;
        }
        let out = std::path::Path::new(&input)
            .with_extension("rust_converted.wav")
            .to_string_lossy()
            .into_owned();
        let cfg = self.build_rt_config();
        let state = Arc::clone(&self.test_state);
        *state.lock().unwrap() = "変換中...".into();
        let cli = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("rcwx-cli.exe").to_string_lossy().into_owned()))
            .unwrap_or_else(|| "rcwx-cli.exe".into());
        std::thread::spawn(move || {
            let mut cmd = std::process::Command::new(&cli);
            cmd.arg("run").arg(&input).arg("-o").arg(&out)
                .arg("--pitch").arg(cfg.pitch.to_string())
                .arg("--f0-method").arg(&cfg.f0_method)
                .arg("--moe").arg(cfg.moe_boost.to_string())
                .arg("--index-rate").arg(cfg.index_rate.to_string())
                .arg("--index-k").arg(cfg.index_k.to_string());
            let _ = no_window(&mut cmd);
            match cmd.output() {
                Ok(o) if o.status.success() => {
                    *state.lock().unwrap() = format!("完了: {out}");
                }
                Ok(o) => {
                    let msg = String::from_utf8_lossy(&o.stderr);
                    *state.lock().unwrap() =
                        format!("失敗: {}", msg.chars().take(160).collect::<String>());
                }
                Err(e) => {
                    *state.lock().unwrap() =
                        format!("起動失敗: {e} (target/release/rcwx-cli.exe を確認)");
                }
            }
        });
    }
}

fn no_window(cmd: &mut std::process::Command) -> &mut std::process::Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd
}

fn rfd_pick_wav() -> Option<String> {
    // OS 標準のファイルダイアログ (PowerShell 経由・依存追加なし)
    let script = "Add-Type -AssemblyName System.Windows.Forms; $d = New-Object System.Windows.Forms.OpenFileDialog; $d.Filter = 'WAV files (*.wav)|*.wav'; if ($d.ShowDialog() -eq 'OK') { Write-Output $d.FileName }";
    let mut cmd = std::process::Command::new("powershell");
    cmd.args(["-NoProfile", "-STA", "-Command", script]);
    let _ = no_window(&mut cmd);
    let out = cmd.output().ok()?;
    if out.status.success() {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            return Some(s);
        }
    }
    None
}

fn section_heading(ui: &mut egui::Ui, title: &str) {
    ui.add_space(6.0);
    ui.heading(title);
    ui.separator();
}

fn level_bar(ui: &mut egui::Ui, level: f32, label: &str) {
    let db = if level > 1e-6 { 20.0 * level.log10() } else { -60.0 };
    let frac = ((db + 60.0) / 60.0).clamp(0.0, 1.0);
    let color = if frac > 0.9 {
        egui::Color32::RED
    } else if frac > 0.75 {
        egui::Color32::YELLOW
    } else {
        egui::Color32::GREEN
    };
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(220.0, 12.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 2.0, egui::Color32::DARK_GRAY);
        let mut fill = rect;
        fill.set_width(rect.width() * frac);
        ui.painter().rect_filled(fill, 2.0, color);
        ui.label(format!("{label}: {db:5.1} dB"));
    });
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let running = self.running.load(Ordering::Acquire);
        if self.done_flag.load(Ordering::Acquire) && running {
            self.running.store(false, Ordering::Release);
            if let Some(err) = self.thread_err.lock().unwrap().take() {
                self.error = Some(err);
            }
        }

        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                let st = self.stats.lock().unwrap().clone();
                if running {
                    if st.hops == 0 {
                        ui.colored_label(egui::Color32::YELLOW, "● モデル読込中...");
                    } else {
                        ui.colored_label(egui::Color32::GREEN, "● Running");
                    }
                } else {
                    ui.label("○ Stopped");
                }
                ui.separator();
                ui.label(format!(
                    "hops {} | p50 {:.1}ms p99 {:.1}ms | underruns {} | queue_full {}",
                    st.hops, st.p50, st.p99, st.underruns, st.queue_full
                ));
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                if let Some(err) = &self.error {
                    ui.colored_label(egui::Color32::RED, format!("error: {err}"));
                }
                ui.heading("RCWX — Rust Voice Changer");
                ui.add_space(4.0);

                section_heading(ui, "■ モデル / オーディオ");
                ui.horizontal(|ui| {
                    ui.label("RVCモデル");
                    egui::ComboBox::from_id_salt("model")
                        .selected_text(
                            self.s.model.rsplit(['/', '\\']).next().unwrap_or("— 選択 —").to_string(),
                        )
                        .show_ui(ui, |ui| {
                            for m in self.models.clone() {
                                ui.selectable_value(
                                    &mut self.s.model,
                                    m.clone(),
                                    m.rsplit(['/', '\\']).next().unwrap_or("").to_string(),
                                );
                            }
                        });
                    if ui.button("再読込").clicked() {
                        self.models = scan_models(&self.s.model);
                        let (i, o) = scan_devices();
                        self.inputs = i;
                        self.outputs = o;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("入力");
                    egui::ComboBox::from_id_salt("in")
                        .selected_text(if self.s.input_device.is_empty() {
                            "(既定)".into()
                        } else {
                            self.s.input_device.clone()
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.s.input_device, String::new(), "(既定)");
                            for d in self.inputs.clone() {
                                ui.selectable_value(&mut self.s.input_device, d.clone(), d);
                            }
                        });
                    ui.label("出力");
                    egui::ComboBox::from_id_salt("out")
                        .selected_text(if self.s.output_device.is_empty() {
                            "(既定)".into()
                        } else {
                            self.s.output_device.clone()
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.s.output_device, String::new(), "(既定)");
                            for d in self.outputs.clone() {
                                ui.selectable_value(&mut self.s.output_device, d.clone(), d);
                            }
                        });
                });
                ui.horizontal(|ui| {
                    ui.label("入力ch");
                    egui::ComboBox::from_id_salt("inch")
                        .selected_text(match self.s.input_channel.as_str() {
                            "0" => "Ch 1 (モノラル)".to_string(),
                            "1" => "Ch 2".to_string(),
                            _ => "自動 (平均)".to_string(),
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.s.input_channel, "auto".into(), "自動 (平均)");
                            ui.selectable_value(&mut self.s.input_channel, "0".into(), "Ch 1 (モノラル)");
                            ui.selectable_value(&mut self.s.input_channel, "1".into(), "Ch 2");
                        });
                });
                ui.add(egui::Slider::new(&mut self.s.input_gain_db, -30.0..=24.0).text("入力ゲイン dB"));
                ui.add(egui::Slider::new(&mut self.s.output_gain_db, -30.0..=24.0).text("出力ゲイン dB"));
                if running {
                    let raw = f32::from_bits(self.input_level.load(Ordering::Relaxed));
                    level_bar(ui, raw, "入力レベル");
                }

                section_heading(ui, "■ レイテンシ");
                let mut chunk_ms = self.s.chunk_sec * 1000.0;
                let lo = if self.s.aggressive { 20.0 } else { 40.0 };
                ui.add(egui::Slider::new(&mut chunk_ms, lo..=300.0).text("チャンクサイズ (ms)"));
                self.s.chunk_sec = (chunk_ms / 20.0).round() * 20.0 / 1000.0;
                ui.checkbox(&mut self.s.aggressive, "Aggressive (低遅延)");
                let ov_ms = (self.s.chunk_sec * 1000.0).clamp(60.0, 300.0);
                let cf_ms = (ov_ms * 0.1).clamp(10.0, 20.0);
                let pre = if self.s.aggressive { 2 } else { 1 };
                let batch = if self.s.chunk_sec < 0.05 { 4 } else { 1 };
                ui.label(format!(
                    "自動設定: オーバーラップ {ov_ms:.0}ms / クロスフェード {cf_ms:.0}ms / プリバッファ {pre} / SOLA探索 10ms / 推論バッチ {batch} hop"
                ));
                let est = batch as f64 * self.s.chunk_sec * 1000.0 + 35.0;
                ui.label(format!("推定レイテンシ: ~{est:.0} ms"));

                section_heading(ui, "■ Pitch / F0");
                ui.add(egui::Slider::new(&mut self.s.pitch, -24.0..=24.0).text("ピッチ (半音)"));
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.s.use_f0, "F0あり");
                    ui.add_enabled_ui(self.s.use_f0, |ui| {
                        egui::ComboBox::from_id_salt("f0")
                            .selected_text(self.s.f0_method.clone())
                            .show_ui(ui, |ui| {
                                for m in ["swiftf0", "rmvpe"] {
                                    ui.selectable_value(&mut self.s.f0_method, m.into(), m);
                                }
                            });
                    });
                });
                ui.add(egui::Slider::new(&mut self.s.moe, 0.0..=1.0).text("Moe Boost (声質)"));

                section_heading(ui, "■ Index 検索");
                ui.checkbox(&mut self.s.use_index, "Indexを使用");
                ui.add_enabled(self.s.use_index, egui::Slider::new(&mut self.s.index_rate, 0.0..=1.0).text("Index率"));
                ui.add_enabled(self.s.use_index, egui::Slider::new(&mut self.s.index_k, 1.0..=8.0).text("検索数 k"));

                section_heading(ui, "■ ノイズキャンセリング");
                ui.horizontal(|ui| {
                    let on = self.s.denoise != "none";
                    let mut on2 = on;
                    ui.checkbox(&mut on2, "有効化");
                    self.s.denoise = if on2 { "spectral".into() } else { "none".into() };
                    ui.label("方式:");
                    egui::ComboBox::from_id_salt("dn")
                        .selected_text(if on2 { "spectral".to_string() } else { "OFF".to_string() })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.s.denoise, "spectral".into(), "spectral");
                            ui.selectable_value(&mut self.s.denoise, "none".into(), "OFF");
                        });
                });
                ui.add_enabled(
                    self.s.denoise != "none",
                    egui::Slider::new(&mut self.s.denoise_strength, 0.5..=2.0).text("除去強度"),
                );

                section_heading(ui, "■ 入力ノイズゲート");
                ui.checkbox(&mut self.s.noise_gate, "有効化 (ノイズ誤変換防止)");
                ui.horizontal(|ui| {
                    ui.label("感度:");
                    ui.add_enabled_ui(self.s.noise_gate, |ui| {
                        egui::ComboBox::from_id_salt("gsens")
                            .selected_text(match self.s.noise_gate_sensitivity {
                                0 => "低".to_string(),
                                2 => "高".to_string(),
                                _ => "中".to_string(),
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.s.noise_gate_sensitivity, 0, "低 (+10dB)");
                                ui.selectable_value(&mut self.s.noise_gate_sensitivity, 1, "中 (+6dB)");
                                ui.selectable_value(&mut self.s.noise_gate_sensitivity, 2, "高 (+3dB)");
                            });
                    });
                });

                section_heading(ui, "■ ポスト処理");
                ui.checkbox(&mut self.s.postprocess, "有効化");
                ui.add_enabled_ui(self.s.postprocess, |ui| {
                    ui.add(egui::Slider::new(&mut self.s.treble_boost_db, 0.0..=12.0).text("高域ブースト dB"));
                    ui.add(egui::Slider::new(&mut self.s.treble_cutoff_hz, 1000.0..=6000.0).text("カットオフ Hz"));
                    ui.add(egui::Slider::new(&mut self.s.limiter_threshold_db, -12.0..=0.0).text("リミッター閾値 dB"));
                    ui.add(egui::Slider::new(&mut self.s.limiter_release_ms, 20.0..=300.0).text("リリース ms"));
                    ui.checkbox(&mut self.s.normalizer_enabled, "音量正規化");
                    ui.add_enabled(
                        self.s.normalizer_enabled,
                        egui::Slider::new(&mut self.s.normalizer_target_rms, 0.02..=0.3).text("目標RMS"),
                    );
                });

                section_heading(ui, "■ WAV 変換テスト");
                ui.horizontal(|ui| {
                    ui.label("入力:");
                    ui.text_edit_singleline(&mut self.test_input);
                    if ui.button("参照...").clicked() {
                        if let Some(p) = rfd_pick_wav() {
                            self.test_input = p;
                        }
                    }
                    if ui.button("変換").clicked() {
                        self.run_file_test();
                    }
                });
                ui.label(self.test_state.lock().unwrap().clone());

                ui.add_space(10.0);
                ui.separator();
                ui.horizontal(|ui| {
                    if running {
                        if ui.button("■ 停止").clicked() {
                            self.stop();
                        }
                    } else if ui.button("▶ 開始").clicked() {
                        save_settings(&self.s);
                        self.start();
                    }
                });
            });
        });
        if running {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(1000));
        }
    }
}

fn main() -> eframe::Result {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([760.0, 640.0])
            .with_title("RCWX (Rust)"),
        ..Default::default()
    };
    eframe::run_native("RCWX", opts, Box::new(|_cc| Ok(Box::new(App::new()))))
}
