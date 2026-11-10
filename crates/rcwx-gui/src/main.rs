//! rcwx-gui — egui リアルタイムボイスチェンジャー GUI (M7)。
//!
//! Python GUI の主要機能を単一バイナリで提供:
//! モデル/デバイス選択・chunk (レイテンシ)・pitch・F0・denoise・ゲート・
//! moe・index・開始/停止・統計表示。設定は ~/.config/rcwx/config.json 互換。

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct GuiSettings {
    model: String,
    input_device: String,
    output_device: String,
    chunk_sec: f64,
    aggressive: bool,
    pitch: f64,
    f0_method: String,
    denoise: bool,
    noise_gate: bool,
    moe: f64,
    index_rate: f64,
    input_gain_db: f64,
    output_gain_db: f64,
}

impl Default for GuiSettings {
    fn default() -> Self {
        Self {
            model: String::new(),
            input_device: String::new(),
            output_device: String::new(),
            chunk_sec: 0.06,
            aggressive: true,
            pitch: 0.0,
            f0_method: "swiftf0".into(),
            denoise: false,
            noise_gate: false,
            moe: 0.0,
            index_rate: 0.0,
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
    last: f64,
}

struct App {
    settings: GuiSettings,
    models: Vec<String>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    running: Arc<AtomicBool>,
    stats: Arc<Mutex<LiveStats>>,
    error: Option<String>,
    // 裏スレッドのエラーと終了検知
    thread_err: Arc<Mutex<Option<String>>>,
    done_flag: Arc<AtomicBool>,
}

fn models_dir() -> std::path::PathBuf {
    let cfg_path = std::path::PathBuf::from(
        std::env::var("USERPROFILE").unwrap_or_default(),
    )
    .join(".config/rcwx/config.json");
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(&cfg_path).unwrap_or_default(),
    ) {
        if let Some(dir) = v["rvc_models_dir"].as_str() {
            return dir.into();
        }
    }
    std::path::PathBuf::from("model")
}

fn load_settings() -> GuiSettings {
    let mut s = GuiSettings::default();
    let cfg_path = std::path::PathBuf::from(
        std::env::var("USERPROFILE").unwrap_or_default(),
    )
    .join(".config/rcwx/config.json");
    let Ok(v) = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(&cfg_path).unwrap_or_default(),
    ) else {
        return s;
    };
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
            s.f0_method = x.into();
        }
        if let Some(x) = i["moe_boost"].as_f64() {
            s.moe = x;
        }
        if let Some(x) = i["index_ratio"].as_f64() {
            s.index_rate = x;
        }
    }
    s
}

fn save_settings(s: &GuiSettings) {
    let cfg_path = std::path::PathBuf::from(
        std::env::var("USERPROFILE").unwrap_or_default(),
    )
    .join(".config/rcwx/config.json");
    let Ok(mut v) = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(&cfg_path).unwrap_or_default(),
    ) else {
        return;
    };
    v["last_model_path"] = serde_json::json!(s.model);
    v["audio"]["input_device_name"] = serde_json::json!(s.input_device);
    v["audio"]["output_device_name"] = serde_json::json!(s.output_device);
    v["audio"]["chunk_sec"] = serde_json::json!(s.chunk_sec);
    v["audio"]["latency_mode"] =
        serde_json::json!(if s.aggressive { "aggressive" } else { "normal" });
    v["audio"]["input_gain_db"] = serde_json::json!(s.input_gain_db);
    v["audio"]["output_gain_db"] = serde_json::json!(s.output_gain_db);
    v["inference"]["f0_method"] = serde_json::json!(s.f0_method);
    v["inference"]["moe_boost"] = serde_json::json!(s.moe);
    v["inference"]["index_ratio"] = serde_json::json!(s.index_rate);
    let _ = std::fs::write(&cfg_path, serde_json::to_string_pretty(&v).unwrap_or_default());
}

impl App {
    fn new() -> Self {
        let mut settings = load_settings();
        let models = scan_models(&settings.model);
        let (inputs, outputs) = scan_devices();
        // config.json の ASIO 名など実在しないデバイス名は既定に戻す
        if !settings.input_device.is_empty()
            && !inputs.iter().any(|d| d.contains(&settings.input_device))
        {
            settings.input_device = String::new();
        }
        if !settings.output_device.is_empty()
            && !outputs.iter().any(|d| d.contains(&settings.output_device))
        {
            settings.output_device = String::new();
        }
        Self {
            settings,
            models,
            inputs,
            outputs,
            running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(Mutex::new(LiveStats::default())),
            error: None,
            thread_err: Arc::new(Mutex::new(None)),
            done_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    fn start(&mut self) {
        if self.running.load(Ordering::Acquire) {
            return;
        }
        let batch_hops = if self.settings.chunk_sec < 0.05 { 4 } else { 1 };
        let cfg = rcwx_rt::RtConfig {
            chunk_sec: self.settings.chunk_sec,
            batch_hops,
            pitch: self.settings.pitch,
            f0_method: self.settings.f0_method.clone(),
            denoise: if self.settings.denoise { "spectral".into() } else { "none".into() },
            noise_gate: self.settings.noise_gate,
            index_rate: self.settings.index_rate as f32,
            moe_boost: self.settings.moe,
            input_gain_db: self.settings.input_gain_db,
            output_gain_db: self.settings.output_gain_db,
            buffer_margin: if self.settings.aggressive { 0.1 } else { 0.25 },
            prebuffer_chunks: if self.settings.aggressive { 2 } else { 1 },
            ..Default::default()
        };
        let in_dev = if self.settings.input_device.is_empty() {
            None
        } else {
            Some(self.settings.input_device.clone())
        };
        let out_dev = if self.settings.output_device.is_empty() {
            None
        } else {
            Some(self.settings.output_device.clone())
        };
        let stop = Arc::clone(&self.running);
        stop.store(true, Ordering::Release);
        let stats = Arc::clone(&self.stats);
        let stats_cb: Box<dyn Fn(&rcwx_rt::RtConfig, u64, u64, u64, f64, f64, f64) + Send> =
            Box::new(move |_, hops, underruns, qfull, p50, p99, last| {
                *stats.lock().unwrap() = LiveStats {
                    hops,
                    underruns,
                    queue_full: qfull,
                    p50,
                    p99,
                    last,
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
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    let mut ins = Vec::new();
    let mut outs = Vec::new();
    if let Ok(ds) = host.input_devices() {
        ins = ds.map(|d| d.to_string()).collect();
    }
    if let Ok(ds) = host.output_devices() {
        outs = ds.map(|d| d.to_string()).collect();
    }
    (ins, outs)
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let running = self.running.load(Ordering::Acquire);
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
                    "hops {} | infer p50 {:.1}ms p99 {:.1}ms last {:.1}ms | underruns {} | queue_full {}",
                    st.hops, st.p50, st.p99, st.last, st.underruns, st.queue_full
                ));
            });
        });
        // 裏スレッドの終了/エラー反映
        if self.done_flag.load(Ordering::Acquire) {
            if self.running.load(Ordering::Acquire) {
                self.running.store(false, Ordering::Release);
            }
            if let Some(err) = self.thread_err.lock().unwrap().take() {
                self.error = Some(err);
            }
        }
        egui::CentralPanel::default().show(ui, |ui| {
            if let Some(err) = &self.error {
                ui.colored_label(egui::Color32::RED, format!("error: {err}"));
                ui.separator();
            }
            ui.heading("RCWX — Rust Voice Changer");
            ui.separator();

            ui.columns(2, |cols| {
                // 左: モデルとデバイス
                let c = &mut cols[0];
                c.heading("Model / Devices");
                c.label("RVC model (.pth)");
                egui::ComboBox::from_id_salt("model")
                    .selected_text(if self.settings.model.is_empty() {
                        "— select —".to_string()
                    } else {
                        self.settings
                            .model
                            .rsplit(['/', '\\'])
                            .next()
                            .unwrap_or("")
                            .to_string()
                    })
                    .show_ui(c, |ui| {
                        for m in self.models.clone() {
                            ui.selectable_value(&mut self.settings.model, m.clone(), m.rsplit(['/', '\\']).next().unwrap_or(""));
                        }
                    });
                c.label("Input device");
                egui::ComboBox::from_id_salt("in")
                    .selected_text(self.settings.input_device.clone())
                    .show_ui(c, |ui| {
                        ui.selectable_value(&mut self.settings.input_device, String::new(), "(default)");
                        for d in self.inputs.clone() {
                            ui.selectable_value(&mut self.settings.input_device, d.clone(), d);
                        }
                    });
                c.label("Output device");
                egui::ComboBox::from_id_salt("out")
                    .selected_text(self.settings.output_device.clone())
                    .show_ui(c, |ui| {
                        ui.selectable_value(&mut self.settings.output_device, String::new(), "(default)");
                        for d in self.outputs.clone() {
                            ui.selectable_value(&mut self.settings.output_device, d.clone(), d);
                        }
                    });

                // 右: レイテンシと音質
                let c = &mut cols[1];
                c.heading("Latency / Voice");
                let mut chunk_ms = self.settings.chunk_sec * 1000.0;
                let lo = if self.settings.aggressive { 20.0 } else { 40.0 };
                c.add(egui::Slider::new(&mut chunk_ms, lo..=300.0).text("chunk (ms)"));
                self.settings.chunk_sec = (chunk_ms / 20.0).round() * 20.0 / 1000.0;
                c.checkbox(&mut self.settings.aggressive, "Aggressive (低遅延)");
                c.add(
                    egui::Slider::new(&mut self.settings.pitch, -24.0..=24.0)
                        .text("pitch (半音)"),
                );
                egui::ComboBox::from_id_salt("f0")
                    .selected_text(self.settings.f0_method.clone())
                    .show_ui(c, |ui| {
                        for m in ["swiftf0", "rmvpe"] {
                            ui.selectable_value(&mut self.settings.f0_method, m.into(), m);
                        }
                    });
                c.checkbox(&mut self.settings.denoise, "ノイズキャンセリング (spectral)");
                c.checkbox(&mut self.settings.noise_gate, "ノイズゲート");
                c.add(egui::Slider::new(&mut self.settings.moe, 0.0..=1.0).text("moe"));
                c.add(egui::Slider::new(&mut self.settings.index_rate, 0.0..=1.0).text("index rate"));
                c.add(egui::Slider::new(&mut self.settings.input_gain_db, -30.0..=24.0).text("入力ゲイン dB"));
                c.add(egui::Slider::new(&mut self.settings.output_gain_db, -30.0..=24.0).text("出力ゲイン dB"));
            });

            ui.separator();
            ui.horizontal(|ui| {
                if running {
                    if ui.button("■ 停止").clicked() {
                        self.stop();
                    }
                } else if ui.button("▶ 開始").clicked() {
                    save_settings(&self.settings);
                    self.start();
                }
                if ui.button("再読み込み").clicked() {
                    self.models = scan_models(&self.settings.model);
                    let (i, o) = scan_devices();
                    self.inputs = i;
                    self.outputs = o;
                }
            });
        });
        if running {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
    }
}

fn main() -> eframe::Result {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([720.0, 420.0])
            .with_title("RCWX (Rust)"),
        ..Default::default()
    };
    eframe::run_native("RCWX", opts, Box::new(|_cc| Ok(Box::new(App::new()))))
}
