//! GPUI の設定画面。メイン / オーディオ / 詳細設定と、実行中に効くパラメータ。
//! 見た目は theme.rs (トワイライト・オーロラのグラスモーフィズム) に一任。

use crate::settings::{
    BundleInfo, GuiSettings, LiveStats, ModelEntry, bundle_info, convert_wav, device_rate,
    host_names, load_settings, models_root, pick_file, pick_folder, pick_save, recommended_gain_db,
    resolve_device, save_settings, scan_devices, scan_models, sidecar_index,
};
use crate::theme::{self, HeroKind, PillTab};
use gpui_kit::assets::IconName;
use gpui_kit::base::Disableable as _;
use gpui_kit::component::button::{Button, ButtonVariants};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::Icon;
use gpui_kit::component::input::{Input, InputState};
use gpui_kit::component::scroll::ScrollableElement as _;
use gpui_kit::component::select::{SearchableVec, Select, SelectEvent, SelectItem, SelectState};
use gpui_kit::component::slider::{Slider, SliderEvent, SliderState};
use gpui_kit::component::{ActiveTheme as _, IndexPath, Sizable as _, h_flex, v_flex};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use rcwx_rt::audio::{RtHooks, play_wav, record_seconds};
use rcwx_rt::{LiveParams, RtReport};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
struct Choice {
    label: SharedString,
    value: SharedString,
}

impl SelectItem for Choice {
    type Value = SharedString;

    fn title(&self) -> SharedString {
        self.label.clone()
    }

    fn value(&self) -> &Self::Value {
        &self.value
    }
}

type ChoiceSelect = SelectState<SearchableVec<Choice>>;

struct Sliders {
    input_gain: Entity<SliderState>,
    output_gain: Entity<SliderState>,
    chunk: Entity<SliderState>,
    pitch: Entity<SliderState>,
    moe: Entity<SliderState>,
    index_rate: Entity<SliderState>,
    index_k: Entity<SliderState>,
    denoise_strength: Entity<SliderState>,
    treble_boost: Entity<SliderState>,
    treble_cutoff: Entity<SliderState>,
    limiter_threshold: Entity<SliderState>,
    limiter_release: Entity<SliderState>,
    normalizer_rms: Entity<SliderState>,
    energy: Entity<SliderState>,
    slew: Entity<SliderState>,
}

struct CountMark {
    at: Instant,
    underruns: u64,
    trims: u64,
    queue_full: u64,
}

pub struct RcwxApp {
    s: GuiSettings,
    models: Vec<ModelEntry>,
    hosts: Vec<String>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    bundle: BundleInfo,
    index_beside: bool,
    input_rate: Option<u32>,
    output_rate: Option<u32>,
    tab: usize,
    model_select: Entity<ChoiceSelect>,
    host_select: Entity<ChoiceSelect>,
    input_select: Entity<ChoiceSelect>,
    output_select: Entity<ChoiceSelect>,
    channel_select: Entity<ChoiceSelect>,
    output_channel_select: Entity<ChoiceSelect>,
    f0_select: Entity<ChoiceSelect>,
    denoise_select: Entity<ChoiceSelect>,
    gate_select: Entity<ChoiceSelect>,
    voice_select: Entity<ChoiceSelect>,
    backend_select: Entity<ChoiceSelect>,
    sliders: Sliders,
    wav: Entity<InputState>,
    wav_in: Entity<InputState>,
    /// エンジンは true で停止する。開始時は false。
    stop_flag: Arc<AtomicBool>,
    session_live: Arc<AtomicBool>,
    running: bool,
    live: Option<Arc<Mutex<LiveParams>>>,
    stats_src: Arc<Mutex<LiveStats>>,
    stats: LiveStats,
    level_src: Arc<AtomicU32>,
    peak_src: Arc<AtomicU32>,
    level: f32,
    peak: f32,
    adapter_src: Arc<Mutex<String>>,
    adapter: String,
    thread_err: Arc<Mutex<Option<String>>>,
    done_flag: Arc<AtomicBool>,
    test_busy: Arc<AtomicBool>,
    play_stop: Arc<AtomicBool>,
    error: Option<String>,
    test_label: String,
    converted: String,
    buffer_banner: Option<String>,
    seen_underruns: u64,
    seen_trims: u64,
    seen_queue: u64,
    count_hist: VecDeque<CountMark>,
    polling: bool,
    _poll: Task<()>,
    _subs: Vec<Subscription>,
}

fn choice(label: impl Into<SharedString>, value: impl Into<SharedString>) -> Choice {
    Choice {
        label: label.into(),
        value: value.into(),
    }
}

fn index_path(row: usize) -> IndexPath {
    IndexPath::new(row)
}

fn choices_from_pairs<L, V, I>(pairs: I) -> Vec<Choice>
where
    L: Into<SharedString>,
    V: Into<SharedString>,
    I: IntoIterator<Item = (L, V)>,
{
    pairs.into_iter().map(|(label, value)| choice(label, value)).collect()
}

fn model_choices(models: &[ModelEntry]) -> Vec<Choice> {
    models
        .iter()
        .map(|m| choice(m.label.clone(), m.path.clone()))
        .collect()
}

fn device_choices(devices: &[String]) -> Vec<Choice> {
    let mut items = vec![choice("(既定)", "")];
    items.extend(devices.iter().cloned().map(|d| choice(d.clone(), d)));
    items
}

fn host_choices(hosts: &[String]) -> Vec<Choice> {
    let mut items = vec![choice("(既定)", "")];
    items.extend(hosts.iter().cloned().map(|name| choice(name.clone(), name)));
    items
}

fn selected_row(items: &[Choice], value: &str) -> Option<IndexPath> {
    items
        .iter()
        .position(|item| item.value.as_ref() == value)
        .map(index_path)
}

fn open_select(
    items: Vec<Choice>,
    current: &str,
    searchable: bool,
    window: &mut Window,
    cx: &mut Context<RcwxApp>,
) -> Entity<ChoiceSelect> {
    let row = selected_row(&items, current);
    cx.new(|cx| SelectState::new(SearchableVec::new(items), row, window, cx).searchable(searchable))
}

fn new_slider(cx: &mut Context<RcwxApp>, min: f32, max: f32, value: f32, step: f32) -> Entity<SliderState> {
    cx.new(|_| SliderState::new().min(min).max(max).step(step).default_value(value))
}

fn opt_name(name: &str) -> Option<&str> {
    if name.is_empty() { None } else { Some(name) }
}

fn mark(on: bool) -> &'static str {
    if on { "あり" } else { "なし" }
}

impl RcwxApp {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut s = load_settings();
        let mut hosts = host_names();
        if !s.host.is_empty() && !hosts.iter().any(|name| name.eq_ignore_ascii_case(&s.host)) {
            s.host.clear();
        }
        hosts.sort();
        let (inputs, outputs) = scan_devices(&s.host);
        s.input_device = resolve_device(&s.input_device, &inputs);
        s.output_device = resolve_device(&s.output_device, &outputs);
        s.chunk_sec = s.snapped_chunk_sec();
        let root = models_root(&s.rvc_models_dir);
        let models = scan_models(&root, &s.model);
        let bundle = bundle_info();
        let index_beside = sidecar_index(&s.model);
        let input_rate = device_rate(&s.host, &s.input_device, true);
        let output_rate = device_rate(&s.host, &s.output_device, false);

        let model_select = open_select(model_choices(&models), &s.model, true, window, cx);
        let host_select = open_select(host_choices(&hosts), &s.host, false, window, cx);
        let input_select = open_select(device_choices(&inputs), &s.input_device, true, window, cx);
        let output_select = open_select(device_choices(&outputs), &s.output_device, true, window, cx);
        let channel_select = open_select(
            choices_from_pairs([("自動 (平均)", "auto"), ("Ch 1", "0"), ("Ch 2", "1")]),
            &s.input_channel,
            false,
            window,
            cx,
        );
        let output_channel_select = open_select(
            choices_from_pairs([("両チャンネル", "all"), ("Ch 1", "0"), ("Ch 2", "1")]),
            &s.output_channel,
            false,
            window,
            cx,
        );
        let f0_select = open_select(
            choices_from_pairs([("swiftf0", "swiftf0"), ("rmvpe", "rmvpe"), ("なし", "none")]),
            s.f0_choice(),
            false,
            window,
            cx,
        );
        let denoise_select = open_select(
            choices_from_pairs([("spectral", "spectral"), ("OFF", "none")]),
            &s.denoise,
            false,
            window,
            cx,
        );
        let gate_select = open_select(
            choices_from_pairs([("低 (+10dB)", "0"), ("中 (+6dB)", "1"), ("高 (+3dB)", "2")]),
            &s.noise_gate_sensitivity.to_string(),
            false,
            window,
            cx,
        );
        let voice_select = open_select(
            choices_from_pairs([
                ("off", "off"),
                ("strict", "strict"),
                ("expand", "expand"),
                ("energy", "energy"),
            ]),
            &s.voice_gate_mode,
            false,
            window,
            cx,
        );
        let backend_select = open_select(
            choices_from_pairs([
                ("Auto", "auto"),
                ("DirectX 12 (Arc)", "dx12"),
                ("Vulkan", "vulkan"),
            ]),
            &s.backend,
            false,
            window,
            cx,
        );
        let sliders = Sliders {
            input_gain: new_slider(cx, -30.0, 24.0, s.input_gain_db as f32, 0.5),
            output_gain: new_slider(cx, -30.0, 24.0, s.output_gain_db as f32, 0.5),
            chunk: new_slider(cx, 20.0, 600.0, (s.chunk_sec * 1000.0) as f32, 20.0),
            pitch: new_slider(cx, -24.0, 24.0, s.pitch as f32, 0.5),
            moe: new_slider(cx, 0.0, 1.0, s.moe as f32, 0.01),
            index_rate: new_slider(cx, 0.0, 1.0, s.index_rate as f32, 0.01),
            index_k: new_slider(cx, 1.0, 8.0, s.index_k as f32, 1.0),
            denoise_strength: new_slider(cx, 0.5, 2.0, s.denoise_strength as f32, 0.05),
            treble_boost: new_slider(cx, 0.0, 12.0, s.treble_boost_db as f32, 0.5),
            treble_cutoff: new_slider(cx, 1000.0, 6000.0, s.treble_cutoff_hz as f32, 50.0),
            limiter_threshold: new_slider(cx, -12.0, 0.0, s.limiter_threshold_db as f32, 0.5),
            limiter_release: new_slider(cx, 20.0, 300.0, s.limiter_release_ms as f32, 5.0),
            normalizer_rms: new_slider(cx, 0.02, 0.3, s.normalizer_target_rms as f32, 0.01),
            energy: new_slider(cx, 0.01, 0.20, s.energy_threshold as f32, 0.01),
            slew: new_slider(cx, 1.0, 6.0, s.f0_slew_max_step as f32, 0.1),
        };
        let wav = cx.new(|cx| InputState::new(window, cx).placeholder("変換する WAV"));
        let wav_in = cx.new(|cx| InputState::new(window, cx).placeholder("ループ再生する WAV"));

        let mut app = Self {
            s,
            models,
            hosts,
            inputs,
            outputs,
            bundle,
            index_beside,
            input_rate,
            output_rate,
            tab: 0,
            model_select,
            host_select,
            input_select,
            output_select,
            channel_select,
            output_channel_select,
            f0_select,
            denoise_select,
            gate_select,
            voice_select,
            backend_select,
            sliders,
            wav,
            wav_in,
            stop_flag: Arc::new(AtomicBool::new(true)),
            session_live: Arc::new(AtomicBool::new(false)),
            running: false,
            live: None,
            stats_src: Arc::new(Mutex::new(LiveStats::default())),
            stats: LiveStats::default(),
            level_src: Arc::new(AtomicU32::new(0)),
            peak_src: Arc::new(AtomicU32::new(0)),
            level: 0.0,
            peak: 0.0,
            adapter_src: Arc::new(Mutex::new(String::new())),
            adapter: String::new(),
            thread_err: Arc::new(Mutex::new(None)),
            done_flag: Arc::new(AtomicBool::new(false)),
            test_busy: Arc::new(AtomicBool::new(false)),
            play_stop: Arc::new(AtomicBool::new(false)),
            error: None,
            test_label: String::new(),
            converted: String::new(),
            buffer_banner: None,
            seen_underruns: 0,
            seen_trims: 0,
            seen_queue: 0,
            count_hist: VecDeque::new(),
            polling: false,
            _poll: Task::ready(()),
            _subs: Vec::new(),
        };
        app.bind(window, cx);
        app
    }

    fn bind(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let model = self.model_select.clone();
        let host = self.host_select.clone();
        let input = self.input_select.clone();
        let output = self.output_select.clone();
        let channel = self.channel_select.clone();
        let output_channel = self.output_channel_select.clone();
        let f0 = self.f0_select.clone();
        let denoise = self.denoise_select.clone();
        let gate = self.gate_select.clone();
        let voice = self.voice_select.clone();
        let backend = self.backend_select.clone();

        self._subs.push(Self::bind_select(window, cx, &model, |this, value| {
            this.s.model = value.to_string();
            this.index_beside = sidecar_index(&this.s.model);
        }));
        self._subs.push(cx.subscribe_in(
            &host,
            window,
            |this, _, event: &SelectEvent<SearchableVec<Choice>>, window, cx| {
                if let SelectEvent::Confirm(Some(value)) = event {
                    this.s.host = value.to_string();
                    this.reload_devices(window, cx);
                    cx.notify();
                }
            },
        ));
        self._subs.push(cx.subscribe_in(
            &input,
            window,
            |this, _, event: &SelectEvent<SearchableVec<Choice>>, _, cx| {
                if let SelectEvent::Confirm(Some(value)) = event {
                    this.s.input_device = value.to_string();
                    this.refresh_rates();
                    this.push_live();
                    cx.notify();
                }
            },
        ));
        self._subs.push(cx.subscribe_in(
            &output,
            window,
            |this, _, event: &SelectEvent<SearchableVec<Choice>>, _, cx| {
                if let SelectEvent::Confirm(Some(value)) = event {
                    this.s.output_device = value.to_string();
                    this.refresh_rates();
                    this.push_live();
                    cx.notify();
                }
            },
        ));
        self._subs.push(Self::bind_select(window, cx, &channel, |this, value| {
            this.s.input_channel = value.to_string();
        }));
        self._subs.push(Self::bind_select(window, cx, &output_channel, |this, value| {
            this.s.output_channel = value.to_string();
        }));
        self._subs.push(cx.subscribe_in(
            &f0,
            window,
            |this, _, event: &SelectEvent<SearchableVec<Choice>>, window, cx| {
                if let SelectEvent::Confirm(Some(value)) = event {
                    if value.as_ref() == "none" {
                        this.s.use_f0 = false;
                    } else {
                        this.s.use_f0 = true;
                        this.s.f0_method = value.to_string();
                    }
                    this.sync_chunk_slider(window, cx);
                    cx.notify();
                }
            },
        ));
        self._subs.push(Self::bind_select(window, cx, &denoise, |this, value| {
            this.s.denoise = value.to_string();
        }));
        self._subs.push(Self::bind_select(window, cx, &gate, |this, value| {
            this.s.noise_gate_sensitivity = match value {
                "0" => 0,
                "2" => 2,
                _ => 1,
            };
        }));
        self._subs.push(Self::bind_select(window, cx, &voice, |this, value| {
            this.s.voice_gate_mode = value.to_string();
        }));
        self._subs.push(Self::bind_select(window, cx, &backend, |this, value| {
            this.s.backend = value.to_string();
            this.s.preserve_device = None;
        }));

        self.watch_slider(window, cx, &self.sliders.input_gain.clone(), |this, v| this.s.input_gain_db = f64::from(v));
        self.watch_slider(window, cx, &self.sliders.output_gain.clone(), |this, v| this.s.output_gain_db = f64::from(v));
        self.watch_slider(window, cx, &self.sliders.chunk.clone(), |this, v| {
            this.s.chunk_sec = f64::from(v) / 1000.0;
            this.s.chunk_sec = this.s.snapped_chunk_sec();
        });
        self.watch_slider(window, cx, &self.sliders.pitch.clone(), |this, v| this.s.pitch = f64::from(v));
        self.watch_slider(window, cx, &self.sliders.moe.clone(), |this, v| this.s.moe = f64::from(v));
        self.watch_slider(window, cx, &self.sliders.index_rate.clone(), |this, v| this.s.index_rate = f64::from(v));
        self.watch_slider(window, cx, &self.sliders.index_k.clone(), |this, v| this.s.index_k = f64::from(v));
        self.watch_slider(window, cx, &self.sliders.denoise_strength.clone(), |this, v| {
            this.s.denoise_strength = f64::from(v);
        });
        self.watch_slider(window, cx, &self.sliders.treble_boost.clone(), |this, v| this.s.treble_boost_db = f64::from(v));
        self.watch_slider(window, cx, &self.sliders.treble_cutoff.clone(), |this, v| {
            this.s.treble_cutoff_hz = f64::from(v);
        });
        self.watch_slider(window, cx, &self.sliders.limiter_threshold.clone(), |this, v| {
            this.s.limiter_threshold_db = f64::from(v);
        });
        self.watch_slider(window, cx, &self.sliders.limiter_release.clone(), |this, v| {
            this.s.limiter_release_ms = f64::from(v);
        });
        self.watch_slider(window, cx, &self.sliders.normalizer_rms.clone(), |this, v| {
            this.s.normalizer_target_rms = f64::from(v);
        });
        self.watch_slider(window, cx, &self.sliders.energy.clone(), |this, v| this.s.energy_threshold = f64::from(v));
        self.watch_slider(window, cx, &self.sliders.slew.clone(), |this, v| this.s.f0_slew_max_step = f64::from(v));
    }

    fn bind_select(
        window: &mut Window,
        cx: &mut Context<Self>,
        select: &Entity<ChoiceSelect>,
        apply: fn(&mut Self, &str),
    ) -> Subscription {
        cx.subscribe_in(
            select,
            window,
            move |this, _, event: &SelectEvent<SearchableVec<Choice>>, _, cx| {
                if let SelectEvent::Confirm(Some(value)) = event {
                    apply(this, value.as_ref());
                    this.push_live();
                    cx.notify();
                }
            },
        )
    }

    fn watch_slider(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
        slider: &Entity<SliderState>,
        apply: fn(&mut Self, f32),
    ) {
        let sub = cx.subscribe_in(slider, window, move |this, _, event: &SliderEvent, _, cx| {
            if let SliderEvent::Change(value) = event {
                apply(this, value.start());
                this.push_live();
                cx.notify();
            }
        });
        self._subs.push(sub);
    }

    fn push_live(&self) {
        let Some(live) = &self.live else {
            return;
        };
        let mut guard = live.lock().unwrap_or_else(|err| err.into_inner());
        *guard = self.s.to_live();
    }

    fn refresh_rates(&mut self) {
        self.input_rate = device_rate(&self.s.host, &self.s.input_device, true);
        self.output_rate = device_rate(&self.s.host, &self.s.output_device, false);
    }

    fn replace_choices(
        select: &Entity<ChoiceSelect>,
        items: Vec<Choice>,
        current: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let current: SharedString = current.to_string().into();
        select.update(cx, |state, cx| {
            state.set_items(SearchableVec::new(items), window, cx);
            state.set_selected_value(&current, window, cx);
        });
    }

    fn reload_devices(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let (inputs, outputs) = scan_devices(&self.s.host);
        self.s.input_device = resolve_device(&self.s.input_device, &inputs);
        self.s.output_device = resolve_device(&self.s.output_device, &outputs);
        self.inputs = inputs;
        self.outputs = outputs;
        Self::replace_choices(&self.input_select, device_choices(&self.inputs), &self.s.input_device, window, cx);
        Self::replace_choices(
            &self.output_select,
            device_choices(&self.outputs),
            &self.s.output_device,
            window,
            cx,
        );
        self.refresh_rates();
    }

    fn rescan(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.hosts = host_names();
        self.models = scan_models(&models_root(&self.s.rvc_models_dir), &self.s.model);
        self.bundle = bundle_info();
        self.index_beside = sidecar_index(&self.s.model);
        Self::replace_choices(&self.model_select, model_choices(&self.models), &self.s.model, window, cx);
        Self::replace_choices(&self.host_select, host_choices(&self.hosts), &self.s.host, window, cx);
        self.reload_devices(window, cx);
        cx.notify();
    }

    fn set_model_path(&mut self, path: String, window: &mut Window, cx: &mut Context<Self>) {
        self.s.model = path;
        self.models = scan_models(&models_root(&self.s.rvc_models_dir), &self.s.model);
        self.index_beside = sidecar_index(&self.s.model);
        Self::replace_choices(&self.model_select, model_choices(&self.models), &self.s.model, window, cx);
        cx.notify();
    }

    fn sync_chunk_slider(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.s.chunk_sec = self.s.snapped_chunk_sec();
        let ms = (self.s.chunk_sec * 1000.0) as f32;
        self.sliders.chunk.update(cx, |state, cx| {
            state.set_value(ms, window, cx);
        });
    }

    fn set_pitch(&mut self, pitch: f32, window: &mut Window, cx: &mut Context<Self>) {
        self.s.pitch = f64::from(pitch);
        self.sliders.pitch.update(cx, |state, cx| state.set_value(pitch, window, cx));
        self.push_live();
    }

    fn ensure_poll(&mut self, cx: &mut Context<Self>) {
        if self.polling {
            return;
        }
        self.polling = true;
        self._poll = cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(Duration::from_millis(200)).await;
                let keep = this
                    .update(cx, |view, cx| {
                        view.poll();
                        let keep = view.session_live.load(Ordering::Acquire)
                            || view.test_busy.load(Ordering::Acquire);
                        if !keep {
                            view.polling = false;
                        }
                        cx.notify();
                        keep
                    })
                    .unwrap_or(false);
                if !keep {
                    break;
                }
            }
        });
    }

    fn poll(&mut self) {
        if let Ok(stats) = self.stats_src.lock() {
            self.stats = stats.clone();
        }
        self.level = f32::from_bits(self.level_src.load(Ordering::Relaxed));
        self.peak = f32::from_bits(self.peak_src.load(Ordering::Relaxed));
        if let Ok(name) = self.adapter_src.lock() {
            self.adapter = name.clone();
        }
        self.note_counts();
        if self.done_flag.swap(false, Ordering::AcqRel) {
            self.running = false;
            if let Some(err) = self.thread_err.lock().ok().and_then(|mut guard| guard.take()) {
                self.error = Some(err);
            }
        }
    }

    fn note_counts(&mut self) {
        let now = Instant::now();
        if self.running || self.session_live.load(Ordering::Acquire) {
            let mut parts = Vec::new();
            if self.stats.underruns > self.seen_underruns {
                parts.push(format!("underrun +{}", self.stats.underruns - self.seen_underruns));
            }
            if self.stats.queue_full > self.seen_queue {
                parts.push(format!("queue full +{}", self.stats.queue_full - self.seen_queue));
            }
            if self.stats.trims > self.seen_trims {
                parts.push(format!("trim +{}", self.stats.trims - self.seen_trims));
            }
            if !parts.is_empty() {
                self.buffer_banner = Some(format!("バッファ: {}", parts.join(" / ")));
            }
        }
        self.seen_underruns = self.stats.underruns;
        self.seen_trims = self.stats.trims;
        self.seen_queue = self.stats.queue_full;
        self.count_hist.push_back(CountMark {
            at: now,
            underruns: self.stats.underruns,
            trims: self.stats.trims,
            queue_full: self.stats.queue_full,
        });
        while self
            .count_hist
            .front()
            .is_some_and(|mark| now.duration_since(mark.at).as_secs_f64() > 60.0)
        {
            self.count_hist.pop_front();
        }
    }

    fn since(&self, seconds: f64, pick: fn(&CountMark) -> u64, current: u64) -> u64 {
        let now = Instant::now();
        let baseline = self
            .count_hist
            .iter()
            .rev()
            .find(|mark| now.duration_since(mark.at).as_secs_f64() >= seconds)
            .map(pick)
            .or_else(|| self.count_hist.front().map(pick))
            .unwrap_or(current);
        current.saturating_sub(baseline)
    }

    fn start(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.session_live.load(Ordering::Acquire) {
            return;
        }
        self.play_stop.store(true, Ordering::Release);
        self.sync_chunk_slider(window, cx);
        let typed = self.wav_in.read(cx).value().to_string();
        if !typed.trim().is_empty() {
            self.s.wav_input = typed;
        }
        save_settings(&self.s);
        let cfg = self.s.to_rt_config();
        let backend = match self.s.backend.as_str() {
            "dx12" => rcwx_nn::engine::Backend::Dx12,
            "vulkan" => rcwx_nn::engine::Backend::Vulkan,
            _ => rcwx_nn::engine::Backend::Auto,
        };
        let in_dev = self.s.input_device.clone();
        let out_dev = self.s.output_device.clone();
        let stop = Arc::new(AtomicBool::new(false));
        self.stop_flag = Arc::clone(&stop);
        let session = Arc::clone(&self.session_live);
        session.store(true, Ordering::Release);
        self.running = true;
        self.error = None;
        self.buffer_banner = None;
        self.stats = LiveStats::default();
        self.seen_underruns = 0;
        self.seen_trims = 0;
        self.seen_queue = 0;
        self.count_hist.clear();
        *self.stats_src.lock().expect("stats") = LiveStats::default();
        self.level = 0.0;
        self.peak = 0.0;
        self.level_src.store(0, Ordering::Relaxed);
        self.peak_src.store(0, Ordering::Relaxed);
        self.adapter.clear();
        if let Ok(mut name) = self.adapter_src.lock() {
            name.clear();
        }
        *self.thread_err.lock().expect("err") = None;
        self.done_flag.store(false, Ordering::Release);

        let live = Arc::new(Mutex::new(self.s.to_live()));
        self.live = Some(Arc::clone(&live));
        let stats_src = Arc::clone(&self.stats_src);
        let level = Arc::clone(&self.level_src);
        let peak = Arc::clone(&self.peak_src);
        let adapter = Arc::clone(&self.adapter_src);
        let thread_err = Arc::clone(&self.thread_err);
        let done = Arc::clone(&self.done_flag);
        let stats_cb = Box::new(move |_: &rcwx_rt::RtConfig, report: RtReport| {
            if let Ok(mut slot) = stats_src.lock() {
                *slot = LiveStats {
                    hops: report.hops,
                    underruns: report.underruns,
                    queue_full: report.queue_full,
                    trims: report.trims,
                    p50: report.p50,
                    p95: report.p95,
                    p99: report.p99,
                    last_ms: report.last_ms,
                };
            }
        });
        let hooks = RtHooks {
            on_stats: Some(stats_cb),
            input_level: Some(level),
            input_peak: Some(peak),
            live: Some(live),
            adapter_name: Some(adapter),
        };
        std::thread::spawn(move || {
            let result = rcwx_rt::audio::run_realtime(
                backend,
                cfg,
                opt_name(&in_dev),
                opt_name(&out_dev),
                stop,
                hooks,
            );
            if let Err(err) = result
                && let Ok(mut slot) = thread_err.lock()
            {
                *slot = Some(format!("{err:#}"));
            }
            done.store(true, Ordering::Release);
            session.store(false, Ordering::Release);
        });
        self.ensure_poll(cx);
        cx.notify();
    }

    fn stop(&mut self, cx: &mut Context<Self>) {
        self.stop_flag.store(true, Ordering::Release);
        self.running = false;
        cx.notify();
    }

    fn browse(&mut self, window: &mut Window, cx: &mut Context<Self>, kind: &'static str) {
        cx.spawn_in(window, async move |this, cx| {
            let picked = cx
                .background_spawn(async move {
                    match kind {
                        "wav" | "loop" => pick_file("WAV を選択", "WAV files (*.wav)|*.wav"),
                        "dir" => pick_folder("RVC モデルのフォルダ"),
                        _ => pick_file("RVCモデルを選択", "RVC model (*.pth)|*.pth"),
                    }
                })
                .await;
            let Some(path) = picked else {
                return;
            };
            this.update_in(cx, |view, window, cx| {
                match kind {
                    "wav" => {
                        view.wav.update(cx, |state, cx| state.set_value(path, window, cx));
                        view.test_label.clear();
                    }
                    "loop" => {
                        view.s.wav_input = path.clone();
                        view.s.use_wav_input = true;
                        view.wav_in.update(cx, |state, cx| state.set_value(path, window, cx));
                    }
                    "dir" => {
                        view.s.rvc_models_dir = path;
                        view.rescan(window, cx);
                    }
                    _ => view.set_model_path(path, window, cx),
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn convert(&mut self, cx: &mut Context<Self>) {
        if self.test_busy.load(Ordering::Acquire) {
            return;
        }
        let input = self.wav.read(cx).value().to_string();
        if input.trim().is_empty() || !std::path::Path::new(input.trim()).exists() {
            self.test_label = "ファイルがありません".into();
            cx.notify();
            return;
        }
        self.test_busy.store(true, Ordering::Release);
        self.test_label = "変換中...".into();
        let cfg = self.s.to_rt_config();
        let busy = Arc::clone(&self.test_busy);
        self.ensure_poll(cx);
        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move { convert_wav(input.trim(), &cfg) })
                .await;
            this.update(cx, |view, cx| {
                view.test_label = match &result {
                    Ok(path) => {
                        view.converted = path.clone();
                        format!("完了: {path}")
                    }
                    Err(err) => err.clone(),
                };
                busy.store(false, Ordering::Release);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn play_converted(&mut self, cx: &mut Context<Self>) {
        if self.converted.is_empty() || self.test_busy.load(Ordering::Acquire) || self.session_live.load(Ordering::Acquire)
        {
            self.test_label = "再生できる変換結果がありません".into();
            cx.notify();
            return;
        }
        self.play_stop.store(false, Ordering::Release);
        self.test_busy.store(true, Ordering::Release);
        self.test_label = "再生中...".into();
        let path = self.converted.clone();
        let host = self.s.host.clone();
        let device = self.s.output_device.clone();
        let stop = Arc::clone(&self.play_stop);
        let busy = Arc::clone(&self.test_busy);
        self.ensure_poll(cx);
        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move { play_wav(&host, opt_name(&device), std::path::Path::new(&path), &stop) })
                .await;
            this.update(cx, |view, cx| {
                view.test_label = match result {
                    Ok(()) => "再生が終わりました".into(),
                    Err(err) => format!("{err:#}"),
                };
                busy.store(false, Ordering::Release);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn save_converted(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let src = self.converted.clone();
        if src.is_empty() {
            self.test_label = "先に変換してください".into();
            cx.notify();
            return;
        }
        cx.spawn_in(window, async move |this, cx| {
            let dest = cx
                .background_spawn(async move { pick_save("変換結果の保存", "converted.wav") })
                .await;
            let Some(dest) = dest else {
                return;
            };
            let copied = std::fs::copy(&src, &dest);
            this.update(cx, |view, cx| {
                view.test_label = match copied {
                    Ok(_) => format!("保存: {dest}"),
                    Err(err) => format!("保存失敗: {err}"),
                };
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn audio_test(&mut self, cx: &mut Context<Self>) {
        if self.session_live.load(Ordering::Acquire) {
            self.test_label = "実行中はテストできません".into();
            cx.notify();
            return;
        }
        if self.test_busy.load(Ordering::Acquire) {
            self.play_stop.store(true, Ordering::Release);
            return;
        }
        self.play_stop.store(false, Ordering::Release);
        self.test_busy.store(true, Ordering::Release);
        self.test_label = "録音中 (3秒)...".into();
        let host = self.s.host.clone();
        let input = self.s.input_device.clone();
        let output = self.s.output_device.clone();
        let channel = match self.s.input_channel.as_str() {
            "0" => Some(0usize),
            "1" => Some(1usize),
            _ => None,
        };
        let cfg = self.s.to_rt_config();
        let stop = Arc::clone(&self.play_stop);
        let busy = Arc::clone(&self.test_busy);
        let rec = std::env::temp_dir().join("rcwx-audio-test.wav");
        self.ensure_poll(cx);
        cx.spawn(async move |this, cx| {
            let rec_for_task = rec.clone();
            let stop_rec = Arc::clone(&stop);
            let recorded = cx
                .background_spawn(async move {
                    record_seconds(&host, opt_name(&input), channel, 3.0, &rec_for_task, &stop_rec)
                })
                .await;
            if recorded.is_err() || stop.load(Ordering::Acquire) {
                this.update(cx, |view, cx| {
                    view.test_label = match recorded {
                        Ok(()) => "録音を停止しました".into(),
                        Err(err) => format!("{err:#}"),
                    };
                    busy.store(false, Ordering::Release);
                    cx.notify();
                })
                .ok();
                return;
            }
            this.update(cx, |view, cx| {
                view.test_label = "変換中...".into();
                cx.notify();
            })
            .ok();
            let rec_path = rec.to_string_lossy().into_owned();
            let converted = cx
                .background_spawn(async move { convert_wav(&rec_path, &cfg) })
                .await;
            let Ok(out) = converted else {
                this.update(cx, |view, cx| {
                    view.test_label = converted.err().unwrap_or_else(|| "変換失敗".into());
                    busy.store(false, Ordering::Release);
                    cx.notify();
                })
                .ok();
                return;
            };
            let shown = out.clone();
            this.update(cx, |view, cx| {
                view.converted = shown;
                view.test_label = "再生中...".into();
                cx.notify();
            })
            .ok();
            let host = this.update(cx, |view, _| view.s.host.clone()).unwrap_or_default();
            let label = out.clone();
            let play_result = cx
                .background_spawn(async move {
                    play_wav(&host, opt_name(&output), std::path::Path::new(&out), &stop)
                })
                .await;
            this.update(cx, |view, cx| {
                view.test_label = match play_result {
                    Ok(()) => format!("テスト完了: {label}"),
                    Err(err) => format!("{err:#}"),
                };
                busy.store(false, Ordering::Release);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn apply_settings(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        save_settings(&self.s);
        self.rescan(window, cx);
        self.test_label = "設定を保存し、モデル一覧を読み直しました".into();
    }

    fn apply_recommended(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(gain) = recommended_gain_db(self.peak) else {
            return;
        };
        self.s.input_gain_db = gain;
        self.sliders.input_gain.update(cx, |state, cx| {
            state.set_value(gain as f32, window, cx);
        });
        self.push_live();
    }

    fn same_device(&self) -> bool {
        !self.s.input_device.is_empty() && self.s.input_device == self.s.output_device
    }
}

/// 色つぼみアイコンつきのガラスカード。
fn card(
    icon: IconName,
    tint: Hsla,
    title: &'static str,
    cx: &App,
    body: impl IntoElement,
) -> AnyElement {
    theme::glass_panel(
        v_flex()
            .w_full()
            .p_4()
            .gap_2p5()
            .child(theme::card_head(icon, tint, title))
            .child(body),
        cx,
    )
    .into_any_element()
}

fn field_label(text: &'static str) -> AnyElement {
    div()
        .w(px(132.))
        .flex_shrink_0()
        .text_sm()
        .text_color(theme::muted_lilac())
        .child(text)
        .into_any_element()
}

fn select_row(label: &'static str, select: &Entity<ChoiceSelect>) -> AnyElement {
    h_flex()
        .flex_1()
        .items_center()
        .gap_3()
        .child(field_label(label))
        .child(div().flex_1().min_w_0().child(Select::new(select)))
        .into_any_element()
}

fn value_text(text: impl Into<SharedString>) -> AnyElement {
    theme::value_chip(text)
}

fn note(text: impl Into<SharedString>) -> AnyElement {
    div()
        .text_xs()
        .text_color(theme::muted_lilac().opacity(0.85))
        .child(text.into())
        .into_any_element()
}

fn slider_row(label: &'static str, value: String, slider: impl IntoElement) -> AnyElement {
    h_flex()
        .w_full()
        .items_center()
        .gap_3()
        .child(field_label(label))
        .child(div().flex_1().min_w(px(80.)).child(slider))
        .child(value_text(value))
        .into_any_element()
}

fn banner(icon: IconName, tint: Hsla, text: impl Into<SharedString>) -> AnyElement {
    theme::glass_banner(icon, tint, text)
}

fn rate_text(rate: Option<u32>) -> String {
    match rate {
        Some(rate) => format!("{rate} Hz"),
        None => "-- Hz".into(),
    }
}

impl Render for RcwxApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let session = self.session_live.load(Ordering::Acquire);
        div()
            .relative()
            .size_full()
            .text_color(cx.theme().foreground)
            .child(theme::aurora_backdrop())
            .child(
                v_flex()
                    .relative()
                    .size_full()
                    .text_color(cx.theme().foreground)
                    .child(self.header(session, cx))
                    .child(self.tabs(cx))
                    .child(self.body(cx))
                    .child(self.footer(session)),
            )
    }
}

impl RcwxApp {
    fn status_note(&self) -> (&'static str, Hsla, bool) {
        if self.running && self.stats.hops == 0 {
            ("モデル読込中", theme::amber(), true)
        } else if self.running {
            ("Running", theme::mint(), true)
        } else if self.session_live.load(Ordering::Acquire) {
            ("停止中", theme::amber(), false)
        } else {
            ("Stopped", theme::muted_lilac(), false)
        }
    }

    fn status_pill(&self) -> AnyElement {
        let (label, color, live) = self.status_note();
        h_flex()
            .items_center()
            .gap_1p5()
            .px_3()
            .py_1p5()
            .rounded_full()
            .bg(theme::white(0.06))
            .border_1()
            .border_color(theme::white(0.10))
            .child(theme::pulse_dot(color, "status-dot", live))
            .child(div().text_xs().text_color(color).child(label))
            .into_any_element()
    }

    fn header(&mut self, session: bool, cx: &mut Context<Self>) -> AnyElement {
        let hero = if self.running {
            theme::hero_button("stop", "停止", HeroKind::Danger, cx.listener(|this, _, _, cx| this.stop(cx)))
        } else if session {
            theme::hero_button("stopping", "停止中", HeroKind::Ghost, |_, _, _| {})
        } else {
            theme::hero_button(
                "start",
                "開始",
                HeroKind::Primary,
                cx.listener(|this, _, window, cx| this.start(window, cx)),
            )
        };
        v_flex()
            .w_full()
            .px_6()
            .pt_5()
            .pb_2()
            .gap_2()
            .child(
                h_flex()
                    .w_full()
                    .items_center()
                    .justify_between()
                    .child(
                        h_flex()
                            .items_center()
                            .gap_3()
                            .child(
                                div()
                                    .size(px(40.))
                                    .rounded_full()
                                    .bg(linear_gradient(
                                        135.,
                                        linear_color_stop(theme::sakura(), 0.),
                                        linear_color_stop(theme::lavender_deep(), 1.),
                                    ))
                                    .shadow(vec![
                                        BoxShadow::new(px(0.), px(2.), theme::sakura().opacity(0.45))
                                            .blur_radius(px(16.)),
                                    ])
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .child(
                                        Icon::new(IconName::Sparkles)
                                            .text_color(theme::plum())
                                            .with_size(px(18.)),
                                    ),
                            )
                            .child(
                                v_flex()
                                    .gap_0p5()
                                    .child(
                                        div()
                                            .text_xl()
                                            .font_weight(FontWeight::BOLD)
                                            .text_color(theme::porcelain())
                                            .child("RCWX"),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(theme::muted_lilac())
                                            .child("Real-time Voice Changer"),
                                    ),
                            ),
                    )
                    .child(h_flex().items_center().gap_2().child(self.status_pill()).child(hero)),
            )
            .child(note(
                "ピッチ、ゲイン、Index、ノイズ、ゲート、ポスト処理は実行中に反映されます。F0方式、チャンク、デバイス、ホスト、WAV入力は再開始で反映されます。",
            ))
            .into_any_element()
    }

    fn tabs(&mut self, cx: &mut Context<Self>) -> AnyElement {
        div()
            .px_6()
            .pb_1()
            .child(theme::pill_tabs(
                &[
                    PillTab { id: "tab-main", icon: IconName::AudioWaveform, label: "メイン" },
                    PillTab { id: "tab-audio", icon: IconName::WavesHorizontal, label: "オーディオ" },
                    PillTab { id: "tab-settings", icon: IconName::SlidersHorizontal, label: "詳細設定" },
                ],
                self.tab,
                cx.listener(|this, index: &usize, _, cx| {
                    this.tab = *index;
                    cx.notify();
                }),
            ))
            .into_any_element()
    }

    fn body(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let page = match self.tab {
            1 => self.tab_audio(cx),
            2 => self.tab_settings(cx),
            _ => self.tab_main(cx),
        };
        v_flex()
            .id("rcwx-scroll")
            .flex_1()
            .min_h_0()
            .w_full()
            .overflow_y_scrollbar()
            .overflow_x_hidden()
            .child(
                v_flex()
                    .px_6()
                    .py_4()
                    .gap_3()
                    .w_full()
                    .when_some(self.error.clone(), |this, err| {
                        this.child(banner(IconName::CircleAlert, theme::rose(), err))
                    })
                    .when(self.same_device(), |this| {
                        this.child(banner(
                            IconName::TriangleAlert,
                            theme::amber(),
                            "入力と出力が同じデバイスです。ヘッドホンを使うとハウリングを避けられます。",
                        ))
                    })
                    .when_some(self.buffer_banner.clone(), |this, text| {
                        let text = text.clone();
                        this.child(
                            h_flex()
                                .w_full()
                                .items_start()
                                .gap_2()
                                .child(div().flex_1().child(banner(IconName::Bell, theme::amber(), text)))
                                .child(
                                    Button::new("dismiss-buffer")
                                        .label("閉じる")
                                        .on_click(cx.listener(|this, _, _, cx| {
                                            this.buffer_banner = None;
                                            cx.notify();
                                        })),
                                ),
                        )
                    })
                    .child(page),
            )
            .into_any_element()
    }

    fn tab_main(&mut self, cx: &mut Context<Self>) -> AnyElement {
        h_flex()
            .w_full()
            .items_start()
            .gap_3()
            .child(
                v_flex()
                    .flex_grow(1.)
                    .flex_shrink(1.)
                    .min_w_0()
                    .gap_3()
                    .bg(hsla(120., 0.9, 0.35, 0.85))
                            .child(self.section_model(cx))
                    .child(self.section_pitch(cx))
                    .child(self.section_index(cx))
                    .child(self.section_denoise(cx))
                    .child(self.section_voice(cx)),
            )
            .child(
                v_flex()
                    .w(px(420.))
                    .flex_shrink_0()
                    .gap_3()
                    .bg(hsla(0., 0.9, 0.45, 0.85))
                            .child(self.section_latency(cx))
                    .child(self.section_device_summary(cx))
                    .child(self.section_test(cx)),
            )
            .into_any_element()
    }

    fn tab_audio(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let recommend = match recommended_gain_db(self.peak) {
            Some(gain) => format!("推奨 {gain:+.0} dB（ピークを -6 dB へ）"),
            None => "推奨 -- dB（信号なし）".into(),
        };
        v_flex()
            .w_full()
            .gap_3()
            .child(card(
                IconName::Mic,
                theme::sky(),
                "デバイス",
                cx,
                v_flex()
                    .gap_2()
                    .w_full()
                    .child(select_row("ホスト", &self.host_select))
                    .child(note(
                        "ASIO バッファは、一覧に ASIO ホストがあるときドライバのパネル設定に従います。",
                    ))
                    .child(
                        h_flex().w_full().gap_3().items_start().child(
                            v_flex()
                                .flex_1()
                                .min_w_0()
                                .gap_1()
                                .child(div().text_sm().text_color(theme::porcelain().opacity(0.88)).child("入力"))
                                .child(Select::new(&self.input_select).placeholder("(既定)")),
                        ).child(
                            v_flex()
                                .flex_1()
                                .min_w_0()
                                .gap_1()
                                .child(div().text_sm().text_color(theme::porcelain().opacity(0.88)).child("出力"))
                                .child(Select::new(&self.output_select).placeholder("(既定)")),
                        ),
                    )
                    .child(
                        h_flex()
                            .gap_3()
                            .child(select_row("入力ch", &self.channel_select))
                            .child(select_row("出力ch", &self.output_channel_select)),
                    )
                    .child(note(
                        format!(
                            "サンプルレート  入力 {} / 出力 {}",
                            rate_text(self.input_rate),
                            rate_text(self.output_rate)
                        ),
                    ))
                    .child(
                        Button::new("rescan-audio")
                            .label("デバイスを再読込")
                            .on_click(cx.listener(|this, _, window, cx| this.rescan(window, cx))),
                    ),
            ))
            .child(card(
                IconName::Activity,
                theme::mint(),
                "レベル",
                cx,
                v_flex()
                    .gap_2p5()
                    .w_full()
                    .child(
                        h_flex()
                            .w_full()
                            .items_center()
                            .gap_3()
                            .child(field_label("入力レベル"))
                            .child(theme::vu_meter(self.level)),
                    )
                    .child(slider_row(
                        "入力ゲイン",
                        format!("{:+.1} dB", self.s.input_gain_db),
                        Slider::new(&self.sliders.input_gain),
                    ))
                    .child(
                        h_flex()
                            .items_center()
                            .gap_3()
                            .child(div().text_sm().text_color(theme::muted_lilac()).child(recommend))
                            .child(
                                Button::new("apply-gain")
                                    .label("推奨を適用")
                                    .on_click(cx.listener(|this, _, window, cx| this.apply_recommended(window, cx))),
                            ),
                    )
                    .child(slider_row(
                        "出力ゲイン",
                        format!("{:+.1} dB", self.s.output_gain_db),
                        Slider::new(&self.sliders.output_gain),
                    )),
            ))
            .into_any_element()
    }

    fn tab_settings(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let dir = if self.s.rvc_models_dir.is_empty() {
            "model（作業フォルダ相対）".to_string()
        } else {
            self.s.rvc_models_dir.clone()
        };
        let preserved = self.s.preserve_device.clone();
        v_flex()
            .w_full()
            .gap_3()
            .child(card(
                IconName::Cpu,
                theme::sakura(),
                "推論",
                cx,
                v_flex()
                    .gap_2()
                    .w_full()
                    .child(select_row("バックエンド", &self.backend_select))
                    .child(note("推論精度: f32"))
                    .when_some(preserved, |this, raw| {
                        this.child(note(
                            format!("保存されていた device={raw} は Auto で起動します。選び直すと保存値を更新します。"),
                        ))
                    })
                    .child(note(format!("実行バンドル: {}", self.bundle.dir)))
                    .child(note(
                        format!(
                            "HuBERT {} / Synth {} / Index {}",
                            mark(self.bundle.hubert),
                            mark(self.bundle.synth),
                            mark(self.bundle.ivf)
                        ),
                    ))
                    .child(note(format!("RVC モデルフォルダ: {dir}")))
                    .child(
                        h_flex()
                            .gap_2()
                            .child(
                                Button::new("browse-dir")
                                    .label("フォルダ")
                                    .on_click(cx.listener(|this, _, window, cx| this.browse(window, cx, "dir"))),
                            )
                            .child(
                                Button::new("apply-settings")
                                    .primary()
                                    .label("設定を適用")
                                    .on_click(cx.listener(|this, _, window, cx| this.apply_settings(window, cx))),
                            ),
                    ),
            ))
            .child(self.section_post(cx))
            .child(self.section_convert(cx))
            .into_any_element()
    }

    fn section_model(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let beside = if self.index_beside { "あり" } else { "なし" };
        let runtime = if self.bundle.ivf { "読込可" } else { "バンドルに無い" };
        card(
            IconName::Disc3,
            theme::sakura(),
            "モデル",
            cx,
            v_flex()
                .gap_2()
                .w_full()
                .child(
                    h_flex()
                        .w_full()
                        .items_center()
                        .gap_2()
                        .child(field_label("RVCモデル"))
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .child(Select::new(&self.model_select).placeholder("モデルを選択")),
                        )
                        .child(
                            Button::new("browse-model")
                                .label("開く")
                                .on_click(cx.listener(|this, _, window, cx| this.browse(window, cx, "model"))),
                        )
                        .child(
                            Button::new("rescan")
                                .label("再読込")
                                .on_click(cx.listener(|this, _, window, cx| this.rescan(window, cx))),
                        ),
                )
                .child(note(
                    format!("モデル隣の .index: {beside} / 実行 Index: {runtime}"),
                ))
                .child(note(
                    "一覧の .pth は記録です。変換に使う声は実行バンドルです。",
                )),
        )
    }

    fn section_latency(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let chunk_ms = self.s.snapped_chunk_sec() * 1000.0;
        let (lo, hi) = self.s.chunk_bounds_ms();
        card(
            IconName::Gauge,
            theme::sky(),
            "レイテンシ",
            cx,
            v_flex()
                .gap_2()
                .w_full()
                .child(slider_row(
                    "チャンク",
                    format!("{chunk_ms:.0} ms"),
                    Slider::new(&self.sliders.chunk),
                ))
                .child(
                    Checkbox::new("aggressive")
                        .label("Aggressive (低遅延)")
                        .checked(self.s.aggressive)
                        .on_change(cx.listener(|this, checked, window, cx| {
                            this.s.aggressive = *checked;
                            this.sync_chunk_slider(window, cx);
                            cx.notify();
                        })),
                )
                .child(note(format!("範囲 {lo:.0}–{hi:.0} ms（20 ms 刻み）")))
                .child(note(self.s.latency_line()))
                .child(
                    h_flex()
                        .items_center()
                        .gap_2()
                        .child(div().text_sm().text_color(theme::muted_lilac()).child("推定レイテンシ"))
                        .child(value_text(format!("~{:.0} ms", self.s.estimate_ms()))),
                ),
        )
    }

    fn section_pitch(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let presets = h_flex()
            .gap_2()
            .child(self.pitch_preset("-12", -12.0, cx))
            .child(self.pitch_preset("-5", -5.0, cx))
            .child(self.pitch_preset("0", 0.0, cx))
            .child(self.pitch_preset("+5", 5.0, cx))
            .child(self.pitch_preset("+12", 12.0, cx));
        card(
            IconName::AudioWaveform,
            theme::lavender(),
            "Pitch / F0",
            cx,
            v_flex()
                .gap_2()
                .w_full()
                .child(slider_row(
                    "ピッチ",
                    format!("{:+.1} st", self.s.pitch),
                    Slider::new(&self.sliders.pitch),
                ))
                .child(presets)
                .child(select_row("F0", &self.f0_select))
                .child(slider_row(
                    "Moe Boost",
                    format!("{:.2}", self.s.moe),
                    Slider::new(&self.sliders.moe),
                ))
                .child(
                    Checkbox::new("octave")
                        .label("オクターブ飛びを抑える")
                        .checked(self.s.octave_suppress)
                        .on_change(cx.listener(|this, checked, _, cx| {
                            this.s.octave_suppress = *checked;
                            this.push_live();
                            cx.notify();
                        })),
                )
                .child(
                    Checkbox::new("slew")
                        .label("F0 の急変を制限")
                        .checked(self.s.f0_slew_limit)
                        .on_change(cx.listener(|this, checked, _, cx| {
                            this.s.f0_slew_limit = *checked;
                            this.push_live();
                            cx.notify();
                        })),
                )
                .child(slider_row(
                    "最大ステップ",
                    format!("{:.1} st", self.s.f0_slew_max_step),
                    Slider::new(&self.sliders.slew).disabled(!self.s.f0_slew_limit),
                ))
                .child(note("ノイズスケール: 0（決定的な合成）"))
                .child(note("倍音位相: 固定")),
        )
    }

    fn pitch_preset(&mut self, label: &'static str, pitch: f32, cx: &mut Context<Self>) -> AnyElement {
        Button::new(SharedString::from(format!("pitch-{label}")))
            .label(label)
            .on_click(cx.listener(move |this, _, window, cx| {
                this.set_pitch(pitch, window, cx);
                cx.notify();
            }))
            .into_any_element()
    }

    fn section_index(&mut self, cx: &mut Context<Self>) -> AnyElement {
        card(
            IconName::Sparkles,
            theme::peach(),
            "Index 検索",
            cx,
            v_flex()
                .gap_2()
                .w_full()
                .child(
                    Checkbox::new("use-index")
                        .label("Indexを使用")
                        .checked(self.s.use_index)
                        .on_change(cx.listener(|this, checked, _, cx| {
                            this.s.use_index = *checked;
                            this.push_live();
                            cx.notify();
                        })),
                )
                .child(slider_row(
                    "Index率",
                    format!("{:.2}", self.s.index_rate),
                    Slider::new(&self.sliders.index_rate).disabled(!self.s.use_index),
                ))
                .child(slider_row(
                    "検索数 k",
                    format!("{:.0}", self.s.index_k),
                    Slider::new(&self.sliders.index_k).disabled(!self.s.use_index),
                )),
        )
    }

    fn section_denoise(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let denoise_on = self.s.denoise != "none";
        card(
            IconName::MicVocal,
            theme::mint(),
            "ノイズキャンセリング",
            cx,
            v_flex()
                .gap_2()
                .w_full()
                .child(
                    h_flex()
                        .items_center()
                        .gap_3()
                        .child(
                            Checkbox::new("denoise")
                                .label("有効化")
                                .checked(denoise_on)
                                .on_change(cx.listener(|this, checked, window, cx| {
                                    this.s.denoise = if *checked { "spectral".into() } else { "none".into() };
                                    let value: SharedString = this.s.denoise.clone().into();
                                    this.denoise_select.update(cx, |state, cx| {
                                        state.set_selected_value(&value, window, cx);
                                    });
                                    this.push_live();
                                    cx.notify();
                                })),
                        )
                        .child(div().w(px(160.)).child(Select::new(&self.denoise_select))),
                )
                .child(note("除去方式: spectral（ストリーミング）"))
                .child(slider_row(
                    "除去強度",
                    format!("{:.2}", self.s.denoise_strength),
                    Slider::new(&self.sliders.denoise_strength).disabled(!denoise_on),
                ))
                .child(
                    h_flex()
                        .items_center()
                        .gap_3()
                        .child(
                            Checkbox::new("noise-gate")
                                .label("入力ノイズゲート")
                                .checked(self.s.noise_gate)
                                .on_change(cx.listener(|this, checked, _, cx| {
                                    this.s.noise_gate = *checked;
                                    this.push_live();
                                    cx.notify();
                                })),
                        )
                        .child(div().text_sm().text_color(theme::muted_lilac()).child("感度"))
                        .child(
                            div().w(px(150.)).child(Select::new(&self.gate_select).disabled(!self.s.noise_gate)),
                        ),
                ),
        )
    }

    fn section_voice(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let energy = self.s.voice_gate_mode == "energy";
        card(
            IconName::Heart,
            theme::rose(),
            "ボイスゲート（変換後）",
            cx,
            v_flex()
                .gap_2()
                .w_full()
                .child(select_row("モード", &self.voice_select))
                .when(energy, |this| {
                    this.child(slider_row(
                        "閾値",
                        format!("{:.2}", self.s.energy_threshold),
                        Slider::new(&self.sliders.energy),
                    ))
                })
                .child(note(
                    "off は全通過、strict は F0、expand は前後を含める、energy はエネルギーも通します。",
                )),
        )
    }

    fn section_device_summary(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let input = if self.s.input_device.is_empty() {
            "(既定)".to_string()
        } else {
            self.s.input_device.clone()
        };
        let output = if self.s.output_device.is_empty() {
            "(既定)".to_string()
        } else {
            self.s.output_device.clone()
        };
        let gpu = if self.adapter.is_empty() {
            "開始後に表示".to_string()
        } else {
            self.adapter.clone()
        };
        card(
            IconName::Mic,
            theme::sky(),
            "オーディオデバイス",
            cx,
            v_flex()
                .gap_1p5()
                .w_full()
                .child(
                    h_flex()
                        .items_center()
                        .gap_2()
                        .child(div().text_xs().text_color(theme::muted_lilac()).child("入力"))
                        .child(div().text_sm().text_color(theme::porcelain().opacity(0.88)).child(input)),
                )
                .child(
                    h_flex()
                        .items_center()
                        .gap_2()
                        .child(div().text_xs().text_color(theme::muted_lilac()).child("出力"))
                        .child(div().text_sm().text_color(theme::porcelain().opacity(0.88)).child(output)),
                )
                .child(note(
                    format!(
                        "{} / {}",
                        rate_text(self.input_rate),
                        rate_text(self.output_rate)
                    ),
                ))
                .child(note(format!("推論デバイス: {gpu}"))),
        )
    }

    fn section_test(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let busy = self.test_busy.load(Ordering::Acquire);
        let session = self.session_live.load(Ordering::Acquire);
        card(
            IconName::CirclePlay,
            theme::mint(),
            "オーディオテスト",
            cx,
            v_flex()
                .gap_2()
                .w_full()
                .child(
                    Button::new("audio-test")
                        .label(if busy { "停止" } else { "テスト (3秒録音→再生)" })
                        .disabled(session)
                        .on_click(cx.listener(|this, _, _, cx| this.audio_test(cx))),
                )
                .child(
                    Checkbox::new("wav-loop")
                        .label("WAVファイルをループ入力")
                        .checked(self.s.use_wav_input)
                        .on_change(cx.listener(|this, checked, _, cx| {
                            this.s.use_wav_input = *checked;
                            cx.notify();
                        })),
                )
                .when(self.s.use_wav_input, |this| {
                    this.child(
                        h_flex()
                            .w_full()
                            .gap_2()
                            .child(div().flex_1().min_w_0().child(Input::new(&self.wav_in)))
                            .child(
                                Button::new("browse-loop")
                                    .label("参照")
                                    .on_click(cx.listener(|this, _, window, cx| this.browse(window, cx, "loop"))),
                            ),
                    )
                })
                .when(!self.test_label.is_empty(), |this| {
                    let label = self.test_label.clone();
                    this.child(banner(IconName::Info, theme::lavender(), label))
                })
                .when(self.test_label.is_empty(), |this| {
                    this.child(note("マイクを 3 秒録って、今の設定で変換し、出力へ再生します。"))
                }),
        )
    }

    fn section_post(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let enabled = self.s.postprocess;
        card(
            IconName::SlidersHorizontal,
            theme::lavender(),
            "ポスト処理",
            cx,
            v_flex()
                .gap_2()
                .w_full()
                .child(
                    Checkbox::new("post")
                        .label("有効化")
                        .checked(enabled)
                        .on_change(cx.listener(|this, checked, _, cx| {
                            this.s.postprocess = *checked;
                            this.push_live();
                            cx.notify();
                        })),
                )
                .child(slider_row(
                    "高域ブースト",
                    format!("{:+.1} dB", self.s.treble_boost_db),
                    Slider::new(&self.sliders.treble_boost).disabled(!enabled),
                ))
                .child(slider_row(
                    "カットオフ",
                    format!("{:.0} Hz", self.s.treble_cutoff_hz),
                    Slider::new(&self.sliders.treble_cutoff).disabled(!enabled),
                ))
                .child(slider_row(
                    "リミッター",
                    format!("{:.1} dB", self.s.limiter_threshold_db),
                    Slider::new(&self.sliders.limiter_threshold).disabled(!enabled),
                ))
                .child(slider_row(
                    "リリース",
                    format!("{:.0} ms", self.s.limiter_release_ms),
                    Slider::new(&self.sliders.limiter_release).disabled(!enabled),
                ))
                .child(
                    Checkbox::new("normalizer")
                        .label("音量正規化")
                        .checked(self.s.normalizer_enabled)
                        .disabled(!enabled)
                        .on_change(cx.listener(|this, checked, _, cx| {
                            this.s.normalizer_enabled = *checked;
                            this.push_live();
                            cx.notify();
                        })),
                )
                .child(slider_row(
                    "目標RMS",
                    format!("{:.2}", self.s.normalizer_target_rms),
                    Slider::new(&self.sliders.normalizer_rms).disabled(!enabled || !self.s.normalizer_enabled),
                )),
        )
    }

    fn section_convert(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let busy = self.test_busy.load(Ordering::Acquire);
        let session = self.session_live.load(Ordering::Acquire);
        card(
            IconName::WandSparkles,
            theme::peach(),
            "WAV 変換",
            cx,
            v_flex()
                .gap_2()
                .w_full()
                .child(
                    h_flex()
                        .w_full()
                        .items_center()
                        .gap_2()
                        .child(field_label("入力"))
                        .child(div().flex_1().min_w_0().child(Input::new(&self.wav)))
                        .child(
                            Button::new("browse-wav")
                                .label("参照")
                                .on_click(cx.listener(|this, _, window, cx| this.browse(window, cx, "wav"))),
                        ),
                )
                .child(
                    h_flex()
                        .gap_2()
                        .child(
                            Button::new("convert")
                                .primary()
                                .label(if busy { "作業中" } else { "変換" })
                                .disabled(busy)
                                .on_click(cx.listener(|this, _, _, cx| this.convert(cx))),
                        )
                        .child(
                            Button::new("play-wav")
                                .label("再生")
                                .disabled(busy || session || self.converted.is_empty())
                                .on_click(cx.listener(|this, _, _, cx| this.play_converted(cx))),
                        )
                        .child(
                            Button::new("stop-play")
                                .label("停止")
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.play_stop.store(true, Ordering::Release);
                                    cx.notify();
                                })),
                        )
                        .child(
                            Button::new("save-wav")
                                .label("保存")
                                .disabled(self.converted.is_empty())
                                .on_click(cx.listener(|this, _, window, cx| this.save_converted(window, cx))),
                        ),
                )
                .when(!self.test_label.is_empty(), |this| {
                    let label = self.test_label.clone();
                    this.child(banner(IconName::Info, theme::lavender(), label))
                })
                .when(self.test_label.is_empty(), |this| {
                    this.child(note("現在の Pitch / F0 / Index / Moe でオフライン変換します。"))
                }),
        )
    }

    fn footer(&self, session: bool) -> AnyElement {
        let (_, status_color, live) = self.status_note();
        let status_label = if self.running && self.stats.hops == 0 {
            "モデル読込中"
        } else if self.running {
            "Running"
        } else if session {
            "停止中"
        } else {
            "Stopped"
        };
        let recent_under = self.since(5.0, |mark| mark.underruns, self.stats.underruns);
        let recent_trim = self.since(5.0, |mark| mark.trims, self.stats.trims);
        let recent_queue = self.since(5.0, |mark| mark.queue_full, self.stats.queue_full);
        let window_under = self.since(60.0, |mark| mark.underruns, self.stats.underruns);
        let window_trim = self.since(60.0, |mark| mark.trims, self.stats.trims);
        let window_queue = self.since(60.0, |mark| mark.queue_full, self.stats.queue_full);
        let buffer_color = if recent_under > 0 {
            theme::rose()
        } else if recent_trim > 0 || recent_queue > 0 {
            theme::amber()
        } else {
            theme::muted_lilac()
        };
        let index_text = if !self.bundle.ivf {
            "Index: 未配置".to_string()
        } else if self.s.use_index {
            format!("Index: {:.2}", self.s.index_rate)
        } else {
            "Index: off".to_string()
        };
        let gpu = if self.adapter.is_empty() {
            "--".to_string()
        } else {
            self.adapter.chars().take(28).collect()
        };
        v_flex()
            .w_full()
            .px_6()
            .py_3()
            .gap_1p5()
            .border_t_1()
            .border_color(theme::white(0.08))
            .bg(linear_gradient(
                90.,
                linear_color_stop(theme::white(0.055), 0.),
                linear_color_stop(theme::white(0.015), 1.),
            ))
            .child(
                h_flex()
                    .w_full()
                    .gap_2()
                    .items_center()
                    .child(
                        h_flex()
                            .items_center()
                            .gap_1p5()
                            .child(theme::pulse_dot(status_color, "footer-dot", live))
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(status_color)
                                    .child(status_label),
                            ),
                    )
                    .child(theme::stat_chip(format!("GPU: {gpu}"), theme::lavender()))
                    .child(theme::stat_chip(
                        format!("遅延 ~{:.0} ms", self.s.estimate_ms()),
                        theme::sakura_light(),
                    ))
                    .child(theme::stat_chip(
                        format!(
                            "推論 {:.1} ms (p95 {:.1})",
                            self.stats.last_ms, self.stats.p95
                        ),
                        theme::sky(),
                    )),
            )
            .child(
                h_flex()
                    .w_full()
                    .gap_2()
                    .items_center()
                    .child(div().text_xs().text_color(theme::muted_lilac()).child(index_text))
                    .child(theme::stat_chip(
                        format!(
                            "underrun {window_under} / queue {window_queue} / trim {window_trim}"
                        ),
                        buffer_color,
                    ))
                    .child(theme::stat_chip(
                        format!(
                            "hops {}  p50 {:.1}  p99 {:.1}",
                            self.stats.hops, self.stats.p50, self.stats.p99
                        ),
                        theme::muted_lilac(),
                    )),
            )
            .into_any_element()
    }
}
