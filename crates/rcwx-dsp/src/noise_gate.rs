//! 入力側ノイズゲート (rcwx/audio/noise_gate.py 相当)。guitar NS 風ダウンワードエクスパンダー。
//!
//! RMS エンベロープ (τ=20ms) + 3dB ヒステリシス + 60ms hold。減衰下限 -40dB。
//! 自動しきい値: 1s 窓最小フロア + 感度マージン、-30dBFS 超で無条件開放。
//! Python 側はスカラー演算が f64 のため状態は f64 で保持。

const ENV_TAU_MS: f64 = 20.0;
const GAIN_ATTACK_MS: f64 = 2.0;
const GAIN_RELEASE_MS: f64 = 80.0;
const HOLD_MS: f64 = 60.0;
const HYSTERESIS_DB: f64 = 3.0;
const RANGE_DB: f64 = 40.0;

const FLOOR_WINDOW_SEC: f64 = 1.0;
const FLOOR_BLOCK_MS: f64 = 50.0;
const FLOOR_WARMUP_BLOCKS: usize = 2;
const FLOOR_SEED: f64 = 1e-6;
const OPEN_GUARD_POWER: f64 = 1e-3; // 10^(-30/10)

pub const SENSITIVITY_MARGIN_LOW_DB: f64 = 10.0;
pub const SENSITIVITY_MARGIN_MID_DB: f64 = 6.0;
pub const SENSITIVITY_MARGIN_HIGH_DB: f64 = 3.0;

fn one_pole_coeff(time_ms: f64, sample_rate: u32) -> f64 {
    if time_ms <= 0.0 {
        return 0.0;
    }
    (-1.0 / (time_ms / 1000.0 * sample_rate as f64)).exp()
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Sensitivity {
    Low,
    Mid,
    High,
}

impl Sensitivity {
    fn margin_db(self) -> f64 {
        match self {
            Sensitivity::Low => SENSITIVITY_MARGIN_LOW_DB,
            Sensitivity::Mid => SENSITIVITY_MARGIN_MID_DB,
            Sensitivity::High => SENSITIVITY_MARGIN_HIGH_DB,
        }
    }
}

pub struct NoiseGate {
    env_coeff: f64,
    gain_attack: f64,
    gain_release: f64,
    hold_samples: usize,
    floor_gain: f64,
    block_len: usize,
    window_blocks: usize,
    hyst_close: f64,
    pub auto_threshold: bool,
    pub margin_db: f64,
    threshold_db: f64,
    open_level: f64,
    close_level: f64,
    // 状態
    env_power: f64,
    open: bool,
    hold_count: usize,
    gain: f64,
    floor_hist: std::collections::VecDeque<f64>,
    hist_min: f64,
    block_min: f64,
    block_pos: usize,
    blocks_seen: usize,
}

impl NoiseGate {
    pub fn new(sample_rate: u32, threshold_db: f64) -> Self {
        let window_blocks = ((FLOOR_WINDOW_SEC * 1000.0 / FLOOR_BLOCK_MS).round() as usize).max(1);
        let mut g = Self {
            env_coeff: one_pole_coeff(ENV_TAU_MS, sample_rate),
            gain_attack: one_pole_coeff(GAIN_ATTACK_MS, sample_rate),
            gain_release: one_pole_coeff(GAIN_RELEASE_MS, sample_rate),
            hold_samples: ((sample_rate as f64 * HOLD_MS / 1000.0) as usize).max(1),
            floor_gain: 10f64.powf(-RANGE_DB / 20.0),
            block_len: ((sample_rate as f64 * FLOOR_BLOCK_MS / 1000.0) as usize).max(1),
            window_blocks,
            hyst_close: 10f64.powf(-HYSTERESIS_DB / 10.0),
            auto_threshold: false,
            margin_db: SENSITIVITY_MARGIN_MID_DB,
            threshold_db: -40.0,
            open_level: 0.0,
            close_level: 0.0,
            env_power: 0.0,
            open: false,
            hold_count: 0,
            gain: 0.0,
            floor_hist: std::collections::VecDeque::new(),
            hist_min: f64::INFINITY,
            block_min: f64::INFINITY,
            block_pos: 0,
            blocks_seen: 0,
        };
        g.set_threshold_db(threshold_db);
        g.reset();
        g
    }

    pub fn set_threshold_db(&mut self, value: f64) {
        self.threshold_db = value;
        self.open_level = 10f64.powf(value / 10.0);
        self.close_level = 10f64.powf((value - HYSTERESIS_DB) / 10.0);
    }

    pub fn threshold_db(&self) -> f64 {
        self.threshold_db
    }

    pub fn set_sensitivity(&mut self, s: Sensitivity) {
        self.margin_db = s.margin_db();
    }

    pub fn reset(&mut self) {
        self.env_power = 0.0;
        self.open = false;
        self.hold_count = 0;
        self.gain = self.floor_gain;
        self.floor_hist.clear();
        self.hist_min = f64::INFINITY;
        self.block_min = f64::INFINITY;
        self.block_pos = 0;
        self.blocks_seen = 0;
    }

    /// `audio` (モノラル f32) をゲート。同長を返す。
    pub fn process(&mut self, audio: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; audio.len()];
        let auto = self.auto_threshold;
        let margin = 10f64.powf(self.margin_db / 10.0);
        let mut env = self.env_power;
        let mut is_open = self.open;
        let mut hold = self.hold_count;
        let mut gain = self.gain;

        for (o, &x) in out.iter_mut().zip(audio) {
            let p = (x as f64) * (x as f64);
            env = self.env_coeff * env + (1.0 - self.env_coeff) * p;

            let (open_lvl, close_lvl);
            if auto {
                if env < self.block_min {
                    self.block_min = env;
                }
                self.block_pos += 1;
                if self.block_pos >= self.block_len {
                    if self.blocks_seen >= FLOOR_WARMUP_BLOCKS {
                        self.floor_hist.push_back(self.block_min);
                        if self.floor_hist.len() > self.window_blocks {
                            self.floor_hist.pop_front();
                        }
                        self.hist_min = self
                            .floor_hist
                            .iter()
                            .cloned()
                            .fold(f64::INFINITY, f64::min);
                    }
                    self.blocks_seen += 1;
                    self.block_min = f64::INFINITY;
                    self.block_pos = 0;
                }
                let noise_floor = self.hist_min.min(self.block_min).max(FLOOR_SEED);
                open_lvl = noise_floor * margin;
                close_lvl = open_lvl * self.hyst_close;
            } else {
                open_lvl = self.open_level;
                close_lvl = self.close_level;
            }

            if auto && env >= OPEN_GUARD_POWER {
                is_open = true;
                hold = 0;
            } else if !is_open && env >= open_lvl {
                is_open = true;
                hold = 0;
            } else if is_open {
                if env < close_lvl {
                    hold += 1;
                    if hold >= self.hold_samples {
                        is_open = false;
                        hold = 0;
                    }
                } else {
                    hold = 0;
                }
            }

            gain = if is_open {
                self.gain_attack * gain + (1.0 - self.gain_attack) * 1.0
            } else {
                self.gain_release * gain + (1.0 - self.gain_release) * self.floor_gain
            };
            *o = (x as f64 * gain) as f32;
        }

        self.env_power = env;
        self.open = is_open;
        self.hold_count = hold;
        self.gain = gain;
        out
    }
}
