//! フロアベース・スキップ専用ドリフト制御 (rcwx/pipeline/drift_control.py 相当)。

use std::collections::VecDeque;

/// ポスト読み出しフロア上限・トリム後目標・ガード。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShedPolicy {
    pub threshold: i64,
    pub target: i64,
    pub guard_ms: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn compute_shed_policy(
    deadline_mode: bool,
    hop_samples_out: i64,
    output_sample_rate: u32,
    chunk_sec: f64,
    buffer_margin: f64,
    inference_sample_count: usize,
    inference_p50_ms: f64,
    inference_p99_ms: f64,
    requested_callback_sec: f64,
    observed_callback_sec: f64,
) -> ShedPolicy {
    if deadline_mode {
        if inference_sample_count < 20 {
            return ShedPolicy {
                threshold: hop_samples_out * 5 / 4,
                target: hop_samples_out,
                guard_ms: chunk_sec * 1000.0,
            };
        }
        let hop_ms = chunk_sec * 1000.0;
        let jitter_ms = (inference_p99_ms - inference_p50_ms).max(0.0);
        let callback_ms = requested_callback_sec.max(observed_callback_sec) * 1000.0;
        let mut guard_ms = (hop_ms * 0.5).max(jitter_ms + callback_ms);
        guard_ms = guard_ms.min(hop_ms * 0.875);
        let target = ((guard_ms * output_sample_rate as f64 / 1000.0).round() as i64).max(1);
        let band = (hop_samples_out / 4).max((callback_ms * output_sample_rate as f64 / 1000.0).round() as i64);
        let threshold = (hop_samples_out * 5 / 4).min(target + band);
        return ShedPolicy {
            threshold: threshold.max(1),
            target: target.max(0),
            guard_ms: target as f64 * 1000.0 / output_sample_rate as f64,
        };
    }

    let margin = buffer_margin.clamp(0.1, 2.0);
    let threshold = ((0.5 + margin) * hop_samples_out as f64) as i64;
    let target = hop_samples_out / 4;
    ShedPolicy {
        threshold: threshold.max(1),
        target: target.max(0),
        guard_ms: target as f64 * 1000.0 / output_sample_rate as f64,
    }
}

/// ポスト読み出しリングレベルのスライディング窓最小値 (~2 チャンク分)。
pub struct FloorTracker {
    window: VecDeque<i64>,
    window_frames: usize,
    maxlen: usize,
    pub floor_samples: i64,
}

impl Default for FloorTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl FloorTracker {
    pub fn new() -> Self {
        Self {
            window: VecDeque::new(),
            window_frames: 0,
            maxlen: 0,
            floor_samples: 0,
        }
    }

    /// デバイスブロックサイズ変更時に窓を再作成。
    pub fn ensure_window(&mut self, frames: usize, hop_samples_out: i64) {
        if self.window_frames != frames || self.maxlen == 0 {
            self.window_frames = frames;
            // ceil(2*hop/frames)、最小 4
            let window_len = ((2 * hop_samples_out + frames.max(1) as i64 - 1) / frames.max(1) as i64).max(4) as usize;
            self.window.clear();
            self.maxlen = window_len;
            self.floor_samples = 0;
        }
    }

    pub fn is_full(&self) -> bool {
        self.maxlen > 0 && self.window.len() == self.maxlen
    }

    pub fn floor(&self) -> i64 {
        self.window.iter().cloned().min().unwrap_or(0)
    }

    pub fn record_post_read(&mut self, level: i64) {
        if self.maxlen == 0 {
            return;
        }
        self.window.push_back(level);
        if self.window.len() > self.maxlen {
            self.window.pop_front();
        }
        if self.window.len() == self.maxlen {
            self.floor_samples = self.window.iter().cloned().min().unwrap_or(0);
        }
    }

    pub fn after_skip(&mut self, target: i64) {
        self.window.clear();
        self.floor_samples = target;
    }

    pub fn clear(&mut self) {
        self.window.clear();
        self.floor_samples = 0;
    }

    pub fn reset(&mut self) {
        self.window.clear();
        self.window_frames = 0;
        self.maxlen = 0;
        self.floor_samples = 0;
    }

    pub fn prime(&mut self, frames: usize, hop_samples_out: i64, floor: i64) {
        self.ensure_window(frames, hop_samples_out);
        self.window.clear();
        for _ in 0..self.maxlen {
            self.window.push_back(floor);
        }
        self.floor_samples = floor;
    }
}
