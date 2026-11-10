//! 音量エンベロープ混合 (upstream RVC realtime_gui の rms_mix_rate 相当)。
//!
//! 出力のラウドネス包絡を入力側へ追従させる:
//! `gain = (rms_in / max(rms_out, FLOOR)) ^ (1 - rms_mix_rate)`。
//! `rms_mix_rate = 1` で無効 (gain = 1)、`0` で完全追従 (upstream と同一意味論)。
//! 入力静音時は出力の残留ノイズを抑える方向へ働く。
//!
//! リアルタイム側は入力 (16k) と出力 (model_sr) が異なるレートで到着するため、
//! それぞれのチャンク長から継続時間を計算して EMA (τ=50ms) で包絡を追跡する。
//! オフライン用に移動窓 RMS による全長版 `apply_envelope_mix_full` も持つ。

/// 包絡追跡の EMA 時定数 (秒)。
const ENV_TAU_SEC: f64 = 0.05;
/// 出力 RMS の下限 (upstream realtime_gui と同じ 1e-3)。
const RMS_FLOOR: f64 = 1e-3;
const MAX_GAIN_DB: f64 = 12.0;
const MIN_GAIN_DB: f64 = -12.0;
/// ゲイン変化の cosine ランプ長。
const RAMP_SEC: f64 = 0.005;

#[derive(Clone)]
pub struct VolumeEnvelopeMixer {
    sample_rate: u32,
    /// upstream の rms_mix_rate (1 = 無効)。
    rate: f64,
    in_rms: f64,
    out_rms: f64,
    prev_gain: f64,
    ramp_samples: usize,
}

impl VolumeEnvelopeMixer {
    pub fn new(sample_rate: u32, rate: f64) -> Self {
        Self {
            sample_rate,
            rate: rate.clamp(0.0, 1.0),
            in_rms: 0.0,
            out_rms: 0.0,
            prev_gain: 1.0,
            ramp_samples: ((sample_rate as f64 * RAMP_SEC) as usize).max(1),
        }
    }

    pub fn reset(&mut self) {
        self.in_rms = 0.0;
        self.out_rms = 0.0;
        self.prev_gain = 1.0;
    }

    pub fn set_rate(&mut self, rate: f64) {
        self.rate = rate.clamp(0.0, 1.0);
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// 入力側チャンク (16k 等、`sample_rate` と異なってよい) で包絡を更新。
    pub fn note_input(&mut self, audio: &[f32], sample_rate: u32) {
        if audio.is_empty() {
            return;
        }
        let rms = (audio.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()
            / audio.len() as f64)
            .sqrt();
        let chunk_sec = audio.len() as f64 / sample_rate as f64;
        self.in_rms = ema_update(self.in_rms, rms, chunk_sec);
    }

    /// 出力にゲインを適用し、出力包絡を更新。`rate >= 1` なら無効。
    pub fn process(&mut self, audio: &mut [f32]) {
        if audio.is_empty() {
            return;
        }
        let rms = (audio.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()
            / audio.len() as f64)
            .sqrt();
        let chunk_sec = audio.len() as f64 / self.sample_rate as f64;
        // ゲイン計算には更新前の包絡を使い (回路を避ける)、適用後に更新する。
        let gain = self.target_gain(self.in_rms, rms);

        let ramp_len = self.ramp_samples.min(audio.len());
        if (gain - self.prev_gain).abs() > 1e-6 && ramp_len > 1 {
            for (i, o) in audio.iter_mut().enumerate() {
                let x = *o as f64;
                if i < ramp_len {
                    let t = i as f64 / ramp_len as f64;
                    let ramp = self.prev_gain
                        + (gain - self.prev_gain) * 0.5 * (1.0 - (std::f64::consts::PI * t).cos());
                    *o = (x * ramp) as f32;
                } else {
                    *o = (x * gain) as f32;
                }
            }
        } else {
            for o in audio.iter_mut() {
                *o = (*o as f64 * gain) as f32;
            }
        }
        self.prev_gain = gain;
        self.out_rms = ema_update(self.out_rms, rms, chunk_sec);
    }

    fn target_gain(&self, in_rms: f64, out_rms: f64) -> f64 {
        if self.rate >= 1.0 {
            return 1.0;
        }
        let exp = 1.0 - self.rate;
        let gain = (in_rms / out_rms.max(RMS_FLOOR)).powf(exp);
        gain.clamp(
            10f64.powf(MIN_GAIN_DB / 20.0),
            10f64.powf(MAX_GAIN_DB / 20.0),
        )
    }
}

fn ema_update(prev: f64, value: f64, chunk_sec: f64) -> f64 {
    if prev <= 0.0 {
        return value;
    }
    let alpha = 1.0 - (-chunk_sec / ENV_TAU_SEC).exp();
    prev + alpha * (value - prev)
}

fn moving_rms(x: &[f32], frame: usize, hop: usize) -> Vec<f64> {
    // Python `librosa.feature.rms` 相当 (center=True: 前後を zero pad)。
    let n = x.len();
    let frames = if n == 0 { 0 } else { x.len().div_ceil(hop) };
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        let center = f * hop;
        let end = (center + frame + 1).min(n); // 対称にするため frame 分の幅を確保
        let start = end.saturating_sub(frame);
        let mut sum = 0.0f64;
        let mut cnt = 0usize;
        for &v in &x[start..end] {
            sum += (v as f64) * (v as f64);
            cnt += 1;
        }
        out.push(if cnt > 0 {
            (sum / cnt as f64).sqrt()
        } else {
            0.0
        });
    }
    out
}

fn interp_to(envelope: &[f64], len: usize) -> Vec<f64> {
    // PyTorch interpolate(linear, align_corners=False) 相当。
    if envelope.is_empty() {
        return vec![0.0; len];
    }
    if envelope.len() == 1 {
        return vec![envelope[0]; len];
    }
    let scale = (envelope.len() - 1) as f64 / len as f64;
    (0..len)
        .map(|i| {
            let src = i as f64 * scale;
            let i0 = src.floor() as usize;
            let i1 = (i0 + 1).min(envelope.len() - 1);
            let t = src - i0 as f64;
            envelope[i0] * (1.0 - t) + envelope[i1] * t
        })
        .collect()
}

/// オフライン版: 入力 (in_sr) と出力 (out_sr) それぞれの移動窓 RMS 包絡から
/// ゲイン曲線を作り `output` に適用する (upstream `change_rms` 相当)。
pub fn apply_envelope_mix_full(
    input: &[f32],
    in_sr: u32,
    output: &mut [f32],
    out_sr: u32,
    rate: f64,
) {
    let rate = rate.clamp(0.0, 1.0);
    if rate >= 1.0 || input.is_empty() || output.is_empty() {
        return;
    }
    // 窓は両レートで ~20ms / hop ~10ms (upstream は frame=4*zc, hop=zc)。
    let in_frame = (in_sr as usize / 50).max(1);
    let out_frame = (out_sr as usize / 50).max(1);
    let in_hop = (in_frame / 2).max(1);
    let out_hop = (out_frame / 2).max(1);
    let rms_in = interp_to(&moving_rms(input, in_frame, in_hop), output.len());
    let rms_out = interp_to(&moving_rms(output, out_frame, out_hop), output.len());
    let exp = 1.0 - rate;
    let max_gain = 10f64.powf(MAX_GAIN_DB / 20.0);
    let min_gain = 10f64.powf(MIN_GAIN_DB / 20.0);
    for (o, (ri, ro)) in output.iter_mut().zip(rms_in.iter().zip(rms_out.iter())) {
        let gain = (ri / ro.max(RMS_FLOOR)).powf(exp).clamp(min_gain, max_gain);
        *o = (*o as f64 * gain) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48000;

    fn sine(freq: f64, amp: f64, len: usize, sr: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                (amp * (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin()) as f32
            })
            .collect()
    }

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64).sqrt()
    }

    #[test]
    fn rate_one_is_identity() {
        let mut m = VolumeEnvelopeMixer::new(SR, 1.0);
        let inp = sine(220.0, 0.5, 4800, SR);
        let mut out = sine(220.0, 0.05, 4800, SR);
        let before = out.clone();
        m.note_input(&inp, SR);
        m.process(&mut out);
        assert_eq!(out, before);
    }

    #[test]
    fn follows_input_loudness() {
        let mut m = VolumeEnvelopeMixer::new(SR, 0.0);
        let inp = sine(220.0, 0.4, SR as usize / 10, SR); // 100ms
        let mut out = sine(220.0, 0.1, SR as usize / 10, SR);
        // 2 回回して包絡を収束させる。
        m.note_input(&inp, SR);
        m.process(&mut out);
        m.note_input(&inp, SR);
        m.process(&mut out);
        let target = rms(&inp);
        let got = rms(&out);
        assert!(
            (got - target).abs() / target < 0.2,
            "got {got}, target {target}"
        );
    }

    #[test]
    fn gain_clamped() {
        let mut m = VolumeEnvelopeMixer::new(SR, 0.0);
        let loud = sine(220.0, 1.0, SR as usize / 10, SR);
        let mut tiny = sine(220.0, 0.001, SR as usize / 10, SR);
        let before = rms(&tiny);
        m.note_input(&loud, SR);
        m.process(&mut tiny);
        // 最大 +12dB (ランプを考慮して緩めに)。
        assert!(
            rms(&tiny) <= before * 4.2,
            "before {before}, after {}",
            rms(&tiny)
        );
    }

    #[test]
    fn silent_input_ducks_output() {
        let mut m = VolumeEnvelopeMixer::new(SR, 0.0);
        let silent = vec![0.0f32; SR as usize / 10];
        let mut out = sine(220.0, 0.2, SR as usize / 10, SR);
        let before = rms(&out);
        m.note_input(&silent, SR);
        m.process(&mut out);
        assert!(
            rms(&out) < before * 0.4,
            "before {before}, after {}",
            rms(&out)
        );
    }

    #[test]
    fn no_nan_with_zeros() {
        let mut m = VolumeEnvelopeMixer::new(SR, 0.5);
        let mut out = vec![0.0f32; 1000];
        m.note_input(&[], SR);
        m.process(&mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        m.process(&mut out);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn full_version_follows() {
        let inp = sine(220.0, 0.4, SR as usize, SR);
        let mut out = sine(220.0, 0.1, SR as usize, SR);
        apply_envelope_mix_full(&inp, SR, &mut out, SR, 0.0);
        let target = rms(&inp);
        let got = rms(&out);
        assert!(
            (got - target).abs() / target < 0.15,
            "got {got}, target {target}"
        );
    }

    #[test]
    fn full_version_rate_one_identity() {
        let inp = sine(220.0, 0.4, SR as usize, SR);
        let mut out = sine(220.0, 0.1, SR as usize, SR);
        let before = out.clone();
        apply_envelope_mix_full(&inp, SR, &mut out, SR, 1.0);
        assert_eq!(out, before);
    }
}
