//! 変換後ボイスゲート (rcwx/pipeline/inference.py `apply_voice_gate` 相当)。
//!
//! `frame_mask` は出力が覆う区間の有声マスク (1 = 有声)。expand は前後 2 フレーム
//! を膨張、energy は短時間エネルギーとの OR。線形アップサンプルのあと 5ms で平滑する。

/// `audio` をその場でゲートする。`mode` が `off` のときは何もしない。
pub fn apply_voice_gate(
    audio: &mut [f32],
    frame_mask: &[f32],
    mode: &str,
    energy_threshold: f32,
    sample_rate: u32,
) {
    if audio.is_empty() || frame_mask.is_empty() || mode == "off" {
        return;
    }
    let mut mask = frame_mask.to_vec();
    if mode == "expand" {
        dilate(&mut mask, 2);
    } else if mode == "energy" {
        or_energy(&mut mask, audio, energy_threshold);
    }
    let mut up = upsample_linear(&mask, audio.len());
    smooth(&mut up, sample_rate, audio.len());
    for (sample, gain) in audio.iter_mut().zip(up) {
        *sample *= gain.clamp(0.0, 1.0);
    }
}

/// max-pool 相当。`radius` 2 ならカーネル 5・padding 2。
fn dilate(mask: &mut [f32], radius: usize) {
    if mask.is_empty() || radius == 0 {
        return;
    }
    let src = mask.to_vec();
    for (i, slot) in mask.iter_mut().enumerate() {
        let a = i.saturating_sub(radius);
        let b = (i + radius + 1).min(src.len());
        *slot = src[a..b].iter().copied().fold(0.0f32, f32::max);
    }
}

fn or_energy(mask: &mut [f32], audio: &[f32], threshold: f32) {
    let frames = mask.len();
    let frame_size = audio.len() / frames;
    if frame_size == 0 {
        return;
    }
    let mut energy = vec![0.0f32; frames];
    for (frame, slot) in energy.iter_mut().enumerate() {
        let start = frame * frame_size;
        let end = start + frame_size;
        let sum: f32 = audio[start..end].iter().map(|v| v * v).sum();
        *slot = sum / frame_size as f32;
    }
    let peak = energy.iter().copied().fold(0.0f32, f32::max).max(1e-8);
    for (slot, e) in mask.iter_mut().zip(energy) {
        if e / peak > threshold {
            *slot = slot.max(1.0);
        }
    }
}

/// PyTorch `interpolate(..., mode="linear", align_corners=False)`。
fn upsample_linear(mask: &[f32], out_len: usize) -> Vec<f32> {
    if out_len == 0 {
        return Vec::new();
    }
    if mask.len() == 1 {
        return vec![mask[0]; out_len];
    }
    let in_len = mask.len();
    let scale = in_len as f32 / out_len as f32;
    let last = in_len - 1;
    (0..out_len)
        .map(|i| {
            let src = scale * (i as f32 + 0.5) - 0.5;
            if src <= 0.0 {
                mask[0]
            } else if src >= last as f32 {
                mask[last]
            } else {
                let i0 = src.floor() as usize;
                let i1 = (i0 + 1).min(last);
                let t = src - i0 as f32;
                mask[i0] + (mask[i1] - mask[i0]) * t
            }
        })
        .collect()
}

/// 一様カーネルの conv1d (zero pad)。偶数長は右側を切り落として入力長に戻す。
fn smooth(mask: &mut Vec<f32>, sample_rate: u32, out_len: usize) {
    let win = (sample_rate as f32 * 0.005) as usize;
    if win <= 1 || mask.is_empty() {
        mask.truncate(out_len);
        return;
    }
    let src = mask.clone();
    let pad = win / 2;
    let mut out = vec![0.0f32; src.len() + 2 * pad + 1 - win];
    let inv = 1.0 / win as f32;
    for (i, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for k in 0..win {
            let idx = i as isize + k as isize - pad as isize;
            if idx >= 0 && (idx as usize) < src.len() {
                acc += src[idx as usize];
            }
        }
        *slot = acc * inv;
    }
    out.truncate(out_len);
    if out.len() < out_len {
        out.resize(out_len, 0.0);
    }
    *mask = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_dilates_two_frames() {
        let mut mask = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        dilate(&mut mask, 2);
        assert_eq!(mask, vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn strict_keeps_silence_and_passes_voiced() {
        // 16 kHz の 5ms 平滑は 80 サンプル。無声側と有声側の中央が残る長さにする。
        let mut audio = vec![1.0; 1600];
        apply_voice_gate(&mut audio, &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], "strict", 0.2, 16_000);
        assert!(audio[..400].iter().all(|v| *v < 0.05), "{:?}", &audio[..8]);
        assert!(audio[1000..1400].iter().all(|v| *v > 0.9), "{:?}", &audio[1000..1008]);
    }

    #[test]
    fn energy_keeps_a_loud_unvoiced_frame() {
        let mut audio = vec![0.0; 800];
        audio[..400].fill(0.8);
        apply_voice_gate(&mut audio, &[0.0, 0.0], "energy", 0.2, 16_000);
        assert!(audio[200] > 0.7, "{}", audio[200]);
        assert!(audio[700] < 0.05, "{}", audio[700]);
    }

    #[test]
    fn off_leaves_audio() {
        let mut audio = vec![0.25, -0.5];
        apply_voice_gate(&mut audio, &[0.0], "off", 0.2, 48_000);
        assert_eq!(audio, vec![0.25, -0.5]);
    }

    #[test]
    fn smooth_preserves_length() {
        // 48 kHz の平滑窓は 240 サンプル。ゼロパッドで端は下がるので中央を見る。
        let mut audio = vec![1.0; 2000];
        apply_voice_gate(&mut audio, &[1.0, 1.0, 1.0, 1.0], "strict", 0.2, 48_000);
        assert_eq!(audio.len(), 2000);
        assert!(audio[400..1600].iter().all(|v| *v > 0.95), "{}", audio[1000]);
    }
}
