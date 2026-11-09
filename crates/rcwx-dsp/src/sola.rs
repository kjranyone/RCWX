//! SOLA (Synchronized Overlap-Add) クロスフェード (rcwx/audio/sola.py 相当)。
//!
//! ホールドバック設計: 各チャンク出力の直後 `crossfade` サンプルを留保し、
//! 次チャンクの先頭と相互相関で最適位置を探して Hann クロスフェードする。
//! `target_len > 0` で出力長を厳密に強制し、累積レイテンシドリフトを防ぐ。

/// SOLA 状態。ホールドバック + Hann 曲線は同じ長さで一体として扱う。
#[derive(Default, Clone)]
pub struct SolaState {
    buffer: Option<Vec<f32>>,
    crossfade_samples: usize,
    search_samples: usize,
    hann_fade_in: Option<Vec<f32>>,
    hann_fade_out: Option<Vec<f32>>,
    pub last_offset: usize,
    pub last_in_len: usize,
    pub last_out_len: usize,
}

impl SolaState {
    pub fn new(crossfade_samples: usize, search_samples: usize) -> Self {
        Self {
            crossfade_samples,
            search_samples,
            ..Default::default()
        }
    }

    /// 窓サイズ変更。crossfade 変更時はホールドバックを落として链を再始動。
    pub fn resized(&self, crossfade_samples: usize, search_samples: usize) -> Self {
        let mut s = Self::new(crossfade_samples, search_samples);
        if crossfade_samples == self.crossfade_samples {
            s.buffer = self.buffer.clone();
            s.hann_fade_in = self.hann_fade_in.clone();
            s.hann_fade_out = self.hann_fade_out.clone();
        }
        s
    }

    fn ensure_window(&mut self) {
        if self.hann_fade_in.is_none() && self.crossfade_samples > 0 {
            let cf = self.crossfade_samples;
            // np.hanning(2*cf): 対称 Hann 0.5 - 0.5*cos(2πn/(2cf-1))
            let m = (2 * cf - 1) as f64;
            let mut fade_in = Vec::with_capacity(cf);
            let mut fade_out = Vec::with_capacity(cf);
            for n in 0..cf {
                let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / m).cos();
                fade_in.push(w as f32);
            }
            for n in cf..2 * cf {
                let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / m).cos();
                fade_out.push(w as f32);
            }
            self.hann_fade_in = Some(fade_in);
            self.hann_fade_out = Some(fade_out);
        }
    }
}

/// 1 チャンクに SOLA クロスフェードを適用。
pub fn sola_crossfade(audio: &[f32], state: &mut SolaState, target_len: usize) -> Vec<f32> {
    let cf = state.crossfade_samples;
    let search = state.search_samples;
    state.last_in_len = audio.len();

    if cf == 0 {
        state.last_out_len = audio.len();
        return audio.to_vec();
    }
    state.ensure_window();

    // 先頭チャンク: 出力 + ホールドバック
    if state.buffer.is_none() {
        state.last_offset = 0;
        let out = if target_len > 0 && audio.len() >= target_len + cf {
            state.buffer = Some(audio[target_len..target_len + cf].to_vec());
            audio[..target_len].to_vec()
        } else if audio.len() > cf {
            state.buffer = Some(audio[audio.len() - cf..].to_vec());
            audio[..audio.len() - cf].to_vec()
        } else {
            // 留保するほど長くない → 全部ホールド
            state.buffer = Some(audio.to_vec());
            Vec::new()
        };
        state.last_out_len = out.len();
        return out;
    }

    // 以降: cf 未満のチャンクは buffer 連結で終了
    if audio.len() < cf {
        let mut result = state.buffer.take().unwrap_or_default();
        result.extend_from_slice(audio);
        state.last_offset = 0;
        state.last_out_len = result.len();
        return result;
    }

    let prev_tail = state.buffer.as_deref().unwrap();
    let search_end = (cf + search).min(audio.len());
    let search_region = &audio[..search_end];

    let offset = find_best_offset(prev_tail, search_region, cf, search);
    state.last_offset = offset;

    let fade_in = state.hann_fade_in.as_deref().unwrap();
    let fade_out = state.hann_fade_out.as_deref().unwrap();
    let mut crossfaded = vec![0.0f32; cf];
    for i in 0..cf {
        let curr = audio.get(offset + i).copied().unwrap_or(0.0);
        crossfaded[i] = prev_tail[i] * fade_out[i] + curr * fade_in[i];
    }

    let remaining = &audio[(offset + cf).min(audio.len())..];

    // 出力長: cf (クロスフェード) + take
    if target_len > 0 {
        let take = target_len as isize - cf as isize;
        if take >= 0 && (take as usize) + cf <= remaining.len() {
            let take = take as usize;
            state.buffer = Some(remaining[take..take + cf].to_vec());
            let mut result = crossfaded;
            result.extend_from_slice(&remaining[..take]);
            state.last_out_len = result.len();
            return result;
        }
    }

    let mut result = crossfaded;
    if remaining.len() >= cf {
        state.buffer = Some(remaining[remaining.len() - cf..].to_vec());
        result.extend_from_slice(&remaining[..remaining.len() - cf]);
    } else {
        state.buffer = Some(audio[audio.len() - cf..].to_vec());
        result.extend_from_slice(remaining);
    }
    state.last_out_len = result.len();
    result
}

/// ストリーム終端で留保中のサンプルを返す。
pub fn sola_flush(state: &mut SolaState) -> Vec<f32> {
    state.buffer.take().unwrap_or_default()
}

/// 正規化相互相関による最適スプライス位置 (最小シフト選択・tol=0.02)。
fn find_best_offset(prev_tail: &[f32], search_region: &[f32], cf: usize, search: usize) -> usize {
    if search == 0 || search_region.len() <= cf {
        return 0;
    }
    let max_offset = search.min(search_region.len() - cf);
    if max_offset == 0 {
        return 0;
    }

    // prev_tail の中心化とノルム (f64)
    let pt_mean = prev_tail.iter().map(|v| *v as f64).sum::<f64>() / cf as f64;
    let pt_centered: Vec<f64> = prev_tail.iter().map(|v| *v as f64 - pt_mean).collect();
    let pt_norm = pt_centered.iter().map(|v| v * v).sum::<f64>().sqrt();
    if pt_norm < 1e-8 {
        return 0;
    }

    let region = &search_region[..cf + max_offset];
    let rl = region.len();

    // np.correlate(region, pt_centered, "valid"): 長さ max_offset+1
    let mut dots = vec![0.0f64; max_offset + 1];
    for (shift, slot) in dots.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for i in 0..cf {
            acc += region[shift + i] as f64 * pt_centered[i];
        }
        *slot = acc;
    }

    // 窓ノルム (累積和)
    let mut cumsum = vec![0.0f64; rl + 1];
    let mut cumsum_sq = vec![0.0f64; rl + 1];
    for i in 0..rl {
        let v = region[i] as f64;
        cumsum[i + 1] = cumsum[i] + v;
        cumsum_sq[i + 1] = cumsum_sq[i] + v * v;
    }

    let mut best = f64::NEG_INFINITY;
    let mut best_offset = 0usize;
    let tol = 0.02f64;
    let mut peak = f64::NEG_INFINITY;
    let mut corrs = vec![f64::NEG_INFINITY; max_offset + 1];
    for s in 0..=max_offset {
        let ws = cumsum[s + cf] - cumsum[s];
        let wss = cumsum_sq[s + cf] - cumsum_sq[s];
        let norms_sq = (wss - ws * ws / cf as f64).max(0.0);
        let norm = norms_sq.sqrt();
        if norm > 1e-8 {
            let c = dots[s] / (pt_norm * norm);
            corrs[s] = c;
            if c > peak {
                peak = c;
            }
        }
    }
    // ピークとの差が tol 以内の最小オフセット
    if peak != f64::NEG_INFINITY {
        for (s, &c) in corrs.iter().enumerate() {
            if c >= peak - tol {
                best = c;
                best_offset = s;
                break;
            }
        }
    }
    let _ = best;
    best_offset
}
