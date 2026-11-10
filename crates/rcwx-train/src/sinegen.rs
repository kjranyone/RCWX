//! 学習用 SineGen (legacy アルゴリズム・CPU・確率的・no_grad 相当)。
//! 参照: rcwx/models/infer_pack/models.py SineGen::_f02sine_legacy + forward。
//! 調波数 1 (harmonic_num=0)。勾配が流れるのは l_linear のスカラーのみで、
//! 正弦波・ノイズは定数 (tape には const として与える)。

/// 乱数供給 (黄金テストでは固定列を差し込める)。
pub trait SineRng {
    fn next_f32(&mut self) -> f32;
}

/// xorshift64* による简易一様乱数。
#[derive(Clone)]
pub struct LcgRng(pub u64);

impl SineRng for LcgRng {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545F4914F6CDD1D);
        ((v >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

pub struct SineGenCfg {
    pub samp_rate: u32,
    pub sine_amp: f32,
    pub noise_std: f32,
}

impl Default for SineGenCfg {
    fn default() -> Self {
        Self {
            samp_rate: 48000,
            sine_amp: 0.1,
            noise_std: 0.003,
        }
    }
}

/// f0 [batch, t_frames] (Hz) → 励振 [batch, t_frames*upp]。
///
/// legacy アルゴリズム (harmonic_num=0, rand_ini=0):
///   rad[t] = (f0/sr) % 1
///   cum = cumsum(rad)
///   tmp[j] = (lin_interp_align_corners(cum, out=T·upp)[j] · upp) % 1
///   rad_nearest[j] = rad[floor(j/upp)]
///   shift[j] = (j>0 && tmp[j] < tmp[j-1]) ? -1 : 0
///   phase[j] = cumsum(rad_nearest + shift)[j]
///   sine[j] = sin(2π·phase[j])·sine_amp
///   uv[j] = f0[floor(j/upp)] > 0
///   noise[j] = N(0,1)·(uv ? noise_std : sine_amp/3)
///   out[j] = uv ? sine[j] + noise[j] : noise[j]
pub fn sinegen_train(
    cfg: &SineGenCfg,
    f0: &[f32],
    batch: usize,
    t_frames: usize,
    upp: usize,
    rng: &mut dyn SineRng,
) -> Vec<f32> {
    let n_out = t_frames * upp;
    let mut out = vec![0.0f32; batch * n_out];
    // Box-Muller (一様 2 個 → 正規 1 個)
    let mut normal = |rng: &mut dyn SineRng| -> f32 {
        let u1 = rng.next_f32().max(1e-7);
        let u2 = rng.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    };
    for b in 0..batch {
        let rad: Vec<f32> = (0..t_frames)
            .map(|t| (f0[b * t_frames + t] / cfg.samp_rate as f32) % 1.0)
            .collect();
        let mut cum = vec![0.0f32; t_frames];
        let mut acc = 0.0f32;
        for t in 0..t_frames {
            acc += rad[t];
            cum[t] = acc;
        }
        // tmp_over_one: align_corners=True 線形補間後 % 1
        let mut tmp = vec![0.0f32; n_out];
        for j in 0..n_out {
            let in_pos = if n_out == 1 {
                0.0f32
            } else {
                j as f32 * (t_frames as f32 - 1.0) / (n_out as f32 - 1.0)
            };
            let t0 = in_pos.floor();
            let frac = in_pos - t0;
            let c0 = cum[t0 as usize];
            let c1 = cum[(t0 as usize + 1).min(t_frames - 1)];
            tmp[j] = ((c0 + (c1 - c0) * frac) * upp as f32) % 1.0;
        }
        // phase = cumsum(nearest(rad) + shift)
        let mut acc2 = 0.0f32;
        for j in 0..n_out {
            let t = (j / upp).min(t_frames - 1);
            let s = if j > 0 && tmp[j] < tmp[j - 1] {
                -1.0
            } else {
                0.0
            };
            acc2 += rad[t] + s;
            let uv = f0[b * t_frames + t] > 0.0;
            let sine = (2.0 * std::f32::consts::PI * acc2).sin() * cfg.sine_amp;
            let noise_amp = if uv {
                cfg.noise_std
            } else {
                cfg.sine_amp / 3.0
            };
            let noise = normal(rng) * noise_amp;
            out[b * n_out + j] = if uv { sine + noise } else { noise };
        }
    }
    out
}
