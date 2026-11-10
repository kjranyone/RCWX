//! RVC 学習 (Issue #8 M-T2+)。rcwx-nn の tape autograd 上に
//! Synthesizer (enc_p/enc_q/flow/dec) と MPD 判別器・損失を構築する。

pub mod data;
pub mod losses;
pub mod modules;
pub mod sinegen;
pub mod train_loop;

pub use modules::mpd;
pub use modules::stft_mel;
pub use modules::synth_train;

/// 平坦化 cosine 類似度 (検証用)。
pub fn cosine_pub(a: &[f32], b: &[f32]) -> f64 {
    rcwx_nn::cosine(a, b)
}

pub mod export;

pub mod index;
