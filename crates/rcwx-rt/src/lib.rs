//! rcwx-rt — リアルタイムボイスチェンジャーエンジン (M6)。
//!
//! 構成:
//! - `config`: RtConfig / RtStats
//! - `ring`: SPSC リングバッファ (出力)
//! - `engine`: ストリーミング推論 (HuBERT 履歴 + F0 コンテキスト + 持ち越し位相 sinegen
//!   + SOLA + ドリフト制御) — 1 hop = `chunk_sec`
//! - `audio`: cpal 入出力 + 推論スレッド

pub mod audio;
pub mod config;
pub mod engine;
pub mod ring;

pub use config::{RtConfig, RtStats};
