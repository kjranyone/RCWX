//! rcwx-gui — GPUI リアルタイムボイスチェンジャー。
//!
//! モデル / オーディオ / レイテンシ / Pitch・F0 / Index / ノイズキャンセリング /
//! 入力ノイズゲート / ポスト処理 / 入力レベル / 統計 / WAV 変換。
//! 設定は ~/.config/rcwx/config.json と互換。
//! 見た目は theme.rs のグラスモーフィズム (トワイライト・オーロラ) に一任。

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod settings;
mod theme;
mod view;

use gpui_kit::*;

fn main() {
    gpui_kit::application()
        .with_assets(gpui_kit::assets::AllAssets)
        .run(|cx| {
            gpui_kit::init(cx);
            theme::apply(cx);
            let bounds = Bounds::centered(None, size(px(1180.), px(860.)), cx);
            let options = WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_background: WindowBackgroundAppearance::Blurred,
                ..Default::default()
            };
            gpui_kit::open_window(options, cx, |window, cx| {
                window.set_window_title("RCWX");
                cx.new(|cx| view::RcwxApp::new(window, cx))
            })
            .expect("RCWX ウィンドウを開けませんでした");
        });
}
