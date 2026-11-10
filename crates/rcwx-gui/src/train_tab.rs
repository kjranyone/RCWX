//! 学習タブ (Issue #8 M-T6)。前処理 → 学習 → エクスポート → index を
//! rcwx-cli サブコマンドでバックグラウンド実行する。

use gpui_kit::assets::IconName;
use gpui_kit::component::button::Button;
use gpui_kit::component::input::InputState;
use gpui_kit::component::{h_flex, v_flex};
use gpui_kit::*;

use crate::theme;
use crate::view::RcwxApp;

/// 学習タブの状態。
pub struct TrainTab {
    pub dataset_dir: Entity<InputState>,
    pub exp_name: Entity<InputState>,
    pub base_g: Entity<InputState>,
    pub base_d: Entity<InputState>,
    pub epochs: Entity<InputState>,
    pub busy: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub log: String,
}

impl TrainTab {
    pub fn new(window: &mut Window, cx: &mut Context<RcwxApp>) -> Self {
        let home = std::env::var("USERPROFILE").unwrap_or_default();
        Self {
            dataset_dir: cx.new(|cx| {
                InputState::new(window, cx).placeholder("wav フォルダ (例: C:\\data\\myvoice)")
            }),
            exp_name: cx.new(|cx| {
                InputState::new(window, cx)
                    .placeholder("myvoice")
                    .default_value("myvoice")
            }),
            base_g: cx.new(|cx| {
                InputState::new(window, cx)
                    .placeholder("base G .pth")
                    .default_value(&format!(
                        "{home}\\.cache\\rcwx\\train\\pretrained\\f0G48k_v2.pth"
                    ))
            }),
            base_d: cx.new(|cx| {
                InputState::new(window, cx)
                    .placeholder("base D .pth")
                    .default_value(&format!(
                        "{home}\\.cache\\rcwx\\train\\pretrained\\f0D48k_v2.pth"
                    ))
            }),
            epochs: cx.new(|cx| {
                InputState::new(window, cx)
                    .placeholder("20")
                    .default_value("20")
            }),
            busy: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            log: "待機中".into(),
        }
    }

    pub fn render(&mut self, cx: &mut Context<RcwxApp>) -> AnyElement {
        let busy = self.busy.load(std::sync::atomic::Ordering::Acquire);
        let log = self.log.clone();
        v_flex()
            .gap_3()
            .w_full()
            .child(self.card_dataset(cx, busy))
            .child(self.card_train(cx, busy))
            .child(self.card_progress(log, cx))
            .into_any_element()
    }

    fn card_dataset(&mut self, cx: &mut Context<RcwxApp>, busy: bool) -> AnyElement {
        let dir = self.dataset_dir.clone();
        let exp = self.exp_name.clone();
        let busy_flag = self.busy.clone();
        let body = v_flex()
            .gap_2()
            .w_full()
            .child(field_row("wav フォルダ", dir, cx))
            .child(field_row("実験名", exp, cx))
            .child(
                h_flex().gap_2().child(
                    Button::new("pp-run")
                        .label(if busy {
                            "実行中..."
                        } else {
                            "前処理を実行"
                        })
                        .on_click(cx.listener(move |this: &mut RcwxApp, _, _, cx| {
                            if this
                                .train
                                .busy
                                .swap(true, std::sync::atomic::Ordering::AcqRel)
                            {
                                return;
                            }
                            let dir = this.train.dataset_dir.clone().read(cx).value().to_string();
                            let exp = this.train.exp_name.clone().read(cx).value().to_string();
                            this.train.log = "前処理を実行中...".into();
                            this.ensure_poll_pub(cx);
                            let flag = busy_flag.clone();
                            cx.spawn(async move |this, cx| {
                                let out = cx
                                    .background_spawn(async move {
                                        run_cli(&["train-preprocess", &dir, "--exp", &exp]).await
                                    })
                                    .await;
                                let _ = flag.store(false, std::sync::atomic::Ordering::Release);
                                this.update(cx, |v: &mut RcwxApp, cx| {
                                    v.train.log = out;
                                    cx.notify();
                                })
                                .ok();
                            })
                            .detach();
                        })),
                ),
            );
        crate::view::card(
            IconName::FolderOpen,
            theme::lavender(),
            "データセット",
            cx,
            body,
        )
    }

    fn card_train(&mut self, cx: &mut Context<RcwxApp>, busy: bool) -> AnyElement {
        let g = self.base_g.clone();
        let d = self.base_d.clone();
        let ep = self.epochs.clone();
        let busy_flag = self.busy.clone();
        let exp_flag = self.busy.clone();
        let body = v_flex()
                .gap_2()
                .w_full()
                .child(field_row("base G (.pth)", g, cx))
                .child(field_row("base D (.pth)", d, cx))
                .child(field_row("エポック数", ep, cx))
                .child(
                    h_flex()
                        .gap_2()
                        .child(
                            Button::new("train-run")
                                .label(if busy { "実行中..." } else { "学習を開始" })
                                .on_click(cx.listener(
                                    move |this: &mut RcwxApp, _, _, cx| {
                                        if this.train.busy.swap(true, std::sync::atomic::Ordering::AcqRel) {
                                            return;
                                        }
                                        let exp = this.train.exp_name.clone().read(cx).value().to_string();
                                        let g = this.train.base_g.clone().read(cx).value().to_string();
                                        let d = this.train.base_d.clone().read(cx).value().to_string();
                                        let ep = this.train.epochs.clone().read(cx).value().to_string();
                                        this.train.log = "学習を実行中... (長時間)".into();
                                        this.ensure_poll_pub(cx);
                                        let flag = busy_flag.clone();
                                        cx.spawn(async move |this, cx| {
                                            let out = cx
                                                .background_spawn(async move {
                                                    run_cli(&[
                                                        "train", "--exp", &exp, "--base-g", &g,
                                                        "--base-d", &d, "--epochs", &ep,
                                                    ])
                                                    .await
                                                })
                                                .await;
                                            let _ = flag.store(false, std::sync::atomic::Ordering::Release);
                                            this.update(cx, |v: &mut RcwxApp, cx| {
                                                v.train.log = out;
                                                cx.notify();
                                            })
                                            .ok();
                                        })
                                        .detach();
                                    },
                                )),
                        )
                        .child(
                            Button::new("train-export")
                                .label("エクスポート")
                                .on_click(cx.listener(
                                    move |this: &mut RcwxApp, _, _, cx| {
                                        if this.train.busy.swap(true, std::sync::atomic::Ordering::AcqRel) {
                                            return;
                                        }
                                        let exp = this.train.exp_name.clone().read(cx).value().to_string();
                                        this.train.log = "エクスポート中...".into();
                                        this.ensure_poll_pub(cx);
                                        let flag = exp_flag.clone();
                                        cx.spawn(async move |this, cx| {
                                            let native = run_cli(&["train-export", "--exp", &exp]).await;
                                            let idx = run_cli(&["train-index", "--exp", &exp]).await;
                                            let _ = flag.store(false, std::sync::atomic::Ordering::Release);
                                            this.update(cx, |v: &mut RcwxApp, cx| {
                                                v.train.log = format!("{native}\n{idx}");
                                                cx.notify();
                                            })
                                            .ok();
                                        })
                                        .detach();
                                    },
                                )),
                        )
                        .child(crate::view::note(
                            "エクスポートで model/ に synth.safetensors + index が生成されます (rt で即使用可)",
                        )),
                );
        crate::view::card(IconName::GraduationCap, theme::sakura(), "学習", cx, body)
    }

    fn card_progress(&mut self, log: String, cx: &App) -> AnyElement {
        crate::view::card(
            IconName::Terminal,
            theme::sky(),
            "進捗",
            cx,
            gpui::div()
                .p_3()
                .rounded_md()
                .text_size(gpui::px(12.))
                .child(log),
        )
    }
}

fn field_row(label: &str, input: Entity<InputState>, _cx: &mut Context<RcwxApp>) -> AnyElement {
    h_flex()
        .gap_2()
        .items_center()
        .child(
            gpui::div()
                .w(gpui::px(110.))
                .flex_shrink_0()
                .text_size(gpui::px(12.))
                .child(label.to_string()),
        )
        .child(input.clone())
        .into_any_element()
}

async fn run_cli(args: &[&str]) -> String {
    let cli = crate::settings::cli_path();
    let mut cmd = std::process::Command::new(&cli);
    cmd.args(args);
    crate::settings::hide_window(&mut cmd);
    match cmd.output() {
        Ok(o) => {
            let mut s = String::from_utf8_lossy(&o.stdout).to_string();
            s.push_str(&String::from_utf8_lossy(&o.stderr));
            if o.status.success() {
                s
            } else {
                format!("失敗:\n{}", s)
            }
        }
        Err(e) => format!("起動失敗: {e} (rcwx-cli.exe を確認)"),
    }
}
