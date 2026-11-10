//! グラスモーフィズム風テーマ (トワイライト・オーロラ)。
//!
//! 深い藍紫のグラデーションに桜・ラベンダー・空色のオーロラを漂わせ、
//! 半透明ガラスパネルを浮かべる配色。gpui-kit コンポーネントは
//! `apply()` で Theme global を塗り替えることで全体が追従する。

use gpui_kit::assets::IconName;
use gpui_kit::component::Icon;
use gpui_kit::component::Sizable as _;
use gpui_kit::component::{h_flex, v_flex};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use std::f32::consts::TAU;
use std::time::Duration;

// ---- パレット ------------------------------------------------------------

/// 桜 (プライマリ)
pub fn sakura() -> Hsla {
    rgb(0xF7A8D8).into()
}
pub fn sakura_light() -> Hsla {
    rgb(0xFFBFDF).into()
}
pub fn sakura_deep() -> Hsla {
    rgb(0xEE96C8).into()
}
/// ラベンダー (アクセント)
pub fn lavender() -> Hsla {
    rgb(0xC9A6FF).into()
}
pub fn lavender_deep() -> Hsla {
    rgb(0xA987F0).into()
}
/// 空色
pub fn sky() -> Hsla {
    rgb(0x8FCBFF).into()
}
/// ミント
pub fn mint() -> Hsla {
    rgb(0x7DDFB2).into()
}
/// ピーチ
pub fn peach() -> Hsla {
    rgb(0xFFB4A2).into()
}
/// ローズ (danger 系)
pub fn rose() -> Hsla {
    rgb(0xFB7185).into()
}
pub fn rose_light() -> Hsla {
    rgb(0xFFA3AE).into()
}
/// 琥珀 (warning 系)
pub fn amber() -> Hsla {
    rgb(0xF5B26B).into()
}
pub fn amber_light() -> Hsla {
    rgb(0xFFCB8C).into()
}
/// テキスト白 (ラベンダーがかった)
pub fn porcelain() -> Hsla {
    rgb(0xF3EEFF).into()
}
pub fn plum() -> Hsla {
    rgb(0x2A1030).into()
}
pub fn muted_lilac() -> Hsla {
    rgb(0xB3A6D6).into()
}

fn ink(alpha: f32) -> Hsla {
    hsla(265., 0.45, 0.04, alpha)
}
/// 完全透明。
pub fn clear() -> Hsla {
    hsla(0., 0., 0., 0.)
}
/// 半透明の白 (ガラスの面・縁に重ねる光)。
pub fn white(alpha: f32) -> Hsla {
    hsla(0., 0., 1., alpha)
}

// ---- Theme global の塗り替え ----------------------------------------------

/// テーマをダークに固定し、トワイライト・ガラスの配色へ置き換える。
/// `gpui_kit::init` の直後に一度呼ぶ。
pub fn apply(cx: &mut App) {
    use gpui_kit::component::{Theme, ThemeMode};
    Theme::change(ThemeMode::Dark, None, cx);
    Theme::update(cx, |t| {
        let w = |a: f32| hsla(0., 0., 1., a);

        t.background = rgb(0x0E0A1C).into();
        t.foreground = porcelain();
        t.muted = rgb(0x2A2140).into();
        t.muted_foreground = muted_lilac();
        t.border = w(0.10);
        t.input = w(0.14);
        t.ring = sakura().opacity(0.55);
        t.selection = sakura().opacity(0.28);
        t.caret = sakura_light();
        t.overlay = ink(0.55);

        t.secondary = w(0.08);
        t.secondary_hover = w(0.13);
        t.secondary_active = w(0.17);
        t.secondary_foreground = porcelain();

        t.primary = sakura();
        t.primary_hover = sakura_light();
        t.primary_active = sakura_deep();
        t.primary_foreground = plum();
        t.accent = lavender();
        t.accent_foreground = rgb(0x1D1230).into();

        t.danger = rose();
        t.danger_hover = rose_light();
        t.danger_active = rgb(0xE86177).into();
        t.danger_foreground = rgb(0x2A0A12).into();
        t.warning = amber();
        t.warning_hover = amber_light();
        t.warning_active = rgb(0xE6A15A).into();
        t.warning_foreground = rgb(0x2A1405).into();
        t.success = mint();
        t.success_hover = rgb(0x92EBC2).into();
        t.success_active = rgb(0x6BD1A2).into();
        t.success_foreground = rgb(0x0A2018).into();

        t.popover = rgb(0x1A1430).into();
        t.popover_foreground = porcelain();

        t.button = w(0.07);
        t.button_hover = w(0.12);
        t.button_active = w(0.16);
        t.button_foreground = porcelain();
        t.button_primary = sakura();
        t.button_primary_hover = sakura_light();
        t.button_primary_active = sakura_deep();
        t.button_primary_foreground = plum();

        t.title_bar = t.background;
        t.status_bar = w(0.05);
        t.status_bar_border = w(0.08);
        t.tab_bar = clear();
        t.tab_bar_segmented = w(0.06);
        t.tab = clear();
        t.tab_active = w(0.13);
        t.tab_active_foreground = sakura_light();
        t.tab_foreground = muted_lilac();

        t.slider_bar = w(0.16);
        t.slider_thumb = sakura_light();
        t.progress_bar = sakura();
        t.switch = sakura();
        t.switch_thumb = porcelain();
        t.scrollbar = w(0.04);
        t.scrollbar_thumb = w(0.14);
        t.scrollbar_thumb_hover = w(0.22);
        t.skeleton = w(0.06);
        t.drop_target = sakura().opacity(0.15);
        t.window_border = w(0.10);
        t.link = rgb(0x9EC8FF).into();
        t.link_hover = sky();
        t.link_active = sky();

        t.radius = px(12.);
        t.radius_lg = px(18.);
        t.window_radius = px(12.);
    });
}

// ---- 背景オーブ ----------------------------------------------------------

struct Orb {
    x: f32,
    y: f32,
    size: f32,
    color: fn() -> Hsla,
    strength: f32,
    period: f64,
    phase: f32,
    id: &'static str,
}

/// オーロラの光源。同心円を重ねてガウス的なぼけを擬似している。
const ORBS: &[Orb] = &[
    Orb {
        x: -220.,
        y: -240.,
        size: 760.,
        color: sakura,
        strength: 1.0,
        period: 11.0,
        phase: 0.00,
        id: "orb-sakura",
    },
    Orb {
        x: 420.,
        y: -160.,
        size: 620.,
        color: lavender,
        strength: 0.9,
        period: 13.0,
        phase: 0.35,
        id: "orb-lavender",
    },
    Orb {
        x: -160.,
        y: 380.,
        size: 560.,
        color: mint,
        strength: 0.55,
        period: 15.0,
        phase: 0.60,
        id: "orb-mint",
    },
    Orb {
        x: 560.,
        y: 300.,
        size: 640.,
        color: sky,
        strength: 0.7,
        period: 12.0,
        phase: 0.80,
        id: "orb-sky",
    },
    Orb {
        x: 170.,
        y: 60.,
        size: 430.,
        color: peach,
        strength: 0.45,
        period: 17.0,
        phase: 0.50,
        id: "orb-peach",
    },
];

fn orb_layer(size: f32, alpha: f32, color: Hsla) -> AnyElement {
    div()
        .rounded_full()
        .size(px(size))
        .bg(color.opacity(alpha))
        .into_any_element()
}

fn orb(spec: &Orb) -> AnyElement {
    let (period, phase, strength, size) = (spec.period, spec.phase, spec.strength, spec.size);
    let color = spec.color;
    let breath = Animation::new(Duration::from_secs_f64(period))
        .repeat_synced()
        .with_easing(move |t| 0.5 - 0.5 * (t * TAU + phase * TAU).cos())
        .with_max_fps(20.);
    let name: &'static str = spec.id;
    div()
        .absolute()
        .left(px(spec.x))
        .top(px(spec.y))
        .with_animation(SharedString::from(name), breath, move |el, value| {
            el.opacity(0.45 + 0.55 * value)
        })
        .child(orb_layer(size, 0.055 * strength, color()))
        .child({
            let inner = size * 0.72;
            let offset = (size - inner) / 2.;
            div()
                .absolute()
                .left(px(offset))
                .top(px(offset))
                .child(orb_layer(inner, 0.085 * strength, color()))
        })
        .child({
            let inner = size * 0.46;
            let offset = (size - inner) / 2.;
            div()
                .absolute()
                .left(px(offset))
                .top(px(offset))
                .child(orb_layer(inner, 0.13 * strength, color()))
        })
        .into_any_element()
}

/// 窓の下地: トワイライトのグラデーション + 漂うオーロラ。
/// アクリル窓 (DWM blur) の上に半透明で重ねる。
pub fn aurora_backdrop() -> AnyElement {
    let base = linear_gradient(
        175.,
        linear_color_stop(hsla(262., 0.42, 0.13, 0.82), 0.),
        linear_color_stop(hsla(268., 0.50, 0.07, 0.90), 1.),
    );
    div()
        .absolute()
        .inset_0()
        .overflow_hidden()
        .bg(base)
        .children(ORBS.iter().map(orb))
        .child(
            // 上端に一本の光
            div()
                .absolute()
                .top_0()
                .left_0()
                .w_full()
                .h(px(1.5))
                .bg(linear_gradient(
                    90.,
                    linear_color_stop(clear(), 0.),
                    linear_color_stop(sakura_light().opacity(0.55), 0.5),
                )),
        )
        .into_any_element()
}

// ---- ガラスカード --------------------------------------------------------

/// グラデーションの縁 (リムライト) を持つガラスパネル。
pub fn glass_panel(body: impl IntoElement, _cx: &App) -> Div {
    div()
        .w_full()
        .rounded(px(20.))
        .p(px(1.))
        .bg(linear_gradient(
            160.,
            linear_color_stop(white(0.30), 0.),
            linear_color_stop(white(0.05), 1.),
        ))
        .shadow(vec![
            BoxShadow::new(px(0.), px(16.), ink(0.30)).blur_radius(px(36.)),
            BoxShadow::new(px(0.), px(2.), ink(0.20)).blur_radius(px(6.)),
        ])
        .child(
            v_flex()
                .w_full()
                .rounded(px(19.))
                .bg(linear_gradient(
                    168.,
                    linear_color_stop(white(0.10), 0.),
                    linear_color_stop(white(0.035), 1.),
                ))
                .child(body),
        )
}

/// セクション見出し: 色つぼみのアイコン + タイトル + 光る罫線。
pub fn card_head(icon: IconName, tint: Hsla, title: &'static str) -> AnyElement {
    h_flex()
        .w_full()
        .items_center()
        .gap_2()
        .child(
            div()
                .size(px(26.))
                .rounded_full()
                .bg(tint.opacity(0.16))
                .border_1()
                .border_color(tint.opacity(0.30))
                .flex()
                .items_center()
                .justify_center()
                .shadow(vec![
                    BoxShadow::new(px(0.), px(0.), tint.opacity(0.25)).blur_radius(px(10.)),
                ])
                .child(Icon::new(icon).text_color(tint).with_size(px(14.))),
        )
        .child(
            div()
                .text_sm()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(porcelain().opacity(0.92))
                .child(title),
        )
        .child(div().flex_1().h(px(1.)).ml_1().bg(linear_gradient(
            90.,
            linear_color_stop(white(0.14), 0.),
            linear_color_stop(clear(), 1.),
        )))
        .into_any_element()
}

// ---- 小部品 ---------------------------------------------------------------

/// 値を表示するガラスチップ (スライダーの右側など)。
pub fn value_chip(text: impl Into<SharedString>) -> AnyElement {
    div()
        .w(px(92.))
        .px_2()
        .py_0p5()
        .rounded_full()
        .bg(white(0.07))
        .border_1()
        .border_color(white(0.10))
        .text_xs()
        .text_color(sakura_light())
        .text_align(TextAlign::Right)
        .child(text.into())
        .into_any_element()
}

/// フッター用の小さな計測チップ。
pub fn stat_chip(text: impl Into<SharedString>, color: Hsla) -> AnyElement {
    div()
        .px_2p5()
        .py_1()
        .rounded_full()
        .bg(white(0.06))
        .border_1()
        .border_color(white(0.09))
        .text_xs()
        .text_color(color)
        .child(text.into())
        .into_any_element()
}

/// 桜→ラベンダーのグラデーションが光る主ボタン。
pub enum HeroKind {
    /// グラデーションが光る「開始」。
    Primary,
    /// ローズガラスの「停止」。
    Danger,
    /// 押せない玻璃。
    Ghost,
}

pub fn hero_button(
    id: &'static str,
    label: &'static str,
    kind: HeroKind,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> AnyElement {
    let (bg, border, text, glows, glows_hover): (Fill, Hsla, Hsla, Vec<BoxShadow>, Vec<BoxShadow>) =
        match kind {
            HeroKind::Primary => {
                let idle = vec![
                    BoxShadow::new(px(0.), px(4.), sakura().opacity(0.42)).blur_radius(px(22.)),
                    BoxShadow::new(px(0.), px(1.), lavender().opacity(0.40)).blur_radius(px(14.)),
                ];
                let hover = vec![
                    BoxShadow::new(px(0.), px(6.), sakura().opacity(0.60)).blur_radius(px(26.)),
                    BoxShadow::new(px(0.), px(2.), lavender().opacity(0.55)).blur_radius(px(18.)),
                ];
                (
                    linear_gradient(
                        100.,
                        linear_color_stop(sakura(), 0.),
                        linear_color_stop(lavender(), 1.),
                    )
                    .into(),
                    clear(),
                    plum(),
                    idle,
                    hover,
                )
            }
            HeroKind::Danger => {
                let idle =
                    vec![BoxShadow::new(px(0.), px(3.), rose().opacity(0.30)).blur_radius(px(16.))];
                let hover =
                    vec![BoxShadow::new(px(0.), px(4.), rose().opacity(0.45)).blur_radius(px(20.))];
                (
                    rose().opacity(0.20).into(),
                    rose().opacity(0.38),
                    rose_light(),
                    idle,
                    hover,
                )
            }
            HeroKind::Ghost => (
                white(0.05).into(),
                white(0.10),
                muted_lilac(),
                vec![],
                vec![BoxShadow::new(px(0.), px(2.), white(0.08)).blur_radius(px(10.))],
            ),
        };
    div()
        .id(SharedString::from(id))
        .cursor_pointer()
        .px_5()
        .py_2()
        .rounded_full()
        .bg(bg)
        .border_1()
        .border_color(border)
        .shadow(glows)
        .text_sm()
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(text)
        .hover(move |s| s.shadow(glows_hover.clone()))
        .on_click(on_click)
        .child(label)
        .into_any_element()
}

/// 脈打つステータスドット。
pub fn pulse_dot(color: Hsla, id: &'static str, live: bool) -> AnyElement {
    let dot = div().size(px(9.)).rounded_full().bg(color).shadow(vec![
        BoxShadow::new(px(0.), px(0.), color.opacity(0.8)).blur_radius(px(6.)),
    ]);
    if live {
        let breath = Animation::new(Duration::from_millis(1600))
            .repeat_synced()
            .with_easing(|t| 0.5 - 0.5 * (t * TAU).cos())
            .with_max_fps(20.);
        dot.with_animation(SharedString::from(id), breath, |el, value| {
            el.opacity(0.45 + 0.55 * value)
        })
        .into_any_element()
    } else {
        dot.into_any_element()
    }
}

/// セグメント型 VU メーター。左からミント→空→ラベンダー→桜→ローズに灯る。
pub fn vu_meter(level: f32) -> AnyElement {
    const SEGMENTS: usize = 26;
    let db = if level > 1e-6 {
        20.0 * f64::from(level).log10()
    } else {
        -60.0
    };
    let frac = ((db + 60.0) / 60.0).clamp(0.0, 1.0);
    let lit = (frac * SEGMENTS as f64).round() as usize;
    let color_at = |i: usize| -> Hsla {
        if i < 14 {
            mint()
        } else if i < 19 {
            sky()
        } else if i < 23 {
            lavender()
        } else if i < 25 {
            sakura()
        } else {
            rose()
        }
    };
    h_flex()
        .items_center()
        .gap_3()
        .child(
            h_flex()
                .w(px(190.))
                .h(px(16.))
                .items_center()
                .gap(px(2.))
                .rounded_full()
                .bg(ink(0.35))
                .border_1()
                .border_color(white(0.08))
                .px(px(3.))
                .child({
                    let bars: Vec<AnyElement> = (0..SEGMENTS)
                        .map(|i| {
                            let hot = i >= SEGMENTS - 3;
                            let mut bar = div().flex_1().h(px(8.)).rounded_full().bg(if i < lit {
                                color_at(i)
                            } else {
                                white(0.07)
                            });
                            if i < lit && hot {
                                bar = bar.shadow(vec![
                                    BoxShadow::new(px(0.), px(0.), color_at(i).opacity(0.7))
                                        .blur_radius(px(5.)),
                                ]);
                            }
                            bar.into_any_element()
                        })
                        .collect();
                    div().flex_1().flex().gap(px(2.)).children(bars)
                }),
        )
        .child(value_chip(format!("{db:5.1} dB")))
        .into_any_element()
}

/// 通知バナー: 色つきガラス + アイコン。
pub fn glass_banner(icon: IconName, tint: Hsla, text: impl Into<SharedString>) -> AnyElement {
    h_flex()
        .w_full()
        .items_start()
        .gap_2p5()
        .px_3()
        .py_2p5()
        .rounded_xl()
        .bg(tint.opacity(0.13))
        .border_1()
        .border_color(tint.opacity(0.28))
        .child(
            Icon::new(icon)
                .text_color(tint)
                .with_size(px(15.))
                .mt(px(1.)),
        )
        .child(
            div()
                .flex_1()
                .text_sm()
                .text_color(porcelain().opacity(0.90))
                .child(text.into()),
        )
        .into_any_element()
}

// ---- タブ -----------------------------------------------------------------

pub struct PillTab {
    pub id: &'static str,
    pub icon: IconName,
    pub label: &'static str,
}

/// ガラスの舟に並ぶピル型タブ。`on_select` には `cx.listener` をそのまま渡せる。
pub fn pill_tabs(
    tabs: &[PillTab],
    selected: usize,
    on_select: impl Fn(&usize, &mut Window, &mut App) + 'static,
) -> AnyElement {
    let on_select = std::rc::Rc::new(on_select);
    let pills: Vec<AnyElement> = tabs
        .iter()
        .enumerate()
        .map(|(index, tab)| {
            let active = index == selected;
            let (fg, icon_color, glow) = if active {
                (
                    sakura_light(),
                    sakura_light(),
                    vec![
                        BoxShadow::new(px(0.), px(1.), white(0.18))
                            .blur_radius(px(4.))
                            .inset(),
                    ],
                )
            } else {
                (muted_lilac(), muted_lilac().opacity(0.8), vec![])
            };
            let on_select = std::rc::Rc::clone(&on_select);
            div()
                .id(SharedString::from(tab.id))
                .cursor_pointer()
                .px_4()
                .py_1p5()
                .rounded_full()
                .when(active, |el| el.bg(white(0.13)).shadow(glow))
                .hover(|s| s.bg(white(0.07)))
                .on_click(move |_, window, cx| on_select(&index, window, cx))
                .child(
                    h_flex()
                        .items_center()
                        .gap_1p5()
                        .child(
                            Icon::new(tab.icon)
                                .text_color(icon_color)
                                .with_size(px(13.)),
                        )
                        .child(div().text_sm().text_color(fg).child(tab.label)),
                )
                .into_any_element()
        })
        .collect();
    h_flex()
        .items_center()
        .gap_1()
        .px_1()
        .py_1()
        .rounded_full()
        .bg(ink(0.30))
        .border_1()
        .border_color(white(0.09))
        .children(pills)
        .into_any_element()
}
