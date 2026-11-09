//! drift_control の手計算値検証 (tests/integration/test_realtime_drift_control.py 相当)。

use rcwx_dsp::drift::{compute_shed_policy, FloorTracker};

#[test]
fn normal_mode_policy() {
    // hop 960 @48k (20ms), margin 0.25 → threshold=int(0.75*960)=720, target=240
    let p = compute_shed_policy(false, 960, 48000, 0.02, 0.25, 100, 5.0, 6.0, 0.01, 0.01);
    assert_eq!(p.threshold, 720);
    assert_eq!(p.target, 240);
    assert!((p.guard_ms - 240.0 * 1000.0 / 48000.0).abs() < 1e-9);
}

#[test]
fn deadline_mode_initial_guard() {
    // サンプル不足 → 1 hop ガード
    let p = compute_shed_policy(true, 960, 48000, 0.02, 0.1, 5, 5.0, 9.0, 0.0025, 0.003);
    assert_eq!(p.threshold, 960 * 5 / 4);
    assert_eq!(p.target, 960);
    assert!((p.guard_ms - 20.0).abs() < 1e-9);
}

#[test]
fn deadline_mode_adaptive() {
    // jitter 4ms + callback 3ms → guard=max(10, 7)=10ms < 0.875*20=17.5 → target=480
    let p = compute_shed_policy(true, 960, 48000, 0.02, 0.1, 100, 5.0, 9.0, 0.0025, 0.003);
    assert_eq!(p.target, 480);
    // band = max(240, int(3ms*48)=144) = 240 → threshold = min(1200, 480+240)=720
    assert_eq!(p.threshold, 720);
    // guard が hop の 87.5% を超える場合は cap
    let p2 = compute_shed_policy(true, 960, 48000, 0.02, 0.1, 100, 5.0, 30.0, 0.01, 0.01);
    // jitter 25ms + 10ms = 35 → capped at 17.5ms → 840 samples
    assert_eq!(p2.target, 840);
}

#[test]
fn floor_tracker_window_min() {
    let mut ft = FloorTracker::new();
    // frames=240 @hop960 → window = ceil(1920/240)=8
    ft.ensure_window(240, 960);
    assert!(!ft.is_full());
    for v in [900, 500, 800] {
        ft.record_post_read(v);
    }
    assert!(!ft.is_full());
    assert_eq!(ft.floor_samples, 0); // 満杯前は未確定
    for v in [700, 600, 1000, 950, 850] {
        ft.record_post_read(v);
    }
    assert!(ft.is_full());
    assert_eq!(ft.floor_samples, 500);
    // 最小値が窓から抜けると更新
    for _ in 0..8 {
        ft.record_post_read(650);
    }
    assert_eq!(ft.floor_samples, 650);
    // スキップ後は再計測
    ft.after_skip(240);
    assert_eq!(ft.floor_samples, 240);
    assert!(!ft.is_full());
}
