//! interp_time カーネル (formant shift の z 伸長) の GPU/CPU 一致検証。
//! 実 GPU 必要 (Vulkan)。

use rcwx_nn::engine::{Backend, Engine};
use rcwx_nn::synth::stretch_time;

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let v = (*state >> 33) as f64 / (1u64 << 33) as f64;
    ((v - 0.5) * 2.0) as f32
}

fn check(eng: &Engine, ch: u32, t_in: u32, t_out: u32) {
    let mut state = 0x1234_5678_9abc_def0u64;
    let src: Vec<f32> = (0..ch as usize * t_in as usize)
        .map(|_| lcg(&mut state))
        .collect();

    let src_t = eng.tensor_from_f32(&src, vec![]);
    let dst = eng.tensor_uninit(ch as usize * t_out as usize);
    let mut g = eng.graph();
    g.interp_time(&src_t, &dst, ch, t_in, t_out);
    g.submit().expect("submit");
    let gpu = eng.readback_f32(&dst);

    let mut expect = vec![0.0f32; ch as usize * t_out as usize];
    for c in 0..ch as usize {
        let row = stretch_time(
            &src[c * t_in as usize..(c + 1) * t_in as usize],
            t_in as usize,
            t_out as usize,
        );
        expect[c * t_out as usize..(c + 1) * t_out as usize].copy_from_slice(&row);
    }
    let maxdiff = gpu
        .iter()
        .zip(expect.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        maxdiff < 1e-5,
        "ch={ch} t_in={t_in} t_out={t_out} maxdiff={maxdiff}"
    );
}

#[test]
fn interp_time_matches_cpu() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    // アップサンプリング (formant > 0) とダウンサンプリング (formant < 0) 両方。
    check(&eng, 192, 40, 54); // ceil(40 * 2^(5/12)) 相当
    check(&eng, 192, 54, 40); // 逆方向
    check(&eng, 8, 1, 8); // 極小入力
    check(&eng, 4, 16, 16); // 同一長 (バイパス相当)
    check(&eng, 512, 100, 133); // 大きめ
}

#[test]
fn stretch_time_edges() {
    // t_out=1, t_in=1, 縮小・拡大でパニックしない + 端クランプ。
    assert_eq!(stretch_time(&[1.0, 2.0], 2, 2), vec![1.0, 2.0]);
    assert_eq!(stretch_time(&[3.0], 1, 1), vec![3.0]);
    let up = stretch_time(&[0.0, 1.0], 2, 5);
    assert_eq!(up.len(), 5);
    assert!((up[0] - 0.0).abs() < 1e-6, "src<0 は index 0 へクランプ");
    let down = stretch_time(&[0.0, 0.5, 1.0, 0.5, 0.0], 5, 2);
    assert_eq!(down.len(), 2);
}
