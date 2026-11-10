//! matmul_tiled 単体ベンチ (FXC/DXC コードゲン比較用)。
//! cargo run -p rcwx-nn --release --example bench_matmul -- [iters]
use rcwx_nn::engine::{Backend, Engine};

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let eng = Engine::new(Backend::Dx12).expect("engine");
    // リアルタイム HuBERT 相当の qkv GEMM: m=56, n=2304, k=768
    let m = 56u32;
    let n = 2304u32;
    let k = 768u32;
    let a = eng.tensor_from_f32(&vec![0.01f32; (m * k) as usize], vec![]);
    let w = eng.tensor_from_f32(&vec![0.02f32; (k * n) as usize], vec![]);
    let b = eng.tensor_from_f32(&vec![0.0f32; n as usize], vec![]);
    let c = eng.tensor_uninit((m * n) as usize);
    // ウォームアップ + プラン
    let plan = {
        let mut g = eng.graph();
        g.matmul(
            &a,
            &w,
            Some(&b),
            &c,
            m,
            n,
            k,
            1,
            k,
            1,
            0,
            0,
            k,
            1,
            0,
            0,
            n,
            1,
            0,
            0,
            1.0,
        );
        g.into_plan().expect("plan")
    };
    for _ in 0..5 {
        eng.replay(&plan, None);
        let _ = eng.readback_f32(&c);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        eng.replay(&plan, None);
    }
    let submit = t0.elapsed();
    let v = eng.readback_f32(&c); // 最後の fence 待ち
    let total = t0.elapsed();
    let zeros: Vec<usize> = (0..v.len()).filter(|&i| v[i] == 0.0).collect();
    println!(
        "zeros: {} of {} (first: {:?})",
        zeros.len(),
        v.len(),
        &zeros[..zeros.len().min(12)]
    );
    let per = total.as_secs_f64() / iters as f64 * 1000.0;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    println!(
        "matmul_tiled m={m} n={n} k={k}: {per:.3} ms/op ({:.2} GFLOPS equiv, submit {:.1}ms total)",
        flops / (per / 1000.0) / 1e9,
        submit.as_secs_f64() * 1000.0
    );
}
