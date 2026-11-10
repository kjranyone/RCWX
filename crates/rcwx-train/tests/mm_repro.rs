//! conv1x1 (mm) の batch=2 最小再現。CPU 検算と比較。
use rcwx_nn::engine::tape::Tape;
use rcwx_nn::engine::{Backend, Engine};

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (1u64 << 33) as f32 - 0.5) * 2.0
}

#[test]
fn mm_batch2_repro() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (batch, oc, ic, t) = (2u32, 16u32, 16u32, 13u32);
    let mut st = 9u64;
    let x_d: Vec<f32> = (0..(batch * ic * t) as usize)
        .map(|_| lcg(&mut st))
        .collect();
    let w_d: Vec<f32> = (0..(oc * ic) as usize).map(|_| lcg(&mut st)).collect();
    let b_d: Vec<f32> = (0..oc as usize).map(|_| lcg(&mut st)).collect();

    let mut store = rcwx_nn::engine::tape::ParamStore::new();
    store.add(&eng, "w", &w_d, vec![oc as usize, ic as usize]);
    store.add(&eng, "b", &b_d, vec![oc as usize]);

    let mut tape = Tape::new(&eng);
    let pv = tape.adopt_all(&store);
    let x = tape.const_f32s(&x_d, vec![batch as usize, ic as usize, t as usize]);
    // mm: out[b, o, t] = Σ_c W[o,c]·x[b,c,t]
    let out = tape.mm(
        pv[0],
        x,
        None,
        (batch * oc * t) as usize,
        oc,
        t,
        ic,
        batch,
        ic,
        1,
        0,
        0,
        1,
        t,
        0,
        ic * t,
        t,
        1,
        oc * t,
        1.0,
    );
    let out = tape.add_ch_bias(out, pv[1], oc, t);
    let out_t = tape.var_tensor(out);
    tape.submit().expect("submit");
    let got = eng.readback_f32(&out_t);

    // CPU
    let mut cpu = vec![0.0f32; got.len()];
    for b in 0..batch {
        for o in 0..oc {
            for tt in 0..t {
                let mut acc = b_d[o as usize];
                for c in 0..ic {
                    acc += w_d[(o * ic + c) as usize] * x_d[((b * ic + c) * t + tt) as usize];
                }
                cpu[((b * oc + o) * t + tt) as usize] = acc;
            }
        }
    }
    let c = rcwx_train::cosine_pub(&got, &cpu);
    println!("mm_batch2: cos={c:.6}");
    assert!(c >= 0.9999, "mm batch2 broken: {c}");
}
