//! conv_transpose1d (stride=4, k=8, pad=2) の CPU 直接比較。
use rcwx_nn::engine::tape::{ParamStore, Tape, Var};
use rcwx_nn::engine::{Backend, Engine};

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (1u64 << 33) as f32 - 0.5) * 2.0
}

#[test]
fn conv_t_stride4_k8() {
    let eng = Engine::new(Backend::Vulkan).expect("engine");
    let (batch, in_c, out_c, k, u, pad) = (2u32, 16u32, 8u32, 8u32, 4u32, 2u32);
    let t_in = 11u32;
    let t_out = (t_in - 1) * u + k - 2 * pad;
    let mut st = 5u64;
    let x_d: Vec<f32> = (0..(batch * in_c * t_in) as usize)
        .map(|_| lcg(&mut st))
        .collect();
    let w_d: Vec<f32> = (0..(in_c * out_c * k) as usize)
        .map(|_| lcg(&mut st))
        .collect();
    let b_d: Vec<f32> = (0..out_c as usize).map(|_| lcg(&mut st)).collect();

    let mut store = ParamStore::new();
    store.add(
        &eng,
        "w",
        &w_d,
        vec![in_c as usize, out_c as usize, k as usize],
    );
    store.add(&eng, "b", &b_d, vec![out_c as usize]);
    let mut tape = Tape::new(&eng);
    let pv = tape.adopt_all(&store);
    let x = tape.const_f32s(&x_d, vec![batch as usize, in_c as usize, t_in as usize]);
    let out = tape.conv_transpose1d(
        x,
        pv[0],
        Some(pv[1]),
        out_c,
        in_c,
        k,
        u,
        1,
        pad,
        batch,
        t_in,
        t_out,
    );
    let out_t = tape.var_tensor(out);
    tape.submit().expect("submit");
    let got = eng.readback_f32(&out_t);

    let mut cpu = vec![0.0f32; (batch * out_c * t_out) as usize];
    for b in 0..batch as usize {
        for o in 0..out_c as usize {
            for p in 0..t_out as usize {
                let mut acc = b_d[o];
                for ci in 0..in_c as usize {
                    for j in 0..k as usize {
                        let sv = p as i64 + pad as i64 - j as i64;
                        if sv >= 0 && sv % u as i64 == 0 {
                            let t = (sv / u as i64) as usize;
                            if t < t_in as usize {
                                acc += w_d[ci * (out_c * k) as usize + o * k as usize + j]
                                    * x_d[b * (in_c * t_in) as usize + ci * t_in as usize + t];
                            }
                        }
                    }
                }
                cpu[b * (out_c * t_out) as usize + o * t_out as usize + p] = acc;
            }
        }
    }
    let cos = rcwx_train::cosine_pub(&got, &cpu);
    println!(
        "convT s4k8: cos={cos:.6} got[0]={:.5} cpu[0]={:.5}",
        got[0], cpu[0]
    );
    assert!(cos >= 0.9999, "convT broken: {cos}");
}
