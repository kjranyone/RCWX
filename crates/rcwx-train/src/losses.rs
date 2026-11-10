//! 学習損失 (LSGAN / feature matching / mel L1 / KL)。上流 RVC losses.py 相当。
//! 各損失はスカラー Var を生成し、逆伝達クロージャを tape に登録する。

use rcwx_nn::engine::tape::{Tape, Var};

/// mean((x − c)²)。dx = dy · 2(x−c)/N。
pub fn mse_mean(tape: &mut Tape<'_>, x: Var, c: f32) -> Var {
    let len = tape.numel_pub(x);
    let n_partials = len.div_ceil(256);
    let x_t = tape.var_tensor(x);
    let partials = tape.alloc_pub(n_partials as usize);
    {
        let mut words = [0u32; 32];
        words[0..3].copy_from_slice(&[len, 4, c.to_bits()]);
        tape.dispatch_pub(
            "t_reduce_partials",
            &[&x_t, &x_t, &partials],
            words,
            n_partials,
            1,
            1,
        );
    }
    let out = tape.alloc_pub(1);
    {
        let mut words = [0u32; 32];
        words[0..2].copy_from_slice(&[n_partials, (1.0f32 / len as f32).to_bits()]);
        tape.dispatch_pub("t_reduce_final", &[&partials, &out], words, 1, 1, 1);
    }
    let scalar = tape.push_scalar(out);
    let x_id = x.id;
    tape.push_loss_op(
        scalar,
        Box::new(move |ctx: &mut rcwx_nn::engine::tape::BwdCtx<'_, '_>| {
            let Some(dy) = ctx.grad_of(scalar.id) else {
                return;
            };
            let x_t = ctx.node_tensor(x_id);
            let dx = ctx.alloc(len as usize);
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[len, c.to_bits(), (1.0f32 / len as f32).to_bits(), 0]);
            ctx.dispatch(
                "t_mse_bwd",
                &[&x_t, &dx, &dy],
                words,
                len.div_ceil(256),
                1,
                1,
            );
            ctx.accum(x_id, dx);
        }),
    );
    scalar
}

/// LSGAN 判別器損失: Σ mean((r−1)²) + Σ mean(f²)。
pub fn lsgan_d_loss(tape: &mut Tape<'_>, real: &[Var], fake: &[Var]) -> Var {
    let mut total: Option<Var> = None;
    for r in real {
        let l = mse_mean(tape, *r, 1.0);
        total = Some(match total {
            Some(t) => tape.add_scalar(t, l),
            None => l,
        });
    }
    for f in fake {
        let l = mse_mean(tape, *f, 0.0);
        total = Some(match total {
            Some(t) => tape.add_scalar(t, l),
            None => l,
        });
    }
    total.expect("no logits")
}

/// LSGAN 生成器損失: Σ mean((f−1)²)。
pub fn lsgan_g_loss(tape: &mut Tape<'_>, fake: &[Var]) -> Var {
    let mut total: Option<Var> = None;
    for f in fake {
        let l = mse_mean(tape, *f, 1.0);
        total = Some(match total {
            Some(t) => tape.add_scalar(t, l),
            None => l,
        });
    }
    total.expect("no logits")
}

/// mean|a − b|。da = dy·sign(a−b)/N、db = −da。
pub fn l1_mean(tape: &mut Tape<'_>, a: Var, b: Var) -> Var {
    let len = tape.numel_pub(a);
    let n_partials = len.div_ceil(256);
    let a_t = tape.var_tensor(a);
    let b_t = tape.var_tensor(b);
    let partials = tape.alloc_pub(n_partials as usize);
    {
        let mut words = [0u32; 32];
        words[0..2].copy_from_slice(&[len, 3]);
        tape.dispatch_pub(
            "t_reduce_partials",
            &[&a_t, &b_t, &partials],
            words,
            n_partials,
            1,
            1,
        );
    }
    let out = tape.alloc_pub(1);
    {
        let mut words = [0u32; 32];
        words[0..2].copy_from_slice(&[n_partials, (1.0f32 / len as f32).to_bits()]);
        tape.dispatch_pub("t_reduce_final", &[&partials, &out], words, 1, 1, 1);
    }
    let scalar = tape.push_scalar(out);
    let a_id = a.id;
    let b_id = b.id;
    tape.push_loss_op(
        scalar,
        Box::new(move |ctx: &mut rcwx_nn::engine::tape::BwdCtx<'_, '_>| {
            let Some(dy) = ctx.grad_of(scalar.id) else {
                return;
            };
            let a_t = ctx.node_tensor(a_id);
            let b_t = ctx.node_tensor(b_id);
            let da = ctx.alloc(len as usize);
            let mut words = [0u32; 32];
            words[0..3].copy_from_slice(&[len, (1.0f32 / len as f32).to_bits(), 0]);
            ctx.dispatch(
                "t_l1_sign",
                &[&a_t, &b_t, &da, &dy],
                words,
                len.div_ceil(256),
                1,
                1,
            );
            ctx.accum(a_id, da.clone_handle());
            let db = ctx.alloc(len as usize);
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[len, 5, 0.0f32.to_bits(), 0]);
            ctx.dispatch("t_pw_fwd", &[&da, &db], words, len.div_ceil(256), 1, 1);
            ctx.accum(b_id, db);
        }),
    );
    scalar
}

/// feature matching: Σ 2·mean|fm_r − fm_g|。
pub fn feature_matching(tape: &mut Tape<'_>, real: &[Var], generated: &[Var]) -> Var {
    let mut total: Option<Var> = None;
    for (r, g) in real.iter().zip(generated.iter()) {
        let l = l1_mean(tape, *r, *g);
        let l = tape.scale_scalar(l, 2.0);
        total = Some(match total {
            Some(t) => tape.add_scalar(t, l),
            None => l,
        });
    }
    total.expect("no fmaps")
}

/// mel 損失 (L1 × coef)。
pub fn mel_l1_loss(tape: &mut Tape<'_>, y_mel: Var, y_hat_mel: Var, coef: f32) -> Var {
    let l = l1_mean(tape, y_mel, y_hat_mel);
    tape.scale_scalar(l, coef)
}

/// KL 損失: mean_masked((logs_p − logs_q − 0.5 + 0.5·d²·exp(−2·logs_p)))。
/// mask は [b, t_len]、他は [b, inter, t_len]。
#[allow(clippy::too_many_arguments)]
pub fn kl_loss(
    tape: &mut Tape<'_>,
    z_p: Var,
    logs_q: Var,
    m_p: Var,
    logs_p: Var,
    mask: Var,
    len: u32,
    t_len: u32,
    inter: u32,
    n_mask: f32,
) -> Var {
    let zp_t = tape.var_tensor(z_p);
    let lq_t = tape.var_tensor(logs_q);
    let mp_t = tape.var_tensor(m_p);
    let lp_t = tape.var_tensor(logs_p);
    let m_t = tape.var_tensor(mask);
    let elems = tape.alloc_pub(len as usize);
    {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[len, t_len, inter, 0]);
        tape.dispatch_pub(
            "t_kl_elem",
            &[&zp_t, &lq_t, &mp_t, &lp_t, &m_t, &elems],
            words,
            len.div_ceil(256),
            1,
            1,
        );
    }
    let n_partials = len.div_ceil(256);
    let partials = tape.alloc_pub(n_partials as usize);
    {
        let mut words = [0u32; 32];
        words[0..2].copy_from_slice(&[len, 0]);
        tape.dispatch_pub(
            "t_reduce_partials",
            &[&elems, &elems, &partials],
            words,
            n_partials,
            1,
            1,
        );
    }
    let out = tape.alloc_pub(1);
    {
        let mut words = [0u32; 32];
        words[0..2].copy_from_slice(&[n_partials, (1.0f32 / n_mask).to_bits()]);
        tape.dispatch_pub("t_reduce_final", &[&partials, &out], words, 1, 1, 1);
    }
    let scalar = tape.push_scalar(out);
    let zp_id = z_p.id;
    let lq_id = logs_q.id;
    let mp_id = m_p.id;
    let lp_id = logs_p.id;
    let mk_id = mask.id;
    tape.push_loss_op(
        scalar,
        Box::new(move |ctx: &mut rcwx_nn::engine::tape::BwdCtx<'_, '_>| {
            let Some(dy) = ctx.grad_of(scalar.id) else {
                return;
            };
            let zp_t = ctx.node_tensor(zp_id);
            let _lq_t = ctx.node_tensor(lq_id);
            let mp_t = ctx.node_tensor(mp_id);
            let lp_t = ctx.node_tensor(lp_id);
            let m_t = ctx.node_tensor(mk_id);
            let dz = ctx.alloc(len as usize);
            let dm = ctx.alloc(len as usize);
            let dlp = ctx.alloc(len as usize);
            let dlq = ctx.alloc(len as usize);
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[len, (1.0f32 / n_mask).to_bits(), t_len, inter]);
            ctx.dispatch(
                "t_kl_bwd_a",
                &[&zp_t, &mp_t, &lp_t, &m_t, &dy, &dz, &dm, &dlp],
                words,
                len.div_ceil(256),
                1,
                1,
            );
            ctx.dispatch(
                "t_kl_bwd_b",
                &[&m_t, &dy, &dlq],
                words,
                len.div_ceil(256),
                1,
                1,
            );
            ctx.accum(zp_id, dz);
            ctx.accum(mp_id, dm);
            ctx.accum(lp_id, dlp);
            ctx.accum(lq_id, dlq);
        }),
    );
    scalar
}
