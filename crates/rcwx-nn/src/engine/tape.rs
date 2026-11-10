//! 学習用 autograd tape (Issue #8 M-T0/M-T1)。
//!
//! 設計:
//! - `ParamStore` が重み・勾配・AdamW 状態を学習全体で永続保持する。
//!   1 学習ステップごとに `Tape` を生成し、`adopt_param` / `adopt_wn` で
//!   ストアのバッファをノードとして取り込む (バッファ共有・コピーなし)。
//! - `Tape` は `Graph` を内包し、各 op の forward dispatch と同時に backward 用
//!   クロージャを op リストに積む。`backward_from` はリストを逆走査し、出力勾配
//!   が存在する op だけ逆伝搬カーネルを同じ Graph へ追加する
//!   (1 ステップ = 1 Graph = 1 submit。pass 内 dispatch は順序実行される)。
//! - 中間テンソルは全 op で新規確保 (in-place なし) なので保存活性が壊れない。
//! - weight_norm は `_v`/`_g` を学習パラメータとし、ステップ冒頭の adopt 時に
//!   fold (w = v·g/||v||) を dispatch、backward で dv/dg に逆伝搬する。
//!
//! 推論経路 (rt/cli) はこのモジュールを一切使わない。

use std::collections::HashMap;

use super::{Engine, GpuTensor, Graph, Plan};

pub type VarId = u32;

/// tape 変数ハンドル (Copy)。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Var {
    pub id: VarId,
}

struct Node {
    t: GpuTensor,
    shape: Vec<usize>,
    /// 勾配を流すか (定数入力 / detach 済みは false)
    req_grad: bool,
}

struct OpEntry {
    out: VarId,
    bwd: Box<dyn FnOnce(&mut BwdCtx<'_, '_>)>,
}

/// 通常パラメータ 1 個分。
pub struct ParamEntry {
    pub name: String,
    pub value: GpuTensor,
    pub grad: GpuTensor,
}

/// weight_norm パラメータ群 (v・g は `ParamEntry`、w は fold 結果)。
pub struct WnEntry {
    pub v_idx: usize,
    pub g_idx: usize,
    /// fold 済み実効重み (forward はこれを使う)
    pub w: GpuTensor,
    normsq: GpuTensor,
    partials: GpuTensor,
    shape: Vec<usize>,
    len: u32,
}

/// 学習パラメータストア (net_g / net_d に 1 個ずつ)。
pub struct ParamStore {
    pub entries: Vec<ParamEntry>,
    pub wns: Vec<WnEntry>,
    pub adam: AdamW,
}

impl ParamStore {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            wns: Vec::new(),
            adam: AdamW::new(1e-4),
        }
    }

    /// パラメータ追加 (ゼロ初期化の勾配バッファも確保)。戻り値は entry index。
    pub fn add(&mut self, eng: &Engine, name: &str, data: &[f32], shape: Vec<usize>) -> usize {
        let n: usize = shape.iter().product();
        assert_eq!(n, data.len());
        let value = eng.tensor_from_f32(data, shape.clone());
        let grad = eng.tensor_zeros(n, vec![n]);
        self.entries.push(ParamEntry {
            name: name.to_string(),
            value,
            grad,
        });
        self.entries.len() - 1
    }

    /// weight_norm パラメータ追加。forward には戻り値 (wn index) を
    /// `Tape::adopt_wn` に渡して得られる Var を使う。
    pub fn add_weight_norm(
        &mut self,
        eng: &Engine,
        name: &str,
        v_data: &[f32],
        g: f32,
        shape: Vec<usize>,
    ) -> usize {
        let v_idx = self.add(eng, &format!("{name}.weight_v"), v_data, shape.clone());
        let g_idx = self.add(eng, &format!("{name}.weight_g"), &[g], vec![1]);
        let n: u32 = shape.iter().map(|&v| v as u32).product();
        let w = eng.tensor_uninit(n as usize);
        let normsq = eng.tensor_zeros(1, vec![1]);
        let partials = eng.tensor_uninit(n.div_ceil(256) as usize);
        self.wns.push(WnEntry {
            v_idx,
            g_idx,
            w,
            normsq,
            partials,
            shape,
            len: n,
        });
        self.wns.len() - 1
    }
}

impl Default for ParamStore {
    fn default() -> Self {
        Self::new()
    }
}

/// AdamW 状態。
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: u64,
    m: HashMap<usize, GpuTensor>,
    v: HashMap<usize, GpuTensor>,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.8,
            beta2: 0.99,
            eps: 1e-9,
            weight_decay: 0.0,
            step: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

/// backward 実行コンテキスト。各 op のクロージャに渡される。
pub struct BwdCtx<'a, 'e> {
    g: &'a mut Graph<'e>,
    nodes: &'a HashMap<VarId, Node>,
    grads: &'a mut HashMap<VarId, GpuTensor>,
}

impl BwdCtx<'_, '_> {
    pub fn node_tensor(&self, id: VarId) -> GpuTensor {
        self.nodes[&id].t.clone_handle()
    }
    fn len_of(&self, id: VarId) -> u32 {
        self.nodes[&id].t.len as u32
    }
    fn req_grad(&self, id: VarId) -> bool {
        self.nodes[&id].req_grad
    }
    pub fn alloc(&self, len: usize) -> GpuTensor {
        self.g.engine_tensor_uninit(len)
    }
    /// 逆伝達カーネルの直接 dispatch (losses.rs 用)。
    pub fn dispatch(
        &mut self,
        kernel: &'static str,
        bufs: &[&GpuTensor],
        words: [u32; 32],
        wx: u32,
        wy: u32,
        wz: u32,
    ) {
        self.g.dispatch_raw(kernel, bufs, words, wx, wy, wz);
    }
    /// 勾配バッファの参照 (存在すれば)。
    pub fn grad_of(&self, id: VarId) -> Option<GpuTensor> {
        self.grads.get(&id).map(|t| t.clone_handle())
    }
    /// id への勾配蓄積。req_grad=false なら破棄。初回は contribution をそのまま
    /// 登録する (bwd クロージャは専用の新規バッファを渡すこと)。
    pub fn accum(&mut self, id: VarId, contrib: GpuTensor) {
        if !self.req_grad(id) {
            return;
        }
        match self.grads.get(&id) {
            Some(existing) => {
                let len = self.len_of(id);
                let existing = existing.clone_handle();
                self.g
                    .add_off3(&existing, &contrib, &existing, len, 0, 0, 0, false);
            }
            None => {
                self.grads.insert(id, contrib);
            }
        }
    }
}

pub struct Tape<'e> {
    eng: &'e Engine,
    g: Graph<'e>,
    nodes: HashMap<VarId, Node>,
    ops: Vec<OpEntry>,
    next_id: VarId,
    grads: HashMap<VarId, GpuTensor>,
    /// adopt 済みパラメータ (store idx, Var)
    adopted: Vec<(usize, Var)>,
}

impl<'e> Tape<'e> {
    pub fn new(eng: &'e Engine) -> Self {
        Self {
            eng,
            g: eng.graph(),
            nodes: HashMap::new(),
            ops: Vec::new(),
            next_id: 0,
            grads: HashMap::new(),
            adopted: Vec::new(),
        }
    }

    fn push_node(&mut self, t: GpuTensor, shape: Vec<usize>, req_grad: bool) -> Var {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, Node { t, shape, req_grad });
        Var { id }
    }

    fn t(&self, v: Var) -> GpuTensor {
        self.nodes[&v.id].t.clone_handle()
    }

    /// テンソル実体の参照 (readback 用)。
    pub fn var_tensor(&self, v: Var) -> GpuTensor {
        self.t(v)
    }

    pub fn shape(&self, v: Var) -> Vec<usize> {
        self.nodes[&v.id].shape.clone()
    }

    fn numel(&self, v: Var) -> u32 {
        self.nodes[&v.id].t.len as u32
    }

    /// 定数入力。
    pub fn const_f32s(&mut self, data: &[f32], shape: Vec<usize>) -> Var {
        let n: usize = shape.iter().product();
        assert_eq!(n, data.len());
        let t = self.eng.tensor_from_f32(data, shape.clone());
        self.push_node(t, shape, false)
    }

    /// 勾配を流す入力 (gradcheck の対象)。
    pub fn input_f32s(&mut self, data: &[f32], shape: Vec<usize>) -> Var {
        let n: usize = shape.iter().product();
        assert_eq!(n, data.len());
        let t = self.eng.tensor_from_f32(data, shape.clone());
        self.push_node(t, shape, true)
    }

    /// detach: 同じテンソルを指すが勾配は流さない新規 Var。
    pub fn detach(&mut self, v: Var) -> Var {
        let t = self.t(v);
        let shape = self.shape(v);
        self.push_node(t, shape, false)
    }

    /// ストアのパラメータをノードとして取り込む (勾配バッファも共有)。
    pub fn adopt_param(&mut self, store: &ParamStore, idx: usize) -> Var {
        let p = &store.entries[idx];
        let var = self.push_node(p.value.clone_handle(), p.value.shape.clone(), true);
        self.grads.insert(var.id, p.grad.clone_handle());
        self.adopted.push((idx, var));
        var
    }

    /// ストアの全パラメータを取り込む。戻り値は store.entries と index 整列。
    /// weight_norm の v/g も含まれるので、adopt_wn の前に呼ぶこと。
    pub fn adopt_all(&mut self, store: &ParamStore) -> Vec<Var> {
        (0..store.entries.len())
            .map(|i| self.adopt_param(store, i))
            .collect()
    }

    /// weight_norm パラメータを取り込み、fold を dispatch して実効重み Var を返す。
    /// `v`/`g` は adopt_all/adopt_param で得た該当エントリの Var (勾配はそこへ流れる)。
    /// ステップ冒頭 (モジュール forward 前) に 1 回呼ぶ。
    pub fn adopt_wn(&mut self, store: &ParamStore, wn_idx: usize, v: Var, g: Var) -> Var {
        let wn = &store.wns[wn_idx];
        let len = wn.len;
        let n_partials = len.div_ceil(256);
        // normsq = Σ v²  (partials → final)
        let v_t = self.t(v);
        let partials = wn.partials.clone_handle();
        let normsq = wn.normsq.clone_handle();
        {
            let mut words = [0u32; 32];
            words[0..2].copy_from_slice(&[len, 1]);
            self.g.dispatch_raw(
                "t_reduce_partials",
                &[&v_t, &v_t, &partials],
                words,
                n_partials,
                1,
                1,
            );
        }
        {
            let mut words = [0u32; 32];
            words[0..2].copy_from_slice(&[n_partials, 1.0f32.to_bits()]);
            self.g
                .dispatch_raw("t_reduce_final", &[&partials, &normsq], words, 1, 1, 1);
        }
        let w = self.push_node(wn.w.clone_handle(), wn.shape.clone(), true);
        {
            let g_t = self.t(g);
            let w_t = self.t(w);
            let mut words = [0u32; 32];
            words[0] = len;
            self.g.dispatch_raw(
                "t_wn_fold",
                &[&v_t, &g_t, &normsq, &w_t],
                words,
                len.div_ceil(256),
                1,
                1,
            );
        }
        // backward: dw → dv, dg (fold はステップ先頭に記録されるので逆順処理で
        // 最後に実行され、蓄積済み dw をすべて消費する)
        let w_id = w.id;
        let v_id = v.id;
        let g_id = g.id;
        let normsq_c = wn.normsq.clone_handle();
        self.ops.push(OpEntry {
            out: w_id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dw) = ctx.grad_of(w_id) else {
                    return;
                };
                let v_t = ctx.node_tensor(v_id);
                let len = dw.len as u32;
                let n_partials = len.div_ceil(256);
                // dot = Σ v·dw
                let partials = ctx.alloc(n_partials as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..2].copy_from_slice(&[len, 2]);
                    ctx.g.dispatch_raw(
                        "t_reduce_partials",
                        &[&v_t, &dw, &partials],
                        words,
                        n_partials,
                        1,
                        1,
                    );
                }
                let dot = ctx.alloc(1);
                {
                    let mut words = [0u32; 32];
                    words[0..2].copy_from_slice(&[n_partials, 1.0f32.to_bits()]);
                    ctx.g
                        .dispatch_raw("t_reduce_final", &[&partials, &dot], words, 1, 1, 1);
                }
                let dv = ctx.alloc(len as usize);
                let dg = ctx.alloc(1);
                let g_t = ctx.node_tensor(g_id);
                let mut words = [0u32; 32];
                words[0] = len;
                ctx.g.dispatch_raw(
                    "t_wn_bwd",
                    &[&v_t, &dw, &dot, &normsq_c, &g_t, &dv, &dg],
                    words,
                    len.div_ceil(256),
                    1,
                    1,
                );
                ctx.accum(v_id, dv);
                ctx.accum(g_id, dg);
            }),
        });
        w
    }

    // ---- ops ----

    /// 線形層: out[m,n] = x[m,k]·W[n,k] + b[n]。
    pub fn linear(&mut self, x: Var, w: Var, b: Option<Var>, m: u32, k: u32, n: u32) -> Var {
        let out_t = self.eng.tensor_uninit((m * n) as usize);
        let out = self.push_node(out_t, vec![m as usize, n as usize], true);
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let w_t = self.t(w);
        let b_t = b.map(|v| self.t(v));
        self.g.matmul_ex(
            &x_t,
            &w_t,
            b_t.as_ref(),
            &out_ref,
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
            0,
        );
        let x_id = x.id;
        let w_id = w.id;
        let b_id = b.map(|v| v.id);
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let w_t = ctx.node_tensor(w_id);
                let x_t = ctx.node_tensor(x_id);
                // dX[m,k] = Σ_n dY[m,n]·W[n,k]
                let dx = ctx.alloc((m * k) as usize);
                ctx.g.matmul_ex(
                    &dy, &w_t, None, &dx, m, k, n, 1, n, 1, 0, 0, 1, k, 0, 0, k, 1, 0, 0, 1.0, 0,
                );
                ctx.accum(x_id, dx);
                // dW[n,k] = Σ_m dY[m,n]·X[m,k]
                let dw = ctx.alloc((n * k) as usize);
                ctx.g.matmul_ex(
                    &dy, &x_t, None, &dw, n, k, m, 1, 1, n, 0, 0, 1, k, 0, 0, k, 1, 0, 0, 1.0, 0,
                );
                ctx.accum(w_id, dw);
                // db[n] = Σ_m dY[m,n]
                if let Some(b_id) = b_id {
                    let db = ctx.alloc(n as usize);
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[0, m, n, 0]);
                    ctx.g.dispatch_raw("t_colsum", &[&dy, &db], words, n, 1, 1);
                    ctx.accum(b_id, db);
                }
            }),
        });
        out
    }

    /// pointwise 活性。op: 0=tanh 1=lrelu 2=gelu 3=sigmoid 4=exp 5=neg 6=relu 7=scale
    pub fn pw(&mut self, x: Var, op: u32, param: f32) -> Var {
        let len = self.numel(x);
        let shape = self.shape(x);
        let out_t = self.eng.tensor_uninit(len as usize);
        let out = self.push_node(out_t, shape, true);
        let out_ref = self.t(out);
        let x_t = self.t(x);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[len, op, param.to_bits(), 0]);
            self.g.dispatch_raw(
                "t_pw_fwd",
                &[&x_t, &out_ref],
                words,
                len.div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let x_t = ctx.node_tensor(x_id);
                let y_t = ctx.node_tensor(out.id);
                let dx = ctx.alloc(len as usize);
                let mut words = [0u32; 32];
                words[0..4].copy_from_slice(&[len, op, param.to_bits(), 0]);
                ctx.g.dispatch_raw(
                    "t_pw_bwd",
                    &[&x_t, &y_t, &dy, &dx],
                    words,
                    len.div_ceil(256),
                    1,
                    1,
                );
                ctx.accum(x_id, dx);
            }),
        });
        out
    }

    pub fn tanh(&mut self, x: Var) -> Var {
        self.pw(x, 0, 0.0)
    }
    pub fn lrelu(&mut self, x: Var, slope: f32) -> Var {
        self.pw(x, 1, slope)
    }
    pub fn gelu(&mut self, x: Var) -> Var {
        self.pw(x, 2, 0.0)
    }
    pub fn sigmoid(&mut self, x: Var) -> Var {
        self.pw(x, 3, 0.0)
    }
    pub fn exp(&mut self, x: Var) -> Var {
        self.pw(x, 4, 0.0)
    }
    pub fn relu(&mut self, x: Var) -> Var {
        self.pw(x, 6, 0.0)
    }

    // ---- conv 系 (im2col + GEMM 統一。backward で im2col を再計算) ----

    /// 1D convolution (zero pad)。レイアウト:
    /// x [batch, in_c, in_len]、w [out_c, in_c*kernel] (PyTorch Conv1d と同一連続配置)、
    /// b [out_c]、out [batch, out_c, out_len]。
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &mut self,
        x: Var,
        w: Var,
        b: Option<Var>,
        out_c: u32,
        in_c: u32,
        kernel: u32,
        stride: u32,
        dilation: u32,
        pad_l: u32,
        batch: u32,
        in_len: u32,
        out_len: u32,
    ) -> Var {
        let ck = in_c * kernel;
        let out_t = self.eng.tensor_uninit((batch * out_c * out_len) as usize);
        let out = self.push_node(
            out_t,
            vec![batch as usize, out_c as usize, out_len as usize],
            true,
        );
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let w_t = self.t(w);
        let b_t = b.map(|v| self.t(v));
        // col [batch, ck, out_len]
        let col = self.eng.tensor_uninit((batch * ck * out_len) as usize);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[batch, in_c, kernel, dilation]);
            words[4..8].copy_from_slice(&[stride, pad_l, in_len, out_len]);
            words[8] = 0; // x_base
            let total = batch * ck * out_len;
            self.g
                .dispatch_raw("t_im2col", &[&x_t, &col], words, total.div_ceil(256), 1, 1);
        }
        // out[b, oc, t] = Σ_ck W[oc, ck] · col[b, ck, t]
        self.g.matmul_ex(
            &w_t,
            &col,
            b_t.as_ref(),
            &out_ref,
            out_c,
            out_len,
            ck,
            batch,
            ck,
            1,
            0,
            0,
            1,
            out_len,
            0,
            ck * out_len,
            out_len,
            1,
            0,
            out_c * out_len,
            1.0,
            1,
        );
        let x_id = x.id;
        let w_id = w.id;
        let b_id = b.map(|v| v.id);
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let x_t = ctx.node_tensor(x_id);
                let w_t = ctx.node_tensor(w_id);
                // dcol[b, ck, t] = Σ_oc W[oc, ck] · dy[b, oc, t]
                let dcol = ctx.alloc((batch * ck * out_len) as usize);
                ctx.g.matmul_ex(
                    &w_t,
                    &dy,
                    None,
                    &dcol,
                    ck,
                    out_len,
                    out_c,
                    batch,
                    1,
                    ck,
                    0,
                    0,
                    1,
                    out_len,
                    0,
                    out_c * out_len,
                    out_len,
                    1,
                    0,
                    ck * out_len,
                    1.0,
                    0,
                );
                // dX[b, c, pos] = col2im(dcol)
                let dx = ctx.alloc((batch * in_c * in_len) as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[batch, in_c, kernel, dilation]);
                    words[4..8].copy_from_slice(&[stride, pad_l, in_len, out_len]);
                    ctx.g.dispatch_raw(
                        "t_col2im",
                        &[&dx, &dcol],
                        words,
                        (batch * in_c * in_len).div_ceil(256),
                        1,
                        1,
                    );
                }
                ctx.accum(x_id, dx);
                // dW = Σ_b dy · colᵀ (col を再計算)
                let col = ctx.alloc((batch * ck * out_len) as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[batch, in_c, kernel, dilation]);
                    words[4..8].copy_from_slice(&[stride, pad_l, in_len, out_len]);
                    words[8] = 0;
                    ctx.g.dispatch_raw(
                        "t_im2col",
                        &[&x_t, &col],
                        words,
                        (batch * ck * out_len).div_ceil(256),
                        1,
                        1,
                    );
                }
                let dw_partial = ctx.alloc((batch * out_c * ck) as usize);
                ctx.g.matmul_ex(
                    &dy,
                    &col,
                    None,
                    &dw_partial,
                    out_c,
                    ck,
                    out_len,
                    batch,
                    out_len,
                    1,
                    0,
                    out_c * out_len,
                    out_len,
                    1,
                    0,
                    ck * out_len,
                    ck,
                    1,
                    0,
                    out_c * ck,
                    1.0,
                    0,
                );
                let dw = ctx.alloc((out_c * ck) as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[0, batch, out_c * ck, 0]);
                    ctx.g
                        .dispatch_raw("t_colsum", &[&dw_partial, &dw], words, out_c * ck, 1, 1);
                }
                ctx.accum(w_id, dw);
                // db[oc] = Σ_{b,t} dy
                if let Some(b_id) = b_id {
                    let bt_oc = ctx.alloc((batch * out_c) as usize);
                    {
                        let mut words = [0u32; 32];
                        words[0..4].copy_from_slice(&[1, batch * out_c, out_len, 0]);
                        ctx.g
                            .dispatch_raw("t_colsum", &[&dy, &bt_oc], words, batch * out_c, 1, 1);
                    }
                    let db = ctx.alloc(out_c as usize);
                    {
                        let mut words = [0u32; 32];
                        words[0..4].copy_from_slice(&[0, batch, out_c, 0]);
                        ctx.g
                            .dispatch_raw("t_colsum", &[&bt_oc, &db], words, out_c, 1, 1);
                    }
                    ctx.accum(b_id, db);
                }
            }),
        });
        out
    }

    /// 1D 転置畳み込み。レイアウト:
    /// x [batch, in_c, in_len]、w [in_c, out_c*kernel] (PyTorch ConvTranspose1d と同一)、
    /// b [out_c]、out [batch, out_c, out_len]。
    /// out_len = (in_len-1)*stride + dilation*(kernel-1) + 1 - pad_l - pad_r 相当
    /// (pad_r 分は col2im が自然に落とす)。
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d(
        &mut self,
        x: Var,
        w: Var,
        b: Option<Var>,
        out_c: u32,
        in_c: u32,
        kernel: u32,
        stride: u32,
        dilation: u32,
        pad_l: u32,
        batch: u32,
        in_len: u32,
        out_len: u32,
    ) -> Var {
        let ok_dim = out_c * kernel;
        let out_t = self.eng.tensor_uninit((batch * out_c * out_len) as usize);
        let out = self.push_node(
            out_t,
            vec![batch as usize, out_c as usize, out_len as usize],
            true,
        );
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let w_t = self.t(w);
        let b_t = b.map(|v| self.t(v));
        // M[b, (oc,j), t] = Σ_ci W[ci, (oc,j)] · x[b, ci, t]
        let m_buf = self.eng.tensor_uninit((batch * ok_dim * in_len) as usize);
        self.g.matmul_ex(
            &w_t,
            &x_t,
            None,
            &m_buf,
            ok_dim,
            in_len,
            in_c,
            batch,
            1,
            ok_dim,
            0,
            0,
            1,
            in_len,
            0,
            in_c * in_len,
            in_len,
            1,
            0,
            ok_dim * in_len,
            1.0,
            0,
        );
        // out[b, oc, p] = col2im(M)
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[batch, out_c, kernel, dilation]);
            words[4..8].copy_from_slice(&[stride, pad_l, out_len, in_len]);
            self.g.dispatch_raw(
                "t_col2im",
                &[&out_ref, &m_buf],
                words,
                (batch * out_c * out_len).div_ceil(256),
                1,
                1,
            );
        }
        if let Some(bias) = b_t.as_ref() {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[batch * out_c * out_len, out_c, out_len, 0]);
            self.g.dispatch_raw(
                "t_add_ch_bias",
                &[&out_ref, bias],
                words,
                (batch * out_c * out_len).div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        let w_id = w.id;
        let b_id = b.map(|v| v.id);
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let x_t = ctx.node_tensor(x_id);
                let w_t = ctx.node_tensor(w_id);
                // dM[b, (oc,j), t] = im2col(dy)
                let dm = ctx.alloc((batch * ok_dim * in_len) as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[batch, out_c, kernel, dilation]);
                    words[4..8].copy_from_slice(&[stride, pad_l, out_len, in_len]);
                    ctx.g.dispatch_raw(
                        "t_im2col",
                        &[&dy, &dm],
                        words,
                        (batch * ok_dim * in_len).div_ceil(256),
                        1,
                        1,
                    );
                }
                // dX[b, ci, t] = Σ_(oc,j) W[ci, (oc,j)] · dM[b, (oc,j), t]
                let dx = ctx.alloc((batch * in_c * in_len) as usize);
                ctx.g.matmul_ex(
                    &w_t,
                    &dm,
                    None,
                    &dx,
                    in_c,
                    in_len,
                    ok_dim,
                    batch,
                    ok_dim,
                    1,
                    0,
                    0,
                    1,
                    in_len,
                    0,
                    ok_dim * in_len,
                    in_len,
                    1,
                    0,
                    in_c * in_len,
                    1.0,
                    0,
                );
                ctx.accum(x_id, dx);
                // dW_partial[b, ci, (oc,j)] = Σ_t x[b, ci, t] · dM[b, (oc,j), t]
                let dw_partial = ctx.alloc((batch * in_c * ok_dim) as usize);
                ctx.g.matmul_ex(
                    &x_t,
                    &dm,
                    None,
                    &dw_partial,
                    in_c,
                    ok_dim,
                    in_len,
                    batch,
                    in_len,
                    1,
                    0,
                    in_c * in_len,
                    in_len,
                    1,
                    0,
                    ok_dim * in_len,
                    ok_dim,
                    1,
                    0,
                    in_c * ok_dim,
                    1.0,
                    0,
                );
                let dw = ctx.alloc((in_c * ok_dim) as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[0, batch, in_c * ok_dim, 0]);
                    ctx.g
                        .dispatch_raw("t_colsum", &[&dw_partial, &dw], words, in_c * ok_dim, 1, 1);
                }
                ctx.accum(w_id, dw);
                // db[oc] = Σ_{b,p} dy
                if let Some(b_id) = b_id {
                    let bp_oc = ctx.alloc((batch * out_c) as usize);
                    {
                        let mut words = [0u32; 32];
                        words[0..4].copy_from_slice(&[1, batch * out_c, out_len, 0]);
                        ctx.g
                            .dispatch_raw("t_colsum", &[&dy, &bp_oc], words, batch * out_c, 1, 1);
                    }
                    let db = ctx.alloc(out_c as usize);
                    {
                        let mut words = [0u32; 32];
                        words[0..4].copy_from_slice(&[0, batch, out_c, 0]);
                        ctx.g
                            .dispatch_raw("t_colsum", &[&bp_oc, &db], words, out_c, 1, 1);
                    }
                    ctx.accum(b_id, db);
                }
            }),
        });
        out
    }

    /// out = a ± b (要素ごと、同一 shape)。
    pub fn add(&mut self, a: Var, b: Var, subtract: bool) -> Var {
        let len = self.numel(a);
        let shape = self.shape(a);
        let out_t = self.eng.tensor_uninit(len as usize);
        let out = self.push_node(out_t, shape, true);
        let out_ref = self.t(out);
        let a_t = self.t(a);
        let b_t = self.t(b);
        self.g
            .add_off3(&a_t, &b_t, &out_ref, len, 0, 0, 0, subtract);
        let a_id = a.id;
        let b_id = b.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let da = ctx.alloc(len as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[len, 0, 0, 0]);
                    ctx.g
                        .dispatch_raw("t_copy", &[&dy, &da], words, len.div_ceil(256), 1, 1);
                }
                ctx.accum(a_id, da);
                let db = if subtract {
                    let neg = ctx.alloc(len as usize);
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[len, 5, 0.0f32.to_bits(), 0]);
                    ctx.g
                        .dispatch_raw("t_pw_fwd", &[&dy, &neg], words, len.div_ceil(256), 1, 1);
                    neg
                } else {
                    let c = ctx.alloc(len as usize);
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[len, 0, 0, 0]);
                    ctx.g
                        .dispatch_raw("t_copy", &[&dy, &c], words, len.div_ceil(256), 1, 1);
                    c
                };
                ctx.accum(b_id, db);
            }),
        });
        out
    }

    /// out = a ⊙ b (要素ごと)。
    pub fn mul2(&mut self, a: Var, b: Var) -> Var {
        let len = self.numel(a);
        let shape = self.shape(a);
        let out_t = self.eng.tensor_uninit(len as usize);
        let out = self.push_node(out_t, shape, true);
        let out_ref = self.t(out);
        let a_t = self.t(a);
        let b_t = self.t(b);
        {
            let mut words = [0u32; 32];
            words[0] = len;
            self.g.dispatch_raw(
                "t_mul2",
                &[&a_t, &b_t, &out_ref],
                words,
                len.div_ceil(256),
                1,
                1,
            );
        }
        let a_id = a.id;
        let b_id = b.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let a_t = ctx.node_tensor(a_id);
                let b_t = ctx.node_tensor(b_id);
                let da = ctx.alloc(len as usize);
                {
                    let mut words = [0u32; 32];
                    words[0] = len;
                    ctx.g
                        .dispatch_raw("t_mul2", &[&dy, &b_t, &da], words, len.div_ceil(256), 1, 1);
                }
                ctx.accum(a_id, da);
                let db = ctx.alloc(len as usize);
                {
                    let mut words = [0u32; 32];
                    words[0] = len;
                    ctx.g
                        .dispatch_raw("t_mul2", &[&dy, &a_t, &db], words, len.div_ceil(256), 1, 1);
                }
                ctx.accum(b_id, db);
            }),
        });
        out
    }

    /// row softmax (行ごと、rows × cols)。保存活性 = 出力。
    pub fn softmax(&mut self, x: Var, rows: u32, cols: u32) -> Var {
        let len = self.numel(x);
        let out_t = self.eng.tensor_uninit(len as usize);
        let out = self.push_node(out_t, vec![rows as usize, cols as usize], true);
        let out_ref = self.t(out);
        let x_t = self.t(x);
        {
            // in-place softmax カーネルに渡すため x を out へコピー
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[len, 0, 0, 0]);
            self.g
                .dispatch_raw("t_copy", &[&x_t, &out_ref], words, len.div_ceil(256), 1, 1);
        }
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[rows, cols, 0, 0]);
            self.g
                .dispatch_raw("softmax", &[&out_ref], words, rows, 1, 1);
        }
        let x_id = x.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let y_t = ctx.node_tensor(out.id);
                let dx = ctx.alloc(len as usize);
                let mut words = [0u32; 32];
                words[0..4].copy_from_slice(&[rows, cols, 0, 0]);
                ctx.g
                    .dispatch_raw("t_softmax_bwd", &[&y_t, &dy, &dx], words, rows, 1, 1);
                ctx.accum(x_id, dx);
            }),
        });
        out
    }

    /// LayerNorm (最終軸)。gamma/beta は 1 次元パラメータ。
    pub fn layernorm(
        &mut self,
        x: Var,
        gamma: Var,
        beta: Var,
        rows: u32,
        cols: u32,
        eps: f32,
    ) -> Var {
        self.layernorm_strided(x, gamma, beta, rows, cols, cols, 1, eps)
    }

    /// stride 指定版。x/out は (in_row, in_k) ストライド ([B,C,T] の
    /// チャネル軸 LN は rows=B·T, cols=C, in_row=1, in_k=T)。
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm_strided(
        &mut self,
        x: Var,
        gamma: Var,
        beta: Var,
        rows: u32,
        cols: u32,
        in_row: u32,
        in_k: u32,
        eps: f32,
    ) -> Var {
        let len = self.numel(x);
        let out_t = self.eng.tensor_uninit(len as usize);
        let out = self.push_node(out_t, vec![rows as usize, cols as usize], true);
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let gamma_t = self.t(gamma);
        let beta_t = self.t(beta);
        self.g.layer_norm(
            &x_t, &gamma_t, &beta_t, &out_ref, rows, cols, in_row, in_k, in_row, in_k, eps,
        );
        let x_id = x.id;
        let gamma_id = gamma.id;
        let beta_id = beta.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let x_t = ctx.node_tensor(x_id);
                let gamma_t = ctx.node_tensor(gamma_id);
                let dx = ctx.alloc(len as usize);
                let xhat = ctx.alloc(len as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[rows, cols, eps.to_bits(), in_row]);
                    words[4] = in_k;
                    ctx.g.dispatch_raw(
                        "t_layernorm_bwd",
                        &[&x_t, &dy, &gamma_t, &dx, &xhat],
                        words,
                        rows,
                        1,
                        1,
                    );
                }
                ctx.accum(x_id, dx);
                let dgamma = ctx.alloc(cols as usize);
                let dbeta = ctx.alloc(cols as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[cols, rows, in_row, in_k]);
                    ctx.g.dispatch_raw(
                        "t_ln_param_grad",
                        &[&xhat, &dy, &dgamma, &dbeta],
                        words,
                        cols,
                        1,
                        1,
                    );
                }
                ctx.accum(gamma_id, dgamma);
                ctx.accum(beta_id, dbeta);
            }),
        });
        out
    }

    // ---- 汎用 strided matmul / 補助 op (M-T2) ----

    /// 汎用 batched matmul: C[b, m, n] = Σ_k A[b, m, k]·W[b, n, k] (+ bias[n])。
    /// 全オペランドは (row_stride, k_stride, base, batch_stride) で任意レイアウト
    /// (matmul_ex と同一規約。W の batch_stride=0 で全バッチ共有)。
    /// out は out_len 要素の新規バッファ。
    #[allow(clippy::too_many_arguments)]
    pub fn mm(
        &mut self,
        a: Var,
        w: Var,
        bias: Option<Var>,
        out_len: usize,
        m: u32,
        n: u32,
        k: u32,
        batch: u32,
        a_row: u32,
        a_k: u32,
        a_base: u32,
        a_bs: u32,
        w_row: u32,
        w_k: u32,
        w_base: u32,
        w_bs: u32,
        c_row: u32,
        c_n: u32,
        c_bs: u32,
        scale: f32,
    ) -> Var {
        let out_t = self.eng.tensor_uninit(out_len);
        let out = self.push_node(out_t, vec![batch as usize, m as usize, n as usize], true);
        let out_ref = self.t(out);
        let a_t = self.t(a);
        let w_t = self.t(w);
        let b_t = bias.map(|v| self.t(v));
        if std::env::var("MM_DEBUG").is_ok() {
            eprintln!(
                "[mm] m={m} n={n} k={k} b={batch} a=({a_row},{a_k},{a_base},{a_bs}) w=({w_row},{w_k},{w_base},{w_bs}) c=({c_row},{c_n},0,{c_bs}) alen={} wlen={} clen={} bias={}",
                a_t.len, w_t.len, out_ref.len, b_t.is_some()
            );
        }
        self.g.matmul_ex(
            &a_t,
            &w_t,
            b_t.as_ref(),
            &out_ref,
            m,
            n,
            k,
            batch,
            a_row,
            a_k,
            a_base,
            a_bs,
            w_row,
            w_k,
            w_base,
            w_bs,
            c_row,
            c_n,
            0,
            c_bs,
            scale,
            0,
        );
        let a_id = a.id;
        let w_id = w.id;
        let b_id = bias.map(|v| v.id);
        let a_len = self.numel(a);
        let w_len = self.numel(w);
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let w_t = ctx.node_tensor(w_id);
                let a_t = ctx.node_tensor(a_id);
                // dA = dy · W (A のレイアウトに書き出し)。scale は dy 側に織り込む。
                let dy_eff;
                let dy_ref: &GpuTensor = if scale != 1.0 {
                    let s = ctx.alloc(dy.len as usize);
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[dy.len as u32, 7, scale.to_bits(), 0]);
                    ctx.g.dispatch_raw(
                        "t_pw_fwd",
                        &[&dy, &s],
                        words,
                        (dy.len as u32).div_ceil(256),
                        1,
                        1,
                    );
                    dy_eff = s;
                    &dy_eff
                } else {
                    &dy
                };
                let da = ctx.alloc(a_len as usize);
                ctx.g.matmul_ex(
                    dy_ref, &w_t, None, &da, m, k, n, batch, c_row, c_n, 0, c_bs, w_k, w_row,
                    w_base, w_bs, a_row, a_k, 0, a_bs, 1.0, 0,
                );
                ctx.accum(a_id, da);
                let dw = ctx.alloc(w_len as usize);
                ctx.g.matmul_ex(
                    dy_ref, &a_t, None, &dw, n, k, m, batch, c_n, c_row, 0, c_bs, a_k, a_row,
                    a_base, a_bs, w_row, w_k, 0, w_bs, 1.0, 0,
                );
                ctx.accum(w_id, dw);
                if let Some(b_id) = b_id {
                    // bias[n] = Σ_{b,m} dy (out が [b, m, n] 連続前提)
                    debug_assert!(c_row == n && c_n == 1);
                    let bm = ctx.alloc((batch * m) as usize);
                    {
                        let mut words = [0u32; 32];
                        words[0..4].copy_from_slice(&[1, batch * m, n, 0]);
                        ctx.g
                            .dispatch_raw("t_colsum", &[dy_ref, &bm], words, batch * m, 1, 1);
                    }
                    let db = ctx.alloc(n as usize);
                    {
                        let mut words = [0u32; 32];
                        words[0..4].copy_from_slice(&[0, batch, n, 0]);
                        ctx.g.dispatch_raw("t_colsum", &[&bm, &db], words, n, 1, 1);
                    }
                    ctx.accum(b_id, db);
                }
            }),
        });
        out
    }

    /// gating (WN): x [b, 2h, t] → tanh(a)·sigmoid(b) [b, h, t]。
    pub fn gating(&mut self, x: Var, batch: u32, h: u32, t_len: u32) -> Var {
        let out_t = self.eng.tensor_uninit((batch * h * t_len) as usize);
        let out = self.push_node(
            out_t,
            vec![batch as usize, h as usize, t_len as usize],
            true,
        );
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let total = batch * h * t_len;
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[batch, h, t_len, 0]);
            self.g.dispatch_raw(
                "t_gating",
                &[&x_t, &out_ref],
                words,
                total.div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        let x_len = self.numel(x);
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let x_t = ctx.node_tensor(x_id);
                let dx = ctx.alloc(x_len as usize);
                let mut words = [0u32; 32];
                words[0..4].copy_from_slice(&[batch, h, t_len, 0]);
                ctx.g.dispatch_raw(
                    "t_gating_bwd",
                    &[&x_t, &dy, &dx],
                    words,
                    total.div_ceil(256),
                    1,
                    1,
                );
                ctx.accum(x_id, dx);
            }),
        });
        out
    }

    /// rel-pos skew。mode 0: [bh, t, 2ws+1] → [bh, t, t]。mode 1: [bh, t, t] → [bh, t, 2ws+1]。
    pub fn skew(&mut self, x: Var, mode: u32, batch: u32, t_len: u32, ws: u32) -> Var {
        let other = if mode == 0 { t_len } else { 2 * ws + 1 };
        let out_t = self.eng.tensor_uninit((batch * t_len * other) as usize);
        let out = self.push_node(
            out_t,
            vec![batch as usize, t_len as usize, other as usize],
            true,
        );
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let total = batch * t_len * other;
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[mode, batch, t_len, ws]);
            self.g.dispatch_raw(
                "t_skew",
                &[&x_t, &out_ref],
                words,
                total.div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        let x_len = self.numel(x);
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let dx = ctx.alloc(x_len as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[x_len, 0, 0.0f32.to_bits(), 0]);
                    ctx.g
                        .dispatch_raw("t_fill", &[&dx], words, x_len.div_ceil(256), 1, 1);
                }
                let mut words = [0u32; 32];
                words[0..4].copy_from_slice(&[mode, batch, t_len, ws]);
                ctx.g
                    .dispatch_raw("t_skew_bwd", &[&dy, &dx], words, total.div_ceil(256), 1, 1);
                ctx.accum(x_id, dx);
            }),
        });
        out
    }

    /// チャネルスライス: in [b, c_in, t] の [c0, c0+c_out) を新規テンソル化。
    pub fn chslice(
        &mut self,
        x: Var,
        c_in: u32,
        c0: u32,
        c_out: u32,
        batch: u32,
        t_len: u32,
    ) -> Var {
        let out_t = self.eng.tensor_uninit((batch * c_out * t_len) as usize);
        let out = self.push_node(
            out_t,
            vec![batch as usize, c_out as usize, t_len as usize],
            true,
        );
        let out_ref = self.t(out);
        let x_t = self.t(x);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[batch, c_in, t_len, c_out]);
            words[4] = c0;
            self.g.dispatch_raw(
                "t_chslice",
                &[&x_t, &out_ref],
                words,
                (batch * c_out * t_len).div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        let x_len = self.numel(x);
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let dx = ctx.alloc(x_len as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[x_len, 0, 0.0f32.to_bits(), 0]);
                    ctx.g
                        .dispatch_raw("t_fill", &[&dx], words, x_len.div_ceil(256), 1, 1);
                }
                let mut words = [0u32; 32];
                words[0..4].copy_from_slice(&[batch, c_in, t_len, c_out]);
                words[4] = c0;
                ctx.g.dispatch_raw(
                    "t_chslice_bwd",
                    &[&dy, &dx],
                    words,
                    (batch * c_out * t_len).div_ceil(256),
                    1,
                    1,
                );
                ctx.accum(x_id, dx);
            }),
        });
        out
    }

    /// ランダムセグメント抽出: in [b, ch, t_len] から out [b, ch, seg]。
    /// start は [batch] の f32 バッファ (start_scale を掛けて丸めたオフセット)。
    /// 逆伝達は in へ scatter-add (書き込み先は要素ごとに一意)。
    pub fn gather_frames(
        &mut self,
        x: Var,
        start: &GpuTensor,
        ch: u32,
        t_len: u32,
        seg: u32,
        start_scale: f32,
    ) -> Var {
        let batch = self.shape(x)[0] as u32;
        let out_t = self.eng.tensor_uninit((batch * ch * seg) as usize);
        let out = self.push_node(out_t, vec![batch as usize, ch as usize, seg as usize], true);
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let total = batch * ch * seg;
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[batch, ch, t_len, seg]);
            words[4] = start_scale.to_bits();
            self.g.dispatch_raw(
                "t_gather_frames",
                &[start, &x_t, &out_ref],
                words,
                total.div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        let x_len = self.numel(x);
        let start_c = start.clone_handle();
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let dx = ctx.alloc(x_len as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[x_len, 0, 0.0f32.to_bits(), 0]);
                    ctx.g
                        .dispatch_raw("t_fill", &[&dx], words, x_len.div_ceil(256), 1, 1);
                }
                let mut words = [0u32; 32];
                words[0..4].copy_from_slice(&[batch, ch, t_len, seg]);
                words[4] = start_scale.to_bits();
                ctx.g.dispatch_raw(
                    "t_gather_frames_bwd",
                    &[&start_c, &dy, &dx],
                    words,
                    total.div_ceil(256),
                    1,
                    1,
                );
                ctx.accum(x_id, dx);
            }),
        });
        out
    }

    /// チャネル bias 加算: out [b, ch, t] = x + bias[ch]。
    pub fn add_ch_bias(&mut self, x: Var, bias: Var, ch: u32, t_len: u32) -> Var {
        let batch = self.shape(x)[0] as u32;
        let total = batch * ch * t_len;
        let out_t = self.eng.tensor_uninit(total as usize);
        let out = self.push_node(out_t, self.shape(x), true);
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let b_t = self.t(bias);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[total, 0, 0, 0]);
            self.g.dispatch_raw(
                "t_copy",
                &[&x_t, &out_ref],
                words,
                total.div_ceil(256),
                1,
                1,
            );
        }
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[total, ch, t_len, 0]);
            self.g.dispatch_raw(
                "t_add_ch_bias",
                &[&out_ref, &b_t],
                words,
                total.div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        let b_id = bias.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let da = ctx.alloc(total as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[total, 0, 0, 0]);
                    ctx.g
                        .dispatch_raw("t_copy", &[&dy, &da], words, total.div_ceil(256), 1, 1);
                }
                ctx.accum(x_id, da);
                let bt = ctx.alloc((batch * ch) as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[1, batch * ch, t_len, 0]);
                    ctx.g
                        .dispatch_raw("t_colsum", &[&dy, &bt], words, batch * ch, 1, 1);
                }
                let db = ctx.alloc(ch as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[0, batch, ch, 0]);
                    ctx.g.dispatch_raw("t_colsum", &[&bt, &db], words, ch, 1, 1);
                }
                ctx.accum(b_id, db);
            }),
        });
        out
    }

    /// フレームマスク乗算: out[b, c, t] = x[b, c, t]·mask[b, t]。mask は定数。
    pub fn mul_frame_mask(&mut self, x: Var, mask: Var, ch: u32, t_len: u32) -> Var {
        let batch = self.shape(x)[0] as u32;
        let total = batch * ch * t_len;
        let out_t = self.eng.tensor_uninit(total as usize);
        let out = self.push_node(out_t, self.shape(x), true);
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let m_t = self.t(mask);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[batch, ch, t_len, 0]);
            self.g.dispatch_raw(
                "t_mul_frame_mask",
                &[&x_t, &m_t, &out_ref],
                words,
                total.div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        let m_id = mask.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let m_t = ctx.node_tensor(m_id);
                let dx = ctx.alloc(total as usize);
                let mut words = [0u32; 32];
                words[0..4].copy_from_slice(&[batch, ch, t_len, 0]);
                ctx.g.dispatch_raw(
                    "t_mul_frame_mask",
                    &[&dy, &m_t, &dx],
                    words,
                    total.div_ceil(256),
                    1,
                    1,
                );
                ctx.accum(x_id, dx);
            }),
        });
        out
    }

    /// 転置: [b, t, c] → [b, c, t]。
    pub fn transpose_tc_ct(&mut self, x: Var, t_len: u32, c: u32) -> Var {
        let batch = self.shape(x)[0] as u32;
        let total = batch * t_len * c;
        let out_t = self.eng.tensor_uninit(total as usize);
        let out = self.push_node(
            out_t,
            vec![batch as usize, c as usize, t_len as usize],
            true,
        );
        let out_ref = self.t(out);
        let x_t = self.t(x);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[batch, t_len, c, 0]);
            self.g.dispatch_raw(
                "t_transpose",
                &[&x_t, &out_ref],
                words,
                total.div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let dx = ctx.alloc(total as usize);
                let mut words = [0u32; 32];
                words[0..4].copy_from_slice(&[batch, t_len, c, 0]);
                ctx.g
                    .dispatch_raw("t_transpose", &[&dy, &dx], words, total.div_ceil(256), 1, 1);
                ctx.accum(x_id, dx);
            }),
        });
        out
    }

    /// 1D reflect pad: x [b, c, in_len] → [b, c, in_len + 2·pad]。
    pub fn reflect_pad(&mut self, x: Var, pad: u32) -> Var {
        let (batch, c, in_len) = {
            let s = self.shape(x);
            (s[0] as u32, s[1] as u32, s[2] as u32)
        };
        let out_len = in_len + 2 * pad;
        let out_t = self.eng.tensor_uninit((batch * c * out_len) as usize);
        let out = self.push_node(
            out_t,
            vec![batch as usize, c as usize, out_len as usize],
            true,
        );
        let out_ref = self.t(out);
        let x_t = self.t(x);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[batch, c, in_len, pad]);
            self.g.dispatch_raw(
                "t_reflect_pad",
                &[&x_t, &out_ref],
                words,
                (batch * c * out_len).div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let dx = ctx.alloc((batch * c * in_len) as usize);
                let mut words = [0u32; 32];
                words[0..4].copy_from_slice(&[batch, c, in_len, pad]);
                ctx.g.dispatch_raw(
                    "t_reflect_pad_bwd",
                    &[&dy, &dx],
                    words,
                    (batch * c * in_len).div_ceil(256),
                    1,
                    1,
                );
                ctx.accum(x_id, dx);
            }),
        });
        out
    }

    /// grouped 1D convolution (DiscriminatorS 用)。groups > 1 は各グループの
    /// im2col + GEMM を順次 dispatch し 1 つの out へ書き込む。
    /// w は [out_c, (in_c/groups)·kernel]、out [batch, out_c, out_len]。
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_grouped(
        &mut self,
        x: Var,
        w: Var,
        b: Option<Var>,
        out_c: u32,
        in_c: u32,
        kernel: u32,
        stride: u32,
        pad_l: u32,
        groups: u32,
        batch: u32,
        in_len: u32,
        out_len: u32,
    ) -> Var {
        if groups == 1 {
            return self.conv1d(
                x, w, b, out_c, in_c, kernel, stride, 1, pad_l, batch, in_len, out_len,
            );
        }
        let cin_g = in_c / groups;
        let cout_g = out_c / groups;
        let ck = cin_g * kernel;
        let x_total = batch * in_c * in_len;
        let out_t = self.eng.tensor_uninit((batch * out_c * out_len) as usize);
        let out = self.push_node(
            out_t,
            vec![batch as usize, out_c as usize, out_len as usize],
            true,
        );
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let w_t = self.t(w);
        let b_t = b.map(|v| self.t(v));
        let col = self.eng.tensor_uninit((batch * ck * out_len) as usize);
        for g in 0..groups {
            {
                let mut words = [0u32; 32];
                words[0..4].copy_from_slice(&[batch, cin_g, kernel, 1]);
                words[4..8].copy_from_slice(&[stride, pad_l, in_len, out_len]);
                words[8] = g * cin_g * in_len; // x_base
                self.g.dispatch_raw(
                    "t_im2col",
                    &[&x_t, &col],
                    words,
                    (batch * ck * out_len).div_ceil(256),
                    1,
                    1,
                );
            }
            let w_base = g * cout_g * ck;
            let c_base = g * cout_g * out_len;
            self.g.matmul_ex(
                &w_t,
                &col,
                None,
                &out_ref,
                cout_g,
                out_len,
                ck,
                batch,
                ck,
                1,
                w_base,
                0,
                1,
                out_len,
                0,
                0,
                out_len,
                1,
                c_base,
                out_c * out_len,
                1.0,
                0,
            );
        }
        if let Some(bias) = b_t.as_ref() {
            let total = batch * out_c * out_len;
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[total, out_c, out_len, 0]);
            self.g.dispatch_raw(
                "t_add_ch_bias",
                &[&out_ref, bias],
                words,
                total.div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        let w_id = w.id;
        let b_id = b.map(|v| v.id);
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let x_t = ctx.node_tensor(x_id);
                let w_t = ctx.node_tensor(w_id);
                let dx = ctx.alloc(x_total as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[x_total, 0, 0.0f32.to_bits(), 0]);
                    ctx.g
                        .dispatch_raw("t_fill", &[&dx], words, x_total.div_ceil(256), 1, 1);
                }
                let dw = ctx.alloc(w_t.len);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[w_t.len as u32, 0, 0.0f32.to_bits(), 0]);
                    ctx.g.dispatch_raw(
                        "t_fill",
                        &[&dw],
                        words,
                        (w_t.len as u32).div_ceil(256),
                        1,
                        1,
                    );
                }
                for g in 0..groups {
                    let w_base = g * cout_g * ck;
                    let c_base = g * cout_g * out_len;
                    // dcol = W_gᵀ · dy_g
                    let dcol = ctx.alloc((batch * ck * out_len) as usize);
                    ctx.g.matmul_ex(
                        &w_t,
                        &dy,
                        None,
                        &dcol,
                        ck,
                        out_len,
                        cout_g,
                        batch,
                        1,
                        ck,
                        w_base,
                        0,
                        1,
                        out_len,
                        c_base,
                        out_c * out_len,
                        out_len,
                        1,
                        0,
                        ck * out_len,
                        1.0,
                        0,
                    );
                    // dx_g = col2im(dcol) → dx スライスへ加算
                    let dx_g = ctx.alloc((batch * cin_g * in_len) as usize);
                    {
                        let mut words = [0u32; 32];
                        words[0..4].copy_from_slice(&[batch, cin_g, kernel, 1]);
                        words[4..8].copy_from_slice(&[stride, pad_l, in_len, out_len]);
                        ctx.g.dispatch_raw(
                            "t_col2im",
                            &[&dx_g, &dcol],
                            words,
                            (batch * cin_g * in_len).div_ceil(256),
                            1,
                            1,
                        );
                        for b in 0..batch {
                            let base = (b * in_c + g * cin_g) * in_len;
                            ctx.g.add_off3(
                                &dx,
                                &dx_g,
                                &dx,
                                cin_g * in_len,
                                base,
                                b * cin_g * in_len,
                                base,
                                false,
                            );
                        }
                    }
                    // col 再計算 → dW_g
                    let col = ctx.alloc((batch * ck * out_len) as usize);
                    {
                        let mut words = [0u32; 32];
                        words[0..4].copy_from_slice(&[batch, cin_g, kernel, 1]);
                        words[4..8].copy_from_slice(&[stride, pad_l, in_len, out_len]);
                        words[8] = g * cin_g * in_len;
                        ctx.g.dispatch_raw(
                            "t_im2col",
                            &[&x_t, &col],
                            words,
                            (batch * ck * out_len).div_ceil(256),
                            1,
                            1,
                        );
                    }
                    let dwp = ctx.alloc((batch * cout_g * ck) as usize);
                    ctx.g.matmul_ex(
                        &dy,
                        &col,
                        None,
                        &dwp,
                        cout_g,
                        ck,
                        out_len,
                        batch,
                        out_len,
                        1,
                        c_base,
                        out_c * out_len,
                        out_len,
                        1,
                        0,
                        ck * out_len,
                        ck,
                        1,
                        0,
                        cout_g * ck,
                        1.0,
                        0,
                    );
                    let dw_g = ctx.alloc((cout_g * ck) as usize);
                    {
                        let mut words = [0u32; 32];
                        words[0..4].copy_from_slice(&[0, batch, cout_g * ck, 0]);
                        ctx.g
                            .dispatch_raw("t_colsum", &[&dwp, &dw_g], words, cout_g * ck, 1, 1);
                        ctx.g.add_off3(
                            &dw,
                            &dw_g,
                            &dw,
                            cout_g * ck,
                            g * cout_g * ck,
                            0,
                            g * cout_g * ck,
                            false,
                        );
                    }
                }
                ctx.accum(x_id, dx);
                ctx.accum(w_id, dw);
                if let Some(b_id) = b_id {
                    let bt = ctx.alloc((batch * out_c) as usize);
                    {
                        let mut words = [0u32; 32];
                        words[0..4].copy_from_slice(&[1, batch * out_c, out_len, 0]);
                        ctx.g
                            .dispatch_raw("t_colsum", &[&dy, &bt], words, batch * out_c, 1, 1);
                    }
                    let db = ctx.alloc(out_c as usize);
                    {
                        let mut words = [0u32; 32];
                        words[0..4].copy_from_slice(&[0, batch, out_c, 0]);
                        ctx.g
                            .dispatch_raw("t_colsum", &[&bt, &db], words, out_c, 1, 1);
                    }
                    ctx.accum(b_id, db);
                }
            }),
        });
        out
    }

    /// チャネル連結: out [b, ca+cb, t] = a ⊕ b。
    pub fn chcat(&mut self, a: Var, b: Var, ca: u32, cb: u32, batch: u32, t_len: u32) -> Var {
        let out_c = ca + cb;
        let out_t = self.eng.tensor_uninit((batch * out_c * t_len) as usize);
        let out = self.push_node(
            out_t,
            vec![batch as usize, out_c as usize, t_len as usize],
            true,
        );
        let out_ref = self.t(out);
        let a_t = self.t(a);
        let b_t = self.t(b);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[batch, ca, cb, t_len]);
            self.g.dispatch_raw(
                "t_chcat",
                &[&a_t, &b_t, &out_ref],
                words,
                (batch * out_c * t_len).div_ceil(256),
                1,
                1,
            );
        }
        let a_id = a.id;
        let b_id = b.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                // d_a = chslice(dy, 0..ca), d_b = chslice(dy, ca..)
                let da = ctx.alloc((batch * ca * t_len) as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[batch, out_c, t_len, ca]);
                    ctx.g.dispatch_raw(
                        "t_chslice",
                        &[&dy, &da],
                        words,
                        (batch * ca * t_len).div_ceil(256),
                        1,
                        1,
                    );
                }
                ctx.accum(a_id, da);
                let db = ctx.alloc((batch * cb * t_len) as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[batch, out_c, t_len, cb]);
                    words[4] = ca;
                    ctx.g.dispatch_raw(
                        "t_chslice",
                        &[&dy, &db],
                        words,
                        (batch * cb * t_len).div_ceil(256),
                        1,
                        1,
                    );
                }
                ctx.accum(b_id, db);
            }),
        });
        out
    }

    /// 時間軸 broadcast 加算: out [b, c, t] = x [b, c, t] + g [b·c]。
    pub fn add_bcast_t(&mut self, x: Var, g: Var, ch: u32, t_len: u32) -> Var {
        let batch = self.shape(x)[0] as u32;
        let total = batch * ch * t_len;
        let out_t = self.eng.tensor_uninit(total as usize);
        let out = self.push_node(out_t, self.shape(x), true);
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let g_t = self.t(g);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[batch, ch, t_len, 0]);
            self.g.dispatch_raw(
                "t_add_bcast_t",
                &[&x_t, &g_t, &out_ref],
                words,
                total.div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        let g_id = g.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let dx = ctx.alloc(total as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[total, 0, 0, 0]);
                    ctx.g
                        .dispatch_raw("t_copy", &[&dy, &dx], words, total.div_ceil(256), 1, 1);
                }
                ctx.accum(x_id, dx);
                // dg[b·c] = Σ_t dy[b, c, t]
                let dg = ctx.alloc((batch * ch) as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[1, batch * ch, t_len, 0]);
                    ctx.g
                        .dispatch_raw("t_colsum", &[&dy, &dg], words, batch * ch, 1, 1);
                }
                ctx.accum(g_id, dg);
            }),
        });
        out
    }

    /// 末尾 2 次元への bias 加算 (attention mask 用): x [.., r2, c2] += bias [r2, c2]。
    pub fn add_bias2d(&mut self, x: Var, bias: Var, c2: u32) -> Var {
        let total = self.numel(x);
        let r2 = self.numel(bias) / c2.max(1);
        let out_t = self.eng.tensor_uninit(total as usize);
        let out = self.push_node(out_t, self.shape(x), true);
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let b_t = self.t(bias);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[total, r2, c2, 0]);
            self.g.dispatch_raw(
                "t_add_bias2d",
                &[&x_t, &b_t, &out_ref],
                words,
                total.div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        let b_id = bias.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let dx = ctx.alloc(total as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[total, 0, 0, 0]);
                    ctx.g
                        .dispatch_raw("t_copy", &[&dy, &dx], words, total.div_ceil(256), 1, 1);
                }
                ctx.accum(x_id, dx);
                // bias は定数 (勾配不要) — 省略
                let _ = b_id;
            }),
        });
        out
    }

    /// チャネル反転 (Flip): x [b, c, t] → out [b, c, t] (行順反転)。自己逆写像。
    pub fn flip_ch(&mut self, x: Var, batch: u32, ch: u32, t_len: u32) -> Var {
        let total = batch * ch * t_len;
        let out_t = self.eng.tensor_uninit(total as usize);
        let out = self.push_node(out_t, self.shape(x), true);
        let out_ref = self.t(out);
        let x_t = self.t(x);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[total, t_len, ch, 0]);
            self.g.dispatch_raw(
                "flip_ch",
                &[&x_t, &out_ref],
                words,
                total.div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let dx = ctx.alloc(total as usize);
                let mut words = [0u32; 32];
                words[0..4].copy_from_slice(&[total, t_len, ch, 0]);
                ctx.g
                    .dispatch_raw("flip_ch", &[&dy, &dx], words, total.div_ceil(256), 1, 1);
                ctx.accum(x_id, dx);
            }),
        });
        out
    }

    /// フレーム抽出 (STFT 用): x [b, 1, in_len] → col [b, n_fft, n_frames]。
    /// bwd は col2im (scatter、書き込み先一意)。
    #[allow(clippy::too_many_arguments)]
    pub fn im2col_frames(
        &mut self,
        x: Var,
        n_fft: u32,
        hop: u32,
        batch: u32,
        in_len: u32,
        n_frames: u32,
    ) -> Var {
        let out_t = self.eng.tensor_uninit((batch * n_fft * n_frames) as usize);
        let out = self.push_node(
            out_t,
            vec![batch as usize, n_fft as usize, n_frames as usize],
            true,
        );
        let out_ref = self.t(out);
        let x_t = self.t(x);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[batch, 1, n_fft, 1]);
            words[4..8].copy_from_slice(&[hop, 0, in_len, n_frames]);
            words[8] = 0;
            self.g.dispatch_raw(
                "t_im2col",
                &[&x_t, &out_ref],
                words,
                (batch * n_fft * n_frames).div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        let x_len = self.numel(x);
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let dx = ctx.alloc(x_len as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[batch, 1, n_fft, 1]);
                    words[4..8].copy_from_slice(&[hop, 0, in_len, n_frames]);
                    words[8] = 0;
                    ctx.g
                        .dispatch_raw("t_col2im", &[&dx, &dy], words, x_len.div_ceil(256), 1, 1);
                }
                ctx.accum(x_id, dx);
            }),
        });
        out
    }

    /// 窓掛け (k-major): col [b, n_fft, n_frames] × win [n_fft]。win は定数。
    pub fn mul_window(&mut self, x: Var, win: Var, n_fft: u32, n_frames: u32) -> Var {
        let batch = self.shape(x)[0] as u32;
        let total = batch * n_fft * n_frames;
        let out_t = self.eng.tensor_uninit(total as usize);
        let out = self.push_node(out_t, self.shape(x), true);
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let w_t = self.t(win);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[total, n_fft, n_frames, 0]);
            self.g.dispatch_raw(
                "t_mul_window",
                &[&x_t, &w_t, &out_ref],
                words,
                total.div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        let w_id = win.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let w_t = ctx.node_tensor(w_id);
                let dx = ctx.alloc(total as usize);
                let mut words = [0u32; 32];
                words[0..4].copy_from_slice(&[total, n_fft, n_frames, 0]);
                ctx.g.dispatch_raw(
                    "t_mul_window",
                    &[&dy, &w_t, &dx],
                    words,
                    total.div_ceil(256),
                    1,
                    1,
                );
                ctx.accum(x_id, dx);
            }),
        });
        out
    }

    /// 振幅: mag = sqrt(re² + im² + eps)。
    pub fn mag(&mut self, re: Var, im: Var, len: u32, eps: f32) -> Var {
        let out_t = self.eng.tensor_uninit(len as usize);
        let out = self.push_node(out_t, vec![len as usize], true);
        let out_ref = self.t(out);
        let re_t = self.t(re);
        let im_t = self.t(im);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[len, eps.to_bits(), 0, 0]);
            self.g.dispatch_raw(
                "t_mag",
                &[&re_t, &im_t, &out_ref],
                words,
                len.div_ceil(256),
                1,
                1,
            );
        }
        let re_id = re.id;
        let im_id = im.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let re_t = ctx.node_tensor(re_id);
                let im_t = ctx.node_tensor(im_id);
                let dre = ctx.alloc(len as usize);
                let dim = ctx.alloc(len as usize);
                let mut words = [0u32; 32];
                words[0..4].copy_from_slice(&[len, eps.to_bits(), 0, 0]);
                ctx.g.dispatch_raw(
                    "t_mag_bwd",
                    &[&re_t, &im_t, &dre, &dy, &dre, &dim],
                    words,
                    len.div_ceil(256),
                    1,
                    1,
                );
                ctx.accum(re_id, dre);
                ctx.accum(im_id, dim);
            }),
        });
        out
    }

    /// log(clamp(x, min_v))。
    pub fn log_clamp(&mut self, x: Var, min_v: f32) -> Var {
        let len = self.numel(x);
        let out_t = self.eng.tensor_uninit(len as usize);
        let out = self.push_node(out_t, self.shape(x), true);
        let out_ref = self.t(out);
        let x_t = self.t(x);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[len, min_v.to_bits(), 0, 0]);
            self.g.dispatch_raw(
                "t_log_clamp",
                &[&x_t, &out_ref],
                words,
                len.div_ceil(256),
                1,
                1,
            );
        }
        let x_id = x.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let x_t = ctx.node_tensor(x_id);
                let dx = ctx.alloc(len as usize);
                let mut words = [0u32; 32];
                words[0..4].copy_from_slice(&[len, min_v.to_bits(), 0, 0]);
                ctx.g.dispatch_raw(
                    "t_log_clamp_bwd",
                    &[&x_t, &dy, &dx],
                    words,
                    len.div_ceil(256),
                    1,
                    1,
                );
                ctx.accum(x_id, dx);
            }),
        });
        out
    }

    /// 2D convolution (MPD 用)。x [b, cin, h, w] → out [b, cout, oh, ow]。
    /// w は [cout, cin·kh·kw] (PyTorch Conv2d と同一平坦配置)。
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &mut self,
        x: Var,
        w: Var,
        b: Option<Var>,
        out_c: u32,
        in_c: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        pad: u32,
        batch: u32,
        h: u32,
        w_dim: u32,
    ) -> Var {
        let ck = in_c * kh * kw;
        let oh = (h + 2 * pad).saturating_sub(kh) / sh + 1;
        let ow = (w_dim + 2 * pad).saturating_sub(kw) / sw + 1;
        let out_t = self.eng.tensor_uninit((batch * out_c * oh * ow) as usize);
        let out = self.push_node(
            out_t,
            vec![batch as usize, out_c as usize, (oh * ow) as usize],
            true,
        );
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let w_t = self.t(w);
        let b_t = b.map(|v| self.t(v));
        let col = self.eng.tensor_uninit((batch * ck * oh * ow) as usize);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[batch, in_c, h, w_dim]);
            words[4..8].copy_from_slice(&[kh, kw, sh, sw]);
            words[8..12].copy_from_slice(&[pad, oh, ow, 0]);
            self.g.dispatch_raw(
                "t_im2col2d",
                &[&x_t, &col],
                words,
                (batch * ck * oh * ow).div_ceil(256),
                1,
                1,
            );
        }
        self.g.matmul_ex(
            &w_t,
            &col,
            b_t.as_ref(),
            &out_ref,
            out_c,
            oh * ow,
            ck,
            batch,
            ck,
            1,
            0,
            0,
            1,
            oh * ow,
            0,
            0,
            oh * ow,
            1,
            0,
            out_c * oh * ow,
            1.0,
            1,
        );
        let x_id = x.id;
        let w_id = w.id;
        let b_id = b.map(|v| v.id);
        let x_len = self.numel(x);
        let w_len = self.numel(w);
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let x_t = ctx.node_tensor(x_id);
                let w_t = ctx.node_tensor(w_id);
                // dcol = Wᵀ·dy
                let dcol = ctx.alloc((batch * ck * oh * ow) as usize);
                ctx.g.matmul_ex(
                    &w_t,
                    &dy,
                    None,
                    &dcol,
                    ck,
                    oh * ow,
                    out_c,
                    batch,
                    1,
                    ck,
                    0,
                    0,
                    1,
                    oh * ow,
                    0,
                    0,
                    oh * ow,
                    1,
                    0,
                    ck * oh * ow,
                    1.0,
                    0,
                );
                // dx = col2im2d(dcol)
                let dx = ctx.alloc(x_len as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[batch, in_c, h, w_dim]);
                    words[4..8].copy_from_slice(&[kh, kw, sh, sw]);
                    words[8..12].copy_from_slice(&[pad, oh, ow, 0]);
                    ctx.g.dispatch_raw(
                        "t_col2im2d",
                        &[&dx, &dcol],
                        words,
                        x_len.div_ceil(256),
                        1,
                        1,
                    );
                }
                ctx.accum(x_id, dx);
                // dW = Σ_b dy·colᵀ (col 再計算)
                let col = ctx.alloc((batch * ck * oh * ow) as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[batch, in_c, h, w_dim]);
                    words[4..8].copy_from_slice(&[kh, kw, sh, sw]);
                    words[8..12].copy_from_slice(&[pad, oh, ow, 0]);
                    ctx.g.dispatch_raw(
                        "t_im2col2d",
                        &[&x_t, &col],
                        words,
                        (batch * ck * oh * ow).div_ceil(256),
                        1,
                        1,
                    );
                }
                let dwp = ctx.alloc((batch * out_c * ck) as usize);
                ctx.g.matmul_ex(
                    &dy,
                    &col,
                    None,
                    &dwp,
                    out_c,
                    ck,
                    oh * ow,
                    batch,
                    oh * ow,
                    1,
                    0,
                    out_c * oh * ow,
                    oh * ow,
                    1,
                    0,
                    ck * oh * ow,
                    ck,
                    1,
                    0,
                    out_c * ck,
                    1.0,
                    0,
                );
                let dw = ctx.alloc(w_len as usize);
                {
                    let mut words = [0u32; 32];
                    words[0..4].copy_from_slice(&[0, batch, out_c * ck, 0]);
                    ctx.g
                        .dispatch_raw("t_colsum", &[&dwp, &dw], words, out_c * ck, 1, 1);
                }
                ctx.accum(w_id, dw);
                if let Some(b_id) = b_id {
                    let n_ow = oh * ow;
                    let bt = ctx.alloc((batch * out_c) as usize);
                    {
                        let mut words = [0u32; 32];
                        words[0..4].copy_from_slice(&[1, batch * out_c, n_ow, 0]);
                        ctx.g
                            .dispatch_raw("t_colsum", &[&dy, &bt], words, batch * out_c, 1, 1);
                    }
                    let db = ctx.alloc(out_c as usize);
                    {
                        let mut words = [0u32; 32];
                        words[0..4].copy_from_slice(&[0, batch, out_c, 0]);
                        ctx.g
                            .dispatch_raw("t_colsum", &[&bt, &db], words, out_c, 1, 1);
                    }
                    ctx.accum(b_id, db);
                }
            }),
        });
        out
    }

    /// 既存 GpuTensor を定数ノード化 (アップロードなし・バッファ共有)。
    pub fn const_tensor(&mut self, t: &GpuTensor, shape: Vec<usize>) -> Var {
        self.push_node(t.clone_handle(), shape, false)
    }

    // ---- 損失構築用公開 API (rcwx-train::losses から使用) ----

    pub fn numel_pub(&self, v: Var) -> u32 {
        self.numel(v)
    }
    pub fn alloc_pub(&mut self, len: usize) -> GpuTensor {
        self.eng.tensor_uninit(len)
    }
    pub fn dispatch_pub(
        &mut self,
        kernel: &'static str,
        bufs: &[&GpuTensor],
        words: [u32; 32],
        wx: u32,
        wy: u32,
        wz: u32,
    ) {
        self.g.dispatch_raw(kernel, bufs, words, wx, wy, wz);
    }
    /// 生バッファをスカラー var (req_grad) として登録。
    pub fn push_scalar(&mut self, t: GpuTensor) -> Var {
        self.push_node(t, vec![1], true)
    }
    /// スカラー同士の加算。
    pub fn add_scalar(&mut self, a: Var, b: Var) -> Var {
        self.add(a, b, false)
    }
    /// スカラーの定数倍。
    pub fn scale_scalar(&mut self, a: Var, s: f32) -> Var {
        self.pw(a, 7, s)
    }
    /// 損失 op を直接登録 (losses.rs がスカラー生成と逆伝達を対で積む)。
    pub fn push_loss_op(&mut self, scalar: Var, bwd: Box<dyn FnOnce(&mut BwdCtx<'_, '_>)>) {
        self.ops.push(OpEntry {
            out: scalar.id,
            bwd,
        });
    }


    /// masked_fill 相当 (末尾 2 次元 [r2, c2] の 0/1 mask でブロードキャスト)。
    pub fn mask_fill2d(&mut self, x: Var, mask: Var, c2: u32, fill: f32) -> Var {
        let total = self.numel(x);
        let r2 = self.numel(mask) / c2.max(1);
        let out_t = self.eng.tensor_uninit(total as usize);
        let out = self.push_node(out_t, self.shape(x), true);
        let out_ref = self.t(out);
        let x_t = self.t(x);
        let m_t = self.t(mask);
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[total, fill.to_bits(), r2, c2]);
        self.g.dispatch_raw(
            "t_mask_fill",
            &[&x_t, &m_t, &out_ref],
            words,
            total.div_ceil(256),
            1,
            1,
        );
        let x_id = x.id;
        let m_id = mask.id;
        self.ops.push(OpEntry {
            out: out.id,
            bwd: Box::new(move |ctx: &mut BwdCtx<'_, '_>| {
                let Some(dy) = ctx.grad_of(out.id) else {
                    return;
                };
                let m_t = ctx.node_tensor(m_id);
                let dx = ctx.alloc(total as usize);
                let mut w = [0u32; 32];
                w[0..4].copy_from_slice(&[total, 0, r2, c2]);
                ctx.dispatch("t_mask_fill_bwd", &[&dy, &m_t, &dx], w, total.div_ceil(256), 1, 1);
                ctx.accum(x_id, dx);
            }),
        });
        out
    }

    // ---- backward / optimizer ----

    /// `seed` の出力勾配を 1.0 として (すなわち loss = Σ seed) 逆伝搬を記録。
    /// 到達可能な op のみ処理し、未到達の op は残す (後続の backward_from で使用)。
    pub fn backward_from(&mut self, seed: Var) {
        let len = self.numel(seed);
        let seed_grad = self.eng.tensor_uninit(len as usize);
        {
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[len, 1, 1.0f32.to_bits(), 0]);
            self.g
                .dispatch_raw("t_fill", &[&seed_grad], words, len.div_ceil(256), 1, 1);
        }
        self.grads.insert(seed.id, seed_grad);
        let mut ops = std::mem::take(&mut self.ops);
        let mut keep: Vec<OpEntry> = Vec::new();
        while let Some(op) = ops.pop() {
            if self.grads.contains_key(&op.out) {
                let mut ctx = BwdCtx {
                    g: &mut self.g,
                    nodes: &self.nodes,
                    grads: &mut self.grads,
                };
                (op.bwd)(&mut ctx);
            } else {
                keep.push(op);
            }
        }
        keep.reverse();
        self.ops = keep;
    }

    /// adopt 済み全パラメータの勾配をゼロに (ステップ冒頭)。
    pub fn zero_grads(&mut self, store: &ParamStore) {
        for (idx, _) in &self.adopted {
            let g = store.entries[*idx].grad.clone_handle();
            let len = g.len as u32;
            let mut words = [0u32; 32];
            words[0..4].copy_from_slice(&[len, 0, 0.0f32.to_bits(), 0]);
            self.g
                .dispatch_raw("t_fill", &[&g], words, len.div_ceil(256), 1, 1);
        }
    }

    /// adopt 済みパラメータを AdamW で更新 (Graph へ dispatch)。
    /// `only` が None なら全 adopt 済みパラメータ。
    pub fn adamw_step(&mut self, store: &mut ParamStore, only: Option<&[usize]>) {
        store.adam.step += 1;
        let bc1 = 1.0 - store.adam.beta1.powi(store.adam.step as i32);
        let bc2 = 1.0 - store.adam.beta2.powi(store.adam.step as i32);
        let indices: Vec<usize> = match only {
            Some(list) => list.to_vec(),
            None => self.adopted.iter().map(|(i, _)| *i).collect(),
        };
        for idx in indices {
            let len = store.entries[idx].value.len as u32;
            let m = store
                .adam
                .m
                .entry(idx)
                .or_insert_with(|| self.eng.tensor_zeros(len as usize, vec![len as usize]))
                .clone_handle();
            let v = store
                .adam
                .v
                .entry(idx)
                .or_insert_with(|| self.eng.tensor_zeros(len as usize, vec![len as usize]))
                .clone_handle();
            let p = store.entries[idx].value.clone_handle();
            let grad = store.entries[idx].grad.clone_handle();
            let mut words = [0u32; 32];
            words[0] = len;
            words[1] = store.adam.lr.to_bits();
            words[2] = store.adam.beta1.to_bits();
            words[3] = store.adam.beta2.to_bits();
            words[4] = store.adam.eps.to_bits();
            words[5] = store.adam.weight_decay.to_bits();
            words[6] = bc1.to_bits();
            words[7] = bc2.to_bits();
            self.g.dispatch_raw(
                "t_adamw",
                &[&p, &grad, &m, &v],
                words,
                len.div_ceil(256),
                1,
                1,
            );
        }
    }

    /// 勾配テンソルの参照 (検証用)。
    pub fn grad_tensor(&self, v: Var) -> Option<GpuTensor> {
        self.grads.get(&v.id).map(|t| t.clone_handle())
    }

    /// 全要素和 (損失読み出し用。勾配追跡なしのスカラー Var)。
    pub fn sum_all(&mut self, x: Var) -> Var {
        let len = self.numel(x);
        let n_partials = len.div_ceil(256);
        let partials = self.eng.tensor_uninit(n_partials as usize);
        let x_t = self.t(x);
        {
            let mut words = [0u32; 32];
            words[0..2].copy_from_slice(&[len, 0]);
            self.g.dispatch_raw(
                "t_reduce_partials",
                &[&x_t, &x_t, &partials],
                words,
                n_partials,
                1,
                1,
            );
        }
        let out = self.eng.tensor_uninit(1);
        {
            let mut words = [0u32; 32];
            words[0..2].copy_from_slice(&[n_partials, 1.0f32.to_bits()]);
            self.g
                .dispatch_raw("t_reduce_final", &[&partials, &out], words, 1, 1, 1);
        }
        self.push_node(out, vec![1], false)
    }

    pub fn submit(self) -> anyhow::Result<()> {
        self.g.submit()
    }

    pub fn into_plan(self) -> anyhow::Result<Plan> {
        self.g.into_plan()
    }
}
