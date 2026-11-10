//! 学習 (Issue #8) 用 WGSL kernel 群。
//!
//! 推論経路 (shaders.rs) とは分離して管理する:
//! - 全 kernel が out-of-place (保存活性を壊さない)
//! - tape (engine/tape.rs) の forward/backward からのみ使う
//! - uniform は shaders.rs と同じ 128B (8 x vec4<u32>) 規約

use std::sync::LazyLock;

pub(crate) fn theader(n_storage: u32) -> String {
    let mut s = String::from(
        "struct Params {\n  p0: vec4<u32>, p1: vec4<u32>, p2: vec4<u32>, p3: vec4<u32>,\n  p4: vec4<u32>, p5: vec4<u32>, p6: vec4<u32>, p7: vec4<u32>,\n};\n@group(0) @binding(0) var<uniform> P: Params;\nfn f32_of(u: u32) -> f32 { return bitcast<f32>(u); }\nfn flat(gid: vec3<u32>) -> u32 { return gid.x + gid.y * 131072u; }\n",
    );
    for i in 1..=n_storage {
        s.push_str(&format!(
            "@group(0) @binding({i}) var<storage, read_write> B{i}: array<f32>;\n"
        ));
    }
    s
}

const ERF_FN: &str = "fn erf_of(x: f32) -> f32 {\n  let s = select(-1.0, 1.0, x >= 0.0);\n  let ax = abs(x);\n  let t = 1.0 / (1.0 + 0.3275911 * ax);\n  let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;\n  return s * (1.0 - poly * exp(-ax * ax));\n}\n";

/// fill: B1[i] = op==0 ? 0 : param
pub static T_FILL: LazyLock<String> = LazyLock::new(|| {
    theader(1)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let op = P.p0.y; let param = f32_of(P.p0.z);
  let i = flat(gid);
  if (i >= len) { return; }
  B1[i] = select(0.0, param, op == 1u);
}
"#
});

/// copy: B2[dst_base+i] = B1[src_base+i]
pub static T_COPY: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let src_base = P.p0.y; let dst_base = P.p0.z;
  let i = flat(gid);
  if (i >= len) { return; }
  B2[dst_base + i] = B1[src_base + i];
}
"#
});

/// mul2: B3[c_base+i] = B1[a_base+i] * B2[b_base+i]
pub static T_MUL2: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let a_base = P.p0.y; let b_base = P.p0.z; let c_base = P.p0.w;
  let i = flat(gid);
  if (i >= len) { return; }
  B3[c_base + i] = B1[a_base + i] * B2[b_base + i];
}
"#
});

/// pointwise forward (out-of-place)。
/// op: 0=tanh 1=lrelu(param) 2=gelu(erf) 3=sigmoid 4=exp 5=neg 6=relu 7=scale(param)
pub static T_PW_FWD: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + ERF_FN
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let op = P.p0.y; let param = f32_of(P.p0.z);
  let i = flat(gid);
  if (i >= len) { return; }
  let x = B1[i];
  if (op == 0u) { B2[i] = tanh(x); }
  else if (op == 1u) { B2[i] = select(param * x, x, x >= 0.0); }
  else if (op == 2u) { B2[i] = 0.5 * x * (1.0 + erf_of(x * 0.7071067811865476)); }
  else if (op == 3u) { B2[i] = 1.0 / (1.0 + exp(-x)); }
  else if (op == 4u) { B2[i] = exp(x); }
  else if (op == 5u) { B2[i] = -x; }
  else if (op == 6u) { B2[i] = select(0.0, x, x > 0.0); }
  else if (op == 7u) { B2[i] = x * param; }
}
"#
});

/// pointwise backward。B1=x (保存入力), B2=y (保存出力), B3=dy, B4=dx。
/// op: 0=tanh(y) 1=lrelu(x) 2=gelu(x) 3=sigmoid(y) 4=exp(y) 5=neg 6=relu(x) 7=scale
pub static T_PW_BWD: LazyLock<String> = LazyLock::new(|| {
    theader(4)
        + ERF_FN
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let op = P.p0.y; let param = f32_of(P.p0.z);
  let i = flat(gid);
  if (i >= len) { return; }
  let x = B1[i];
  let y = B2[i];
  let dy = B3[i];
  if (op == 0u) { B4[i] = dy * (1.0 - y * y); }
  else if (op == 1u) { B4[i] = select(param * dy, dy, x >= 0.0); }
  else if (op == 2u) {
    let phi = 0.5 * (1.0 + erf_of(x * 0.7071067811865476));
    let pdf = 0.3989422804014327 * exp(-0.5 * x * x);
    B4[i] = dy * (phi + x * pdf);
  }
  else if (op == 3u) { B4[i] = dy * y * (1.0 - y); }
  else if (op == 4u) { B4[i] = dy * y; }
  else if (op == 5u) { B4[i] = -dy; }
  else if (op == 6u) { B4[i] = select(0.0, dy, x > 0.0); }
  else if (op == 7u) { B4[i] = dy * param; }
}
"#
});

/// softmax backward (row per workgroup)。B1=y B2=dy B3=dx。
/// dx = (dy - Σ dy·y) · y
pub static T_SOFTMAX_BWD: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let rows = P.p0.x; let cols = P.p0.y;
  let r = wid.x;
  if (r >= rows) { return; }
  let base = r * cols;
  var acc = 0.0;
  var i = lid.x;
  while (i < cols) { acc = acc + B2[base + i] * B1[base + i]; i = i + 64u; }
  red[lid.x] = acc; workgroupBarrier();
  var s: u32 = 32u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let dot = red[0];
  workgroupBarrier();
  i = lid.x;
  while (i < cols) { B3[base + i] = (B2[base + i] - dot) * B1[base + i]; i = i + 64u; }
}
"#
});

/// LayerNorm backward (stride 対応)。B1=x, B2=dy, B3=gamma, B4=dx, B5=xhat。
/// x/dx/xhat は (in_row, in_k) ストライド、dy は [rows, cols] 連続。
/// dx = rstd * (g·dy - mean(g·dy) - xhat · mean(g·dy·xhat))
pub static T_LAYERNORM_BWD: LazyLock<String> = LazyLock::new(|| {
    theader(5)
        + r#"
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let rows = P.p0.x; let cols = P.p0.y; let eps = f32_of(P.p0.z);
  let in_row = P.p0.w; let in_k = P.p1.x;
  let r = wid.x;
  if (r >= rows) { return; }
  let base = r * in_row;
  var acc = 0.0;
  var i = lid.x;
  while (i < cols) { acc = acc + B1[base + i * in_k]; i = i + 64u; }
  red[lid.x] = acc; workgroupBarrier();
  var s: u32 = 32u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let mean = red[0] / f32(cols);
  workgroupBarrier();
  var acc2 = 0.0;
  i = lid.x;
  while (i < cols) { let d = B1[base + i * in_k] - mean; acc2 = acc2 + d * d; i = i + 64u; }
  red[lid.x] = acc2; workgroupBarrier();
  s = 32u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let rstd = inverseSqrt(red[0] / f32(cols) + eps);
  i = lid.x;
  while (i < cols) { B5[base + i * in_k] = (B1[base + i * in_k] - mean) * rstd; i = i + 64u; }
  workgroupBarrier();
  var a1 = 0.0;
  var a2 = 0.0;
  i = lid.x;
  while (i < cols) {
    let gdy = B3[i] * B2[r * cols + i];
    a1 = a1 + gdy;
    a2 = a2 + gdy * B5[base + i * in_k];
    i = i + 64u;
  }
  red[lid.x] = a1; workgroupBarrier();
  s = 32u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let m1 = red[0] / f32(cols);
  workgroupBarrier();
  red[lid.x] = a2; workgroupBarrier();
  s = 32u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let m2 = red[0] / f32(cols);
  workgroupBarrier();
  i = lid.x;
  while (i < cols) {
    B4[base + i * in_k] = rstd * (B3[i] * B2[r * cols + i] - m1 - B5[base + i * in_k] * m2);
    i = i + 64u;
  }
}
"#
});

/// LayerNorm パラメータ勾配 (stride 対応)。B1=xhat (in_row/in_k), B2=dy(連続),
/// B3=dgamma, B4=dbeta。dgamma[j] = Σ_r dy[r,j]·xhat[r,j]。
pub static T_LN_PARAM_GRAD: LazyLock<String> = LazyLock::new(|| {
    theader(4)
        + r#"
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let cols = P.p0.x; let rows = P.p0.y;
  let in_row = P.p0.z; let in_k = P.p0.w;
  let j = wid.x;
  if (j >= cols) { return; }
  var a1 = 0.0;
  var a2 = 0.0;
  var r = lid.x;
  while (r < rows) {
    let dy = B2[r * cols + j];
    a1 = a1 + dy * B1[r * in_row + j * in_k];
    a2 = a2 + dy;
    r = r + 64u;
  }
  red[lid.x] = a1; workgroupBarrier();
  var s: u32 = 32u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let g1 = red[0];
  workgroupBarrier();
  red[lid.x] = a2; workgroupBarrier();
  s = 32u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid.x == 0u) { B3[j] = g1; B4[j] = red[0]; }
}
"#
});

/// 列/行和。axis 0: db[n] = Σ_m B1[m*ld+n]。axis 1: db[m] = Σ_n B1[m*ld+n]。
pub static T_COLSUM: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let axis = P.p0.x; let rows = P.p0.y; let cols = P.p0.z;
  let j = wid.x;
  if (axis == 0u) {
    if (j >= cols) { return; }
    var a = 0.0;
    var r = lid.x;
    while (r < rows) { a = a + B1[r * cols + j]; r = r + 64u; }
    red[lid.x] = a; workgroupBarrier();
    var s: u32 = 32u;
    while (s > 0u) {
      if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
      workgroupBarrier();
      s = s / 2u;
    }
    if (lid.x == 0u) { B2[j] = red[0]; }
  } else {
    if (j >= rows) { return; }
    var a = 0.0;
    var c = lid.x;
    while (c < cols) { a = a + B1[j * cols + c]; c = c + 64u; }
    red[lid.x] = a; workgroupBarrier();
    var s: u32 = 32u;
    while (s > 0u) {
      if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
      workgroupBarrier();
      s = s / 2u;
    }
    if (lid.x == 0u) { B2[j] = red[0]; }
  }
}
"#
});

/// reduce_partials: workgroup ごとの部分和。op 0=Σx 1=Σx² 2=Σa·b (B2=b)。
/// B2 は op<2 ではダミーでよい (同じバッファを渡す)。
pub static T_REDUCE_PARTIALS: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let op = P.p0.y;
  var v = 0.0;
  if (gid.x < len) {
    if (op == 0u) { v = B1[gid.x]; }
    else if (op == 1u) { v = B1[gid.x] * B1[gid.x]; }
    else if (op == 3u) { v = abs(B1[gid.x] - B2[gid.x]); }
    else if (op == 4u) { let d = B1[gid.x] - f32_of(P.p0.z); v = d * d; }
    else { v = B1[gid.x] * B2[gid.x]; }
  }
  red[lid.x] = v;
  workgroupBarrier();
  var s: u32 = 128u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid.x == 0u) { B3[wid.x] = red[0]; }
}
"#
});

/// reduce_final: B1[0..n] の総和 × scale を B2[0] へ。
pub static T_REDUCE_FINAL: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let n = P.p0.x; let scale = f32_of(P.p0.y);
  var a = 0.0;
  var i = lid.x;
  while (i < n) { a = a + B1[i]; i = i + 256u; }
  red[lid.x] = a; workgroupBarrier();
  var s: u32 = 128u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid.x == 0u) { B2[0] = red[0] * scale; }
}
"#
});

/// AdamW (decoupled weight decay)。
/// B1=param B2=grad B3=m B4=v。全 read_write で in-place 更新。
pub static T_ADAMW: LazyLock<String> = LazyLock::new(|| {
    theader(4)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x;
  let lr = f32_of(P.p0.y);
  let beta1 = f32_of(P.p0.z);
  let beta2 = f32_of(P.p0.w);
  let eps = f32_of(P.p1.x);
  let wd = f32_of(P.p1.y);
  let bc1 = f32_of(P.p1.z);
  let bc2 = f32_of(P.p1.w);
  let i = flat(gid);
  if (i >= len) { return; }
  let g = B2[i];
  let m = beta1 * B3[i] + (1.0 - beta1) * g;
  let v = beta2 * B4[i] + (1.0 - beta2) * g * g;
  B3[i] = m;
  B4[i] = v;
  let mhat = m / bc1;
  let vhat = v / bc2;
  B1[i] = B1[i] - lr * (mhat / (sqrt(vhat) + eps) + wd * B1[i]);
}
"#
});

/// weight_norm fold: w = v · g / ||v||。B1=v B2=g[1] B3=normsq[1] B4=w
pub static T_WN_FOLD: LazyLock<String> = LazyLock::new(|| {
    theader(4)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x;
  let i = flat(gid);
  if (i >= len) { return; }
  let norm = sqrt(B3[0]);
  B4[i] = B1[i] * (B2[0] / norm);
}
"#
});

/// weight_norm backward。
/// B1=v B2=dw(folded) B3=dot(v,dw)[1] B4=normsq[1] B5=g[1] B6=dv B7=dg[1]
/// dv = (g/||v||)·(dw − v·dot/||v||²)、dg = dot/||v||
pub static T_WN_BWD: LazyLock<String> = LazyLock::new(|| {
    theader(7)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x;
  let i = flat(gid);
  let norm = sqrt(B4[0]);
  let norm2 = B4[0];
  let dot = B3[0];
  if (i == 0u) { B7[0] = dot / norm; }
  if (i >= len) { return; }
  let coef = B5[0] / norm;
  B6[i] = coef * (B2[i] - B1[i] * (dot / norm2));
}
"#
});

/// im2col (学習用・バッチ対応)。x [B, cin, in_len] (+x_base) →
/// col [B, cin*k, out_len]。col[b, c*k+j, t] = x[b, c, t*stride - pad + j*dil]。
pub static T_IM2COL: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let c_in = P.p0.y; let k_dim = P.p0.z; let dil = P.p0.w;
  let stride = P.p1.x; let pad_l = P.p1.y; let in_len = P.p1.z; let out_len = P.p1.w;
  let x_base = P.p2.x;
  let gi = flat(gid);
  let per_b = c_in * k_dim * out_len;
  if (gi >= batch * per_b) { return; }
  let t = gi % out_len;
  let ck = (gi / out_len) % (c_in * k_dim);
  let b = gi / per_b;
  let c = ck / k_dim;
  let j = ck % k_dim;
  let pos = i32(t * stride) - i32(pad_l) + i32(j * dil);
  let v = select(0.0, B1[x_base + b * (c_in * in_len) + c * in_len + u32(pos)],
                 pos >= 0 && pos < i32(in_len));
  B2[gi] = v;
}
"#
});

/// col2im (gather 形式・バッチ対応。conv dX / convT fwd 共用)。
/// B1=out [B, c, in_len]、B2=col [B, c*k, out_len]。
/// out[b, c, pos] = Σ_j col[b, c*k+j, t]、t = (pos + pad − j·dil)/stride が整数かつ範囲内。
pub static T_COL2IM: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let c_in = P.p0.y; let k_dim = P.p0.z; let dil = P.p0.w;
  let stride = P.p1.x; let pad_l = P.p1.y; let in_len = P.p1.z; let out_len = P.p1.w;
  let out_base = P.p2.x;
  let gi = flat(gid);
  let per_b = c_in * in_len;
  if (gi >= batch * per_b) { return; }
  let b = gi / per_b;
  let rem = gi % per_b;
  let c = rem / in_len;
  let pos = rem % in_len;
  let col_base = b * (c_in * k_dim * out_len);
  var acc = 0.0;
  for (var j: u32 = 0u; j < k_dim; j = j + 1u) {
    let s = i32(pos) + i32(pad_l) - i32(j * dil);
    if (s >= 0 && s % i32(stride) == 0) {
      let t = u32(s) / stride;
      if (t < out_len) {
        acc = acc + B2[col_base + (c * k_dim + j) * out_len + t];
      }
    }
  }
  B1[out_base + gi] = acc;
}
"#
});

/// チャネル bias 加算 (in-place): x [B, ch, t] の各要素に bias[ch] を加える。
pub static T_ADD_CH_BIAS: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = P.p0.x; let ch = P.p0.y; let t_len = P.p0.z;
  let i = flat(gid);
  if (i >= total) { return; }
  let c = (i / t_len) % ch;
  B1[i] = B1[i] + B2[c];
}
"#
});

/// l1 勾配の種: dx = dy[0] · sign(a − b) / N。B1=a B2=b B3=dx B4=dy[1]
pub static T_L1_SIGN: LazyLock<String> = LazyLock::new(|| {
    theader(4)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let inv_n = f32_of(P.p0.y);
  let i = flat(gid);
  if (i >= len) { return; }
  B3[i] = B4[0] * sign(B1[i] - B2[i]) * inv_n;
}
"#
});

/// mean((x − c)²) 勾配の種: dx = dy[0] · 2·(x − c) / N。B1=x B2=dx B3=dy[1]
pub static T_MSE_BWD: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let c = f32_of(P.p0.y); let inv_n = f32_of(P.p0.z);
  let i = flat(gid);
  if (i >= len) { return; }
  B2[i] = B3[0] * 2.0 * (B1[i] - c) * inv_n;
}
"#
});

/// チャネルスライス: out [b, c_out, t] = in [b, c_in, t] のチャネル [c0, c0+c_out)。
pub static T_CHSLICE: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let c_in = P.p0.y; let t_len = P.p0.z;
  let c_out = P.p0.w; let c0 = P.p1.x;
  let gi = flat(gid);
  if (gi >= batch * c_out * t_len) { return; }
  let t = gi % t_len;
  let c = (gi / t_len) % c_out;
  let b = gi / (c_out * t_len);
  B2[gi] = B1[(b * c_in + c0 + c) * t_len + t];
}
"#
});

/// チャネルスライス逆伝達 (書き込み先は要素ごとに一意なので衝突なし)。
/// gin[b, c0+c, t] += gout[b, c, t]。B1=gout B2=gin。
pub static T_CHSLICE_BWD: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let c_in = P.p0.y; let t_len = P.p0.z;
  let c_out = P.p0.w; let c0 = P.p1.x;
  let gi = flat(gid);
  if (gi >= batch * c_out * t_len) { return; }
  let t = gi % t_len;
  let c = (gi / t_len) % c_out;
  let b = gi / (c_out * t_len);
  B2[(b * c_in + c0 + c) * t_len + t] = B2[(b * c_in + c0 + c) * t_len + t] + B1[gi];
}
"#
});

/// gating (WN): x [b, 2h, t] → acts[b, c<h, t] = tanh(x[b,c,t])·sigmoid(x[b,h+c,t])。
/// B1=in B2=out (out は [b, h, t])。
pub static T_GATING: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let h = P.p0.y; let t_len = P.p0.z;
  let gi = flat(gid);
  if (gi >= batch * h * t_len) { return; }
  let t = gi % t_len;
  let c = (gi / t_len) % h;
  let b = gi / (h * t_len);
  let a = B1[(b * 2u * h + c) * t_len + t];
  let g = B1[(b * 2u * h + h + c) * t_len + t];
  B2[gi] = tanh(a) * (1.0 / (1.0 + exp(-g)));
}
"#
});

/// gating 逆伝達。B1=in(保存) B2=dy B3=dx ([b, 2h, t])。
pub static T_GATING_BWD: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let h = P.p0.y; let t_len = P.p0.z;
  let gi = flat(gid);
  if (gi >= batch * h * t_len) { return; }
  let t = gi % t_len;
  let c = (gi / t_len) % h;
  let b = gi / (h * t_len);
  let a = B1[(b * 2u * h + c) * t_len + t];
  let g = B1[(b * 2u * h + h + c) * t_len + t];
  let dy = B2[gi];
  let ta = tanh(a);
  let sg = 1.0 / (1.0 + exp(-g));
  B3[(b * 2u * h + c) * t_len + t] = dy * (1.0 - ta * ta) * sg;
  B3[(b * 2u * h + h + c) * t_len + t] = dy * ta * sg * (1.0 - sg);
}
"#
});

/// rel-pos skew。mode 0 (rel→abs): out[bh, i, j] = M[bh, i, j-i+ws] (範囲外は 0)。
/// mode 1 (abs→rel): out[bh, i, m] = S[bh, i, i+m-ws] (範囲外は 0)。
pub static T_SKEW: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let mode = P.p0.x; let batch = P.p0.y; let t_len = P.p0.z; let ws = P.p0.w;
  let gi = flat(gid);
  if (mode == 0u) {
    let total_o = batch * t_len * t_len;
    if (gi >= total_o) { return; }
    let j = gi % t_len;
    let i = (gi / t_len) % t_len;
    let bh = gi / (t_len * t_len);
    let r = i32(j) - i32(i) + i32(ws);
    if (r >= 0 && r <= 2 * i32(ws)) {
      B2[gi] = B1[bh * t_len * (2u * ws + 1u) + i * (2u * ws + 1u) + u32(r)];
    } else {
      B2[gi] = 0.0;
    }
  } else {
    let total_o = batch * t_len * (2u * ws + 1u);
    if (gi >= total_o) { return; }
    let m = gi % (2u * ws + 1u);
    let i = (gi / (2u * ws + 1u)) % t_len;
    let bh = gi / (t_len * (2u * ws + 1u));
    let j = i32(i) + i32(m) - i32(ws);
    if (j >= 0 && j < i32(t_len)) {
      B2[gi] = B1[bh * t_len * t_len + i * t_len + u32(j)];
    } else {
      B2[gi] = 0.0;
    }
  }
}
"#
});

/// skew 逆伝達 (書き込み先は要素ごとに一意)。
/// mode 0: dM[bh,i,r] += dS[bh, i, i+r-ws]。mode 1: dS[bh,i,j] += dP[bh, i, j-i+ws]。
/// B1=dout B2=dx (呼び出し側でゼロ初期化)。
pub static T_SKEW_BWD: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let mode = P.p0.x; let batch = P.p0.y; let t_len = P.p0.z; let ws = P.p0.w;
  let gi = flat(gid);
  if (mode == 0u) {
    let total_o = batch * t_len * t_len;
    if (gi >= total_o) { return; }
    let j = gi % t_len;
    let i = (gi / t_len) % t_len;
    let bh = gi / (t_len * t_len);
    let r = i32(j) - i32(i) + i32(ws);
    if (r >= 0 && r <= 2 * i32(ws)) {
      let t = bh * t_len * (2u * ws + 1u) + i * (2u * ws + 1u) + u32(r);
      B2[t] = B2[t] + B1[gi];
    }
  } else {
    let total_o = batch * t_len * (2u * ws + 1u);
    if (gi >= total_o) { return; }
    let m = gi % (2u * ws + 1u);
    let i = (gi / (2u * ws + 1u)) % t_len;
    let bh = gi / (t_len * (2u * ws + 1u));
    let j = i32(i) + i32(m) - i32(ws);
    if (j >= 0 && j < i32(t_len)) {
      let t = bh * t_len * t_len + i * t_len + u32(j);
      B2[t] = B2[t] + B1[gi];
    }
  }
}
"#
});

/// 末尾 2 次元 [r2, c2] の bias 加算 (broadcast)。B3[i] = B1[i] + B2[r, c]。
pub static T_ADD_BIAS2D: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = P.p0.x; let r2 = P.p0.y; let c2 = P.p0.z;
  let i = flat(gid);
  if (i >= total) { return; }
  let c = i % c2;
  let r = (i / c2) % r2;
  B3[i] = B1[i] + B2[r * c2 + c];
}
"#
});

/// 1D reflect pad (両側)。x [b, c, in_len] → out [b, c, in_len+2p]。
/// torch F.pad(mode="reflect") と同一: 左 out[p-1-i] = x[i+1]、右 out[p+len+i] = x[len-2-i]。
pub static T_REFLECT_PAD: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let c = P.p0.y; let in_len = P.p0.z; let pad = P.p0.w;
  let out_len = in_len + 2u * pad;
  let gi = flat(gid);
  if (gi >= batch * c * out_len) { return; }
  let o = gi % out_len;
  let ch = (gi / out_len) % c;
  let b = gi / (c * out_len);
  var src: i32 = i32(o) - i32(pad);
  if (src < 0) { src = -src - 1; }
  if (src >= i32(in_len)) { src = 2 * i32(in_len) - 1 - src; }
  B2[gi] = B1[(b * c + ch) * in_len + u32(src)];
}
"#
});

/// 窓掛け (k-major): col [b, n_fft, n_frames] × win [n_fft]。
/// B3[i] = B1[i] · B2[i / n_frames]。
pub static T_MUL_WINDOW: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = P.p0.x; let n_fft = P.p0.y; let n_frames = P.p0.z;
  let i = flat(gid);
  if (i >= total) { return; }
  B3[i] = B1[i] * B2[i / n_frames];
}
"#
});

/// 振幅: B3 = sqrt(B1² + B2² + eps)。
pub static T_MAG: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let eps = f32_of(P.p0.y);
  let i = flat(gid);
  if (i >= len) { return; }
  B3[i] = sqrt(B1[i] * B1[i] + B2[i] * B2[i] + eps);
}
"#
});

/// 振幅逆伝達: dre = dy·re/m, dim = dy·im/m。B1=re B2=im B4=dy B5=dre B6=dim (B3 未使用)。
pub static T_MAG_BWD: LazyLock<String> = LazyLock::new(|| {
    theader(6)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let eps = f32_of(P.p0.y);
  let i = flat(gid);
  if (i >= len) { return; }
  let m = sqrt(B1[i] * B1[i] + B2[i] * B2[i] + eps);
  B5[i] = B4[i] * B1[i] / m;
  B6[i] = B4[i] * B2[i] / m;
}
"#
});

/// log(clamp(x, min)) とその逆伝達。
pub static T_LOG_CLAMP: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let min_v = f32_of(P.p0.y);
  let i = flat(gid);
  if (i >= len) { return; }
  B2[i] = log(max(B1[i], min_v));
}
"#
});

/// B1=x(保存) B2=dy B3=dx。
pub static T_LOG_CLAMP_BWD: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let min_v = f32_of(P.p0.y);
  let i = flat(gid);
  if (i >= len) { return; }
  B3[i] = B2[i] / max(B1[i], min_v);
}
"#
});

/// KL 要素値: B6 = (logs_p − logs_q − 0.5 + 0.5·(z_p−m_p)²·exp(−2·logs_p))·mask。
/// B1=z_p B2=logs_q B3=m_p B4=logs_p B5=mask B6=out。
pub static T_KL_ELEM: LazyLock<String> = LazyLock::new(|| {
    theader(6)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x;
  let i = flat(gid);
  if (i >= len) { return; }
  let d = B1[i] - B3[i];
  let v = B4[i] - B2[i] - 0.5 + 0.5 * d * d * exp(-2.0 * B4[i]);
  B6[i] = v * B5[i];
}
"#
});

/// KL 逆伝達 A: dz / dm / dlogs_p。scale = dy[0] / Σmask (1/Σmask は uniform)。
/// B1=z_p B2=m_p B3=logs_p B4=mask B5=dy[1] B6=dz B7=dm B8=dlogs_p
pub static T_KL_BWD_A: LazyLock<String> = LazyLock::new(|| {
    theader(8)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let inv_m = f32_of(P.p0.y);
  let i = flat(gid);
  if (i >= len) { return; }
  let d = B1[i] - B2[i];
  let e = exp(-2.0 * B3[i]);
  let sm = B5[0] * inv_m * B4[i];
  B6[i] = sm * d * e;
  B7[i] = -sm * d * e;
  B8[i] = sm * (1.0 - d * d * e);
}
"#
});

/// KL 逆伝達 B: dlogs_q = −dy[0]/Σmask·mask。B1=mask B2=dy[1] B3=dlogs_q
pub static T_KL_BWD_B: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let inv_m = f32_of(P.p0.y);
  let i = flat(gid);
  if (i >= len) { return; }
  B3[i] = -B2[0] * inv_m * B1[i];
}
"#
});

/// ランダムセグメント抽出: out[b, c, i] = in[b, c, round(start[b]·start_scale) + i]。
/// B1=start(f32) B2=in B3=out。
pub static T_GATHER_FRAMES: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let ch = P.p0.y; let t_len = P.p0.z;
  let seg = P.p0.w; let start_scale = f32_of(P.p1.x);
  let gi = flat(gid);
  if (gi >= batch * ch * seg) { return; }
  let i = gi % seg;
  let c = (gi / seg) % ch;
  let b = gi / (ch * seg);
  let start = u32(round(B1[b] * start_scale));
  B3[gi] = B2[(b * ch + c) * t_len + start + i];
}
"#
});

/// セグメント抽出逆伝達 (書き込み先一意)。B1=start B2=gout B3=gin。
pub static T_GATHER_FRAMES_BWD: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let ch = P.p0.y; let t_len = P.p0.z;
  let seg = P.p0.w; let start_scale = f32_of(P.p1.x);
  let gi = flat(gid);
  if (gi >= batch * ch * seg) { return; }
  let i = gi % seg;
  let c = (gi / seg) % ch;
  let b = gi / (ch * seg);
  let start = u32(round(B1[b] * start_scale));
  let t = (b * ch + c) * t_len + start + i;
  B3[t] = B3[t] + B2[gi];
}
"#
});

/// 転置: [b, t, c] ↔ [b, c, t] (mode 0: btc→bct)。逆も同型 (転置の逆は転置)。
pub static T_TRANSPOSE: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let t_len = P.p0.y; let c = P.p0.z;
  let gi = flat(gid);
  if (gi >= batch * t_len * c) { return; }
  let cc = gi % c;
  let t = (gi / c) % t_len;
  let b = gi / (t_len * c);
  B2[(b * c + cc) * t_len + t] = B1[gi];
}
"#
});

/// フレームマスク乗算: out[b, c, t] = x[b, c, t]·mask[b, t]。
pub static T_MUL_FRAME_MASK: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let c = P.p0.y; let t_len = P.p0.z;
  let gi = flat(gid);
  if (gi >= batch * c * t_len) { return; }
  let t = gi % t_len;
  let b = gi / (c * t_len);
  B3[gi] = B1[gi] * B2[b * t_len + t];
}
"#
});

/// 2D im2col (学習用・バッチ/stride 対応)。
/// x [b, cin, h, w] → col [b, cin·kh·kw, oh·ow]。pad は上下左右同一。
pub static T_IM2COL2D: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let cin = P.p0.y; let h = P.p0.z; let w = P.p0.w;
  let kh = P.p1.x; let kw = P.p1.y; let sh = P.p1.z; let sw = P.p1.w;
  let pad = P.p2.x; let oh = P.p2.y; let ow = P.p2.z;
  let gi = flat(gid);
  let per_b = cin * kh * kw * oh * ow;
  if (gi >= batch * per_b) { return; }
  let o = gi % (oh * ow);
  let ck = (gi / (oh * ow)) % (cin * kh * kw);
  let b = gi / per_b;
  let ow2 = o % ow;
  let oh2 = o / ow;
  let c = ck / (kh * kw);
  let rem = ck % (kh * kw);
  let j = rem / kw;
  let i = rem % kw;
  let sy = i32(oh2 * sh) - i32(pad) + i32(j);
  let sx = i32(ow2 * sw) - i32(pad) + i32(i);
  let v = select(
    0.0,
    B1[b * (cin * h * w) + c * (h * w) + u32(sy) * w + u32(sx)],
    sy >= 0 && sy < i32(h) && sx >= 0 && sx < i32(w),
  );
  B2[gi] = v;
}
"#
});

/// 2D col2im (逆伝達用 gather。out_base で出力スライス指定可)。
/// out[b, c, y, x] (+base) = Σ_{j,i} col[b, (c·kh+j)·kw+i, ...] 該当分。
pub static T_COL2IM2D: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let cin = P.p0.y; let h = P.p0.z; let w = P.p0.w;
  let kh = P.p1.x; let kw = P.p1.y; let sh = P.p1.z; let sw = P.p1.w;
  let pad = P.p2.x; let oh = P.p2.y; let ow = P.p2.z; let out_base = P.p2.w;
  let gi = flat(gid);
  if (gi >= batch * cin * h * w) { return; }
  let x = gi % w;
  let y = (gi / w) % h;
  let c = (gi / (h * w)) % cin;
  let b = gi / (cin * h * w);
  let col_base = b * (cin * kh * kw * oh * ow);
  var acc = 0.0;
  for (var j: u32 = 0u; j < kh; j = j + 1u) {
    for (var i: u32 = 0u; i < kw; i = i + 1u) {
      let sy = i32(y) + i32(pad) - i32(j);
      let sx = i32(x) + i32(pad) - i32(i);
      if (sy >= 0 && sy % i32(sh) == 0 && sx >= 0 && sx % i32(sw) == 0) {
        let oy = u32(sy) / sh;
        let ox = u32(sx) / sw;
        if (oy < oh && ox < ow) {
          let ck = (c * kh + j) * kw + i;
          acc = acc + B2[col_base + ck * (oh * ow) + oy * ow + ox];
        }
      }
    }
  }
  B1[out_base + gi] = acc;
}
"#
});

/// reflect pad の随伴: dx[..., src] = dy[..., p+src]
/// + (src < p ? dy[..., p-1-src] : 0) + (in_len-2-p < src <= in_len-2 ? dy[..., p+2*in_len-2-src] : 0)。
pub static T_REFLECT_PAD_BWD: LazyLock<String> = LazyLock::new(|| {
    theader(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let c = P.p0.y; let in_len = P.p0.z; let pad = P.p0.w;
  let out_len = in_len + 2u * pad;
  let gi = flat(gid);
  if (gi >= batch * c * in_len) { return; }
  let src = gi % in_len;
  let base_o = ((gi / in_len) * out_len);
  var v = B1[base_o + pad + src];
  if (src < pad) { v = v + B1[base_o + pad - 1u - src]; }
  let i_right = i32(in_len) - 2 - i32(src);
  if (i_right >= 0 && u32(i_right) < pad) {
    v = v + B1[base_o + pad + in_len + u32(i_right)];
  }
  B2[gi] = v;
}
"#
});

/// 時間軸 broadcast 加算: x [b, c, t] += g [b, c] (WN/dec の条件付け)。
pub static T_ADD_BCAST_T: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let c = P.p0.y; let t_len = P.p0.z;
  let gi = flat(gid);
  if (gi >= batch * c * t_len) { return; }
  let t = gi % t_len;
  let ch = (gi / t_len) % c;
  let b = gi / (c * t_len);
  B3[gi] = B1[gi] + B2[b * c + ch];
}
"#
});

/// チャネル連結: out [b, ca+cb, t] = a ⊕ b。
pub static T_CHCAT: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = P.p0.x; let ca = P.p0.y; let cb = P.p0.z; let t_len = P.p0.w;
  let gi = flat(gid);
  let total = batch * (ca + cb) * t_len;
  if (gi >= total) { return; }
  let t = gi % t_len;
  let c = (gi / t_len) % (ca + cb);
  let b = gi / ((ca + cb) * t_len);
  if (c < ca) {
    B3[gi] = B1[(b * ca + c) * t_len + t];
  } else {
    B3[gi] = B2[(b * cb + (c - ca)) * t_len + t];
  }
}
"#
});

/// masked_fill 相当: out[i] = mask2d[(i/c2)%r2, i%c2] > 0.5 ? x[i] : fill。
/// B1=x B2=mask(0/1) B3=out。
pub static T_MASK_FILL: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let fill = f32_of(P.p0.y);
  let r2 = P.p0.z; let c2 = P.p0.w;
  let i = flat(gid);
  if (i >= len) { return; }
  let c = i % c2;
  let r = (i / c2) % r2;
  B3[i] = select(fill, B1[i], B2[r * c2 + c] > 0.5);
}
"#
});

/// mask_fill の逆伝達: dx[i] = dy[i]·mask2d[(i/c2)%r2, i%c2]。B1=dy B2=mask B3=dx。
pub static T_MASK_FILL_BWD: LazyLock<String> = LazyLock::new(|| {
    theader(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x;
  let r2 = P.p0.z; let c2 = P.p0.w;
  let i = flat(gid);
  if (i >= len) { return; }
  let c = i % c2;
  let r = (i / c2) % r2;
  B3[i] = B1[i] * B2[r * c2 + c];
}
"#
});
