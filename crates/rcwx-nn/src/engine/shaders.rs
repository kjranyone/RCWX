//! WGSL kernel sources. All kernels use a 128B uniform (8 x vec4<u32>).

use std::sync::LazyLock;

fn header(n_storage: u32) -> String {
    let mut s = String::from(
        "struct Params {\n  p0: vec4<u32>, p1: vec4<u32>, p2: vec4<u32>, p3: vec4<u32>,\n  p4: vec4<u32>, p5: vec4<u32>, p6: vec4<u32>, p7: vec4<u32>,\n};\n@group(0) @binding(0) var<uniform> P: Params;\nfn f32_of(u: u32) -> f32 { return bitcast<f32>(u); }\n",
    );
    for i in 1..=n_storage {
        s.push_str(&format!(
            "@group(0) @binding({i}) var<storage, read_write> B{i}: array<f32>;\n"
        ));
    }
    s
}

pub static MATMUL: LazyLock<String> = LazyLock::new(|| {
    header(4)
        + r#"
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m_dim = P.p0.x; let n_dim = P.p0.y; let k_dim = P.p0.z; let batch = P.p0.w;
  let a_row = P.p1.x; let a_k = P.p1.y; let a_base = P.p1.z; let a_bs = P.p1.w;
  let w_row = P.p2.x; let w_k = P.p2.y; let w_base = P.p2.z; let w_bs = P.p2.w;
  let c_row = P.p3.x; let c_n = P.p3.y; let c_base = P.p3.z; let c_bs = P.p3.w;
  let has_bias = P.p4.x; let scale = f32_of(P.p4.y); let bias_axis = P.p4.z;
  let m = gid.y; let n = gid.x; let b = gid.z;
  if (m >= m_dim || n >= n_dim) { return; }
  var acc = 0.0;
  let a0 = a_base + b * a_bs + m * a_row;
  let w0 = w_base + b * w_bs + n * w_row;
  for (var k: u32 = 0u; k < k_dim; k = k + 1u) {
    acc = acc + B1[a0 + k * a_k] * B2[w0 + k * w_k];
  }
  if (has_bias == 1u) {
    if (bias_axis == 0u) { acc = acc + B3[n]; } else { acc = acc + B3[m]; }
  }
  B4[c_base + c_bs * b + c_row * m + c_n * n] = acc * scale;
}
"#
});

pub fn matmul_tiled_src() -> String {
    header(4)
        + r#"
var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;
@compute @workgroup_size(8, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let m_dim = P.p0.x; let n_dim = P.p0.y; let k_dim = P.p0.z; let batch = P.p0.w;
  let a_row = P.p1.x; let a_k = P.p1.y; let a_base = P.p1.z; let a_bs = P.p1.w;
  let w_row = P.p2.x; let w_k = P.p2.y; let w_base = P.p2.z; let w_bs = P.p2.w;
  let c_row = P.p3.x; let c_n = P.p3.y; let c_base = P.p3.z; let c_bs = P.p3.w;
  let has_bias = P.p4.x; let scale = f32_of(P.p4.y); let bias_axis = P.p4.z;
  let b_idx = gid.z;
  let tile_m = wid.y * 16u;
  let tile_n = wid.x * 16u;
  let lx = lid.x * 2u;
  let ly = lid.y * 2u;
  var acc00 = 0.0; var acc01 = 0.0; var acc10 = 0.0; var acc11 = 0.0;
  let m0 = tile_m + lx;
  let m1 = tile_m + lx + 1u;
  let n0 = tile_n + ly;
  let n1 = tile_n + ly + 1u;
  let a_off = a_base + b_idx * a_bs;
  let w_off = w_base + b_idx * w_bs;
  var kt: u32 = 0u;
  while (kt < k_dim) {
    var idx = lid.y * 8u + lid.x;
    for (var rep: u32 = 0u; rep < 4u; rep = rep + 1u) {
      let r = idx / 16u;
      let c = idx % 16u;
      let gm = tile_m + r;
      let gk = kt + c;
      ta[r * 16u + c] = select(0.0, B1[a_off + gm * a_row + gk * a_k], gm < m_dim && gk < k_dim);
      let gn = tile_n + r;
      tb[r * 16u + c] = select(0.0, B2[w_off + gn * w_row + gk * w_k], gn < n_dim && gk < k_dim);
      idx = idx + 64u;
    }
    workgroupBarrier();
    for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
      let a0 = ta[lx * 16u + kk];
      let a1 = ta[(lx + 1u) * 16u + kk];
      let b0 = tb[ly * 16u + kk];
      let b1 = tb[(ly + 1u) * 16u + kk];
      acc00 = acc00 + a0 * b0;
      acc01 = acc01 + a0 * b1;
      acc10 = acc10 + a1 * b0;
      acc11 = acc11 + a1 * b1;
    }
    workgroupBarrier();
    kt = kt + 16u;
  }
  if (has_bias == 1u) {
    let b00 = select(B3[n0], B3[m0], bias_axis == 1u);
    let b01 = select(B3[n1], B3[m0], bias_axis == 1u);
    let b10 = select(B3[n0], B3[m1], bias_axis == 1u);
    let b11 = select(B3[n1], B3[m1], bias_axis == 1u);
    if (m0 < m_dim && n0 < n_dim) { B4[c_base + c_bs * b_idx + c_row * m0 + c_n * n0] = (acc00 + b00) * scale; }
    if (m0 < m_dim && n1 < n_dim) { B4[c_base + c_bs * b_idx + c_row * m0 + c_n * n1] = (acc01 + b01) * scale; }
    if (m1 < m_dim && n0 < n_dim) { B4[c_base + c_bs * b_idx + c_row * m1 + c_n * n0] = (acc10 + b10) * scale; }
    if (m1 < m_dim && n1 < n_dim) { B4[c_base + c_bs * b_idx + c_row * m1 + c_n * n1] = (acc11 + b11) * scale; }
  } else {
    if (m0 < m_dim && n0 < n_dim) { B4[c_base + c_bs * b_idx + c_row * m0 + c_n * n0] = acc00 * scale; }
    if (m0 < m_dim && n1 < n_dim) { B4[c_base + c_bs * b_idx + c_row * m0 + c_n * n1] = acc01 * scale; }
    if (m1 < m_dim && n0 < n_dim) { B4[c_base + c_bs * b_idx + c_row * m1 + c_n * n0] = acc10 * scale; }
    if (m1 < m_dim && n1 < n_dim) { B4[c_base + c_bs * b_idx + c_row * m1 + c_n * n1] = acc11 * scale; }
  }
}
"#
}

pub static CONV1D: LazyLock<String> = LazyLock::new(|| {
    header(4)
        + r#"
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_c = P.p0.x; let in_c = P.p0.y; let k_dim = P.p0.z; let dil = P.p0.w;
  let stride = P.p1.x; let pad_l = P.p1.y; let in_len = P.p1.z; let out_len = P.p1.w;
  let groups = P.p2.x; let has_bias = P.p2.y;
  let ch = gid.y; let t = gid.x;
  if (ch >= out_c || t >= out_len) { return; }
  let cpg_in = in_c / groups;
  let g = ch / (out_c / groups);
  var acc = 0.0;
  for (var cl: u32 = 0u; cl < cpg_in; cl = cl + 1u) {
    let c = g * cpg_in + cl;
    for (var j: u32 = 0u; j < k_dim; j = j + 1u) {
      let pos = i32(t * stride) - i32(pad_l) + i32(j * dil);
      if (pos >= 0 && pos < i32(in_len)) {
        acc = acc + B2[ch * (cpg_in * k_dim) + cl * k_dim + j] * B1[c * in_len + u32(pos)];
      }
    }
  }
  if (has_bias == 1u) { acc = acc + B3[ch]; }
  B4[ch * out_len + t] = acc;
}
"#
});

pub static ADD: LazyLock<String> = LazyLock::new(|| {
    header(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x;
  let sign = select(-1.0, 1.0, P.p0.y == 0u);
  let a_base = P.p0.z;
  let b_base = P.p0.w;
  let c_base = P.p1.x;
  let i = gid.x + gid.y * 131072u;
  if (i >= len) { return; }
  B3[c_base + i] = B1[a_base + i] + sign * B2[b_base + i];
}
"#
});

pub static GELU: LazyLock<String> = LazyLock::new(|| {
    header(1)
        + r#"
fn erf_of(x: f32) -> f32 {
  let s = select(-1.0, 1.0, x >= 0.0);
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return s * (1.0 - poly * exp(-ax * ax));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x;
  let i = gid.x + gid.y * 131072u;
  if (i >= len) { return; }
  let x = B1[i];
  B1[i] = 0.5 * x * (1.0 + erf_of(x * 0.7071067811865476));
}
"#
});

pub static SOFTMAX: LazyLock<String> = LazyLock::new(|| {
    header(1)
        + r#"
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let rows = P.p0.x; let cols = P.p0.y;
  let r = wid.x;
  if (r >= rows) { return; }
  let base = r * cols;
  var mx = -3.0e38;
  var i = lid.x;
  while (i < cols) { mx = max(mx, B1[base + i]); i = i + 64u; }
  red[lid.x] = mx; workgroupBarrier();
  var s: u32 = 32u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = max(red[lid.x], red[lid.x + s]); }
    workgroupBarrier();
    s = s / 2u;
  }
  let m = red[0];
  workgroupBarrier();
  var e_local = 0.0;
  i = lid.x;
  while (i < cols) { e_local = e_local + exp(B1[base + i] - m); i = i + 64u; }
  red[lid.x] = e_local; workgroupBarrier();
  s = 32u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let sum_inv = 1.0 / red[0];
  workgroupBarrier();
  i = lid.x;
  while (i < cols) { B1[base + i] = exp(B1[base + i] - m) * sum_inv; i = i + 64u; }
}
"#
});

pub fn groupnorm_src() -> String {
    header(4)
        + r#"
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let c_dim = P.p0.x; let t_dim = P.p0.y; let eps = f32_of(P.p0.z);
  let ch = wid.x;
  if (ch >= c_dim) { return; }
  let base = ch * t_dim;
  var acc = 0.0;
  var i = lid.x;
  while (i < t_dim) { acc = acc + B1[base + i]; i = i + 64u; }
  red[lid.x] = acc; workgroupBarrier();
  var s: u32 = 32u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let mean = red[0] / f32(t_dim);
  workgroupBarrier();
  var acc2 = 0.0;
  i = lid.x;
  while (i < t_dim) { let d = B1[base + i] - mean; acc2 = acc2 + d * d; i = i + 64u; }
  red[lid.x] = acc2; workgroupBarrier();
  s = 32u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let inv = inverseSqrt(red[0] / f32(t_dim) + eps);
  let g = B2[ch]; let bb = B3[ch];
  i = lid.x;
  while (i < t_dim) { B4[base + i] = (B1[base + i] - mean) * inv * g + bb; i = i + 64u; }
}
"#
}

pub fn layernorm_src() -> String {
    header(4)
        + r#"
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let rows = P.p0.x; let cols = P.p0.y; let in_row = P.p0.z; let in_k = P.p0.w;
  let out_row = P.p1.x; let eps = f32_of(P.p1.y); let out_k = P.p1.z;
  let r = wid.x;
  if (r >= rows) { return; }
  var acc = 0.0;
  var i = lid.x;
  while (i < cols) { acc = acc + B1[r * in_row + i * in_k]; i = i + 64u; }
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
  while (i < cols) { let d = B1[r * in_row + i * in_k] - mean; acc2 = acc2 + d * d; i = i + 64u; }
  red[lid.x] = acc2; workgroupBarrier();
  s = 32u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let inv = inverseSqrt(red[0] / f32(cols) + eps);
  i = lid.x;
  while (i < cols) {
    let v = (B1[r * in_row + i * in_k] - mean) * inv;
    B4[r * out_row + i * out_k] = v * B2[i] + B3[i];
    i = i + 64u;
  }
}
"#
}

pub fn im2col_src() -> String {
    header(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c_in = P.p0.x; let k_dim = P.p0.y; let dil = P.p0.z;
  let stride = P.p1.x; let pad_l = P.p1.y; let in_len = P.p1.z; let out_len = P.p1.w;
  let x_base = P.p0.w;
  let gi = gid.x + gid.y * 131072u;
  let t = gi % out_len;
  let ck = gi / out_len;
  if (t >= out_len || ck >= c_in * k_dim) { return; }
  let c = ck / k_dim;
  let j = ck % k_dim;
  let pos = i32(t * stride) - i32(pad_l) + i32(j * dil);
  let v = select(0.0, B1[x_base + c * in_len + u32(pos)], pos >= 0 && pos < i32(in_len));
  B2[ck * out_len + t] = v;
}
"#
}

pub fn im2col2d_src() -> String {
    header(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c_dim = P.p0.x; let h = P.p0.y; let w = P.p0.z;
  let i = gid.x + gid.y * 131072u;
  let total_out = h * w;
  let total = c_dim * 9u * total_out;
  if (i >= total) { return; }
  let x = i % w;
  let y = (i / w) % h;
  let ck = i / total_out;
  let c = ck / 9u;
  let j = (ck % 9u) / 3u;
  let ii = ck % 3u;
  let sy = i32(y) + i32(j) - 1;
  let sx = i32(x) + i32(ii) - 1;
  let v = select(0.0, B1[c * h * w + u32(sy) * w + u32(sx)], sy >= 0 && sy < i32(h) && sx >= 0 && sx < i32(w));
  B2[i] = v;
}
"#
}

/// 汎用 im2col2d (kernel k・pad p・stride 1)。出力 [c*k*k, h*w]。
pub fn im2col2d_k_src() -> String {
    header(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c_dim = P.p0.x; let h = P.p0.y; let w = P.p0.z; let k_dim = P.p0.w;
  let pad = P.p1.x;
  let i = gid.x + gid.y * 131072u;
  let total_out = h * w;
  let kk = k_dim * k_dim;
  let total = c_dim * kk * total_out;
  if (i >= total) { return; }
  let x = i % w;
  let y = (i / w) % h;
  let ck = i / total_out;
  let c = ck / kk;
  let rem = ck % kk;
  let j = rem / k_dim;
  let ii = rem % k_dim;
  let sy = i32(y) + i32(j) - i32(pad);
  let sx = i32(x) + i32(ii) - i32(pad);
  let v = select(0.0, B1[c * h * w + u32(sy) * w + u32(sx)], sy >= 0 && sy < i32(h) && sx >= 0 && sx < i32(w));
  B2[i] = v;
}
"#
}

pub fn avg_pool2x2_src() -> String {
    header(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c_dim = P.p0.x; let h = P.p0.y; let w = P.p0.z;
  let i = gid.x + gid.y * 131072u;
  let oh = h / 2u; let ow = w / 2u;
  if (i >= c_dim * oh * ow) { return; }
  let x = i % ow;
  let y = (i / ow) % oh;
  let c = i / (oh * ow);
  let a = B1[c * h * w + (2u * y) * w + 2u * x];
  let b = B1[c * h * w + (2u * y) * w + 2u * x + 1u];
  let cc = B1[c * h * w + (2u * y + 1u) * w + 2u * x];
  let d = B1[c * h * w + (2u * y + 1u) * w + 2u * x + 1u];
  B2[i] = (a + b + cc + d) * 0.25;
}
"#
}

pub fn conv_transpose1d_src() -> String {
    header(4)
        + r#"
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_c = P.p0.x; let in_c = P.p0.y; let k_dim = P.p0.z; let stride = P.p0.w;
  let pad_l = P.p1.x; let in_len = P.p1.y; let out_len = P.p1.z; let has_bias = P.p1.w;
  let ch = gid.y; let pos = gid.x;
  if (ch >= out_c || pos >= out_len) { return; }
  var acc = 0.0;
  for (var c: u32 = 0u; c < in_c; c = c + 1u) {
    for (var j: u32 = 0u; j < k_dim; j = j + 1u) {
      let src = i32(pos) + i32(pad_l) - i32(j);
      if (src >= 0 && (u32(src) % stride) == 0u && u32(src) / stride < in_len) {
        let t = u32(src) / stride;
        acc = acc + B2[c * (out_c * k_dim) + ch * k_dim + j] * B1[c * in_len + t];
      }
    }
  }
  if (has_bias == 1u) { acc = acc + B3[ch]; }
  B4[ch * out_len + pos] = acc;
}
"#
}

pub fn conv_transpose2d_src() -> String {
    header(4)
        + r#"
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cin = P.p0.x; let cout = P.p0.y; let h_out = P.p0.z; let w_out = P.p0.w;
  let h_in = P.p1.x; let w_in = P.p1.y; let sh = P.p1.z; let sw = P.p1.w;
  let oph = P.p2.x; let opw = P.p2.y;
  let x = gid.x; let y = gid.y; let co = gid.z;
  if (x >= w_out || y >= h_out || co >= cout) { return; }
  var acc = 0.0;
  for (var ci: u32 = 0u; ci < cin; ci = ci + 1u) {
    for (var j: u32 = 0u; j < 3u; j = j + 1u) {
      for (var ii: u32 = 0u; ii < 3u; ii = ii + 1u) {
        let sy = i32(y) + 1 - i32(j);
        let sx = i32(x) + 1 - i32(ii);
        if (sy % i32(sh) == 0 && sx % i32(sw) == 0 && sy >= 0 && sx >= 0) {
          let ty = u32(sy) / sh;
          let tx = u32(sx) / sw;
          if (ty < h_in && tx < w_in) {
            acc = acc + B2[ci * (cout * 9u) + co * 9u + j * 3u + ii] * B1[ci * h_in * w_in + ty * w_in + tx];
          }
        }
      }
    }
  }
  B3[co * h_out * w_out + y * w_out + x] = acc + B4[co];
}
"#
}

pub fn concat_ch_src() -> String {
    header(3)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = P.p0.x; let ca = P.p0.y; let cb = P.p0.z;
  let i = gid.x + gid.y * 131072u;
  if (i >= (ca + cb) * n) { return; }
  if (i < ca * n) { B3[i] = B1[i]; } else { B3[i] = B2[i - ca * n]; }
}
"#
}

pub fn flip_ch_src() -> String {
    header(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = P.p0.x; let t_len = P.p0.y; let ch = P.p0.z;
  let i = gid.x + gid.y * 131072u;
  if (i >= total) { return; }
  let t = i % t_len;
  let c = i / t_len;
  B2[(ch - 1u - c) * t_len + t] = B1[c * t_len + t];
}
"#
}

pub fn pad2d_src() -> String {
    header(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c_dim = P.p0.x; let h = P.p0.y; let w = P.p0.z;
  let w2 = P.p1.x; let h2 = P.p1.y;
  let i = gid.x + gid.y * 131072u;
  if (i >= c_dim * h * w) { return; }
  let x = i % w;
  let y = (i / w) % h;
  let c = i / (h * w);
  B2[c * h2 * w2 + y * w2 + x] = B1[i];
}
"#
}

pub fn transpose_flatten_src() -> String {
    header(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c_dim = P.p0.x; let t = P.p0.y; let m = P.p0.z;
  let i = gid.x + gid.y * 131072u;
  if (i >= c_dim * t * m) { return; }
  let mm = i % m;
  let tt = (i / m) % t;
  let c = i / (m * t);
  B2[tt * c_dim * m + c * m + mm] = B1[i];
}
"#
}

pub fn rel_attn_src() -> String {
    header(3)
        + r#"
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let mode = P.p0.x; let length = P.p0.y; let d = P.p0.z; let ws = P.p0.w;
  let scale = bitcast<f32>(P.p1.x);
  let heads = P.p1.y;
  if (mode == 0u) {
    let j = gid.x; let i = gid.y; let h = gid.z;
    if (i >= length || j >= length || h >= heads) { return; }
    let r = i32(j) - i32(i) + i32(ws);
    if (r < 0 || r > 2 * i32(ws)) { return; }
    var acc = 0.0;
    for (var dd: u32 = 0u; dd < d; dd = dd + 1u) {
      acc = acc + B1[(h * d + dd) * length + i] * B2[u32(r) * d + dd];
    }
    let sidx = h * length * length + i * length + j;
    B3[sidx] = B3[sidx] + acc * scale;
  } else {
    let dd = gid.x; let i = gid.y; let h = gid.z;
    if (i >= length || dd >= d || h >= heads) { return; }
    var acc = 0.0;
    for (var j: u32 = 0u; j < length; j = j + 1u) {
      let r = i32(j) - i32(i) + i32(ws);
      if (r >= 0 && r <= 2 * i32(ws)) {
        acc = acc + B1[h * length * length + i * length + j] * B2[u32(r) * d + dd];
      }
    }
    let oidx = (h * d + dd) * length + i;
    B3[oidx] = B3[oidx] + acc;
  }
}
"#
}

pub fn pointwise_src() -> String {
    header(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = P.p0.x; let op = P.p0.y;
  let param = bitcast<f32>(P.p0.z);
  let ch = P.p0.w;
  let i = gid.x + gid.y * 131072u;
  if (i >= len) { return; }
  if (op == 1u) {
    let x = B1[i];
    B2[i] = select(param * x, x, x >= 0.0);
  } else if (op == 2u) {
    B2[i] = tanh(B1[i]);
  } else if (op == 3u) {
    let x = B1[i];
    B2[i] = x * (1.0 / (1.0 + exp(-1.702 * x)));
  } else if (op == 4u) {
    B2[i] = B1[i] * param;
  } else if (op == 5u) {
    let t_len = P.p1.x; let h = P.p1.y;
    let t = i % t_len;
    let row = i / t_len;
    let a = B1[row * t_len + t];
    let b = B1[(row + h) * t_len + t];
    B1[row * t_len + t] = tanh(a) * (1.0 / (1.0 + exp(-b)));
  } else if (op == 6u) {
    let t_len = P.p1.x;
    let t = i % t_len;
    let c = i / t_len;
    B2[(ch - 1u - c) * t_len + t] = B1[c * t_len + t];
  } else if (op == 7u) {
    let x = B1[i];
    B2[i] = select(0.0, x, x > 0.0);
  }
}
"#
}

/// implicit-GEMM conv1d (stride 1・tiled matmul 構造・im2col 省略)。
/// A = 重み [cout, k_dim=cin*ck]、B2 = 信号 X [cin, in_len]、出力 [cout, len]。
/// gk→(ci,j) の除算はチャンク毎の SLM テーブルで hot path から除去。
/// post_op (P.p5.w): 0=none 1=lrelu — 畳み込み直後の活性化を fuse。
pub fn matmul_conv_src() -> String {
    header(4)
        + r#"
fn post1(v: f32, op: u32) -> f32 {
  if (op == 1u) { return select(0.1 * v, v, v > 0.0); }
  return v;
}
var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;
var<workgroup> idxci: array<u32, 16>;
var<workgroup> idxj: array<u32, 16>;
@compute @workgroup_size(8, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let m_dim = P.p0.x; let n_dim = P.p0.y; let k_dim = P.p0.z;
  let a_row = P.p1.x; let a_k = P.p1.y; let a_base = P.p1.z;
  let c_row = P.p3.x; let c_n = P.p3.y; let c_base = P.p3.z;
  let has_bias = P.p4.x; let scale = f32_of(P.p4.y); let bias_axis = P.p4.z;
  let conv_k = P.p4.w;
  let pad = P.p5.x; let dil = P.p5.y; let in_len = P.p5.z; let post_op = P.p5.w;
  let tile_m = wid.y * 16u;
  let tile_n = wid.x * 16u;
  let lx = lid.x * 2u;
  let ly = lid.y * 2u;
  var acc00 = 0.0; var acc01 = 0.0; var acc10 = 0.0; var acc11 = 0.0;
  let m0 = tile_m + lx;
  let m1 = tile_m + lx + 1u;
  let n0 = tile_n + ly;
  let n1 = tile_n + ly + 1u;
  var kt: u32 = 0u;
  while (kt < k_dim) {
    if (lid.x == 0u && lid.y == 0u) {
      for (var c2: u32 = 0u; c2 < 16u; c2 = c2 + 1u) {
        let gk2 = kt + c2;
        idxci[c2] = gk2 / conv_k;
        idxj[c2] = gk2 % conv_k;
      }
    }
    workgroupBarrier();
    var idx = lid.y * 8u + lid.x;
    for (var rep: u32 = 0u; rep < 4u; rep = rep + 1u) {
      let r = idx / 16u;
      let c = idx % 16u;
      let gm = tile_m + r;
      let gk = kt + c;
      ta[r * 16u + c] = select(0.0, B1[a_base + gm * a_row + gk * a_k], gm < m_dim && gk < k_dim);
      // B = conv gather: 位置 = gn - pad + j*dil
      let gn = tile_n + r;
      let pos = i32(gn) - i32(pad) + i32(idxj[c] * dil);
      let v = select(0.0, B2[idxci[c] * in_len + u32(pos)],
                     (gn < n_dim && gk < k_dim) && pos >= 0 && pos < i32(in_len));
      tb[r * 16u + c] = v;
      idx = idx + 64u;
    }
    workgroupBarrier();
    for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
      let a0 = ta[lx * 16u + kk];
      let a1 = ta[(lx + 1u) * 16u + kk];
      let b0 = tb[ly * 16u + kk];
      let b1 = tb[(ly + 1u) * 16u + kk];
      acc00 = acc00 + a0 * b0;
      acc01 = acc01 + a0 * b1;
      acc10 = acc10 + a1 * b0;
      acc11 = acc11 + a1 * b1;
    }
    workgroupBarrier();
    kt = kt + 16u;
  }
  if (has_bias == 1u) {
    let b00 = B3[m0];
    let b01 = B3[m0];
    let b10 = B3[m1];
    let b11 = B3[m1];
    if (m0 < m_dim && n0 < n_dim) { B4[c_base + c_row * m0 + c_n * n0] = post1((acc00 + b00) * scale, post_op); }
    if (m0 < m_dim && n1 < n_dim) { B4[c_base + c_row * m0 + c_n * n1] = post1((acc01 + b01) * scale, post_op); }
    if (m1 < m_dim && n0 < n_dim) { B4[c_base + c_row * m1 + c_n * n0] = post1((acc10 + b10) * scale, post_op); }
    if (m1 < m_dim && n1 < n_dim) { B4[c_base + c_row * m1 + c_n * n1] = post1((acc11 + b11) * scale, post_op); }
  } else {
    if (m0 < m_dim && n0 < n_dim) { B4[c_base + c_row * m0 + c_n * n0] = post1(acc00 * scale, post_op); }
    if (m0 < m_dim && n1 < n_dim) { B4[c_base + c_row * m0 + c_n * n1] = post1(acc01 * scale, post_op); }
    if (m1 < m_dim && n0 < n_dim) { B4[c_base + c_row * m1 + c_n * n0] = post1(acc10 * scale, post_op); }
    if (m1 < m_dim && n1 < n_dim) { B4[c_base + c_row * m1 + c_n * n1] = post1(acc11 * scale, post_op); }
  }
}
"#
}

/// チャネル major 行列の末尾 n_out フレーム抽出: out[c, t] = in[c, off+t]。
pub fn slice_ch_src() -> String {
    header(2)
        + r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n_out = P.p0.x; let n_in = P.p0.y; let ch = P.p0.z; let off = P.p0.w;
  let i = gid.x + gid.y * 131072u;
  if (i >= ch * n_out) { return; }
  let c = i / n_out;
  let t = i % n_out;
  B2[i] = B1[c * n_in + off + t];
}
"#
}

/// matmul_conv の fp16 版: SLM タイルと内積を f16 で計算 (16 項ごとに f32 昇格)。
/// 入力バッファは f32 のまま (ロード時に変換)。
pub fn matmul_conv_f16_src() -> String {
    header(4)
        + r#"
enable f16;
fn post1(v: f32, op: u32) -> f32 {
  if (op == 1u) { return select(0.1 * v, v, v > 0.0); }
  return v;
}
var<workgroup> ta: array<f16, 256>;
var<workgroup> tb: array<f16, 256>;
var<workgroup> idxci: array<u32, 16>;
var<workgroup> idxj: array<u32, 16>;
@compute @workgroup_size(8, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let m_dim = P.p0.x; let n_dim = P.p0.y; let k_dim = P.p0.z;
  let a_row = P.p1.x; let a_k = P.p1.y; let a_base = P.p1.z;
  let c_row = P.p3.x; let c_n = P.p3.y; let c_base = P.p3.z;
  let has_bias = P.p4.x; let scale = f32_of(P.p4.y); let bias_axis = P.p4.z;
  let conv_k = P.p4.w;
  let pad = P.p5.x; let dil = P.p5.y; let in_len = P.p5.z; let post_op = P.p5.w;
  let tile_m = wid.y * 16u;
  let tile_n = wid.x * 16u;
  let lx = lid.x * 2u;
  let ly = lid.y * 2u;
  var acc00 = 0.0; var acc01 = 0.0; var acc10 = 0.0; var acc11 = 0.0;
  let m0 = tile_m + lx;
  let m1 = tile_m + lx + 1u;
  let n0 = tile_n + ly;
  let n1 = tile_n + ly + 1u;
  var kt: u32 = 0u;
  while (kt < k_dim) {
    if (lid.x == 0u && lid.y == 0u) {
      for (var c2: u32 = 0u; c2 < 16u; c2 = c2 + 1u) {
        let gk2 = kt + c2;
        idxci[c2] = gk2 / conv_k;
        idxj[c2] = gk2 % conv_k;
      }
    }
    workgroupBarrier();
    var idx = lid.y * 8u + lid.x;
    for (var rep: u32 = 0u; rep < 4u; rep = rep + 1u) {
      let r = idx / 16u;
      let c = idx % 16u;
      let gm = tile_m + r;
      let gk = kt + c;
      ta[r * 16u + c] = f16(select(0.0, B1[a_base + gm * a_row + gk * a_k], gm < m_dim && gk < k_dim));
      let gn = tile_n + r;
      let pos = i32(gn) - i32(pad) + i32(idxj[c] * dil);
      let v = select(0.0, B2[idxci[c] * in_len + u32(pos)],
                     (gn < n_dim && gk < k_dim) && pos >= 0 && pos < i32(in_len));
      tb[r * 16u + c] = f16(v);
      idx = idx + 64u;
    }
    workgroupBarrier();
    var h00 = f16(0.0); var h01 = f16(0.0); var h10 = f16(0.0); var h11 = f16(0.0);
    for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
      let a0 = ta[lx * 16u + kk];
      let a1 = ta[(lx + 1u) * 16u + kk];
      let b0 = tb[ly * 16u + kk];
      let b1 = tb[(ly + 1u) * 16u + kk];
      h00 = h00 + a0 * b0;
      h01 = h01 + a0 * b1;
      h10 = h10 + a1 * b0;
      h11 = h11 + a1 * b1;
    }
    acc00 = acc00 + f32(h00);
    acc01 = acc01 + f32(h01);
    acc10 = acc10 + f32(h10);
    acc11 = acc11 + f32(h11);
    workgroupBarrier();
    kt = kt + 16u;
  }
  if (has_bias == 1u) {
    let b00 = B3[m0];
    let b10 = B3[m1];
    if (m0 < m_dim && n0 < n_dim) { B4[c_base + c_row * m0 + c_n * n0] = post1((acc00 + b00) * scale, post_op); }
    if (m0 < m_dim && n1 < n_dim) { B4[c_base + c_row * m0 + c_n * n1] = post1((acc01 + b00) * scale, post_op); }
    if (m1 < m_dim && n0 < n_dim) { B4[c_base + c_row * m1 + c_n * n0] = post1((acc10 + b10) * scale, post_op); }
    if (m1 < m_dim && n1 < n_dim) { B4[c_base + c_row * m1 + c_n * n1] = post1((acc11 + b10) * scale, post_op); }
  } else {
    if (m0 < m_dim && n0 < n_dim) { B4[c_base + c_row * m0 + c_n * n0] = post1(acc00 * scale, post_op); }
    if (m0 < m_dim && n1 < n_dim) { B4[c_base + c_row * m0 + c_n * n1] = post1(acc01 * scale, post_op); }
    if (m1 < m_dim && n0 < n_dim) { B4[c_base + c_row * m1 + c_n * n0] = post1(acc10 * scale, post_op); }
    if (m1 < m_dim && n1 < n_dim) { B4[c_base + c_row * m1 + c_n * n1] = post1(acc11 * scale, post_op); }
  }
}
"#
}

/// matmul_tiled + 残差加算 (B5 = residual [m, n]、c_stride と同じ配置)。
/// 出力 = acc + bias + residual。残差 fuse で add_off パスを削減。
pub fn matmul_tiled_res_src() -> String {
    header(5)
        + r#"
var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;
@compute @workgroup_size(8, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let m_dim = P.p0.x; let n_dim = P.p0.y; let k_dim = P.p0.z;
  let a_row = P.p1.x; let a_k = P.p1.y; let a_base = P.p1.z;
  let w_row = P.p2.x; let w_k = P.p2.y; let w_base = P.p2.z;
  let c_row = P.p3.x; let c_n = P.p3.y; let c_base = P.p3.z;
  let has_bias = P.p4.x; let scale = f32_of(P.p4.y); let bias_axis = P.p4.z;
  let tile_m = wid.y * 16u;
  let tile_n = wid.x * 16u;
  let lx = lid.x * 2u;
  let ly = lid.y * 2u;
  var acc00 = 0.0; var acc01 = 0.0; var acc10 = 0.0; var acc11 = 0.0;
  let m0 = tile_m + lx;
  let m1 = tile_m + lx + 1u;
  let n0 = tile_n + ly;
  let n1 = tile_n + ly + 1u;
  var kt: u32 = 0u;
  while (kt < k_dim) {
    var idx = lid.y * 8u + lid.x;
    for (var rep: u32 = 0u; rep < 4u; rep = rep + 1u) {
      let r = idx / 16u;
      let c = idx % 16u;
      let gm = tile_m + r;
      let gk = kt + c;
      ta[r * 16u + c] = select(0.0, B1[a_base + gm * a_row + gk * a_k], gm < m_dim && gk < k_dim);
      let gn = tile_n + r;
      tb[r * 16u + c] = select(0.0, B2[w_base + gn * w_row + gk * w_k], gn < n_dim && gk < k_dim);
      idx = idx + 64u;
    }
    workgroupBarrier();
    for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
      let a0 = ta[lx * 16u + kk];
      let a1 = ta[(lx + 1u) * 16u + kk];
      let b0 = tb[ly * 16u + kk];
      let b1 = tb[(ly + 1u) * 16u + kk];
      acc00 = acc00 + a0 * b0;
      acc01 = acc01 + a0 * b1;
      acc10 = acc10 + a1 * b0;
      acc11 = acc11 + a1 * b1;
    }
    workgroupBarrier();
    kt = kt + 16u;
  }
  if (m0 < m_dim && n0 < n_dim) {
    var v = acc00 * scale + B5[c_base + c_row * m0 + c_n * n0];
    if (has_bias == 1u) { v = v + B3[m0]; }
    B4[c_base + c_row * m0 + c_n * n0] = v;
  }
  if (m0 < m_dim && n1 < n_dim) {
    var v = acc01 * scale + B5[c_base + c_row * m0 + c_n * n1];
    if (has_bias == 1u) { v = v + B3[m0]; }
    B4[c_base + c_row * m0 + c_n * n1] = v;
  }
  if (m1 < m_dim && n0 < n_dim) {
    var v = acc10 * scale + B5[c_base + c_row * m1 + c_n * n0];
    if (has_bias == 1u) { v = v + B3[m1]; }
    B4[c_base + c_row * m1 + c_n * n0] = v;
  }
  if (m1 < m_dim && n1 < n_dim) {
    var v = acc11 * scale + B5[c_base + c_row * m1 + c_n * n1];
    if (has_bias == 1u) { v = v + B3[m1]; }
    B4[c_base + c_row * m1 + c_n * n1] = v;
  }
}
"#
}

/// matmul_tiled 汎用拡張版 (5 binding)。post_op:
/// 0=none 1=lrelu 2=relu 3=gelu(erf) 4=+residual(B5) 5=+residual+lrelu。
/// bias_axis は 1 (=B3[m]) のみ (resblock/FFN 用)。
pub fn matmul_tiled_x_src() -> String {
    header(5)
        + r#"
fn erf_of2(x: f32) -> f32 {
  let s = select(-1.0, 1.0, x >= 0.0);
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return s * (1.0 - poly * exp(-ax * ax));
}
fn apply_post(v: f32, op: u32) -> f32 {
  if (op == 1u) { return select(0.1 * v, v, v > 0.0); }
  if (op == 2u) { return select(0.0, v, v > 0.0); }
  if (op == 3u) { return 0.5 * v * (1.0 + erf_of2(v)); }
  return v;
}
var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;
@compute @workgroup_size(8, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let m_dim = P.p0.x; let n_dim = P.p0.y; let k_dim = P.p0.z; let batch = P.p0.w;
  let a_row = P.p1.x; let a_k = P.p1.y; let a_base = P.p1.z; let a_bs = P.p1.w;
  let w_row = P.p2.x; let w_k = P.p2.y; let w_base = P.p2.z; let w_bs = P.p2.w;
  let c_row = P.p3.x; let c_n = P.p3.y; let c_base = P.p3.z; let c_bs = P.p3.w;
  let has_bias = P.p4.x; let scale = f32_of(P.p4.y);
  let post = P.p4.w;
  let b_idx = 0u;
  let tile_m = wid.y * 16u;
  let tile_n = wid.x * 16u;
  let lx = lid.x * 2u;
  let ly = lid.y * 2u;
  var acc00 = 0.0; var acc01 = 0.0; var acc10 = 0.0; var acc11 = 0.0;
  let m0 = tile_m + lx;
  let m1 = tile_m + lx + 1u;
  let n0 = tile_n + ly;
  let n1 = tile_n + ly + 1u;
  var kt: u32 = 0u;
  while (kt < k_dim) {
    var idx = lid.y * 8u + lid.x;
    for (var rep: u32 = 0u; rep < 4u; rep = rep + 1u) {
      let r = idx / 16u;
      let c = idx % 16u;
      let gm = tile_m + r;
      let gk = kt + c;
      ta[r * 16u + c] = select(0.0, B1[a_base + gm * a_row + gk * a_k], gm < m_dim && gk < k_dim);
      let gn = tile_n + r;
      tb[r * 16u + c] = select(0.0, B2[w_base + gn * w_row + gk * w_k], gn < n_dim && gk < k_dim);
      idx = idx + 64u;
    }
    workgroupBarrier();
    for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
      let a0 = ta[lx * 16u + kk];
      let a1 = ta[(lx + 1u) * 16u + kk];
      let b0 = tb[ly * 16u + kk];
      let b1 = tb[(ly + 1u) * 16u + kk];
      acc00 = acc00 + a0 * b0;
      acc01 = acc01 + a0 * b1;
      acc10 = acc10 + a1 * b0;
      acc11 = acc11 + a1 * b1;
    }
    workgroupBarrier();
    kt = kt + 16u;
  }
  if (m0 < m_dim && n0 < n_dim) {
    var v = acc00 * scale;
    if (has_bias == 1u) { v = v + B3[m0]; }
    if (post >= 4u) { v = v + B5[c_base + c_row * m0 + c_n * n0]; }
    B4[c_base + c_row * m0 + c_n * n0] = apply_post(v, post);
  }
  if (m0 < m_dim && n1 < n_dim) {
    var v = acc01 * scale;
    if (has_bias == 1u) { v = v + B3[m0]; }
    if (post >= 4u) { v = v + B5[c_base + c_row * m0 + c_n * n1]; }
    B4[c_base + c_row * m0 + c_n * n1] = apply_post(v, post);
  }
  if (m1 < m_dim && n0 < n_dim) {
    var v = acc10 * scale;
    if (has_bias == 1u) { v = v + B3[m1]; }
    if (post >= 4u) { v = v + B5[c_base + c_row * m1 + c_n * n0]; }
    B4[c_base + c_row * m1 + c_n * n0] = apply_post(v, post);
  }
  if (m1 < m_dim && n1 < n_dim) {
    var v = acc11 * scale;
    if (has_bias == 1u) { v = v + B3[m1]; }
    if (post >= 4u) { v = v + B5[c_base + c_row * m1 + c_n * n1]; }
    B4[c_base + c_row * m1 + c_n * n1] = apply_post(v, post);
  }
}
"#
}

/// LN(x + y): hubert/synth の残差 + LayerNorm を 1 dispatch に融合。
/// x/y は同レイアウト [rows, cols] (in_row/in_k stride)。
pub fn add_layernorm_src() -> String {
    header(5)
        + r#"
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let rows = P.p0.x; let cols = P.p0.y; let in_row = P.p0.z; let in_k = P.p0.w;
  let out_row = P.p1.x; let eps = f32_of(P.p1.y); let out_k = P.p1.z;
  let r = wid.x;
  if (r >= rows) { return; }
  var acc = 0.0;
  var i = lid.x;
  while (i < cols) { acc = acc + B1[r * in_row + i * in_k] + B2[r * in_row + i * in_k]; i = i + 64u; }
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
  while (i < cols) {
    let d = B1[r * in_row + i * in_k] + B2[r * in_row + i * in_k] - mean;
    acc2 = acc2 + d * d;
    i = i + 64u;
  }
  red[lid.x] = acc2; workgroupBarrier();
  s = 32u;
  while (s > 0u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let var_ = red[0] / f32(cols);
  let inv_std = inverseSqrt(var_ + eps);
  i = lid.x;
  while (i < cols) {
    let v = B1[r * in_row + i * in_k] + B2[r * in_row + i * in_k];
    let g = B3[i];
    let b = B4[i];
    B5[r * out_row + i * out_k] = (v - mean) * inv_std * g + b;
    i = i + 64u;
  }
}
"#
}
