//! wgpu + WGSL 独自推論エンジン (Issue #6 セクション4/5)。
//!
//! 設計方針:
//! - fp32 厳密経路を最初に、fp16/subgroup 経路は後段で分離して用意する
//! - 重みはロード時に一度だけ GPU 常駐。演算ごとの CPU 往復なし
//! - 1 forward = 1 command encoder + 1 submit (dispatch 数を増やしても submit は増やさない)
//! - uniform は動的 offset 付きの共有バッファ (submit 前に一括 write_buffer)

pub mod shaders;

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use wgpu::util::DeviceExt;

/// 1 スロット 128B の uniform 共有バッファ。
pub const UNIFORM_SLOT_SIZE: wgpu::BufferAddress = 256; // min_uniform_buffer_offset_alignment=256
const UNIFORM_SLOTS: u32 = 4096;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Vulkan,
    Dx12,
    Auto,
}

pub struct GpuTensor {
    pub buffer: wgpu::Buffer,
    /// 論理 shape (行優先)。演算は strided access で直接扱う
    pub shape: Vec<usize>,
    pub len: usize,
}

impl GpuTensor {
    /// ハンドル複製 (バッファ同一)。
    pub fn clone_handle(&self) -> GpuTensor {
        GpuTensor {
            buffer: self.buffer.clone(),
            shape: self.shape.clone(),
            len: self.len,
        }
    }

    pub fn bytes(&self) -> wgpu::BufferAddress {
        (self.len * 4) as wgpu::BufferAddress
    }
}

pub struct Kernel {
    pub pipeline: wgpu::ComputePipeline,
    pub bgl: wgpu::BindGroupLayout,
    pub n_storage: u32,
}

pub struct Engine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    uniform_buf: wgpu::Buffer,
    kernels: HashMap<&'static str, Kernel>,
    pub has_f16: bool,
}

impl Engine {
    fn uniform_buf_get(&self) -> wgpu::Buffer {
        self.uniform_buf.clone()
    }

    /// プランを再生: 入力 copy (任意) + 単一 compute pass + submit。
    pub fn replay(&self, plan: &Plan, input_copy: Option<(&GpuTensor, &GpuTensor)>) {
        match input_copy {
            Some((src, dst)) => {
                let pair = [(src, dst.clone_handle())];
                self.replay_inputs(plan, &pair);
            }
            None => self.replay_inputs(plan, &[]),
        }
    }

    /// 複数入力 copy 対応のプラン再生。
    pub fn replay_inputs(&self, plan: &Plan, copies: &[(&GpuTensor, GpuTensor)]) {
        for (slot, words) in &plan.uniform_writes {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            self.queue.write_buffer(
                &self.uniform_buf_get(),
                *slot as u64 * UNIFORM_SLOT_SIZE,
                &bytes,
            );
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        for (src, dst) in copies {
            enc.copy_buffer_to_buffer(&src.buffer, 0, &dst.buffer, 0, src.bytes());
        }
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for cmd in &plan.cmds {
                let k = self.kernels.get(cmd.kernel).unwrap();
                cpass.set_pipeline(&k.pipeline);
                cpass.set_bind_group(0, &cmd.bind_group, &[]);
                cpass.dispatch_workgroups(cmd.wx, cmd.wy, cmd.wz);
            }
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// バッファへ f32 データを直接書き込む (永続バッファ再利用用)。
    pub fn write_f32(&self, dst: &GpuTensor, data: &[f32]) {
        self.queue
            .write_buffer(&dst.buffer, 0, bytemuck::cast_slice(data));
    }
}

fn backend_flag(b: Backend) -> wgpu::Backends {
    match b {
        Backend::Vulkan => wgpu::Backends::VULKAN,
        Backend::Dx12 => wgpu::Backends::DX12,
        Backend::Auto => wgpu::Backends::PRIMARY,
    }
}

impl Engine {
    pub fn new(backend: Backend) -> Result<Self> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = backend_flag(backend);
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .with_context(|| "request_adapter")?;
        // 要求バックエンドと異なるアダプタが来たら拒否 (Auto 以外)
        let info = adapter.get_info();
        match (backend, info.backend) {
            (Backend::Vulkan, wgpu::Backend::Vulkan)
            | (Backend::Dx12, wgpu::Backend::Dx12)
            | (Backend::Auto, _) => {}
            (want, got) => bail!("backend mismatch: requested {want:?}, got {got:?}"),
        }
        // SHADER_F16 が使えるなら要求 (fp16 compute カーネル用)
        let features = if adapter
            .features()
            .contains(wgpu::Features::SHADER_F16)
        {
            wgpu::Features::SHADER_F16
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rcwx-engine"),
            required_features: features,
            ..Default::default()
        }))
        .with_context(|| "request_device")?;
        let has_f16 = device.features().contains(wgpu::Features::SHADER_F16);
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: UNIFORM_SLOT_SIZE * UNIFORM_SLOTS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut e = Self {
            device,
            queue,
            adapter_info: info,
            uniform_buf,
            kernels: HashMap::new(),
            has_f16,
        };
        e.compile_all()?;
        Ok(e)
    }

    fn compile(&mut self, name: &'static str, src: &str, n_storage: u32) -> Result<()> {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(src)),
        });
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(128),
            },
            count: None,
        }];
        for i in 1..=n_storage {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(name),
            entries: &entries,
        });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&self.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some(name),
                        bind_group_layouts: &[Some(&bgl)],
                        immediate_size: 0,
                    },
                )),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        self.kernels.insert(
            name,
            Kernel {
                pipeline,
                bgl,
                n_storage,
            },
        );
        Ok(())
    }

    fn compile_all(&mut self) -> Result<()> {
        self.compile("matmul", &shaders::MATMUL, 4)?;
        self.compile("matmul_tiled", &shaders::matmul_tiled_src(), 4)?;
        self.compile("conv1d", &shaders::CONV1D, 4)?;
        self.compile("groupnorm", &shaders::groupnorm_src(), 4)?;
        self.compile("layernorm", &shaders::layernorm_src(), 4)?;
        self.compile("gelu", &shaders::GELU, 1)?;
        self.compile("softmax", &shaders::SOFTMAX, 1)?;
        self.compile("add", &shaders::ADD, 3)?;
        self.compile("im2col", &shaders::im2col_src(), 2)?;
        self.compile("pointwise", &shaders::pointwise_src(), 2)?;
        self.compile("conv_transpose1d", &shaders::conv_transpose1d_src(), 4)?;
        self.compile("rel_attn", &shaders::rel_attn_src(), 3)?;
        self.compile("flip_ch", &shaders::flip_ch_src(), 2)?;
        self.compile("im2col2d", &shaders::im2col2d_src(), 2)?;
        self.compile("im2col2d_k", &shaders::im2col2d_k_src(), 2)?;
        self.compile("matmul_conv", &shaders::matmul_conv_src(), 4)?;
        self.compile("slice_ch", &shaders::slice_ch_src(), 2)?;
        self.compile("avg_pool2x2", &shaders::avg_pool2x2_src(), 2)?;
        self.compile("conv_transpose2d", &shaders::conv_transpose2d_src(), 4)?;
        self.compile("concat_ch", &shaders::concat_ch_src(), 3)?;
        self.compile("transpose_flatten", &shaders::transpose_flatten_src(), 2)?;
        self.compile("pad2d", &shaders::pad2d_src(), 2)?;
        Ok(())
    }

    pub fn tensor_from_f32(&self, data: &[f32], shape: Vec<usize>) -> GpuTensor {
        let len = data.len();
        GpuTensor {
            buffer: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                }),
            shape,
            len,
        }
    }

    pub fn tensor_zeros(&self, len: usize, shape: Vec<usize>) -> GpuTensor {
        self.tensor_from_f32(&vec![0.0f32; len], shape)
    }

    /// 未初期化バッファ (全要素がカーネルで上書きされる前提)。ゼロ転送なし。
    pub fn tensor_uninit(&self, len: usize) -> GpuTensor {
        GpuTensor {
            buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: ((len * 4) as u64).max(16),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            shape: vec![len],
            len,
        }
    }

    pub fn readback_f32(&self, t: &GpuTensor) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: t.bytes(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&t.buffer, 0, &staging, 0, t.bytes());
        let idx = self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None })
            .expect("poll");
        let data = slice.get_mapped_range().expect("map").to_vec();
        staging.unmap();
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// デバッグ用: uniform バッファの先頭 n u32 を読む。
    pub fn dump_uniform(&self, n: usize) -> Vec<u32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform-dump"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.uniform_buf, 0, &staging, 0, (n * 4) as u64);
        let idx = self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).unwrap();
        let data = slice.get_mapped_range().unwrap().to_vec();
        staging.unmap();
        data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// 1 forward 分の記録。uniform は submit 時に一括書き込み。
    pub fn graph(&self) -> Graph<'_> {
        Graph {
            engine: self,
            encoder: self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }),
            uniform_writes: Vec::new(),
            next_slot: 0,
            cmds: Vec::new(),
        }
    }
}

impl<'a> Graph<'a> {
    /// engine 上に未初期化テンソル (プラン構築中の作業領域)。
    pub fn engine_tensor_uninit(&self, len: usize) -> GpuTensor {
        self.engine.tensor_uninit(len)
    }
}

pub struct Graph<'a> {
    pub engine: &'a Engine,
    encoder: wgpu::CommandEncoder,
    uniform_writes: Vec<(u32, [u32; 32])>, // (slot, words)
    next_slot: u32,
    cmds: Vec<Cmd>,
}

pub struct Cmd {
    kernel: &'static str,
    bind_group: wgpu::BindGroup,
    wx: u32,
    wy: u32,
    wz: u32,
}

/// 形状ごとに一度だけ構築する実行プラン (バッファは外で保持、uniform は構築時に書き込み済み)。
pub struct Plan {
    pub cmds: Vec<Cmd>,
    pub n_cmds: usize,
    /// この plan が使う uniform スロットの内容 (replay 時に再書込 — 他モデルが上書き防止)
    pub uniform_writes: Vec<(u32, [u32; 32])>,
}

impl<'a> Graph<'a> {
    fn alloc_slot(&mut self, words: [u32; 32]) -> u32 {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.uniform_writes.push((slot, words));
        slot
    }

    fn dispatch(
        &mut self,
        kernel: &'static str,
        bufs: &[&GpuTensor],
        words: [u32; 32],
        wx: u32,
        wy: u32,
        wz: u32,
    ) {
        let eng = self.engine;
        let k = eng.kernels.get(kernel).unwrap();
        let slot = self.alloc_slot(words);
        let ubuf = eng.uniform_buf_get();
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &ubuf,
                offset: slot as u64 * UNIFORM_SLOT_SIZE,
                size: wgpu::BufferSize::new(128),
            }),
        }];
        for (i, t) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &t.buffer,
                    offset: 0,
                    size: None,
                }),
            });
        }
        let bg = eng.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &k.bgl,
            entries: &entries,
        });
        let _ = slot;
        // 1D grid 上限 (65535) 対応: x を 512 列 × y 行に分割
        let (wx, wy) = if wy == 1 && wz == 1 && wx > 65535 {
            (512u32, wx.div_ceil(512))
        } else {
            (wx, wy)
        };
        self.cmds.push(Cmd {
            kernel,
            bind_group: bg,
            wx,
            wy,
            wz,
        });
    }

    /// 汎用 matmul: C[b, m, n] = scale · (Σ_k A[b, m, k] · W[b, n, k]) + bias?
    /// A/W/C は base + 行stride/kstride (要素数単位) で任意レイアウトに対応。
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &mut self,
        a: &GpuTensor,
        w: &GpuTensor,
        bias: Option<&GpuTensor>,
        c: &GpuTensor,
        m: u32,
        n: u32,
        k: u32,
        batch: u32,
        a_row_stride: u32,
        a_k_stride: u32,
        a_base: u32,
        a_batch_stride: u32,
        w_row_stride: u32,
        w_k_stride: u32,
        w_base: u32,
        w_batch_stride: u32,
        c_row_stride: u32,
        c_n_stride: u32,
        c_base: u32,
        c_batch_stride: u32,
        scale: f32,
    ) {
        self.matmul_ex(
            a, w, bias, c, m, n, k, batch,
            a_row_stride, a_k_stride, a_base, a_batch_stride,
            w_row_stride, w_k_stride, w_base, w_batch_stride,
            c_row_stride, c_n_stride, c_base, c_batch_stride,
            scale, 0,
        );
    }

    /// bias_axis: 0 = bias[n] (C が [m, out] レイアウト)、1 = bias[m] (C が [out, m] 転置書き込み)。
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_ex(
        &mut self,
        a: &GpuTensor,
        w: &GpuTensor,
        bias: Option<&GpuTensor>,
        c: &GpuTensor,
        m: u32,
        n: u32,
        k: u32,
        batch: u32,
        a_row_stride: u32,
        a_k_stride: u32,
        a_base: u32,
        a_batch_stride: u32,
        w_row_stride: u32,
        w_k_stride: u32,
        w_base: u32,
        w_batch_stride: u32,
        c_row_stride: u32,
        c_n_stride: u32,
        c_base: u32,
        c_batch_stride: u32,
        scale: f32,
        bias_axis: u32,
    ) {
        let has_bias = bias.is_some();
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[m, n, k, batch]);
        words[4..8].copy_from_slice(&[a_row_stride, a_k_stride, a_base, a_batch_stride]);
        words[8..12].copy_from_slice(&[w_row_stride, w_k_stride, w_base, w_batch_stride]);
        words[12..16].copy_from_slice(&[c_row_stride, c_n_stride, c_base, c_batch_stride]);
        words[16] = has_bias as u32;
        words[17] = scale.to_bits();
        words[18] = bias_axis;
        // bias は binding 3。has_bias=0 でも bind が必要なので dummy として a を渡す
        let dummy = a;
        let bias_ref = bias.unwrap_or(dummy);
        let use_tiled = k >= 128;
        let (kernel, wx, wy) = if use_tiled {
            ("matmul_tiled", n.div_ceil(16), m.div_ceil(16))
        } else {
            ("matmul", n.div_ceil(8), m.div_ceil(8))
        };
        self.dispatch(kernel, &[a, w, bias_ref, c], words, wx, wy, batch);
    }

    /// 1D conv (zero padding, 任意 stride/dilation/groups, bias 任意)。
    /// X/Out は [C, L] レイアウト、W は [out_c, in_c/groups, k]。
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        bias: Option<&GpuTensor>,
        out: &GpuTensor,
        out_c: u32,
        in_c: u32,
        kernel: u32,
        stride: u32,
        dilation: u32,
        pad_l: u32,
        in_len: u32,
        out_len: u32,
        groups: u32,
    ) {
        let has_bias = bias.is_some();
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[out_c, in_c, kernel, dilation]);
        words[4..8].copy_from_slice(&[stride, pad_l, in_len, out_len]);
        words[8..12].copy_from_slice(&[groups, has_bias as u32, 0, 0]);
        let dummy = x;
        let bias_ref = bias.unwrap_or(dummy);
        self.dispatch(
            "conv1d",
            &[x, w, bias_ref, out],
            words,
            out_len.div_ceil(8),
            out_c.div_ceil(8),
            1,
        );
    }

    /// per-channel GroupNorm (groups == channels) over T。in-place。
    pub fn group_norm(&mut self, x: &GpuTensor, gamma: &GpuTensor, beta: &GpuTensor, c: u32, t: u32, eps: f32) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[c, t, eps.to_bits(), 0]);
        self.dispatch("groupnorm", &[x, gamma, beta, x], words, c, 1, 1);
    }

    /// LayerNorm (最終軸 C)。in_row/in_k stride で入力レイアウト、out_row_stride で出力
    /// (転置書き込み可能)。rows = 行数。
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(
        &mut self,
        x: &GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        out: &GpuTensor,
        rows: u32,
        cols: u32,
        in_row_stride: u32,
        in_k_stride: u32,
        out_row_stride: u32,
        out_k_stride: u32,
        eps: f32,
    ) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[rows, cols, in_row_stride, in_k_stride]);
        words[4..8].copy_from_slice(&[out_row_stride, eps.to_bits(), out_k_stride, 0]);
        self.dispatch("layernorm", &[x, gamma, beta, out], words, rows, 1, 1);
    }

    /// pointwise: op 0=copy 1=lrelu 2=tanh 3=gelu_approx 4=mul_scalar 5=gating 6=flip_ch(out へ)
    pub fn pointwise(&mut self, x: &GpuTensor, out: Option<&GpuTensor>, len: u32, op: u32, param: f32, t_len: u32, ch: u32, h: u32) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[len, op, param.to_bits(), ch]);
        words[4] = t_len;
        words[5] = h;
        match out {
            Some(o) => self.dispatch("pointwise", &[x, o], words, len.div_ceil(256), 1, 1),
            None => self.dispatch("pointwise", &[x, x], words, len.div_ceil(256), 1, 1),
        }
    }

    pub fn relu(&mut self, x: &GpuTensor, len: u32) {
        self.pointwise(x, None, len, 7, 0.0, 0, 0, 0);
    }

    pub fn im2col2d(&mut self, x: &GpuTensor, ic: &GpuTensor, c: u32, h: u32, w: u32) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[c, h, w, 0]);
        let total = c * 9 * h * w;
        let wg = total.div_ceil(256);
        if wg <= 65535 {
            self.dispatch("im2col2d", &[x, ic], words, wg, 1, 1);
        } else {
            // 2D grid: y = 行ブロック、x = 列 (合計 ≥ wg をカバー)
            let cols = 512u32;
            let rows = wg.div_ceil(cols);
            self.dispatch("im2col2d", &[x, ic], words, cols, rows, 1);
        }
    }

    /// チャネル major [ch, n_in] の [off, off+n_out) フレームを [ch, n_out] へ (GPU 直結)。
    pub fn slice_ch(&mut self, src: &GpuTensor, dst: &GpuTensor, ch: u32, n_in: u32, off: u32, n_out: u32) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[n_out, n_in, ch, off]);
        let total = ch * n_out;
        let wg = total.div_ceil(256);
        if wg <= 65535 {
            self.dispatch("slice_ch", &[src, dst], words, wg, 1, 1);
        } else {
            self.dispatch("slice_ch", &[src, dst], words, 512, wg.div_ceil(512), 1);
        }
    }

    /// implicit-GEMM conv1d (stride 1・tiled matmul 構造・im2col 省略)。
    /// a = 重み [cout, k_dim=cin*ck], x = 信号 [cin, in_len], out = [cout, len]。
    /// post_op: 0=none 1=lrelu。
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_conv(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        bias: Option<&GpuTensor>,
        c: &GpuTensor,
        m: u32,
        n: u32,
        k_dim: u32,
        conv_k: u32,
        dil: u32,
        pad: u32,
        in_len: u32,
        post_op: u32,
    ) {
        let has_bias = bias.is_some();
        let mut w = [0u32; 32];
        w[0..4].copy_from_slice(&[m, n, k_dim, 1]);
        w[4..8].copy_from_slice(&[k_dim, 1, 0, 0]); // a_row=k_dim, a_k=1
        w[12..16].copy_from_slice(&[n, 1, 0, 0]);   // c_row=n, c_n=1
        w[16..20].copy_from_slice(&[has_bias as u32, 1.0f32.to_bits(), 1, conv_k]);
        w[20..24].copy_from_slice(&[pad, dil, in_len, post_op]);
        let dummy = a;
        let bias_ref = bias.unwrap_or(dummy);
        // f16 版は現状 DXC が packed 命令を出さず速度劣化 + 数値劣化のため無効
        self.dispatch("matmul_conv", &[a, x, bias_ref, c], w, n.div_ceil(16), m.div_ceil(16), 1);
    }

    /// 汎用 im2col2d (kernel k・pad p)。出力 [c*k*k, h*w]。
    pub fn im2col2d_k(&mut self, x: &GpuTensor, ic: &GpuTensor, c: u32, h: u32, w: u32, k: u32, pad: u32) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[c, h, w, k]);
        words[4] = pad;
        let total = c * k * k * h * w;
        let wg = total.div_ceil(256);
        if wg <= 65535 {
            self.dispatch("im2col2d_k", &[x, ic], words, wg, 1, 1);
        } else {
            let cols = 512u32;
            let rows = wg.div_ceil(cols);
            self.dispatch("im2col2d_k", &[x, ic], words, cols, rows, 1);
        }
    }

    pub fn avg_pool2x2(&mut self, x: &GpuTensor, out: &GpuTensor, c: u32, h: u32, w: u32) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[c, h, w, 0]);
        let total = c * (h / 2) * (w / 2);
        self.dispatch("avg_pool2x2", &[x, out], words, total.div_ceil(256), 1, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose2d(&mut self, x: &GpuTensor, w: &GpuTensor, bias: Option<&GpuTensor>, out: &GpuTensor,
        cin: u32, cout: u32, h_in: u32, w_in: u32, h_out: u32, w_out: u32,
        sh: u32, sw: u32, oph: u32, opw: u32) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[cin, cout, h_out, w_out]);
        words[4..8].copy_from_slice(&[h_in, w_in, sh, sw]);
        words[8..10].copy_from_slice(&[oph, opw]);
        let dummy = x;
        let bias_ref = bias.unwrap_or(dummy);
        self.dispatch("conv_transpose2d", &[x, w, out, bias_ref], words, w_out.div_ceil(8), h_out.div_ceil(8), cout);
    }

    pub fn concat_ch(&mut self, a: &GpuTensor, b: &GpuTensor, out: &GpuTensor, ca: u32, cb: u32, n: u32) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[n, ca, cb, 0]);
        self.dispatch("concat_ch", &[a, b, out], words, ((ca + cb) * n).div_ceil(256), 1, 1);
    }

    pub fn transpose_flatten(&mut self, x: &GpuTensor, out: &GpuTensor, c: u32, t: u32, m: u32) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[c, t, m, 0]);
        self.dispatch("transpose_flatten", &[x, out], words, (c * t * m).div_ceil(256), 1, 1);
    }

    pub fn pad2d(&mut self, src: &GpuTensor, dst: &GpuTensor, c: u32, h: u32, w: u32, h2: u32, w2: u32) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[c, h, w, 0]);
        words[4..6].copy_from_slice(&[w2, h2]);
        self.dispatch("pad2d", &[src, dst], words, (c * h * w).div_ceil(256), 1, 1);
    }

    pub fn lrelu(&mut self, x: &GpuTensor, len: u32) {
        self.pointwise(x, None, len, 1, 0.1, 0, 0, 0);
    }

    pub fn tanh_inplace(&mut self, x: &GpuTensor, len: u32) {
        self.pointwise(x, None, len, 2, 0.0, 0, 0, 0);
    }

    /// gating: x [2H, T] の前半に tanh(a)*sigmoid(b) を書き込み (後半は未使用のまま)
    pub fn gating(&mut self, x: &GpuTensor, t_len: u32, h: u32) {
        let len = h * t_len; // a 側のみ
        self.pointwise(x, None, len, 5, 0.0, t_len, 0, h);
    }

    pub fn flip_ch(&mut self, x: &GpuTensor, out: &GpuTensor, t_len: u32, ch: u32) {
        let total = ch * t_len;
        self.pointwise(x, Some(out), total, 6, 0.0, t_len, ch, 0);
    }

    /// 1D 転置畳み込み (stride 付き, zero pad 左のみ指定)。
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        bias: Option<&GpuTensor>,
        out: &GpuTensor,
        out_c: u32,
        in_c: u32,
        kernel: u32,
        stride: u32,
        pad_l: u32,
        in_len: u32,
        out_len: u32,
    ) {
        let has_bias = bias.is_some();
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[out_c, in_c, kernel, stride]);
        words[4..8].copy_from_slice(&[pad_l, in_len, out_len, has_bias as u32]);
        let dummy = x;
        let bias_ref = bias.unwrap_or(dummy);
        self.dispatch("conv_transpose1d", &[x, w, bias_ref, out], words, out_len.div_ceil(8), out_c.div_ceil(8), 1);
    }

    /// 相対位置 attention 補助。mode0: scores += (q·relk)*scale。mode1: out += p·relv。
    pub fn rel_attn_scores(&mut self, q: &GpuTensor, relk: &GpuTensor, scores: &GpuTensor, length: u32, d: u32, ws: u32, heads: u32, scale: f32) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[0, length, d, ws]);
        words[4..6].copy_from_slice(&[scale.to_bits(), heads]);
        self.dispatch("rel_attn", &[q, relk, scores], words, length.div_ceil(8), length.div_ceil(8), heads);
    }

    pub fn rel_attn_values(&mut self, p: &GpuTensor, relv: &GpuTensor, out: &GpuTensor, length: u32, d: u32, ws: u32, heads: u32) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[1, length, d, ws]);
        words[5] = heads;
        self.dispatch("rel_attn", &[p, relv, out], words, d.div_ceil(8), length.div_ceil(8), heads);
    }

    /// im2col: IC[c*k+j, t] = X[c, t*stride - pad + j*dil]。戻り値バッファは呼び出し側が用意。
    pub fn im2col(
        &mut self,
        x: &GpuTensor,
        ic: &GpuTensor,
        c_in: u32,
        kernel: u32,
        stride: u32,
        dilation: u32,
        pad_l: u32,
        in_len: u32,
        out_len: u32,
    ) {
        self.im2col_off(x, 0, ic, c_in, kernel, stride, dilation, pad_l, in_len, out_len);
    }

    pub fn im2col_off(
        &mut self,
        x: &GpuTensor,
        x_base: u32,
        ic: &GpuTensor,
        c_in: u32,
        kernel: u32,
        stride: u32,
        dilation: u32,
        pad_l: u32,
        in_len: u32,
        out_len: u32,
    ) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[c_in, kernel, dilation, x_base]);
        words[4..8].copy_from_slice(&[stride, pad_l, in_len, out_len]);
        let total = c_in * kernel * out_len;
        self.dispatch("im2col", &[x, ic], words, total.div_ceil(256), 1, 1);
    }

    pub fn gelu(&mut self, x: &GpuTensor, len: u32) {
        let mut words = [0u32; 32];
        words[0] = len;
        self.dispatch("gelu", &[x], words, len.div_ceil(256), 1, 1);
    }

    /// row softmax (in-place)。rows × cols。
    pub fn softmax(&mut self, x: &GpuTensor, rows: u32, cols: u32) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[rows, cols, 0, 0]);
        self.dispatch("softmax", &[x], words, rows, 1, 1);
    }

    pub fn add(&mut self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor, len: u32) {
        self.add_off(a, b, c, len, 0, 0, false);
    }

    /// out[c_base+i] = a[a_base+i] ± b[b_base+i]
    pub fn add_off(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
        len: u32,
        a_base: u32,
        b_base: u32,
        subtract: bool,
    ) {
        self.add_off3(a, b, c, len, a_base, b_base, 0, subtract);
    }

    pub fn add_off3(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
        len: u32,
        a_base: u32,
        b_base: u32,
        c_base: u32,
        subtract: bool,
    ) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[len, subtract as u32, a_base, b_base]);
        words[4] = c_base;
        self.dispatch("add", &[a, b, c], words, len.div_ceil(256), 1, 1);
    }

    /// テスト用: エイリアス無し GN (出力バッファ分離)。
    pub fn dispatch_test_gn_noalias(
        &mut self,
        x: &GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        out: &GpuTensor,
        c: u32,
        t: u32,
    ) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[c, t, 1e-5f32.to_bits(), 0]);
        self.dispatch("groupnorm", &[x, gamma, beta, out], words, c, 1, 1);
    }

    /// uniform を書き込んで cmds を取り出す (毎ホップ再利用するプラン構築用)。
    pub fn into_plan(mut self) -> Result<Plan> {
        let eng = self.engine;
        for (slot, words) in &self.uniform_writes {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            eng.queue.write_buffer(
                &eng.uniform_buf_get(),
                *slot as u64 * UNIFORM_SLOT_SIZE,
                &bytes,
            );
        }
        Ok(Plan {
            n_cmds: self.cmds.len(),
            cmds: std::mem::take(&mut self.cmds),
            uniform_writes: std::mem::take(&mut self.uniform_writes),
        })
    }

    /// プランを再生 (engine 委譲)。
    pub fn replay(&self, plan: &Plan, input_copy: Option<(&GpuTensor, &GpuTensor)>) {
        self.engine.replay(plan, input_copy);
    }


    /// uniform を一括書き込みし、**単一の compute pass** に全 dispatch を記録して submit。
    pub fn submit(mut self) -> Result<()> {
        let eng = self.engine;
        for (slot, words) in &self.uniform_writes {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            eng.queue.write_buffer(
                &eng.uniform_buf_get(),
                *slot as u64 * UNIFORM_SLOT_SIZE,
                &bytes,
            );
        }
        {
            let mut cpass = self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for cmd in &self.cmds {
                let k = eng.kernels.get(cmd.kernel).unwrap();
                cpass.set_pipeline(&k.pipeline);

                cpass.set_bind_group(0, &cmd.bind_group, &[]);
                cpass.dispatch_workgroups(cmd.wx, cmd.wy, cmd.wz);
            }
        }
        eng.queue.submit(Some(self.encoder.finish()));
        Ok(())
    }
}
