//! wgpu + WGSL 独自推論エンジン (Issue #6 セクション4/5)。
//!
//! 設計方針:
//! - fp32 厳密経路を最初に、fp16/subgroup 経路は後段で分離して用意する
//! - 重みはロード時に一度だけ GPU 常駐。演算ごとの CPU 往復なし
//! - 1 forward = 1 command encoder + 1 submit (dispatch 数を増やしても submit は増やさない)
//! - uniform は動的 offset 付きの共有バッファ (submit 前に一括 write_buffer)

pub mod shaders;
pub mod tape;
pub mod train_shaders;

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result, bail};
use wgpu::util::DeviceExt;

/// 1 スロット 128B の uniform 共有バッファ。
pub const UNIFORM_SLOT_SIZE: wgpu::BufferAddress = 256; // min_uniform_buffer_offset_alignment=256
const UNIFORM_SLOTS: u32 = 16384;

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
    /// TIMESTAMP_QUERY が有効か (診断用 GPU 時間計測)。
    pub has_ts: bool,
    /// MAP_READ の作り直しを避ける。サイズごとに 1 本。
    staging_pool: Mutex<Vec<(wgpu::BufferAddress, wgpu::Buffer)>>,
    /// 常設 upload バッファ。write_f32 の transient resource churn を
    /// 正規 map_async/unmap + copy で除去する。
    upload: Mutex<UploadRing>,
    /// submit 単調エポック (リングが安全に map できるかの判定)。
    submit_epoch: std::sync::atomic::AtomicU64,
    /// 完了確認済みエポックの下限 (fence 待ちで更新)。
    drained_epoch: std::sync::atomic::AtomicU64,
    /// プランごとにユニークな uniform スロット。ホップ毎の上書きをしない。
    next_uniform: std::sync::atomic::AtomicU32,
    /// 学習経路で使い捨て Graph が返却したスロットのフリーリスト。
    /// プラン (into_plan) 化されたスロットはここへ戻らない = rt 経路は影響なし。
    free_uniform: Mutex<Vec<u32>>,
}

/// `Queue::write_buffer` は呼び出しごとに transient placed resource を
/// 生成→life tracker 経由で破棄する。この churn がドライバの residency
/// 更新を誘発し、fence 完了が ~50ms 遅れる周期税 (HuBERT 読み戻し側) の
/// 原因 (write を全スキップすると消えることを実証済み)。
/// 常設バッファに正規 map_async で書き、copy を次の submit に carrier として
/// 同梱する (アロケーション churn ゼロ)。
///
/// なお wgpu 生成バッファを背後から生 Map する案は、このドライバでは
/// ポインタが GPU 可視メモリとエイリアスせず書き込みが届かない
/// (GPU 側 readback で全ゼロ) ため採用していない。
///
/// 安全条件: map は走行中 submit の完了を待つ仕様なので、書き込みは
/// 直前 submit が fence 完了済み (drained) のときだけ行う。未ドレン
/// (オフライン一括処理など) は write_buffer にフォールバックする。
struct UploadRing {
    enabled: bool,
    buf: Option<wgpu::Buffer>,
    cap: u64,
    used: u64,
    pending: Vec<PendingUpload>,
}

impl UploadRing {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            buf: None,
            cap: 0,
            used: 0,
            pending: Vec::new(),
        }
    }

    /// 必要容量を保証。不足なら作り直す (旧バッファは pending の src クローンが
    /// flush まで生かすので破棄されない)。
    fn ensure(&mut self, device: &wgpu::Device, needed: u64) -> bool {
        if self.buf.is_some() && self.cap >= needed {
            return true;
        }
        let cap = needed.max(1 << 20);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rcwx-upload"),
            size: cap,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.buf = Some(buf);
        self.cap = cap;
        self.used = 0;
        true
    }
}

struct PendingUpload {
    src: wgpu::Buffer,
    src_off: wgpu::BufferAddress,
    dst: wgpu::Buffer,
    size: wgpu::BufferAddress,
}

impl Engine {
    fn uniform_buf_get(&self) -> wgpu::Buffer {
        self.uniform_buf.clone()
    }

    /// uniform スロット確保。フリーリスト優先、無ければ単調カウンタ。
    fn alloc_uniform_slot(&self) -> u32 {
        if let Some(s) = self
            .free_uniform
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .pop()
        {
            return s;
        }
        let s = self.next_uniform.fetch_add(1, Ordering::Relaxed);
        assert!(s < UNIFORM_SLOTS, "uniform slot overflow");
        s
    }

    /// 使い捨て Graph のスロットを回収。直前 submit との競合は queue の FIFO 順で
    /// 保証される (書き戻しは次回 submit 先頭の write_buffer として適用される)。
    fn recycle_uniform_slots(&self, slots: impl Iterator<Item = u32>) {
        let mut free = self
            .free_uniform
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        free.extend(slots);
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
        // uniform はプラン構築時にスロットを分けて書いてある。再生のたびに書き直すと
        // DX12 の staging belt が埋まり、readback 待ちが ~50ms 伸びる。
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
        self.submit_flushed(enc.finish());
    }

    /// 複数プランを 1 回の submit にまとめる。uniform スロットはプラン間で重ならない前提。
    pub fn replay_plans(&self, plans: &[&Plan]) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for plan in plans {
                for cmd in &plan.cmds {
                    let k = self.kernels.get(cmd.kernel).unwrap();
                    cpass.set_pipeline(&k.pipeline);
                    cpass.set_bind_group(0, &cmd.bind_group, &[]);
                    cpass.dispatch_workgroups(cmd.wx, cmd.wy, cmd.wz);
                }
            }
        }
        self.submit_flushed(enc.finish());
    }

    fn write_uniforms(&self, writes: &[(u32, [u32; 32])]) {
        if writes.is_empty() {
            return;
        }
        let min_slot = writes.iter().map(|(slot, _)| *slot).min().unwrap();
        let max_slot = writes.iter().map(|(slot, _)| *slot).max().unwrap();
        let slots = (max_slot - min_slot + 1) as usize;
        let mut bytes = vec![0u8; slots * UNIFORM_SLOT_SIZE as usize];
        for (slot, words) in writes {
            let off = (*slot - min_slot) as usize * UNIFORM_SLOT_SIZE as usize;
            for (i, word) in words.iter().enumerate() {
                bytes[off + i * 4..off + i * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        self.queue.write_buffer(
            &self.uniform_buf_get(),
            min_slot as u64 * UNIFORM_SLOT_SIZE,
            &bytes,
        );
    }

    /// バッファへ f32 データを直接書き込む (永続バッファ再利用用)。
    /// DX12 は常設 upload リング経由 (アロケーションゼロ)。失敗/無効時は
    /// write_buffer (呼び出しごとに transient resource が churn する) に倒れる。
    pub fn write_f32(&self, dst: &GpuTensor, data: &[f32]) {
        if self.upload_ring_write(dst, data) {
            return;
        }
        self.queue
            .write_buffer(&dst.buffer, 0, bytemuck::cast_slice(data));
    }

    /// 常設 upload リングへの書き込み。戻り値 false はフォールバック指示。
    fn upload_ring_write(&self, dst: &GpuTensor, data: &[f32]) -> bool {
        let len = (data.len() * 4) as wgpu::BufferAddress;
        {
            let ring = self.upload.lock().unwrap_or_else(|err| err.into_inner());
            if !ring.enabled {
                return false;
            }
        }
        // map は走行中 submit の完了を内部で待つ仕様。未ドレンなら従来の
        // write_buffer へ (rt は毎バッチ完全ドレンなので常にリングを使う)。
        if self.submit_epoch.load(Ordering::Acquire) > self.drained_epoch.load(Ordering::Acquire) {
            return false;
        }
        let bytes = bytemuck::cast_slice(data);
        let (used, buf) = {
            let mut ring = self.upload.lock().unwrap_or_else(|err| err.into_inner());
            let used = ring.used;
            if !ring.ensure(&self.device, used + len) {
                ring.enabled = false;
                return false;
            }
            (used, ring.buf.as_ref().unwrap().clone())
        };
        if !self.map_write_bytes(&buf, used, bytes) {
            let mut ring = self.upload.lock().unwrap_or_else(|err| err.into_inner());
            ring.enabled = false;
            return false;
        }
        let mut ring = self.upload.lock().unwrap_or_else(|err| err.into_inner());
        ring.pending.push(PendingUpload {
            src: buf,
            src_off: used,
            dst: dst.buffer.clone(),
            size: len,
        });
        ring.used = used + len;
        true
    }

    /// 正規 map_async → 書き込み → unmap。ドレン済みなので即時完了する。
    fn map_write_bytes(&self, buf: &wgpu::Buffer, off: u64, bytes: &[u8]) -> bool {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        buf.slice(..).map_async(wgpu::MapMode::Write, move |_| {
            let _ = tx.send(());
        });
        let started = std::time::Instant::now();
        loop {
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_millis(5)),
            });
            if rx.try_recv().is_ok() {
                break;
            }
            if started.elapsed() > std::time::Duration::from_secs(2) {
                return false;
            }
        }
        {
            let Ok(mut range) = buf.slice(..).get_mapped_range_mut() else {
                return false;
            };
            let lo = off as usize;
            range.slice(lo..lo + bytes.len()).copy_from_slice(bytes);
        }
        buf.unmap();
        true
    }

    /// upload リングの pending copy を carrier として先に流し、本体と同じ
    /// submit で実行する (carrier → 本体の順序はキューの FIFO で保証)。
    /// 戻り値は (submission index, epoch)。epoch は drained 管理に使う。
    fn submit_flushed2(&self, buf: wgpu::CommandBuffer) -> (wgpu::SubmissionIndex, u64) {
        let epoch = self.submit_epoch.fetch_add(1, Ordering::Relaxed) + 1;
        let mut ring = self.upload.lock().unwrap_or_else(|err| err.into_inner());
        if ring.pending.is_empty() {
            let idx = self.queue.submit(Some(buf));
            return (idx, epoch);
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for up in ring.pending.drain(..) {
            enc.copy_buffer_to_buffer(&up.src, up.src_off, &up.dst, 0, up.size);
        }
        let carrier = enc.finish();
        let idx = self.queue.submit([carrier, buf]);
        ring.used = 0;
        (idx, epoch)
    }

    fn submit_flushed(&self, buf: wgpu::CommandBuffer) -> wgpu::SubmissionIndex {
        let (idx, _epoch) = self.submit_flushed2(buf);
        idx
    }

    fn mark_drained(&self, epoch: u64) {
        self.drained_epoch.fetch_max(epoch, Ordering::Release);
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
        // デバイス生成より前。CreateCommandQueue が NORMAL のまま作られると後から上げられない。
        if !matches!(backend, Backend::Vulkan) {
            install_dx12_high_priority_queue();
        }
        boost_gpu_scheduling_priority();
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
        let mut features = if adapter.features().contains(wgpu::Features::SHADER_F16) {
            wgpu::Features::SHADER_F16
        } else {
            wgpu::Features::empty()
        };
        // 診断用 timestamp query (RCWX_TS_PROBE)
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rcwx-engine"),
            required_features: features,
            ..Default::default()
        }))
        .with_context(|| "request_device")?;
        let has_f16 = device.features().contains(wgpu::Features::SHADER_F16);
        let has_ts = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        // 実測 (Arc B570 / 32.0.101.9150): per-write の map_async/unmap は
        // 4 バッチ周期の完了遅延 (~50ms) を機械的に誘発し write_buffer (5 回/バッ
        // チの transient 生成) より悪い。submit 統合 (2 回/バッチ) と組み合わせた
        // write_buffer が最良 (税 2-3 件/48 バッチ)。リングは既定 OFF とし、
        // ドライバ更新時に RCWX_RING=1 で再検証する。
        let use_upload_ring = std::env::var_os("RCWX_RING").is_some();
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: UNIFORM_SLOT_SIZE * UNIFORM_SLOTS as u64,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut e = Self {
            device,
            queue,
            adapter_info: info,
            uniform_buf,
            kernels: HashMap::new(),
            has_f16,
            has_ts,
            staging_pool: Mutex::new(Vec::new()),
            upload: Mutex::new(UploadRing::new(use_upload_ring)),
            submit_epoch: std::sync::atomic::AtomicU64::new(0),
            drained_epoch: std::sync::atomic::AtomicU64::new(0),
            next_uniform: std::sync::atomic::AtomicU32::new(0),
            free_uniform: Mutex::new(Vec::new()),
        };
        disable_d3d12_background_work(&e.queue);
        e.compile_all()?;
        Ok(e)
    }
}

/// wgpu-hal はキューを `D3D12_COMMAND_QUEUE_PRIORITY_NORMAL` で作る。優先度は生成後に
/// 変えられないので、`D3D12CreateDevice` の戻り値の vtable を差し替えて HIGH で作らせる。
/// NORMAL のままだと Arc のドライバが compute を溜めて、fence が約 50ms 遅れる。
fn install_dx12_high_priority_queue() {
    if std::env::var("RCWX_GPU_PRIORITY").ok().as_deref() == Some("0") {
        return;
    }
    #[cfg(windows)]
    unsafe {
        install_dx12_queue_hook();
    }
}

#[cfg(windows)]
unsafe fn install_dx12_queue_hook() {
    use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
    static ONCE: AtomicBool = AtomicBool::new(false);
    if ONCE.swap(true, Ordering::Relaxed) {
        return;
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleW(name: *const u16) -> *mut core::ffi::c_void;
        fn VirtualProtect(
            addr: *mut core::ffi::c_void,
            size: usize,
            new_protect: u32,
            old: *mut u32,
        ) -> i32;
    }
    type GetProcAddressFn =
        unsafe extern "system" fn(*mut core::ffi::c_void, *const u8) -> *mut core::ffi::c_void;
    type CreateDeviceFn = unsafe extern "system" fn(
        *mut core::ffi::c_void,
        u32,
        *const u8,
        *mut *mut core::ffi::c_void,
    ) -> i32;
    type CreateQueueFn = unsafe extern "system" fn(
        *mut core::ffi::c_void,
        *const QueueDesc,
        *const u8,
        *mut *mut core::ffi::c_void,
    ) -> i32;
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct QueueDesc {
        ty: u32,
        priority: i32,
        flags: u32,
        node_mask: u32,
    }
    static ORIG_GPA: AtomicPtr<core::ffi::c_void> = AtomicPtr::new(core::ptr::null_mut());
    static ORIG_CREATE_DEVICE: AtomicPtr<core::ffi::c_void> = AtomicPtr::new(core::ptr::null_mut());
    static ORIG_CREATE_QUEUE: AtomicPtr<core::ffi::c_void> = AtomicPtr::new(core::ptr::null_mut());

    unsafe extern "system" fn create_queue_hook(
        this: *mut core::ffi::c_void,
        desc: *const QueueDesc,
        riid: *const u8,
        out: *mut *mut core::ffi::c_void,
    ) -> i32 {
        let orig: CreateQueueFn =
            unsafe { std::mem::transmute(ORIG_CREATE_QUEUE.load(Ordering::Acquire)) };
        if desc.is_null() {
            return unsafe { orig(this, desc, riid, out) };
        }
        let mut raised = unsafe { *desc };
        // HIGH = 100。GLOBAL_REALTIME (10000) は権限がないと失敗する。
        if raised.priority < 100 {
            raised.priority = 100;
            let hr = unsafe { orig(this, &raised, riid, out) };
            if hr >= 0 {
                eprintln!("[GPU-PRIO] D3D12 command queue priority = HIGH");
                return hr;
            }
        }
        unsafe { orig(this, desc, riid, out) }
    }

    unsafe extern "system" fn create_device_hook(
        adapter: *mut core::ffi::c_void,
        level: u32,
        riid: *const u8,
        out: *mut *mut core::ffi::c_void,
    ) -> i32 {
        let orig: CreateDeviceFn =
            unsafe { std::mem::transmute(ORIG_CREATE_DEVICE.load(Ordering::Acquire)) };
        let hr = unsafe { orig(adapter, level, riid, out) };
        if hr < 0 || out.is_null() {
            return hr;
        }
        let device = unsafe { *out };
        if device.is_null() {
            return hr;
        }
        // ID3D12Device::CreateCommandQueue は vtable index 8
        // (IUnknown 3 + ID3D12Object 4 + GetNodeCount)。
        unsafe {
            let vtable = *(device as *mut *mut *const core::ffi::c_void);
            let slot = vtable.add(8);
            if ORIG_CREATE_QUEUE.load(Ordering::Acquire).is_null() {
                ORIG_CREATE_QUEUE.store(*slot as *mut core::ffi::c_void, Ordering::Release);
            }
            let mut old = 0u32;
            if VirtualProtect(slot as *mut core::ffi::c_void, 8, 0x04, &mut old) != 0 {
                *slot = create_queue_hook as *const core::ffi::c_void;
                let mut ignored = 0u32;
                let _ = VirtualProtect(slot as *mut core::ffi::c_void, 8, old, &mut ignored);
            }
        }
        hr
    }

    unsafe extern "system" fn get_proc_address_hook(
        module: *mut core::ffi::c_void,
        name: *const u8,
    ) -> *mut core::ffi::c_void {
        let orig: GetProcAddressFn =
            unsafe { std::mem::transmute(ORIG_GPA.load(Ordering::Acquire)) };
        let result = unsafe { orig(module, name) };
        if result.is_null() || name.is_null() || (name as usize) <= 0xFFFF {
            return result;
        }
        let lit = b"D3D12CreateDevice\0";
        let matches = (0..lit.len()).all(|i| unsafe { *name.add(i) } == lit[i]);
        if !matches {
            return result;
        }
        ORIG_CREATE_DEVICE.store(result, Ordering::Release);
        create_device_hook as *mut core::ffi::c_void
    }

    unsafe {
        let base = GetModuleHandleW(core::ptr::null());
        if base.is_null() {
            eprintln!("[GPU-PRIO] queue hook missed (no module)");
            return;
        }
        let base_b = base as *const u8;
        let e_lfanew = (base_b.add(0x3C) as *const i32).read_unaligned();
        if e_lfanew <= 0 {
            return;
        }
        let nt = base_b.add(e_lfanew as usize);
        let magic = (nt.add(24) as *const u16).read_unaligned();
        if magic != 0x20B {
            eprintln!("[GPU-PRIO] queue hook missed (not PE32+)");
            return;
        }
        // OptionalHeader + 112 が DataDirectory。index 1 が import。
        let import_rva = (nt.add(24 + 120) as *const u32).read_unaligned();
        let import_size = (nt.add(24 + 124) as *const u32).read_unaligned();
        if import_rva == 0 || import_size < 20 {
            eprintln!("[GPU-PRIO] queue hook missed (no import dir)");
            return;
        }
        let mut desc = base_b.add(import_rva as usize);
        loop {
            let name_rva = (desc.add(12) as *const u32).read_unaligned();
            let first_thunk = (desc.add(16) as *const u32).read_unaligned();
            let original = (desc.add(0) as *const u32).read_unaligned();
            if name_rva == 0 {
                break;
            }
            let int_rva = if original != 0 { original } else { first_thunk };
            let mut index = 0usize;
            loop {
                let thunk =
                    (base_b.add(int_rva as usize + index * 8) as *const u64).read_unaligned();
                if thunk == 0 {
                    break;
                }
                if thunk & (1u64 << 63) == 0 {
                    let name = base_b.add(thunk as usize + 2);
                    let lit = b"GetProcAddress\0";
                    let matches = (0..lit.len()).all(|i| *name.add(i) == lit[i]);
                    if matches {
                        let slot = base_b.add(first_thunk as usize + index * 8)
                            as *mut *mut core::ffi::c_void;
                        ORIG_GPA.store(*slot, Ordering::Release);
                        let mut old = 0u32;
                        if VirtualProtect(slot as *mut core::ffi::c_void, 8, 0x04, &mut old) != 0 {
                            *slot = get_proc_address_hook as *mut core::ffi::c_void;
                            let mut ignored = 0u32;
                            let _ = VirtualProtect(
                                slot as *mut core::ffi::c_void,
                                8,
                                old,
                                &mut ignored,
                            );
                            eprintln!("[GPU-PRIO] D3D12 queue hook installed");
                        }
                        return;
                    }
                }
                index += 1;
            }
            desc = desc.add(20);
        }
        eprintln!("[GPU-PRIO] queue hook missed (GetProcAddress not imported)");
    }
}

/// DWM の合成と EcoQoS が compute の fence 待ちを数十 ms 伸ばすのを止める。
fn boost_gpu_scheduling_priority() {
    if std::env::var("RCWX_GPU_PRIORITY").ok().as_deref() == Some("0") {
        return;
    }
    #[cfg(windows)]
    unsafe {
        pin_process_scheduling();
    }
}

#[cfg(windows)]
unsafe fn pin_process_scheduling() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static ONCE: AtomicBool = AtomicBool::new(false);
    if ONCE.swap(true, Ordering::Relaxed) {
        return;
    }
    #[link(name = "gdi32")]
    unsafe extern "system" {
        fn D3DKMTSetProcessSchedulingPriorityClass(process: isize, priority_class: i32) -> i32;
    }
    #[link(name = "winmm")]
    unsafe extern "system" {
        fn timeBeginPeriod(period: u32) -> u32;
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn SetPriorityClass(process: isize, class: u32) -> i32;
        fn SetProcessInformation(process: isize, class: u32, info: *const u8, len: u32) -> i32;
    }
    #[repr(C)]
    struct PowerThrottle {
        version: u32,
        control: u32,
        state: u32,
    }
    unsafe {
        // 1ms タイマー。既定の 15.6ms だと fence 待ちが 50ms 単位に丸まる。
        let _ = timeBeginPeriod(1);
        let process = GetCurrentProcess();
        // HIGH_PRIORITY_CLASS。EcoQoS の execution-speed / timer スロットルを外す。
        let _ = SetPriorityClass(process, 0x0000_0080);
        let throttle = PowerThrottle {
            version: 1,
            control: 0x1 | 0x4,
            state: 0,
        };
        let _ = SetProcessInformation(
            process,
            4,
            &throttle as *const PowerThrottle as *const u8,
            std::mem::size_of::<PowerThrottle>() as u32,
        );
        for class in [5, 4, 3] {
            if D3DKMTSetProcessSchedulingPriorityClass(process, class) == 0 {
                let name = match class {
                    5 => "REALTIME",
                    4 => "HIGH",
                    _ => "ABOVE_NORMAL",
                };
                eprintln!("[GPU-PRIO] WDDM scheduling priority = {name}");
                return;
            }
        }
    }
}

/// ドライバの裏最適化が compute 待ちを数十 ms 伸ばすのを止める。
fn disable_d3d12_background_work(queue: &wgpu::Queue) {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK,
            D3D12_MEASUREMENTS_ACTION_DISCARD_PREVIOUS, ID3D12Device6,
        };
        let Some(hal_queue) = queue.as_hal::<wgpu::hal::api::Dx12>() else {
            return;
        };
        let mut device: Option<ID3D12Device6> = None;
        if hal_queue.as_raw().GetDevice(&mut device).is_err() {
            return;
        }
        let Some(device) = device else {
            return;
        };
        if device
            .SetBackgroundProcessingMode(
                D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK,
                D3D12_MEASUREMENTS_ACTION_DISCARD_PREVIOUS,
                None,
                None,
            )
            .is_ok()
        {
            eprintln!("[GPU-PRIO] D3D12 background work disabled");
        }
        let desc = hal_queue.as_raw().GetDesc();
        eprintln!("[GPU-PRIO] D3D12 queue priority field = {}", desc.Priority);
    }
    #[cfg(not(windows))]
    {
        let _ = queue;
    }
}

/// 推論スレッドを MMCSS Pro Audio に載せる。呼び出しスレッドだけが対象。
pub fn pin_realtime_thread() {
    if std::env::var("RCWX_GPU_PRIORITY").ok().as_deref() == Some("0") {
        return;
    }
    #[cfg(windows)]
    {
        thread_local! {
            static DONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        }
        if DONE.with(|done| done.replace(true)) {
            return;
        }
        unsafe { pin_mmcss_thread() }
    }
}

#[cfg(windows)]
unsafe fn pin_mmcss_thread() {
    #[link(name = "avrt")]
    unsafe extern "system" {
        fn AvSetMmThreadCharacteristicsW(task: *const u16, index: *mut u32) -> isize;
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadPriority(thread: isize, priority: i32) -> i32;
    }
    unsafe {
        let name: Vec<u16> = "Pro Audio\0".encode_utf16().collect();
        let mut index = 0u32;
        let handle = AvSetMmThreadCharacteristicsW(name.as_ptr(), &mut index);
        if handle == 0 {
            // THREAD_PRIORITY_HIGHEST。MMCSS が使えないときの予備。
            let _ = SetThreadPriority(GetCurrentThread(), 2);
            eprintln!("[GPU-PRIO] MMCSS unavailable, thread priority = HIGHEST");
        } else {
            eprintln!("[GPU-PRIO] MMCSS task = Pro Audio");
        }
    }
}

impl Engine {
    fn compile(&mut self, name: &'static str, src: &str, n_storage: u32) -> Result<()> {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
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
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        self.compile("matmul_tiled_res", &shaders::matmul_tiled_res_src(), 5)?;
        self.compile("matmul_tiled_x", &shaders::matmul_tiled_x_src(), 5)?;
        self.compile("add_layernorm", &shaders::add_layernorm_src(), 5)?;
        self.compile("slice_ch", &shaders::slice_ch_src(), 2)?;
        self.compile("avg_pool2x2", &shaders::avg_pool2x2_src(), 2)?;
        self.compile("conv_transpose2d", &shaders::conv_transpose2d_src(), 4)?;
        self.compile("concat_ch", &shaders::concat_ch_src(), 3)?;
        self.compile("transpose_flatten", &shaders::transpose_flatten_src(), 2)?;
        self.compile("pad2d", &shaders::pad2d_src(), 2)?;
        // 学習 (Issue #8) 用 kernel 群
        self.compile("t_fill", &train_shaders::T_FILL, 1)?;
        self.compile("t_copy", &train_shaders::T_COPY, 2)?;
        self.compile("t_mul2", &train_shaders::T_MUL2, 3)?;
        self.compile("t_pw_fwd", &train_shaders::T_PW_FWD, 2)?;
        self.compile("t_pw_bwd", &train_shaders::T_PW_BWD, 4)?;
        self.compile("t_softmax_bwd", &train_shaders::T_SOFTMAX_BWD, 3)?;
        self.compile("t_layernorm_bwd", &train_shaders::T_LAYERNORM_BWD, 5)?;
        self.compile("t_ln_param_grad", &train_shaders::T_LN_PARAM_GRAD, 4)?;
        self.compile("t_colsum", &train_shaders::T_COLSUM, 2)?;
        self.compile("t_reduce_partials", &train_shaders::T_REDUCE_PARTIALS, 3)?;
        self.compile("t_reduce_final", &train_shaders::T_REDUCE_FINAL, 2)?;
        self.compile("t_adamw", &train_shaders::T_ADAMW, 4)?;
        self.compile("t_wn_fold", &train_shaders::T_WN_FOLD, 4)?;
        self.compile("t_wn_bwd", &train_shaders::T_WN_BWD, 7)?;
        self.compile("t_im2col", &train_shaders::T_IM2COL, 2)?;
        self.compile("t_col2im", &train_shaders::T_COL2IM, 2)?;
        self.compile("t_add_ch_bias", &train_shaders::T_ADD_CH_BIAS, 2)?;
        self.compile("t_l1_sign", &train_shaders::T_L1_SIGN, 4)?;
        self.compile("t_mse_bwd", &train_shaders::T_MSE_BWD, 3)?;
        self.compile("t_chslice", &train_shaders::T_CHSLICE, 2)?;
        self.compile("t_chslice_bwd", &train_shaders::T_CHSLICE_BWD, 2)?;
        self.compile("t_gating", &train_shaders::T_GATING, 2)?;
        self.compile("t_gating_bwd", &train_shaders::T_GATING_BWD, 3)?;
        self.compile("t_skew", &train_shaders::T_SKEW, 2)?;
        self.compile("t_skew_bwd", &train_shaders::T_SKEW_BWD, 2)?;
        self.compile("t_add_bias2d", &train_shaders::T_ADD_BIAS2D, 3)?;
        self.compile("t_mask_fill", &train_shaders::T_MASK_FILL, 3)?;
        self.compile("t_mask_fill_bwd", &train_shaders::T_MASK_FILL_BWD, 3)?;
        self.compile("t_reflect_pad", &train_shaders::T_REFLECT_PAD, 2)?;
        self.compile("t_mul_window", &train_shaders::T_MUL_WINDOW, 3)?;
        self.compile("t_mag", &train_shaders::T_MAG, 3)?;
        self.compile("t_mag_bwd", &train_shaders::T_MAG_BWD, 6)?;
        self.compile("t_log_clamp", &train_shaders::T_LOG_CLAMP, 2)?;
        self.compile("t_log_clamp_bwd", &train_shaders::T_LOG_CLAMP_BWD, 3)?;
        self.compile("t_kl_elem", &train_shaders::T_KL_ELEM, 6)?;
        self.compile("t_kl_bwd_a", &train_shaders::T_KL_BWD_A, 8)?;
        self.compile("t_kl_bwd_b", &train_shaders::T_KL_BWD_B, 3)?;
        self.compile("t_gather_frames", &train_shaders::T_GATHER_FRAMES, 3)?;
        self.compile(
            "t_gather_frames_bwd",
            &train_shaders::T_GATHER_FRAMES_BWD,
            3,
        )?;
        self.compile("t_transpose", &train_shaders::T_TRANSPOSE, 2)?;
        self.compile("t_mul_frame_mask", &train_shaders::T_MUL_FRAME_MASK, 3)?;
        self.compile("t_im2col2d", &train_shaders::T_IM2COL2D, 2)?;
        self.compile("t_col2im2d", &train_shaders::T_COL2IM2D, 2)?;
        self.compile("t_reflect_pad_bwd", &train_shaders::T_REFLECT_PAD_BWD, 2)?;
        self.compile("t_add_bcast_t", &train_shaders::T_ADD_BCAST_T, 3)?;
        self.compile("t_chcat", &train_shaders::T_CHCAT, 3)?;
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

    fn take_staging(&self, size: wgpu::BufferAddress) -> wgpu::Buffer {
        let mut pool = self
            .staging_pool
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        if let Some(i) = pool.iter().position(|(cached, _)| *cached == size) {
            return pool.swap_remove(i).1;
        }
        drop(pool);
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn wait_gpu(&self, idx: wgpu::SubmissionIndex) {
        // 無限 Wait は完了イベントの結合待ちで ~50ms 遅れる。1ms ごとに fence 値を見に行く。
        // (待ち中の空 submit キックも試したが完了処理は進まず効果なし)
        let mode = std::env::var("RCWX_WAIT_MODE").unwrap_or_default();
        let started = std::time::Instant::now();
        match mode.as_str() {
            // 診断: fence 値のホットスピン (イベント登録なし)
            "spin" => loop {
                if matches!(
                    self.device.poll(wgpu::PollType::Poll),
                    Ok(wgpu::PollStatus::QueueEmpty)
                ) {
                    return;
                }
                if started.elapsed() > std::time::Duration::from_secs(5) {
                    panic!("GPU readback exceeded 5s");
                }
                std::hint::spin_loop();
            },
            // 診断: 単発の長いイベント待ち (従来の遅い経路)
            "event" => {
                self.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: Some(idx),
                        timeout: Some(std::time::Duration::from_secs(5)),
                    })
                    .expect("GPU readback wait");
            }
            _ => loop {
                match self.device.poll(wgpu::PollType::Wait {
                    submission_index: Some(idx.clone()),
                    timeout: Some(std::time::Duration::from_millis(1)),
                }) {
                    Ok(status) if status.wait_finished() => return,
                    Ok(_) => {}
                    Err(wgpu::PollError::Timeout) => {
                        if started.elapsed() > std::time::Duration::from_secs(5) {
                            panic!("GPU readback exceeded 5s");
                        }
                    }
                    Err(err) => panic!("GPU readback poll: {err}"),
                }
            },
        }
    }

    /// 診断用: 空コマンドの submit→fence 完了の往復レイテンシ。
    /// 呼び出し時点でキューがドレイン済みなら純粋なドライバ往復が測れる。
    /// 大きければ「コマンドがドライバに保持されている」、小さくて直後の
    /// 重い計算の readback だけ遅いなら「GPU 実行そのものが遅い」。
    pub fn probe_noop(&self) -> f64 {
        let enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let idx = self.queue.submit(Some(enc.finish()));
        let t = std::time::Instant::now();
        self.wait_gpu(idx);
        t.elapsed().as_secs_f64() * 1000.0
    }

    fn mapped_f32(staging: &wgpu::Buffer) -> Vec<f32> {
        let data = staging.slice(..).get_mapped_range().expect("map").to_vec();
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    pub fn readback_f32(&self, t: &GpuTensor) -> Vec<f32> {
        let size = t.bytes();
        let staging = self.take_staging(size);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&t.buffer, 0, &staging, 0, size);
        let (idx, epoch) = self.submit_flushed2(enc.finish());
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.wait_gpu(idx);
        self.mark_drained(epoch);
        let data = Self::mapped_f32(&staging);
        staging.unmap();
        self.staging_pool
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push((size, staging));
        data
    }

    /// プラン再生 + 出力読み戻しを 1 回の submit に統合。
    /// submit 数と fence 境界を減らす (publish 遅延の当たり目を減らす)。
    /// DX12 は sentinel 経路で待つ (fence 公開遅延を回避)。
    pub fn run_plans_readback(&self, plans: &[&Plan], out: &GpuTensor) -> Vec<f32> {
        let size = out.bytes();
        let staging = self.take_staging(size);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for plan in plans {
                for cmd in &plan.cmds {
                    let k = self.kernels.get(cmd.kernel).unwrap();
                    cpass.set_pipeline(&k.pipeline);
                    cpass.set_bind_group(0, &cmd.bind_group, &[]);
                    cpass.dispatch_workgroups(cmd.wx, cmd.wy, cmd.wz);
                }
            }
        }
        enc.copy_buffer_to_buffer(&out.buffer, 0, &staging, 0, size);
        let (idx, epoch) = self.submit_flushed2(enc.finish());
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.wait_gpu(idx);
        self.mark_drained(epoch);
        let data = Self::mapped_f32(&staging);
        staging.unmap();
        self.staging_pool
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push((size, staging));
        data
    }

    /// 入力 copy + 複数プラン + 2 バッファ読み戻しを 1 回の submit に統合
    /// (HuBERT + SwiftF0 の前半まとめ実行用)。
    #[allow(clippy::type_complexity)]
    pub fn run_inputs_plans_readback_two(
        &self,
        input_copies: &[(&GpuTensor, &GpuTensor)],
        plans: &[&Plan],
        a: &GpuTensor,
        b: &GpuTensor,
    ) -> (Vec<f32>, Vec<f32>) {
        let size_a = a.bytes();
        let size_b = b.bytes();
        let staging_a = self.take_staging(size_a);
        let staging_b = self.take_staging(size_b);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (src, dst) in input_copies {
            enc.copy_buffer_to_buffer(&src.buffer, 0, &dst.buffer, 0, src.bytes());
        }
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for plan in plans {
                for cmd in &plan.cmds {
                    let k = self.kernels.get(cmd.kernel).unwrap();
                    cpass.set_pipeline(&k.pipeline);
                    cpass.set_bind_group(0, &cmd.bind_group, &[]);
                    cpass.dispatch_workgroups(cmd.wx, cmd.wy, cmd.wz);
                }
            }
        }
        enc.copy_buffer_to_buffer(&a.buffer, 0, &staging_a, 0, size_a);
        enc.copy_buffer_to_buffer(&b.buffer, 0, &staging_b, 0, size_b);
        let (idx, epoch) = self.submit_flushed2(enc.finish());
        staging_a.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        staging_b.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.wait_gpu(idx);
        self.mark_drained(epoch);
        let data_a = Self::mapped_f32(&staging_a);
        let data_b = Self::mapped_f32(&staging_b);
        staging_a.unmap();
        staging_b.unmap();
        let mut pool = self
            .staging_pool
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        pool.push((size_a, staging_a));
        pool.push((size_b, staging_b));
        (data_a, data_b)
    }

    /// 診断: プラン再生 + 読み戻しを 1 submit に統合し、compute pass の GPU 時間と
    /// CPU 側 fence 待ちを別々に報告する。RCWX_TS_PROBE 用。
    /// 前後を微小 pass (1 workgroup) で挟み、穴の位置を分解する:
    ///   hole_pre  = compute pass 開始前のギャップ
    ///   hole_post = compute pass 終了後のギャップ
    ///   publish   = 実行完了 → fence 公開の遅れ (cpu_wait - 実行区間)
    pub fn run_plans_timed(&self, plans: &[&Plan], out: &GpuTensor) -> Vec<f32> {
        let size = out.bytes();
        let staging = self.take_staging(size);
        let use_ts = self.has_ts;
        let qs = use_ts.then(|| {
            self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                ty: wgpu::QueryType::Timestamp,
                count: 6,
            })
        });
        let ts_buf = use_ts.then(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 48,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let ts_staging = use_ts.then(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 48,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let tiny = plans.first().and_then(|p| p.cmds.first()).map(|cmd| {
            (
                self.kernels.get(cmd.kernel).unwrap().pipeline.clone(),
                cmd.bind_group.clone(),
            )
        });
        if let (Some(qs), Some((pipe, bg))) = (&qs, &tiny) {
            // pass A: 微小 dispatch (1 workgroup)。後段の全 dispatch が上書きする。
            let mut apass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            apass.set_pipeline(pipe);
            apass.set_bind_group(0, bg, &[]);
            apass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: qs.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(2),
                    end_of_pass_write_index: Some(3),
                }),
            });
            for plan in plans {
                for cmd in &plan.cmds {
                    let k = self.kernels.get(cmd.kernel).unwrap();
                    cpass.set_pipeline(&k.pipeline);
                    cpass.set_bind_group(0, &cmd.bind_group, &[]);
                    cpass.dispatch_workgroups(cmd.wx, cmd.wy, cmd.wz);
                }
            }
        }
        if let (Some(qs), Some((pipe, bg))) = (&qs, &tiny) {
            // pass B: 微小 dispatch。audio の一部を壊すが計測専用なので許容。
            let mut bpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(4),
                    end_of_pass_write_index: Some(5),
                }),
            });
            bpass.set_pipeline(pipe);
            bpass.set_bind_group(0, bg, &[]);
            bpass.dispatch_workgroups(1, 1, 1);
        }
        if let (Some(qs), Some(ts_buf), Some(ts_staging)) = (&qs, &ts_buf, &ts_staging) {
            enc.resolve_query_set(qs, 0..6, ts_buf, 0);
            enc.copy_buffer_to_buffer(ts_buf, 0, ts_staging, 0, 48);
        }
        enc.copy_buffer_to_buffer(&out.buffer, 0, &staging, 0, size);
        let (idx, epoch) = self.submit_flushed2(enc.finish());
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        if let Some(ts_staging) = &ts_staging {
            ts_staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        let t = std::time::Instant::now();
        self.wait_gpu(idx);
        self.mark_drained(epoch);
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut line = format!("    [ts] cpu_wait={cpu_ms:.2}ms");
        if let Some(ts_staging) = &ts_staging {
            let raw = Self::mapped_f32(ts_staging);
            let ts: Vec<u64> = (0..6)
                .map(|i| raw[i * 2].to_bits() as u64 | ((raw[i * 2 + 1].to_bits() as u64) << 32))
                .collect();
            let period = self.queue.get_timestamp_period() as f64;
            let ms = |a: u64, b: u64| (b.saturating_sub(a)) as f64 * period / 1e6;
            let span = ms(ts[0], ts[5]);
            line += format!(
                " hole_pre={:.2} pass={:.2} hole_post={:.2} publish={:.2}",
                ms(ts[1], ts[2]),
                ms(ts[2], ts[3]),
                ms(ts[3], ts[4]),
                (cpu_ms - span).max(0.0)
            )
            .as_str();
            ts_staging.unmap();
        }
        eprintln!("{line}");
        let data = Self::mapped_f32(&staging);
        staging.unmap();
        self.staging_pool
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push((size, staging));
        data
    }

    /// 2 バッファを 1 回の submit / 1 回の fence 待ちで読む。
    /// 同じサイズを続けて取るのはプールが空なら新規確保するので問題ない。
    pub fn readback_two(&self, a: &GpuTensor, b: &GpuTensor) -> (Vec<f32>, Vec<f32>) {
        let size_a = a.bytes();
        let size_b = b.bytes();
        let staging_a = self.take_staging(size_a);
        let staging_b = self.take_staging(size_b);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&a.buffer, 0, &staging_a, 0, size_a);
        enc.copy_buffer_to_buffer(&b.buffer, 0, &staging_b, 0, size_b);
        let (idx, epoch) = self.submit_flushed2(enc.finish());
        staging_a.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        staging_b.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.wait_gpu(idx);
        self.mark_drained(epoch);
        let data_a = Self::mapped_f32(&staging_a);
        let data_b = Self::mapped_f32(&staging_b);
        staging_a.unmap();
        staging_b.unmap();
        let mut pool = self
            .staging_pool
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        pool.push((size_a, staging_a));
        pool.push((size_b, staging_b));
        (data_a, data_b)
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
        let (idx, epoch) = self.submit_flushed2(enc.finish());
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            })
            .unwrap();
        self.mark_drained(epoch);
        let data = slice.get_mapped_range().unwrap().to_vec();
        staging.unmap();
        data.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// 1 forward 分の記録。uniform は submit 時に一括書き込み。
    pub fn graph(&self) -> Graph<'_> {
        Graph {
            engine: self,
            encoder: Some(
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }),
            ),
            uniform_writes: Vec::new(),
            cmds: Vec::new(),
        }
    }
}

impl<'a> Graph<'a> {
    /// engine 上に未初期化テンソル (プラン構築中の作業領域)。
    pub fn engine_tensor_uninit(&self, len: usize) -> GpuTensor {
        self.engine.tensor_uninit(len)
    }

    /// 学習 tape からの直接 dispatch (カーネル名・バッファ列・uniform 語)。
    pub(crate) fn dispatch_raw(
        &mut self,
        kernel: &'static str,
        bufs: &[&GpuTensor],
        words: [u32; 32],
        wx: u32,
        wy: u32,
        wz: u32,
    ) {
        self.dispatch(kernel, bufs, words, wx, wy, wz);
    }
}

pub struct Graph<'a> {
    pub engine: &'a Engine,
    encoder: Option<wgpu::CommandEncoder>,
    uniform_writes: Vec<(u32, [u32; 32])>, // (slot, words)
    cmds: Vec<Cmd>,
}

#[derive(Clone)]
pub struct Cmd {
    kernel: &'static str,
    bind_group: wgpu::BindGroup,
    wx: u32,
    wy: u32,
    wz: u32,
}

/// 形状ごとに一度だけ構築する実行プラン (バッファは外で保持、uniform は構築時に書き込み済み)。
/// 統合 submit 用に clone 可能 (bind group は Arc)。
#[derive(Clone)]
pub struct Plan {
    pub cmds: Vec<Cmd>,
    pub n_cmds: usize,
    /// この plan が使う uniform スロットの内容 (replay 時に再書込 — 他モデルが上書き防止)
    pub uniform_writes: Vec<(u32, [u32; 32])>,
}

impl Drop for Graph<'_> {
    /// submit されず破棄された Graph の uniform スロットを回収する。
    /// into_plan/submit は uniform_writes を move 済みなのでここでは空。
    fn drop(&mut self) {
        if !self.uniform_writes.is_empty() {
            let slots = self
                .uniform_writes
                .iter()
                .map(|(s, _)| *s)
                .collect::<Vec<_>>();
            self.engine.recycle_uniform_slots(slots.into_iter());
        }
    }
}

impl<'a> Graph<'a> {
    fn alloc_slot(&mut self, words: [u32; 32]) -> u32 {
        let slot = self.engine.alloc_uniform_slot();
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
            a,
            w,
            bias,
            c,
            m,
            n,
            k,
            batch,
            a_row_stride,
            a_k_stride,
            a_base,
            a_batch_stride,
            w_row_stride,
            w_k_stride,
            w_base,
            w_batch_stride,
            c_row_stride,
            c_n_stride,
            c_base,
            c_batch_stride,
            scale,
            0,
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
        // K が短いデコーダ最終段 (例: 32ch×k3 → K=96) も SLM タイルで重みを再利用する。
        let use_tiled = k >= 32;
        let (kernel, wx, wy) = if use_tiled {
            ("matmul_tiled", n.div_ceil(32), m.div_ceil(32))
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
    pub fn group_norm(
        &mut self,
        x: &GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        c: u32,
        t: u32,
        eps: f32,
    ) {
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
    pub fn pointwise(
        &mut self,
        x: &GpuTensor,
        out: Option<&GpuTensor>,
        len: u32,
        op: u32,
        param: f32,
        t_len: u32,
        ch: u32,
        h: u32,
    ) {
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
    pub fn slice_ch(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        ch: u32,
        n_in: u32,
        off: u32,
        n_out: u32,
    ) {
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

    /// matmul_tiled_x: post_op 0=none 1=lrelu 2=relu 3=gelu 4=+residual 5=+residual+lrelu。
    /// extra = residual (post>=4 のとき)。bias は B3[m]。
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_ex_x(
        &mut self,
        a: &GpuTensor,
        w: &GpuTensor,
        bias: Option<&GpuTensor>,
        extra: Option<&GpuTensor>,
        c: &GpuTensor,
        m: u32,
        n: u32,
        k: u32,
        a_row_stride: u32,
        a_k_stride: u32,
        w_row_stride: u32,
        w_k_stride: u32,
        c_row_stride: u32,
        c_n_stride: u32,
        scale: f32,
        post: u32,
    ) {
        let has_bias = bias.is_some();
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[m, n, k, 1]);
        words[4..8].copy_from_slice(&[a_row_stride, a_k_stride, 0, 0]);
        words[8..12].copy_from_slice(&[w_row_stride, w_k_stride, 0, 0]);
        words[12..16].copy_from_slice(&[c_row_stride, c_n_stride, 0, 0]);
        words[16..20].copy_from_slice(&[has_bias as u32, scale.to_bits(), 0, post]);
        let dummy = a;
        let bias_ref = bias.unwrap_or(dummy);
        let extra_ref = extra.unwrap_or(dummy);
        let (wx, wy) = (n.div_ceil(32), m.div_ceil(32));
        self.dispatch(
            "matmul_tiled_x",
            &[a, w, bias_ref, c, extra_ref],
            words,
            wx,
            wy,
            1,
        );
    }

    /// LN(x + y) を 1 dispatch で (残差 + LayerNorm 融合)。
    #[allow(clippy::too_many_arguments)]
    pub fn add_layer_norm(
        &mut self,
        x: &GpuTensor,
        y: &GpuTensor,
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
        words[4..7].copy_from_slice(&[out_row_stride, eps.to_bits(), out_k_stride]);
        self.dispatch(
            "add_layernorm",
            &[x, y, gamma, beta, out],
            words,
            rows,
            1,
            1,
        );
    }

    /// matmul + 残差加算 (im2col 済み B を想定)。B5 = residual [m, n]。
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_ex_res(
        &mut self,
        a: &GpuTensor,
        w: &GpuTensor,
        bias: Option<&GpuTensor>,
        residual: &GpuTensor,
        c: &GpuTensor,
        m: u32,
        n: u32,
        k: u32,
        a_row_stride: u32,
        a_k_stride: u32,
        w_row_stride: u32,
        w_k_stride: u32,
        c_row_stride: u32,
        c_n_stride: u32,
    ) {
        let has_bias = bias.is_some();
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[m, n, k, 1]);
        words[4..8].copy_from_slice(&[a_row_stride, a_k_stride, 0, 0]);
        words[8..12].copy_from_slice(&[w_row_stride, w_k_stride, 0, 0]);
        words[12..16].copy_from_slice(&[c_row_stride, c_n_stride, 0, 0]);
        words[16] = has_bias as u32;
        words[17] = 1.0f32.to_bits();
        let dummy = a;
        let bias_ref = bias.unwrap_or(dummy);
        let (wx, wy) = (n.div_ceil(32), m.div_ceil(32));
        self.dispatch(
            "matmul_tiled_res",
            &[a, w, bias_ref, c, residual],
            words,
            wx,
            wy,
            1,
        );
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
        w[12..16].copy_from_slice(&[n, 1, 0, 0]); // c_row=n, c_n=1
        w[16..20].copy_from_slice(&[has_bias as u32, 1.0f32.to_bits(), 1, conv_k]);
        w[20..24].copy_from_slice(&[pad, dil, in_len, post_op]);
        let dummy = a;
        let bias_ref = bias.unwrap_or(dummy);
        // f16 版は現状 DXC が packed 命令を出さず速度劣化 + 数値劣化のため無効
        self.dispatch(
            "matmul_conv",
            &[a, x, bias_ref, c],
            w,
            n.div_ceil(16),
            m.div_ceil(16),
            1,
        );
    }

    /// 汎用 im2col2d (kernel k・pad p)。出力 [c*k*k, h*w]。
    pub fn im2col2d_k(
        &mut self,
        x: &GpuTensor,
        ic: &GpuTensor,
        c: u32,
        h: u32,
        w: u32,
        k: u32,
        pad: u32,
    ) {
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
    pub fn conv_transpose2d(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        bias: Option<&GpuTensor>,
        out: &GpuTensor,
        cin: u32,
        cout: u32,
        h_in: u32,
        w_in: u32,
        h_out: u32,
        w_out: u32,
        sh: u32,
        sw: u32,
        oph: u32,
        opw: u32,
    ) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[cin, cout, h_out, w_out]);
        words[4..8].copy_from_slice(&[h_in, w_in, sh, sw]);
        words[8..10].copy_from_slice(&[oph, opw]);
        let dummy = x;
        let bias_ref = bias.unwrap_or(dummy);
        self.dispatch(
            "conv_transpose2d",
            &[x, w, out, bias_ref],
            words,
            w_out.div_ceil(8),
            h_out.div_ceil(8),
            cout,
        );
    }

    pub fn concat_ch(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        out: &GpuTensor,
        ca: u32,
        cb: u32,
        n: u32,
    ) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[n, ca, cb, 0]);
        self.dispatch(
            "concat_ch",
            &[a, b, out],
            words,
            ((ca + cb) * n).div_ceil(256),
            1,
            1,
        );
    }

    pub fn transpose_flatten(&mut self, x: &GpuTensor, out: &GpuTensor, c: u32, t: u32, m: u32) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[c, t, m, 0]);
        self.dispatch(
            "transpose_flatten",
            &[x, out],
            words,
            (c * t * m).div_ceil(256),
            1,
            1,
        );
    }

    pub fn pad2d(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        c: u32,
        h: u32,
        w: u32,
        h2: u32,
        w2: u32,
    ) {
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
        self.dispatch(
            "conv_transpose1d",
            &[x, w, bias_ref, out],
            words,
            out_len.div_ceil(8),
            out_c.div_ceil(8),
            1,
        );
    }

    /// 相対位置 attention 補助。mode0: scores += (q·relk)*scale。mode1: out += p·relv。
    pub fn rel_attn_scores(
        &mut self,
        q: &GpuTensor,
        relk: &GpuTensor,
        scores: &GpuTensor,
        length: u32,
        d: u32,
        ws: u32,
        heads: u32,
        scale: f32,
    ) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[0, length, d, ws]);
        words[4..6].copy_from_slice(&[scale.to_bits(), heads]);
        self.dispatch(
            "rel_attn",
            &[q, relk, scores],
            words,
            length.div_ceil(8),
            length.div_ceil(8),
            heads,
        );
    }

    pub fn rel_attn_values(
        &mut self,
        p: &GpuTensor,
        relv: &GpuTensor,
        out: &GpuTensor,
        length: u32,
        d: u32,
        ws: u32,
        heads: u32,
    ) {
        let mut words = [0u32; 32];
        words[0..4].copy_from_slice(&[1, length, d, ws]);
        words[5] = heads;
        self.dispatch(
            "rel_attn",
            &[p, relv, out],
            words,
            d.div_ceil(8),
            length.div_ceil(8),
            heads,
        );
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
        self.im2col_off(
            x, 0, ic, c_in, kernel, stride, dilation, pad_l, in_len, out_len,
        );
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
        eng.write_uniforms(&self.uniform_writes);
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
    /// (Graph は Drop で uniform slot を回収するため、encoder は mem::take で取り出す)
    pub fn submit(mut self) -> Result<()> {
        let eng = self.engine;
        eng.write_uniforms(&self.uniform_writes);
        let mut encoder = self.encoder.take().expect("graph already submitted");
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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
        eng.submit_flushed(encoder.finish());
        Ok(())
    }
}
