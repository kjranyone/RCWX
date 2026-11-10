//! 出力用 SPSC リングバッファ。書き手 = 推論スレッド、読み手 = オーディオコールバック。
//! 単プロデューサ/単コンシューマを想定し、atomic index のみで同期する。

use std::cell::UnsafeCell;

pub struct RingBuffer {
    buf: UnsafeCell<Vec<f32>>,
    cap: usize,
    head: std::sync::atomic::AtomicUsize, // 書き込み位置
    tail: std::sync::atomic::AtomicUsize, // 読み出し位置
}

// safety: SPSC (単一書き手・単一読み手) で index 同期されている
unsafe impl Sync for RingBuffer {}

impl RingBuffer {
    pub fn new(capacity_samples: usize) -> Self {
        Self {
            buf: UnsafeCell::new(vec![0.0; capacity_samples]),
            cap: capacity_samples,
            head: std::sync::atomic::AtomicUsize::new(0),
            tail: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// 滞留サンプル数。
    pub fn len(&self) -> usize {
        let h = self.head.load(std::sync::atomic::Ordering::Acquire);
        let t = self.tail.load(std::sync::atomic::Ordering::Acquire);
        h.wrapping_sub(t)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 一括書き込み (満杯分は上書きせず破棄)。書けたサンプル数を返す。
    pub fn write(&self, data: &[f32]) -> usize {
        let h = self.head.load(std::sync::atomic::Ordering::Relaxed);
        let t = self.tail.load(std::sync::atomic::Ordering::Acquire);
        let used = h.wrapping_sub(t);
        let free = self.cap - used;
        let n = data.len().min(free);
        // safety: 単一書き手のみがこの領域に書く
        let buf = unsafe { &mut *self.buf.get() };
        for (i, v) in data.iter().take(n).enumerate() {
            let pos = (h + i) % self.cap;
            buf[pos] = *v;
        }
        self.head
            .store(h.wrapping_add(n), std::sync::atomic::Ordering::Release);
        n
    }

    /// 一括読み出し。実際に読めたサンプル数を返す (不足分は埋めない)。
    pub fn read(&self, out: &mut [f32]) -> usize {
        let t = self.tail.load(std::sync::atomic::Ordering::Relaxed);
        let h = self.head.load(std::sync::atomic::Ordering::Acquire);
        let avail = h.wrapping_sub(t);
        let n = out.len().min(avail);
        let buf = unsafe { &*self.buf.get() };
        for (i, slot) in out.iter_mut().take(n).enumerate() {
            let pos = (t + i) % self.cap;
            *slot = buf[pos];
        }
        self.tail
            .store(t.wrapping_add(n), std::sync::atomic::Ordering::Release);
        n
    }

    /// ハードスキップ: 先頭 `n` サンプルを破棄 (n ≤ len)。
    pub fn skip(&self, n: usize) {
        let t = self.tail.load(std::sync::atomic::Ordering::Relaxed);
        let h = self.head.load(std::sync::atomic::Ordering::Acquire);
        let avail = h.wrapping_sub(t);
        let skip = n.min(avail);
        self.tail
            .store(t.wrapping_add(skip), std::sync::atomic::Ordering::Release);
    }
}
