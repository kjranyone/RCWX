//! torch checkpoint (.pt / .pth = zip + pickle) の完全リーダ。
//!
//! `data.pkl` をスタックマシンで解釈し、`_rebuild_tensor_v2` が参照する
//! storage (zip 内 `data/<key>`) からテンソル実体を f32 で読み出す。
//! fairseq checkpoint のカスタムクラス (Dictionary / Namespace 等) は
//! GLOBAL を評価せず無視し、state_dict のテンソルのみを取り出す。
//! RVC .pth も fairseq .pt も同一形式なので 1 つのリーダで両方読める。
//!
//! cli の旧 pth_info.rs (shape のみ) を発展・統合したもの。

use anyhow::bail;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F16,
    F32,
    F64,
    Other,
}

impl DType {
    fn from_storage(name: &str) -> Self {
        // persistent id の storage クラス名 ("torch.FloatStorage" 等) から判定
        match name.rsplit('.').next().unwrap_or(name) {
            "HalfStorage" => DType::F16,
            "FloatStorage" | "UntypedStorage" => DType::F32, // Untyped は dtype 不明 → F32 扱い
            "DoubleStorage" => DType::F64,
            _ => DType::Other,
        }
    }
    fn elem_size(self) -> usize {
        match self {
            DType::F16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
            DType::Other => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StorageRef {
    pub dtype: DType,
    pub key: String,
    pub numel: i64,
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub storage: StorageRef,
    pub offset: i64,
    pub size: Vec<i64>,
    pub stride: Vec<i64>,
}

#[derive(Debug, Clone)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String, String),
    Storage(StorageRef),
    Tensor(TensorMeta),
    Mark,
    Other,
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        if let Value::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        if let Value::Int(i) = self {
            Some(*i)
        } else if let Value::Float(f) = self {
            Some(*f as i64)
        } else {
            None
        }
    }
    pub fn as_dict(&self) -> Option<&Vec<(Value, Value)>> {
        if let Value::Dict(d) = self {
            Some(d)
        } else {
            None
        }
    }
    /// 辞書から str キーで値を引く。
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_dict()?
            .iter()
            .find(|(k, _)| k.as_str() == Some(key))
            .map(|(_, v)| v)
    }
}

pub struct PthFile {
    zip: zip::ZipArchive<std::fs::File>,
    /// data.pkl の親ディレクトリ (zip のアーカイブ名依存)。
    data_prefix: String,
}

impl PthFile {
    pub fn open(path: &std::path::Path) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let zip = zip::ZipArchive::new(file)?;
        let pkl_name = zip
            .file_names()
            .filter_map(|n| n.ok())
            .find(|n| n.ends_with("data.pkl"))
            .ok_or_else(|| anyhow::anyhow!("data.pkl not found in archive"))?
            .into_owned();
        let data_prefix = pkl_name
            .strip_suffix("data.pkl")
            .unwrap_or_default()
            .to_string();
        Ok(Self { zip, data_prefix })
    }

    /// data.pkl をパースしてトップレベル Value を返す (テンソルはメタのみ)。
    pub fn parse(&mut self) -> anyhow::Result<Value> {
        let mut entry = self.zip.by_name(&(self.data_prefix.clone() + "data.pkl"))?;
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut entry, &mut buf)?;
        drop(entry);
        parse_pickle(&buf)
    }

    /// テンソル実体を f32 の連続 (row-major) 配列として読む。
    pub fn read_f32(&mut self, meta: &TensorMeta) -> anyhow::Result<Vec<f32>> {
        let esize = meta.storage.dtype.elem_size();
        if esize == 0 {
            bail!("unsupported dtype for tensor data");
        }
        let total: i64 = meta.size.iter().product::<i64>().max(0);
        let mut raw = self.storage_bytes(&meta.storage.key)?;
        if raw.len() % esize != 0 {
            bail!("storage {} size not aligned", meta.storage.key);
        }
        // storage を要素配列へ (F16 → F32 変換)
        let n_elems = raw.len() / esize;
        let mut elems = Vec::with_capacity(n_elems);
        match meta.storage.dtype {
            DType::F16 => {
                for c in raw.chunks_exact(2) {
                    elems.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
                }
            }
            DType::F32 => {
                raw.shrink_to_fit();
                for c in raw.chunks_exact(4) {
                    elems.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            DType::F64 => {
                for c in raw.chunks_exact(8) {
                    elems.push(
                        f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32,
                    );
                }
            }
            DType::Other => unreachable!(),
        }
        drop(raw);
        // stride が row-major 連続なら一気コピー
        let contiguous = {
            let mut expected = 1i64;
            let mut ok = true;
            for (d, s) in meta.size.iter().zip(meta.stride.iter()).rev() {
                if *s != expected {
                    ok = false;
                    break;
                }
                expected *= d.max(&1);
            }
            ok
        };
        let off = meta.offset as usize;
        let total = total as usize;
        if contiguous {
            if off + total > elems.len() {
                bail!("tensor {} out of storage range", meta.storage.key);
            }
            return Ok(elems[off..off + total].to_vec());
        }
        // 非連続: 多重インデックスで gather
        let mut out = Vec::with_capacity(total);
        let ndim = meta.size.len();
        let mut idx = vec![0i64; ndim];
        for _ in 0..total {
            let mut pos = meta.offset;
            for d in 0..ndim {
                pos += idx[d] * meta.stride[d];
            }
            out.push(*elems.get(pos as usize).unwrap_or(&0.0));
            // インクリメント
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < meta.size[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(out)
    }

    fn storage_bytes(&mut self, key: &str) -> anyhow::Result<Vec<u8>> {
        let name = format!("{}data/{}", self.data_prefix, key);
        let mut entry = self
            .zip
            .by_name(&name)
            .map_err(|e| anyhow::anyhow!("storage {name}: {e}"))?;
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut entry, &mut buf)?;
        Ok(buf)
    }
}

/// half (binary16) → f32。
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 => {
            // subnormal: 2^-24 * frac (f16 subnormal = frac/1024 × 2^-14)
            let v = (frac as f32) / 16777216.0;
            v.copysign(if sign == 1 { -1.0 } else { 1.0 }).to_bits()
        }
        31 => {
            if frac == 0 {
                (sign << 31) | 0x7f80_0000
            } else {
                (sign << 31) | 0x7f80_0000 | (frac << 13)
            }
        }
        _ => (sign << 31) | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    memo: Vec<Value>,
}

fn parse_pickle(buf: &[u8]) -> anyhow::Result<Value> {
    let trace = std::env::var("RCWX_PTH_TRACE").is_ok();
    let mut r = Reader {
        b: buf,
        pos: 0,
        stack: Vec::new(),
        memo: Vec::new(),
    };
    loop {
        let op =
            *r.b.get(r.pos)
                .ok_or_else(|| anyhow::anyhow!("pickle truncated at {}", r.pos))?;
        if trace {
            eprintln!("op {:#04x} at {}", op, r.pos);
        }
        r.pos += 1;
        match op {
            0x80 => {
                r.pos += 1; // PROTO version
            }
            0x95 => {
                let _len = r.take_u64()?; // FRAME
            }
            b'.' => break,
            b'N' => r.stack.push(Value::None),
            0x88 => r.stack.push(Value::Bool(false)),
            0x89 => r.stack.push(Value::Bool(true)),
            b'K' => {
                let v = r.take_u8()? as i64;
                r.stack.push(Value::Int(v));
            }
            b'M' => {
                let v = r.take_u16()? as i64;
                r.stack.push(Value::Int(v));
            }
            b'J' => {
                let v = r.take_i32()? as i64;
                r.stack.push(Value::Int(v));
            }
            0x8a => {
                // LONG1 (符号付き可変長。上位バイトは切り捨て)
                let n = r.take_u8()? as usize;
                let mut v: i64 = 0;
                for i in 0..n {
                    let b = r.take_u8()? as i64;
                    if i < 8 {
                        v |= b << (8 * i);
                    }
                }
                r.stack.push(Value::Int(v));
            }
            b'I' => {
                let line = r.take_line()?;
                let s = String::from_utf8_lossy(&line).trim().to_string();
                r.stack.push(Value::Int(s.parse().unwrap_or(0)));
            }
            b'G' => {
                let bits = r.take_u64()?;
                r.stack.push(Value::Float(f64::from_bits(bits)));
            }
            b'F' => {
                let line = r.take_line()?;
                let s = String::from_utf8_lossy(&line);
                r.stack.push(Value::Float(s.trim().parse().unwrap_or(0.0)));
            }
            0x8c => {
                let n = r.take_u8()? as usize;
                let s = String::from_utf8_lossy(r.take_bytes(n)?).into_owned();
                r.stack.push(Value::Str(s));
            }
            b'X' => {
                let n = r.take_u32()? as usize;
                let s = String::from_utf8_lossy(r.take_bytes(n)?).into_owned();
                r.stack.push(Value::Str(s));
            }
            0x8e => {
                // BINBYTES8 (protocol 5)
                let n = r.take_u64()? as usize;
                let bytes = r.take_bytes(n)?.to_vec();
                r.stack.push(Value::Bytes(bytes));
            }
            b'B' => {
                // BINBYTES
                let n = r.take_u32()? as usize;
                let bytes = r.take_bytes(n)?.to_vec();
                r.stack.push(Value::Bytes(bytes));
            }
            b'c' => {
                let module = String::from_utf8_lossy(&r.take_line()?).into_owned();
                let name = String::from_utf8_lossy(&r.take_line()?).into_owned();
                r.stack.push(Value::Global(module, name));
            }
            0x93 => {
                // STACK_GLOBAL: [..., module, name] → Global
                let name = r.stack.pop();
                let module = r.stack.pop();
                let (m, n) = match (module, name) {
                    (Some(Value::Str(m)), Some(Value::Str(n))) => (m, n),
                    (Some(Value::Global(m1, m2)), Some(Value::Str(n))) => (format!("{m1}.{m2}"), n),
                    _ => bail!("STACK_GLOBAL args not strings"),
                };
                r.stack.push(Value::Global(m, n));
            }
            b'{' | b'}' => r.stack.push(Value::Dict(Vec::new())),
            b']' => r.stack.push(Value::List(Vec::new())),
            b')' => r.stack.push(Value::Tuple(Vec::new())),
            b'(' => r.stack.push(Value::Mark),
            b'l' => {
                let items = r.pop_to_mark();
                r.stack.push(Value::List(items));
            }
            b't' => {
                let items = r.pop_to_mark();
                r.stack.push(Value::Tuple(items));
            }
            b'u' => {
                let items = r.pop_to_mark();
                if let Some(Value::Dict(d)) = r.stack.last_mut() {
                    for pair in items.chunks(2) {
                        if pair.len() == 2 {
                            d.push((pair[0].clone(), pair[1].clone()));
                        }
                    }
                }
            }
            b'e' => {
                let items = r.pop_to_mark();
                if let Some(Value::List(l)) = r.stack.last_mut() {
                    l.extend(items);
                }
            }
            b'a' => {
                let v = r.stack.pop();
                if let (Some(v), Some(Value::List(l))) = (v, r.stack.last_mut()) {
                    l.push(v);
                }
            }
            b's' => {
                let v = r.stack.pop();
                let k = r.stack.pop();
                if let (Some(v), Some(k), Some(Value::Dict(d))) = (v, k, r.stack.last_mut()) {
                    d.push((k, v));
                }
            }
            b'd' => {
                let items = r.pop_to_mark();
                r.stack.push(Value::Dict(
                    items
                        .chunks(2)
                        .filter(|p| p.len() == 2)
                        .map(|p| (p[0].clone(), p[1].clone()))
                        .collect(),
                ));
            }
            0x85 => r.push_tuple(1),
            0x86 => r.push_tuple(2),
            0x87 => r.push_tuple(3),
            b'R' => {
                // REDUCE: [..., func, args] → _rebuild_tensor_v2 なら Tensor メタ
                let args = r.stack.pop();
                let func = r.stack.pop();
                let is_tensor = matches!(&func, Some(Value::Global(m, n))
                    if m == "torch._utils" && (n == "_rebuild_tensor_v2" || n == "_rebuild_tensor"));
                if trace && !is_tensor {
                    eprintln!(
                        "REDUCE func = {:?}",
                        func.as_ref().map(|v| format!("{v:?}")).unwrap_or_default()
                    );
                }
                if is_tensor {
                    let t = tensor_from_args(args.as_ref());
                    if let Some(t) = t {
                        r.stack.push(Value::Tensor(t));
                    } else {
                        if trace {
                            eprintln!(
                                "REDUCE tensor args not parsed: {:?}",
                                args.as_ref().map(|a| format!("{a:?}")).unwrap_or_default()
                            );
                        }
                        r.stack.push(Value::Other);
                    }
                } else if matches!(&func, Some(Value::Global(m, _)) if m == "collections") {
                    // OrderedDict 等は dict として扱う (SETITEMS/BUILD で内容が入る)
                    r.stack.push(Value::Dict(Vec::new()));
                } else {
                    r.stack.push(Value::Other);
                }
            }
            0x81 | 0x82 => {
                // NEWOBJ / NEWOBJ_EX: 引数を捨てて Other
                if op == 0x82 {
                    let _kwargs = r.stack.pop();
                }
                let _args = r.stack.pop();
                let _func = r.stack.pop();
                r.stack.push(Value::Other);
            }
            b'b' => {
                // BUILD (OrderedDict など)。オブジェクト本体はすでに push 済み
                let _state = r.stack.pop();
            }
            b'Q' => {
                // BINPERSID: persistent id タプル → Storage 参照
                let pid = r.stack.pop();
                r.stack.push(storage_from_pid(pid.as_ref()));
            }
            b'q' => {
                let idx = r.take_u8()? as usize;
                if let Some(v) = r.stack.last() {
                    if r.memo.len() <= idx {
                        r.memo.resize(idx + 1, Value::None);
                    }
                    r.memo[idx] = v.clone();
                }
            }
            0x94 => {
                // MEMOIZE: 次の idx = memo.len()
                if let Some(v) = r.stack.last() {
                    r.memo.push(v.clone());
                }
            }
            b'h' => {
                let idx = r.take_u8()? as usize;
                r.push_memo(idx);
            }
            b'j' => {
                // LONG_BINGET (u32 index)
                let idx = r.take_u32()? as usize;
                r.push_memo(idx);
            }
            0x72 => {
                // LONG_BINPUT
                let idx = r.take_u32()? as usize;
                if let Some(v) = r.stack.last() {
                    if r.memo.len() <= idx {
                        r.memo.resize(idx + 1, Value::None);
                    }
                    r.memo[idx] = v.clone();
                }
            }
            b'0' => {
                r.stack.pop();
            }
            b'1' => {
                while let Some(v) = r.stack.pop() {
                    if matches!(v, Value::Mark) {
                        break;
                    }
                }
            }
            _ => {
                r.stack.push(Value::Other);
            }
        }
    }
    r.stack.pop().ok_or_else(|| anyhow::anyhow!("empty pickle"))
}

/// persistent id タプル → Storage。
/// 新形式: `("storage", storage_type, key, location, numel)` (5 要素)
/// 旧形式: `(storage_type, key, location, numel)` (4 要素)
fn storage_from_pid(pid: Option<&Value>) -> Value {
    let Some(Value::Tuple(t)) = pid else {
        return Value::Other;
    };
    let (type_i, key_i, numel_i) =
        if t.len() == 5 && t.first().and_then(|v| v.as_str()) == Some("storage") {
            (1usize, 2usize, 4usize)
        } else if t.len() == 4 {
            (0, 1, 3)
        } else {
            return Value::Other;
        };
    let dtype = t.get(type_i).map(|v| match v {
        Value::Global(_, n) => DType::from_storage(n),
        Value::Str(s) => DType::from_storage(s),
        _ => DType::Other,
    });
    let (Some(dtype), Some(Value::Str(key)), Some(numel)) = (
        dtype,
        t.get(key_i).filter(|v| matches!(v, Value::Str(_))),
        t.get(numel_i).and_then(|v| v.as_i64()),
    ) else {
        return Value::Other;
    };
    Value::Storage(StorageRef {
        dtype,
        key: key.clone(),
        numel,
    })
}

/// `_rebuild_tensor_v2(storage, offset, size, stride, ...)` → TensorMeta。
fn tensor_from_args(args: Option<&Value>) -> Option<TensorMeta> {
    let Value::Tuple(t) = args? else {
        return None;
    };
    let Value::Storage(storage) = t.first()? else {
        return None;
    };
    let offset = t.get(1).and_then(|v| v.as_i64()).unwrap_or(0);
    let nums = |v: Option<&Value>| -> Vec<i64> {
        match v {
            Some(Value::Tuple(items)) => items.iter().filter_map(|x| x.as_i64()).collect(),
            _ => Vec::new(),
        }
    };
    let size = nums(t.get(2));
    if size.is_empty() {
        return None;
    }
    let mut stride = nums(t.get(3));
    if stride.len() != size.len() {
        // 欠落時は row-major を仮定
        let mut acc = 1i64;
        stride = size
            .iter()
            .rev()
            .map(|d| {
                let s = acc;
                acc *= d.max(&1);
                s
            })
            .collect::<Vec<_>>();
        stride.reverse();
    }
    Some(TensorMeta {
        storage: storage.clone(),
        offset,
        size,
        stride,
    })
}

impl<'a> Reader<'a> {
    fn take_u8(&mut self) -> anyhow::Result<u8> {
        let v = *self.b.get(self.pos).ok_or_else(|| anyhow::anyhow!("eof"))?;
        self.pos += 1;
        Ok(v)
    }
    fn take_u16(&mut self) -> anyhow::Result<u16> {
        let lo = self.take_u8()? as u16;
        let hi = self.take_u8()? as u16;
        Ok(lo | (hi << 8))
    }
    fn take_i32(&mut self) -> anyhow::Result<i32> {
        let mut b = [0u8; 4];
        for x in b.iter_mut() {
            *x = self.take_u8()?;
        }
        Ok(i32::from_le_bytes(b))
    }
    fn take_u32(&mut self) -> anyhow::Result<u32> {
        let mut b = [0u8; 4];
        for x in b.iter_mut() {
            *x = self.take_u8()?;
        }
        Ok(u32::from_le_bytes(b))
    }
    fn take_u64(&mut self) -> anyhow::Result<u64> {
        let mut b = [0u8; 8];
        for x in b.iter_mut() {
            *x = self.take_u8()?;
        }
        Ok(u64::from_le_bytes(b))
    }
    fn take_line(&mut self) -> anyhow::Result<Vec<u8>> {
        let start = self.pos;
        while self.pos < self.b.len() && self.b[self.pos] != b'\n' {
            self.pos += 1;
        }
        let mut end = self.pos;
        if self.pos < self.b.len() {
            self.pos += 1;
        }
        if end > start && self.b[end - 1] == b'\r' {
            end -= 1;
        }
        Ok(self.b[start..end].to_vec())
    }
    fn take_bytes(&mut self, n: usize) -> anyhow::Result<&[u8]> {
        if self.pos + n > self.b.len() {
            anyhow::bail!("eof");
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn pop_to_mark(&mut self) -> Vec<Value> {
        let mut items = Vec::new();
        while let Some(v) = self.stack.pop() {
            if matches!(v, Value::Mark) {
                break;
            }
            items.push(v);
        }
        items.reverse();
        items
    }
    fn push_tuple(&mut self, n: usize) {
        let mut t = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(v) = self.stack.pop() {
                t.push(v);
            }
        }
        t.reverse();
        self.stack.push(Value::Tuple(t));
    }
    fn push_memo(&mut self, idx: usize) {
        if let Some(v) = self.memo.get(idx) {
            let v = v.clone();
            self.stack.push(v);
        } else {
            self.stack.push(Value::Other);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-9); // subnormal ではなく 0.5
        assert!((f16_to_f32(0x0400) - 0.00006103515625).abs() < 1e-12); // frac=1024 → 2^-14
        assert!((f16_to_f32(0x0001) - 5.960464477539063e-8).abs() < 1e-14); // 最小 subnormal
    }

    fn home() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("USERPROFILE").unwrap())
    }

    /// 実ファイルに対する構造確認 (開発機のみ): cargo test -p rcwx-nn --lib real_ -- --ignored --nocapture
    #[test]
    #[ignore = "requires local model files"]
    fn real_hubert_pt() {
        let mut f =
            PthFile::open(&home().join(".cache/rcwx/models/hubert/hubert_base.pt")).unwrap();
        let top = f.parse().unwrap();
        let model = top.get("model").unwrap();
        let d = model.as_dict().unwrap();
        println!("hubert tensors: {}", d.len());
        let wv = d
            .iter()
            .find(|(k, _)| k.as_str() == Some("encoder.pos_conv.0.weight_v"))
            .unwrap();
        match &wv.1 {
            Value::Tensor(t) => {
                println!("weight_v size {:?} dtype {:?}", t.size, t.storage.dtype);
                let v = f.read_f32(t).unwrap();
                println!("weight_v[0..4] = {:?}", &v[..4]);
            }
            other => println!("weight_v NOT a tensor: {other:?}"),
        }
        let fp = d
            .iter()
            .find(|(k, _)| k.as_str() == Some("final_proj.weight"))
            .unwrap();
        if let Value::Tensor(t) = &fp.1 {
            println!("final_proj size {:?}", t.size);
        }
    }

    #[test]
    #[ignore = "requires local model files"]
    fn real_rmvpe_pt() {
        let mut f = PthFile::open(&home().join(".cache/rcwx/models/rmvpe/rmvpe.pt")).unwrap();
        let top = f.parse().unwrap();
        let d = top.as_dict().unwrap();
        println!("rmvpe top tensors: {}", d.len());
        let cnn = d
            .iter()
            .find(|(k, _)| k.as_str() == Some("cnn.weight"))
            .unwrap();
        if let Value::Tensor(t) = &cnn.1 {
            println!("cnn.weight size {:?} dtype {:?}", t.size, t.storage.dtype);
            let v = f.read_f32(t).unwrap();
            println!("cnn.weight[0..4] = {:?}", &v[..4]);
        }
    }

    #[test]
    #[ignore = "requires local model files"]
    fn real_rvc_pth() {
        let mut f = PthFile::open(std::path::Path::new(
            "C:/lib/github/grand2-products/RCWX/rvc_models/NORA01-少女Girl/NORA01-少女Girl.pth",
        ))
        .unwrap();
        let top = f.parse().unwrap();
        let config_desc = match top.get("config") {
            Some(Value::List(l)) => format!("list(len={})", l.len()),
            _ => "other".into(),
        };
        println!(
            "version={:?} f0={:?} sr={:?} config={}",
            top.get("version").and_then(|v| v.as_str()),
            top.get("f0").and_then(|v| v.as_i64()),
            top.get("sr").map(|v| format!("{v:?}")),
            config_desc
        );
        let w = top.get("weight").unwrap();
        let d = w.as_dict().unwrap();
        println!("weight tensors: {}", d.len());
        let emb = d
            .iter()
            .find(|(k, _)| k.as_str() == Some("emb_g.weight"))
            .unwrap();
        if let Value::Tensor(t) = &emb.1 {
            println!("emb_g size {:?} dtype {:?}", t.size, t.storage.dtype);
            let v = f.read_f32(t).unwrap();
            println!("emb_g[0..4] = {:?}", &v[..4]);
        }
    }
}
