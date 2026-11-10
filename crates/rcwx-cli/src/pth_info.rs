//! torch .pth (zip + pickle) の最小リーダ — モデル情報取得用。
//!
//! data.pkl を解釈してトップレベル dict の version / config / f0 と
//! weight 内のテンソル名→shape を取り出す。数値ストレージは読まない。

#[derive(Debug, Clone)]
pub enum Value {
    Mark,
    None,
    Bool,
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Tensor { size: Vec<i64> },
    Global(String, String),
    Other,
}

fn get_str(v: &Value) -> Option<&str> {
    if let Value::Str(s) = v {
        Some(s)
    } else {
        None
    }
}

pub struct PthInfo {
    pub version: Option<String>,
    pub config: Vec<f64>,
    pub f0: Option<i64>,
    pub sr: Option<i64>,
    pub emb_g_dim: Option<i64>,
    pub has_nsf: bool,
    pub enc_pre_in: Option<i64>,
}

pub fn read_pth_info(path: &std::path::Path) -> anyhow::Result<PthInfo> {
    let file = std::fs::File::open(path)?;
    let mut zip = zip::ZipArchive::new(file)?;
    // torch zip は <prefix>/data.pkl (prefix はアーカイブ名依存)
    let pkl_name = zip
        .file_names()
        .filter_map(|n| n.ok())
        .find(|n| n.ends_with("data.pkl"))
        .ok_or_else(|| anyhow::anyhow!("data.pkl not found in archive"))?
        .into_owned();
    let mut entry = zip.by_name(&pkl_name)?;
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut entry, &mut buf)?;
    drop(entry);
    let top = parse_pickle(&buf)?;
    let get_str = get_str as fn(&Value) -> Option<&str>;
    let dict = match &top {
        Value::Dict(d) => d,
        _ => anyhow::bail!("unexpected pickle root"),
    };
    let mut info = PthInfo {
        version: None,
        config: Vec::new(),
        f0: None,
        sr: None,
        emb_g_dim: None,
        has_nsf: false,
        enc_pre_in: None,
    };
    for (k, v) in dict {
        match get_str(k) {
            Some("version") => info.version = get_str(v).map(|s| s.to_string()),
            Some("config") => {
                if let Value::List(items) = v {
                    info.config = items
                        .iter()
                        .map(|x| match x {
                            Value::Int(i) => *i as f64,
                            Value::Float(f) => *f,
                            _ => 0.0,
                        })
                        .collect();
                }
            }
            Some("f0") => {
                if let Value::Int(i) = v {
                    info.f0 = Some(*i);
                }
            }
            Some("sr") => {
                if let Value::Int(i) = v {
                    info.sr = Some(*i);
                }
            }
            Some("weight") => {
                if let Value::Dict(w) = v {
                    for (wk, wv) in w {
                        let name = match get_str(wk) {
                            Some(s) => s,
                            None => continue,
                        };
                        if let Value::Tensor { size } = wv {
                            if name == "emb_g.weight" {
                                info.emb_g_dim = size.first().copied();
                            } else if name == "enc_p.pre.weight" {
                                info.enc_pre_in = size.get(1).copied();
                            }
                        }
                        if name == "dec.m_source.l_linear.weight" {
                            info.has_nsf = true;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(info)
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    memo: Vec<Value>,
}

fn parse_pickle(buf: &[u8]) -> anyhow::Result<Value> {
    let mut r = Reader {
        b: buf,
        pos: 0,
        stack: Vec::new(),
        memo: Vec::new(),
    };
    loop {
        let op = *r
            .b
            .get(r.pos)
            .ok_or_else(|| anyhow::anyhow!("pickle truncated"))?;
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
            0x88 => r.stack.push(Value::Bool),
            0x89 => r.stack.push(Value::Bool),
            b'K' => { let v = r.take_u8()? as i64; r.stack.push(Value::Int(v)); }
            b'M' => { let v = r.take_u16()? as i64; r.stack.push(Value::Int(v)); }
            b'J' => { let v = r.take_i32()? as i64; r.stack.push(Value::Int(v)); }
            0x8a => {
                // LONG1
                let n = r.take_u8()? as usize;
                let mut v: i64 = 0;
                for i in 0..n {
                    v |= (r.take_u8()? as i64) << (8 * i);
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
                let v = Value::Float(f64::from_bits(bits));
                r.stack.push(v);
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
            b'c' => {
                let module = String::from_utf8_lossy(&r.take_line()?).into_owned();
                let name = String::from_utf8_lossy(&r.take_line()?).into_owned();
                r.stack.push(Value::Global(module, name));
            }
            b'{' => r.stack.push(Value::Dict(Vec::new())),
            b'}' => r.stack.push(Value::Dict(Vec::new())),
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
                if let Some(Value::List(l)) = r.stack.last_mut() {
                    if let Some(v) = v {
                        l.push(v);
                    }
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
                // REDUCE: args, func → _rebuild_tensor_v2 なら size を持つ Tensor
                let args = r.stack.pop();
                let func = r.stack.pop();
                let is_tensor = matches!(&func, Some(Value::Global(m, n))
                    if m == "torch._utils" && (n == "_rebuild_tensor_v2" || n == "_rebuild_tensor"));
                if is_tensor {
                    let size = match &args {
                        Some(Value::Tuple(t)) if t.len() >= 3 => match &t[2] {
                            Value::Tuple(size) => size
                                .iter()
                                .map(|v| match v {
                                    Value::Int(i) => *i,
                                    _ => 0,
                                })
                                .collect(),
                            _ => Vec::new(),
                        },
                        _ => Vec::new(),
                    };
                    r.stack.push(Value::Tensor { size });
                } else if matches!(&func, Some(Value::Global(m, _)) if m == "collections") {
                    // OrderedDict 等は dict として扱う (SETITEMS/BUILD で内容が入る)
                    r.stack.push(Value::Dict(Vec::new()));
                } else {
                    r.stack.push(Value::Other);
                }
            }
            0x81 | 0x82 => {
                // NEWOBJ / NEWOBJ_EX: 引数を捨てて Other
                let _args = r.stack.pop();
                let _func = r.stack.pop();
                r.stack.push(Value::Other);
            }
            b'b' => {
                // BUILD (OrderedDict など)。オブジェクト本体はすでに push 済み
                let _state = r.stack.pop();
            }
            b'Q' => {
                // BINPERSID: ストレージ参照 → Other (shape は REDUCE 側で持つ)
                let _pid = r.stack.pop();
                r.stack.push(Value::Other);
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
                if let Some(v) = r.memo.get(idx) {
                    let v = v.clone();
                    r.stack.push(v);
                } else {
                    r.stack.push(Value::Other);
                }
            }
            0x8b => {
                // LONG_BINGET
                let idx = r.take_u32()? as usize;
                if let Some(v) = r.memo.get(idx) {
                    let v = v.clone();
                    r.stack.push(v);
                } else {
                    r.stack.push(Value::Other);
                }
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
    r.stack
        .pop()
        .ok_or_else(|| anyhow::anyhow!("empty pickle"))
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
}
