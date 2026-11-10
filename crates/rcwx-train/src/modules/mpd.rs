//! MPD 判別器 (RVC v2: DiscriminatorS + periods [2,3,5,7,11,17,23,37])。
//! 重み名は PyTorch state_dict キーと同一。

use std::collections::HashMap;

use rcwx_nn::engine::Engine;
use rcwx_nn::engine::tape::{ParamStore, Tape, Var};

use crate::modules::Adopted;

pub const PERIODS_V2: [u32; 8] = [2, 3, 5, 7, 11, 17, 23, 37];

/// 重みキー (disc_p.N.*) から実際に存在する disc_p 数を数え、対応する
/// 先頭 periods を返す (テスト用 tiny D は 2 個など)。
pub fn infer_periods(w: &MpdWeights) -> Vec<u32> {
    let mut n = 0usize;
    for k in w.keys() {
        if let Some(rest) = k.strip_prefix("disc_p.") {
            if let Some(idx) = rest.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
                if idx + 1 > n {
                    n = idx + 1;
                }
            }
        }
    }
    let n = n.min(PERIODS_V2.len());
    PERIODS_V2[..n].to_vec()
}

pub struct Mpd {
    pub store: ParamStore,
    pub periods: Vec<u32>,
    pub has_s: bool,
}

pub type MpdWeights = HashMap<String, (Vec<f32>, Vec<usize>)>;

impl Mpd {
    /// w は state_dict 風 ("disc_s.convs.0.weight_v" 等)。空なら判別器なし
    /// (periods も空扱い — smoke 用)。
    pub fn new(eng: &Engine, w: &MpdWeights, periods: &[u32]) -> anyhow::Result<Self> {
        if w.is_empty() {
            return Ok(Self {
                store: ParamStore::new(),
                periods: vec![],
                has_s: false,
            });
        }
        let mut store = ParamStore::new();
        let has_s =
            w.contains_key("disc_s.convs.0.weight") || w.contains_key("disc_s.convs.0.weight_v");
        let keys: Vec<String> = w.keys().cloned().collect();
        for k in keys {
            if k.ends_with(".weight_v") {
                let prefix = k.trim_end_matches(".weight_v").to_string();
                if store
                    .wns
                    .iter()
                    .any(|wn| store.entries[wn.v_idx].name == format!("{prefix}.weight_v"))
                {
                    continue;
                }
                if let Some((v, vs)) = w.get(&format!("{prefix}.weight_v")) {
                    let g = w
                        .get(&format!("{prefix}.weight_g"))
                        .map(|(gv, _)| gv.as_slice())
                        .unwrap_or(&[1.0]);
                    store.add_weight_norm(eng, &prefix, v, g, vs.clone());
                }
            } else if k.ends_with(".weight_g") {
                continue;
            } else if k.ends_with(".weight") {
                let prefix = k.trim_end_matches(".weight").to_string();
                if w.contains_key(&format!("{prefix}.weight_v")) {
                    continue;
                }
                let (d, s) = &w[&k];
                store.add(eng, &k, d, s.clone());
            } else if k.ends_with(".bias") {
                let (d, s) = &w[&k];
                store.add(eng, &k, d, s.clone());
            }
        }
        Ok(Self {
            store,
            periods: periods.to_vec(),
            has_s,
        })
    }

    /// 全パラメータ adopt (WN fold 込み)。
    pub fn adopt(&self, tape: &mut Tape<'_>) -> Adopted {
        let mut a = Adopted::default();
        let wn_names: Vec<String> = self
            .store
            .wns
            .iter()
            .map(|wn| self.store.entries[wn.v_idx].name.clone())
            .collect();
        for name in &wn_names {
            let prefix = name.trim_end_matches(".weight_v");
            let wn_idx = self
                .store
                .wns
                .iter()
                .position(|wn| self.store.entries[wn.v_idx].name == format!("{prefix}.weight_v"))
                .unwrap();
            let wn = &self.store.wns[wn_idx];
            let v = a.adopt_entry(tape, &self.store, wn.v_idx);
            let g = a.adopt_entry(tape, &self.store, wn.g_idx);
            let w = tape.adopt_wn(&self.store, wn_idx, v, g);
            a.p.insert(format!("{prefix}.w"), w);
        }
        for (i, e) in self.store.entries.iter().enumerate() {
            if e.name.ends_with(".weight_v") || e.name.ends_with(".weight_g") {
                continue;
            }
            if e.name.ends_with(".weight") {
                let prefix = e.name.trim_end_matches(".weight");
                if self
                    .store
                    .wns
                    .iter()
                    .any(|wn| self.store.entries[wn.v_idx].name == format!("{prefix}.weight_v"))
                {
                    continue;
                }
                let v = a.adopt_entry(tape, &self.store, i);
                a.p.insert(format!("{prefix}.w"), v);
            } else if e.name.ends_with(".bias") {
                let v = a.adopt_entry(tape, &self.store, i);
                a.p.insert(e.name.clone(), v);
            }
        }
        a
    }
}

/// DiscriminatorS (grouped conv 列 + post)。
pub fn disc_s(tape: &mut Tape<'_>, a: &Adopted, x: Var, batch: u32, t_len: u32) -> (Var, Vec<Var>) {
    let chans = [
        (1u32, 16u32),
        (16, 64),
        (64, 256),
        (256, 1024),
        (1024, 1024),
        (1024, 1024),
    ];
    let ks = [15u32, 41, 41, 41, 41, 5];
    let strides = [1u32, 4, 4, 4, 4, 1];
    let groups = [1u32, 4, 16, 64, 256, 1];
    let mut fmaps = Vec::new();
    let mut x = x;
    let mut in_c = 1u32;
    let mut t = t_len;
    for i in 0..chans.len() {
        let (oc, k, s, g) = (chans[i].1, ks[i], strides[i], groups[i]);
        let out_len = t.saturating_sub(k) / s + 1;
        let w = a.w(&format!("disc_s.convs.{i}"));
        let b = a.bias(&format!("disc_s.convs.{i}"));
        x = tape.conv1d_grouped(x, w, b, oc, in_c, k, s, 0, g, batch, t, out_len);
        x = tape.lrelu(x, 0.1);
        fmaps.push(x);
        in_c = oc;
        t = out_len;
    }
    let out_len = t.saturating_sub(3) + 1;
    let w = a.w("disc_s.conv_post");
    let b = a.bias("disc_s.conv_post");
    let logits = tape.conv1d(x, w, b, 1, in_c, 3, 1, 1, 0, batch, t, out_len);
    (logits, fmaps)
}

/// DiscriminatorP (period リシェイプ + conv2d 列 + post)。
pub fn disc_p(
    tape: &mut Tape<'_>,
    a: &Adopted,
    idx: usize,
    period: u32,
    x: Var,
    batch: u32,
    t_len: u32,
) -> (Var, Vec<Var>) {
    let prefix = format!("disc_p.{idx}");
    let rem = t_len % period;
    let pad = if rem == 0 { 0 } else { period - rem };
    // 末尾 reflect pad (upstream: F.pad(y, (0, n_pad), "reflect"))。
    // reflect_pad_dims は両側 pad のため、pad オフセットの gather で
    // 先頭側を落として末尾のみの pad に変換する。
    let x = if pad > 0 {
        let both = tape.reflect_pad_dims(x, batch, 1, t_len, pad);
        tape.gather_frames_pub(both, pad, 1, t_len + 2 * pad, t_len + pad)
    } else {
        x
    };
    let t2 = t_len + pad;
    let h = t2 / period;
    let chans = [32u32, 128, 512, 1024, 1024];
    let mut fmaps = Vec::new();
    let mut c_in = 1u32;
    let mut hh = h;
    let ww = period;
    let mut cur = x;
    for i in 0..chans.len() {
        let w = a.w(&format!("{prefix}.convs.{i}"));
        let b = a.bias(&format!("{prefix}.convs.{i}"));
        // conv2d は pad_h=2 (padding=(2,0) 相当): oh = (h + 2·2 − 5)/3 + 1
        let oh = (hh + 4 - 5) / 3 + 1;
        cur = tape.conv2d(cur, w, b, chans[i], c_in, 5, 1, 3, 1, 2, 0, batch, hh, ww);
        cur = tape.lrelu(cur, 0.1);
        fmaps.push(cur);
        c_in = chans[i];
        hh = oh;
    }
    let w = a.w(&format!("{prefix}.conv_post"));
    let b = a.bias(&format!("{prefix}.conv_post"));
    let oh = (hh + 2 - 3) / 1 + 1;
    let _ = oh;
    let logits = tape.conv2d(cur, w, b, 1, c_in, 3, 1, 1, 1, 1, 0, batch, hh, ww);
    (logits, fmaps)
}

/// MPD 全体 forward。x [B·T] → (logits, fmaps)。
pub fn mpd_forward(
    tape: &mut Tape<'_>,
    mpd: &Mpd,
    a: &Adopted,
    x: Var,
    batch: u32,
    t_len: u32,
) -> (Vec<Var>, Vec<Var>) {
    let mut logits = Vec::new();
    let mut fmaps = Vec::new();
    if mpd.has_s {
        let (l, f) = disc_s(tape, a, x, batch, t_len);
        logits.push(l);
        fmaps.extend(f);
    }
    for (i, p) in mpd.periods.iter().enumerate() {
        let (l, f) = disc_p(tape, a, i, *p, x, batch, t_len);
        logits.push(l);
        fmaps.extend(f);
    }
    (logits, fmaps)
}
