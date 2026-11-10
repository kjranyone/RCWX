//! MPD 判別器 (RVC v2: DiscriminatorS + periods [2,3,5,7,11,17,23,37])。
//! 重み名は PyTorch state_dict キーと同一。

use std::collections::HashMap;

use rcwx_nn::engine::Engine;
use rcwx_nn::engine::tape::{ParamStore, Tape, Var};

use crate::modules::Adopted;

pub const PERIODS_V2: [u32; 8] = [2, 3, 5, 7, 11, 17, 23, 37];

pub struct Mpd {
    pub store: ParamStore,
    pub periods: Vec<u32>,
    pub has_s: bool,
    /// "discriminators.0.convs.0" 等のプレフィックス集合
    pub prefixes: Vec<String>,
}

pub type MpdWeights = HashMap<String, (Vec<f32>, Vec<usize>)>;

impl Mpd {
    /// w は state_dict 風 ("disc_s.convs.0.weight_v" 等)。
    pub fn new(eng: &Engine, w: &MpdWeights, periods: &[u32]) -> anyhow::Result<Self> {
        let mut store = ParamStore::new();
        // S 判別器の存在判定
        let has_s = w.contains_key("disc_s.convs.0.weight");
        let mut register = |store: &mut ParamStore, prefix: &str, w: &MpdWeights| {
            let wkey = format!("{prefix}.weight");
            if let Some((v, vs)) = w.get(&format!("{prefix}.weight_v")) {
                let g = w
                    .get(&format!("{prefix}.weight_g"))
                    .map(|(gv, _)| gv[0])
                    .unwrap_or(1.0);
                return store.add_weight_norm(eng, prefix, v, g, vs.clone());
            }
            let (d, s) = &w[&wkey];
            store.add(eng, prefix, d, s.clone())
        };
        // WN/param 登録 (名前 → entry idx は Adopted 経由で解決するため
        // ここでは store に全部積むだけ)
        let keys: Vec<&String> = w.keys().collect();
        for k in keys {
            if k.ends_with(".weight_v") || k.ends_with(".weight_g") {
                let prefix = k
                    .trim_end_matches(".weight_v")
                    .trim_end_matches(".weight_g");
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
                        .map(|(gv, _)| gv[0])
                        .unwrap_or(1.0);
                    store.add_weight_norm(eng, prefix, v, g, vs.clone());
                }
            } else if k.ends_with(".weight") || k.ends_with(".bias") {
                // WN 畳み込みの weight は fold 済みなので登録しない
                let prefix = k
                    .trim_end_matches(".weight")
                    .trim_end_matches(".bias")
                    .to_string();
                let is_wn = w.contains_key(&format!("{prefix}.weight_v"));
                if is_wn && k.ends_with(".weight") {
                    continue;
                }
                let (d, s) = &w[k];
                store.add(eng, k, d, s.clone());
            }
        }
        let mut prefixes = Vec::new();
        if has_s {
            prefixes.push("disc_s".to_string());
        }
        for (i, _) in periods.iter().enumerate() {
            prefixes.push(format!("disc_p.{i}"));
        }
        Ok(Self {
            store,
            periods: periods.to_vec(),
            has_s,
            prefixes,
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
            } else {
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
        let out_len = (t + 2 * 0).saturating_sub(k) / s + 1; // pad なし
        let w = a.w(&format!("disc_s.convs.{i}"));
        let b = a.bias(&format!("disc_s.convs.{i}"));
        x = tape.conv1d_grouped(x, w, b, oc, in_c, k, s, 0, g, batch, t, out_len);
        x = tape.lrelu(x, 0.1);
        fmaps.push(x);
        in_c = oc;
        t = out_len;
    }
    let out_len = t.saturating_sub(3) + 1;
    let w = a.w("disc_s.post");
    let b = a.bias("disc_s.post");
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
    // 時間軸を period の倍数へ reflect pad
    let rem = t_len % period;
    let pad = if rem == 0 { 0 } else { period - rem };
    let x = if pad > 0 { tape.reflect_pad(x, pad) } else { x };
    let t2 = t_len + pad;
    let h = t2 / period;
    // 入力を (b, 1, h, period) 扱い: x 平坦 [b, 1, t2] = [b, 1, h·period]
    let chans = [32u32, 128, 512, 1024, 1024];
    let mut fmaps = Vec::new();
    let mut c_in = 1u32;
    let mut hh = h;
    let mut ww = period;
    let mut cur = x;
    for i in 0..chans.len() {
        let w = a.w(&format!("{prefix}.convs.{i}"));
        let b = a.bias(&format!("{prefix}.convs.{i}"));
        let oh = (hh + 2 - 5) / 3 + 1; // pad=2, k=5, s=3
        cur = tape.conv2d(cur, w, b, chans[i], c_in, 5, 1, 3, 1, 2, batch, hh, ww);
        cur = tape.lrelu(cur, 0.1);
        fmaps.push(cur);
        c_in = chans[i];
        hh = oh;
        ww = ww; // 幅方向 stride 1
    }
    let w = a.w(&format!("{prefix}.post"));
    let b = a.bias(&format!("{prefix}.post"));
    let oh = (hh + 2 - 3) / 1 + 1;
    let logits = tape.conv2d(cur, w, b, 1, c_in, 3, 1, 1, 1, 1, batch, hh, ww);
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
    for (i, &p) in mpd.periods.iter().enumerate() {
        let (l, f) = disc_p(tape, a, i, p, x, batch, t_len);
        logits.push(l);
        fmaps.extend(f);
    }
    (logits, fmaps)
}
