//! IVF 検索パリティ: FAISS 検索との一致検証。

use rcwx_nn::ivf::IvfIndex;

fn golden_path() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/golden/ivf_golden.npz")
        .to_string_lossy()
        .into_owned()
}

fn read_npz_f32(path: &str, name: &str, frames: usize) -> Vec<f32> {
    let file = std::fs::File::open(path).expect("npz open");
    let mut zip = zip::ZipArchive::new(file).expect("zip");
    let mut entry = zip.by_name(&format!("{name}.npy")).expect("npy entry");
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut entry, &mut buf).expect("read");
    let hlen = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let data = &buf[10 + hlen..];
    let n = data.len() / 4;
    let mut out = Vec::with_capacity(n);
    for c in data.chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    assert_eq!(out.len(), frames * 768, "unexpected size for {name}");
    out
}

#[test]
fn ivf_matches_faiss() {
    let gp = golden_path();
    if !std::path::Path::new(&gp).exists() {
        eprintln!("golden missing; run tools/convert_ivf.py first");
        return;
    }
    let index = match IvfIndex::load(&rcwx_nn::ivf::default_ivf_dir()) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("ivf index not found ({e}); skipping");
            return;
        }
    };
    let t = 40; // 2 × 20 フレーム
    let query = read_npz_f32(&gp, "query", t);
    let blended_ref = read_npz_f32(&gp, "blended", t);

    let mut feats = query;
    index.blend(&mut feats, t, 0.15, 4);

    // 量子化無しの f32 演算なのでほぼ厳密一致 (top-k 選択の並び差は和で相殺)
    let maxdiff = feats
        .iter()
        .zip(&blended_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cos = rcwx_nn::cosine(&feats, &blended_ref);
    println!("ivf blended maxdiff: {maxdiff:.3e} cosine: {cos:.8}");
    assert!(maxdiff < 1e-3, "ivf maxdiff {maxdiff}");
}
