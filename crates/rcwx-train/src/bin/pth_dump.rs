//! .pth の state_dict キーをダンプ (開発用)。
//! 使い方: cargo run -p rcwx-train --release --bin pth_dump -- <file.pth>

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: pth_dump <file.pth>");
    let mut f = rcwx_nn::pth::PthFile::open(std::path::Path::new(&path))?;
    let top = f.parse()?;
    let sd = top
        .get("model")
        .or_else(|| top.get("weight"))
        .ok_or_else(|| anyhow::anyhow!("no model/weight"))?;
    let d = sd.as_dict().unwrap();
    let mut names: Vec<String> = d
        .iter()
        .filter_map(|(k, _)| k.as_str().map(|s| s.to_string()))
        .collect();
    names.sort();
    println!("total: {}", names.len());
    for n in names.iter().take(12) {
        println!("  {n}");
    }
    Ok(())
}
