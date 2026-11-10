//! base .pth (WN 保持) を native 形式へ変換する開発ツール。
//! 使い方: cargo run -p rcwx-train --release --bin export_base -- <base_g.pth> <out_dir>

use rcwx_train::data::preprocess::pth_loader;
use rcwx_train::modules::synth_train::{SynthTrain, SynthTrainCfg};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: export_base <base_g.pth> <out_dir>");
        std::process::exit(1);
    }
    let src = &args[1];
    let out = std::path::Path::new(&args[2]);
    let eng = rcwx_nn::engine::Engine::new(rcwx_nn::engine::Backend::Vulkan)?;
    let gw = pth_loader::load_pth_weights(std::path::Path::new(src))?;
    let cfg = SynthTrainCfg::v2_48k();
    let synth = SynthTrain::new(&eng, cfg.clone(), &gw)?;
    rcwx_train::export::export_native(&eng, &synth, &cfg, out)?;
    println!("exported: {}", out.display());
    Ok(())
}
