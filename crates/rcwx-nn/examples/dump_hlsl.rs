//! flash_attn WGSL を naga で HLSL に変換して出す (DXC コンパイル時間研究用)。
//! 使い方: cargo run -p rcwx-nn --release --example dump_hlsl -- out.hlsl
fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "flash_attn.hlsl".into());
    let src = match std::env::args().nth(2).as_deref() {
        Some("matmul") => rcwx_nn::engine::shaders::matmul_tiled_src(),
        _ => rcwx_nn::engine::shaders::flash_attn_src(),
    };
    if std::env::var_os("RAW_WGSL").is_some() {
        std::fs::write(&out, &src).unwrap();
        eprintln!("raw wgsl written");
        return;
    }
    let module = naga::front::wgsl::parse_str(&src).expect("wgsl parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator.validate(&module).expect("validate");
    let opts = naga::back::hlsl::Options::default();
    let po = naga::back::hlsl::PipelineOptions::default();
    let mut buf = String::new();
    {
        // Writer は out を消費するため、into_raw で取り出す代わりにスコープを抜けて
        // 直接文字列へ書き込む実装は持たない → write 後に内部 out を奪う
        let mut w = naga::back::hlsl::Writer::new(&mut buf, &opts, &po);
        w.write(&module, &info, None).expect("hlsl write");
    }
    std::fs::write(&out, &buf).expect("write file");
    eprintln!("wrote {out} ({} bytes)", buf.len());
}
