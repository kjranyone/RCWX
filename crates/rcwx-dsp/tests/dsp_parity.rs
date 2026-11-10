//! rcwx-dsp パリティ: tools/dump_dsp_golden.py の golden との一致検証。

use rcwx_dsp::feedback::max_normalized_lag_correlation;
use rcwx_dsp::moe::apply_moe_f0_style;
use rcwx_dsp::noise_gate::{NoiseGate, Sensitivity};
use rcwx_dsp::postprocess::{PostprocessConfig, Postprocessor};
use rcwx_dsp::resample::{resample_linear, resample_poly, StatefulResampler};
use rcwx_dsp::sola::{sola_crossfade, sola_flush, SolaState};
use rcwx_dsp::spectral::StreamingSpectralGate;

fn golden_path() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/golden/dsp_golden.npz")
        .to_string_lossy()
        .into_owned()
}

fn read_npz_f32(path: &str, name: &str) -> Vec<f32> {
    read_npz_raw(path, name, 4, |c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
}

fn read_npz_i64(path: &str, name: &str) -> Vec<i64> {
    read_npz_raw(path, name, 8, |c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
}

fn read_npz_f64(path: &str, name: &str) -> Vec<f64> {
    read_npz_raw(path, name, 8, |c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
}

fn read_npz_raw<T: Copy>(path: &str, name: &str, width: usize, cv: impl Fn(&[u8]) -> T) -> Vec<T> {
    let file = std::fs::File::open(path).expect("npz open");
    let mut zip = zip::ZipArchive::new(file).expect("zip");
    let mut entry = zip.by_name(&format!("{name}.npy")).expect("npy entry");
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut entry, &mut buf).expect("read");
    let hlen = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    buf[10 + hlen..].chunks_exact(width).map(cv).collect()
}

fn maxdiff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

#[test]
fn resample_poly_matches_scipy() {
    let gp = golden_path();
    if !std::path::Path::new(&gp).exists() {
        eprintln!("golden missing; run tools/dump_dsp_golden.py");
        return;
    }
    let in48 = read_npz_f32(&gp, "rs_in48");
    let batch = read_npz_f32(&gp, "rs_batch_48_16");
    let mine = resample_poly(&in48, 16000, 48000);
    assert_eq!(mine.len(), batch.len());
    println!("48→16 maxdiff: {:.3e}", maxdiff(&mine, &batch));
    assert!(maxdiff(&mine, &batch) < 1e-5);

    let in16 = read_npz_f32(&gp, "rs_in16");
    let batch = read_npz_f32(&gp, "rs_batch_16_48");
    let mine = resample_poly(&in16, 48000, 16000);
    assert_eq!(mine.len(), batch.len());
    println!("16→48 maxdiff: {:.3e}", maxdiff(&mine, &batch));
    assert!(maxdiff(&mine, &batch) < 1e-5);

    let lin_ref = read_npz_f32(&gp, "rs_batch_48_16_lin");
    let lin = resample_linear(&in48, 48000, 16000);
    assert_eq!(lin.len(), lin_ref.len());
    assert!(maxdiff(&lin, &lin_ref) < 1e-5, "linear diff {}", maxdiff(&lin, &lin_ref));

    // ストリーミング (overlap-save)
    let mut sr = StatefulResampler::new(48000, 16000);
    let mut stream = Vec::new();
    for c in in48.chunks(1440) {
        stream.extend(sr.resample_chunk(c));
    }
    let ref_stream = read_npz_f32(&gp, "rs_stream_48_16");
    assert_eq!(stream.len(), ref_stream.len());
    println!("48→16 stream maxdiff: {:.3e}", maxdiff(&stream, &ref_stream));
    assert!(maxdiff(&stream, &ref_stream) < 1e-5);

    let mut sr = StatefulResampler::new(16000, 48000);
    let mut stream = Vec::new();
    for c in in16.chunks(480) {
        stream.extend(sr.resample_chunk(c));
    }
    let ref_stream = read_npz_f32(&gp, "rs_stream_16_48");
    assert_eq!(stream.len(), ref_stream.len());
    println!("16→48 stream maxdiff: {:.3e}", maxdiff(&stream, &ref_stream));
    assert!(maxdiff(&stream, &ref_stream) < 1e-5);
}

#[test]
fn sola_matches_python() {
    let gp = golden_path();
    if !std::path::Path::new(&gp).exists() {
        eprintln!("golden missing");
        return;
    }
    let n_in = read_npz_i64(&gp, "sola_in").len() / 1600 * 1600;
    let flat_in = read_npz_f32(&gp, "sola_in");
    let out_lens = read_npz_i64(&gp, "sola_out_lens");
    let out_ref = read_npz_f32(&gp, "sola_out");
    let flush_ref = read_npz_f32(&gp, "sola_flush");
    let _ = n_in;

    let mut state = SolaState::new(480, 480);
    let mut collected = Vec::new();
    let n_chunks = out_lens.len();
    for i in 0..n_chunks {
        let seg = &flat_in[i * 1600..(i + 1) * 1600];
        let out = sola_crossfade(seg, &mut state, 960);
        assert_eq!(out.len(), out_lens[i] as usize, "chunk {i} out len");
        collected.extend(out);
    }
    assert_eq!(collected.len(), out_ref.len());
    println!("sola maxdiff: {:.3e}", maxdiff(&collected, &out_ref));
    assert!(maxdiff(&collected, &out_ref) < 1e-5);
    let flush = sola_flush(&mut state);
    assert_eq!(flush.len(), flush_ref.len());
    println!("sola flush maxdiff: {:.3e}", maxdiff(&flush, &flush_ref));
    assert!(maxdiff(&flush, &flush_ref) < 1e-5);
}

#[test]
fn spectral_gate_matches_python() {
    let gp = golden_path();
    if !std::path::Path::new(&gp).exists() {
        eprintln!("golden missing");
        return;
    }
    let input = read_npz_f32(&gp, "spec_in");
    let out_ref = read_npz_f32(&gp, "spec_out");
    let mut gate = StreamingSpectralGate::new(16000, 6.0, -24.0);
    let mut out = Vec::new();
    for c in input.chunks(320) {
        out.extend(gate.process(c));
    }
    assert_eq!(out.len(), out_ref.len());
    let d = maxdiff(&out, &out_ref);
    let corr = rcwx_dsp::spectral_corr(&out, &out_ref);
    println!("spectral maxdiff: {d:.3e} corr: {corr:.6}");
    // f64 FFT と累積状態の丸め差はあるが位相・ゲート判定は同一のはず
    assert!(d < 1e-3, "spectral diff {d}");
}

#[test]
fn noise_gate_matches_python() {
    let gp = golden_path();
    if !std::path::Path::new(&gp).exists() {
        eprintln!("golden missing");
        return;
    }
    for (tag, auto) in [("man", false), ("auto", true)] {
        let input = read_npz_f32(&gp, &format!("ng_in_{tag}"));
        let out_ref = read_npz_f32(&gp, &format!("ng_out_{tag}"));
        let mut ng = NoiseGate::new(16000, -40.0);
        ng.auto_threshold = auto;
        if auto {
            ng.set_sensitivity(Sensitivity::Mid);
        }
        let mut out = Vec::new();
        for c in input.chunks(320) {
            out.extend(ng.process(c));
        }
        assert_eq!(out.len(), out_ref.len());
        let d = maxdiff(&out, &out_ref);
        println!("noise_gate[{tag}] maxdiff: {d:.3e}");
        assert!(d < 1e-6, "noise gate [{tag}] diff {d}");
    }
}

#[test]
fn postprocess_matches_python() {
    let gp = golden_path();
    if !std::path::Path::new(&gp).exists() {
        eprintln!("golden missing");
        return;
    }
    let flat_in = read_npz_f32(&gp, "pp_in");
    let flat_ref = read_npz_f32(&gp, "pp_out");
    let mut pp = Postprocessor::new(48000, PostprocessConfig::default());
    let mut out = Vec::new();
    for i in 0..10 {
        let c = &flat_in[i * 1440..(i + 1) * 1440];
        let o = pp.process(c);
        out.extend(o);
    }
    let d = maxdiff(&out, &flat_ref);
    println!("postprocess maxdiff: {d:.3e}");
    assert!(d < 1e-4, "postprocess diff {d}");
}

#[test]
fn moe_matches_python() {
    let gp = golden_path();
    if !std::path::Path::new(&gp).exists() {
        eprintln!("golden missing");
        return;
    }
    let mut f0 = read_npz_f32(&gp, "moe_in");
    let ref_out = read_npz_f32(&gp, "moe_out");
    apply_moe_f0_style(&mut f0, 0.45);
    let d = maxdiff(&f0, &ref_out);
    println!("moe maxdiff: {d:.3e}");
    assert!(d < 1e-3, "moe diff {d}");
}

#[test]
fn feedback_corr_matches_python() {
    let gp = golden_path();
    if !std::path::Path::new(&gp).exists() {
        eprintln!("golden missing");
        return;
    }
    let needle = read_npz_f32(&gp, "fb_needle");
    let hay = read_npz_f32(&gp, "fb_hay");
    let hay2 = read_npz_f32(&gp, "fb_hay2");
    let ref1 = read_npz_f64(&gp, "fb_corr")[0];
    let ref2 = read_npz_f64(&gp, "fb_corr2")[0];
    let mine1 = max_normalized_lag_correlation(&needle, &hay);
    let mine2 = max_normalized_lag_correlation(&needle, &hay2);
    let fft1 = rcwx_dsp::feedback::max_normalized_lag_correlation_fft(&needle, &hay);
    let fft2 = rcwx_dsp::feedback::max_normalized_lag_correlation_fft(&needle, &hay2);
    println!("fb fft corr: {fft1:.9} vs {mine1:.9} | {fft2:.9} vs {mine2:.9}");
    assert!((fft1 - mine1).abs() < 1e-6, "fft corr1 mismatch");
    assert!((fft2 - mine2).abs() < 1e-6, "fft corr2 mismatch");
    println!(
        "fb corr: {mine1:.12} vs {ref1:.12} | {mine2:.12} vs {ref2:.12}"
    );
    assert!((mine1 - ref1).abs() < 1e-7);
    assert!((mine2 - ref2).abs() < 1e-7);
}
