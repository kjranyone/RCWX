"""swift-f0 ONNX → safetensors 変換 + golden ダンプ (dev ツール・実行時不要)。

入力: .venv/Lib/site-packages/swift_f0/model.onnx
出力: ~/.cache/rcwx/models/rust/swiftf0.safetensors
      target/golden/swiftf0_golden.npz  (パリティテスト基準)

モデル構造 (opset 20):
  pad 384 両端 (constant 0) → STFT(1024, hop256, periodic-hann) →
  magnitude bins 3..135 (132 bins) → log(mag+1e-8) →
  conv2d 5x5 SAME_UPPER ×5 (1→8→16→32→64→1) + relu →
  freq_projection 1x1 conv (132→200) → softmax(200) →
  argmax + ±9bin 重み付き平均 → pitch_hz / confidence
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

VENV_MODEL = Path(__file__).resolve().parents[1] / ".venv/Lib/site-packages/swift_f0/model.onnx"
OUT_DIR = Path.home() / ".cache/rcwx/models/rust"
GOLDEN_DIR = Path(__file__).resolve().parents[1] / "target/golden"


def save_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        n = arr.nbytes
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [offset, offset + n]}
        blobs.append(arr.tobytes())
        offset += n
    import json

    header_bytes = json.dumps(header).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(blobs))


def main() -> None:
    m = onnx.load(str(VENV_MODEL))
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}

    out: dict[str, np.ndarray] = {}
    out["window"] = inits["onnx::STFT_171"].astype(np.float32)          # [1024]
    out["centers"] = inits["pitch_bin_centers"].astype(np.float32)       # [200]
    conv_map = {
        0: ("onnx::Conv_152", "onnx::Conv_153"),
        1: ("onnx::Conv_155", "onnx::Conv_156"),
        2: ("onnx::Conv_158", "onnx::Conv_159"),
        3: ("onnx::Conv_161", "onnx::Conv_162"),
        4: ("onnx::Conv_164", "onnx::Conv_165"),
    }
    for i, (w, b) in conv_map.items():
        out[f"conv{i}.weight"] = inits[w].astype(np.float32)  # [cout, cin, 5, 5]
        out[f"conv{i}.bias"] = inits[b].astype(np.float32)
    out["proj.weight"] = inits["freq_projection.weight"].astype(np.float32).reshape(200, 132)
    out["proj.bias"] = inits["freq_projection.bias"].astype(np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_safetensors(OUT_DIR / "swiftf0.safetensors", out)
    digest = hashlib.sha256((OUT_DIR / "swiftf0.safetensors").read_bytes()).hexdigest()[:12]
    print(f"saved {OUT_DIR / 'swiftf0.safetensors'} ({digest})")

    # ---- golden: onnxruntime 生出力 + RCWX ラッパー最終出力 ----
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(VENV_MODEL), opts, providers=["CPUExecutionProvider"])

    rng = np.random.RandomState(0)
    sr = 16000
    n = sr  # 1s
    t = np.arange(n) / sr
    seg = np.zeros(n, dtype=np.float32)
    # 0-0.3s: 220Hz、0.3-0.6s: チャープ 150→300Hz、0.6-0.8s: 無音、0.8-1.0s: 440Hz+ノイズ
    seg[: int(0.3 * sr)] = 0.5 * np.sin(2 * np.pi * 220.0 * t[: int(0.3 * sr)])
    c = slice(int(0.3 * sr), int(0.6 * sr))
    f_chirp = 150.0 + 150.0 * (t[c] - 0.3) / 0.3
    seg[c] = 0.5 * np.sin(2 * np.pi * np.cumsum(f_chirp) / sr)
    c = slice(int(0.8 * sr), int(1.0 * sr))
    seg[c] = 0.4 * np.sin(2 * np.pi * 440.0 * t[c]) + 0.02 * rng.randn(int(0.2 * sr)).astype(np.float32)

    pitch_raw, conf_raw = sess.run(None, {"input_audio": seg[None, :]})
    pitch_raw = pitch_raw[0]
    conf_raw = conf_raw[0]

    # RCWX ラッパー最終出力 (hop 256→160 + オクターブ解決)
    import torch  # noqa: F401 (swiftf0 wrapper が torch を要求)
    from rcwx.models.swiftf0 import SwiftF0

    det = SwiftF0(hop_length=160, confidence_threshold=0.35)
    with torch.no_grad():
        f0_final = det.infer(torch.from_numpy(seg), threshold=0.35, f0_min=50.0, f0_max=1100.0)
    f0_final = f0_final.numpy()[0]

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        GOLDEN_DIR / "swiftf0_golden.npz",
        audio=seg,
        pitch_raw=pitch_raw,
        conf_raw=conf_raw,
        f0_final=f0_final,
    )
    voiced = f0_final > 0
    print(f"golden: swift frames={len(pitch_raw)} target frames={len(f0_final)} voiced={voiced.sum()}")
    print(f"  f0_final voiced mean={f0_final[voiced].mean():.1f}Hz" if voiced.any() else "  no voiced")


if __name__ == "__main__":
    main()
