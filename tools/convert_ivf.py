"""FAISS IVF index → safetensors 変換 + golden ダンプ (dev ツール)。

入力: model/kurumi/kurumi.index (IndexIVFFlat, L2, nprobe=1)
出力: ~/.cache/rcwx/models/rust/ivf.safetensors
      target/golden/ivf_golden.npz  (検索クエリと重み付き平均出力)
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "model/kurumi/kurumi.index"
OUT = Path.home() / ".cache/rcwx/models/rust/ivf.safetensors"
GOLDEN = ROOT / "target/golden/ivf_golden.npz"


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
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def main() -> None:
    import faiss

    index = faiss.read_index(str(INDEX))
    assert int(getattr(index, "metric_type", -1)) == int(faiss.METRIC_L2)
    nlist = int(index.nlist)
    d = int(index.d)
    features = index.reconstruct_n(0, index.ntotal)  # [ntotal, d] float32

    sizes = np.array([index.invlists.list_size(i) for i in range(nlist)], np.int64)
    largest = int(sizes.max())
    assert largest <= 256, f"list size {largest} > 256 cap"
    centroids = index.quantizer.reconstruct_n(0, nlist).astype(np.float32)

    # 各リストを固定長 256 にパディング (候補 ID は範囲外 = 無効を ntotal で表す)
    cap = 256
    packed = np.full((nlist * cap,), index.ntotal, dtype=np.int64)
    starts = np.empty(nlist, np.int64)
    for lid in range(nlist):
        n = int(sizes[lid])
        starts[lid] = lid * cap
        if n:
            ids = faiss.rev_swig_ptr(index.invlists.get_ids(lid), n)
            packed[lid * cap : lid * cap + n] = ids[:n]
    # features は ID 順テーブルをそのまま保存 (各ベクトルは必ずどこかのリストに 1 回現れる)
    packed_features = features

    save_safetensors(
        OUT,
        {
            "centroids": centroids,  # [nlist, d]
            "packed_features": packed_features,  # [ntotal, d] (ID = 行番号)
        },
    )
    # int64 スロットは safetensors に f32 として保存できないため別途 raw で
    OUT.with_suffix(".slots.bin").write_bytes(packed.reshape(nlist, cap).tobytes())
    meta = {"nlist": nlist, "d": d, "cap": cap, "ntotal": int(index.ntotal), "dim768": d == 768}
    OUT.with_suffix(".json").write_text(json.dumps(meta))
    print(f"saved {OUT} (lists={nlist} vectors={index.ntotal} largest={largest} d={d})")

    # ---- golden: 特徴量クエリ → FAISS 検索 → 重み付き平均ブレンド ----
    rng = np.random.RandomState(3)
    # 実際の HuBERT 特徴に近い統計でダミー生成
    feats = rng.randn(2, 20, d).astype(np.float32) * 2.0
    k = 4
    score, ix = index.search(feats.reshape(-1, d), k=k)
    weight = np.square(1.0 / (score + 1e-6))
    weight /= weight.sum(axis=1, keepdims=True)
    retrieved = np.sum(features[ix] * weight[:, :, None], axis=1).reshape(2, 20, d)
    blended = 0.15 * retrieved + 0.85 * feats  # index_rate=0.15 既定

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    np.savez(GOLDEN, query=feats, blended=blended, retrieved=retrieved)
    print(f"golden: query {feats.shape} → blended {blended.shape}")


if __name__ == "__main__":
    main()
