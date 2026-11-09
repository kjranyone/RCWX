"""f0_processing 参照ダンプ (M3-b パリティ用)。"""
import numpy as np
import torch
from safetensors.torch import save_file
import sys
sys.path.insert(0, ".")
from rcwx.pipeline.f0_processing import apply_f0_filter_chain, quantize_f0_to_pitch

rng = np.random.default_rng(7)
T = 100
cases = {}
# steady: 持続母音 + 小ノイズ
f = 180.0 + 6.0 * np.sin(np.arange(T) * 0.3) + rng.standard_normal(T) * 1.5
cases["steady"] = f.astype(np.float32)
# octave flip: 途中で一時的に 1 オクターブ上
f = np.full(T, 200.0, dtype=np.float32)
f[30:34] = 400.0
f[33] = 405.0
cases["octave_flip"] = f
# slew: 急激なジャンプ
f = np.full(T, 150.0, dtype=np.float32)
f[50:56] = np.linspace(150, 320, 6)
cases["slew"] = f
# holes: 短い無声穴
f = np.full(T, 220.0, dtype=np.float32)
f[20] = 0.0
f[21] = 0.0
f[60:64] = 0.0  # 4fr > 3 → 埋めない
cases["holes"] = f
# mixed: 全部混ぜ + 無声区間
f = 170.0 + 40.0 * np.sin(np.arange(T) * 0.08) + rng.standard_normal(T) * 3.0
f = f.astype(np.float32)
f[:10] = 0.0
f[70:75] = 0.0
f[40] = 0.0
f[41] = 0.0
f[55:58] *= 2.0  # 一時オクターブ上
cases["mixed"] = f

out = {}
for name, f in cases.items():
    t = torch.from_numpy(f).unsqueeze(0)
    out[f"{name}_in"] = t.clone()
    y = apply_f0_filter_chain(
        t.clone(),
        f0_lowpass_cutoff_hz=16.0,
        enable_octave_flip_suppress=True,
        enable_f0_slew_limit=True,
        f0_slew_max_step_st=3.6,
        f0_hole_fill_ms=30.0,
    )
    out[f"{name}_out"] = y
    print(name, "voiced:", int((y[0] > 0).sum()))

# quantize
f0 = cases["mixed"]
q, _ = quantize_f0_to_pitch(torch.from_numpy(f0).unsqueeze(0))
out["quant_in"] = torch.from_numpy(f0).unsqueeze(0)
out["quant_out"] = q.float()
save_file(out, "target/golden/f0_ref.safetensors")
print("saved")
