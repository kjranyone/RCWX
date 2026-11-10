"""M-T2 golden 2: enc_q + flow + decoder (GeneratorNSF) の fwd+bwd (Issue #8)。

小さめのランダム重みでリポジトリ参照実装 (rcwx/models/infer_pack) を動かし、
重み・入力・出力・勾配を safetensors に保存。Rust 側
(crates/rcwx-train/tests/module_parity2.rs) と突き合わせる。

使い方: uv run python tools/train_ref2.py
出力: ~/.cache/rcwx/train/golden/generator_golden.safetensors
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rcwx.models.infer_pack import models as im  # noqa: E402

B, T = 2, 11
SPEC_BINS = 16
INTER = 6
HIDDEN = 8
GIN = 4
SR = 48000
UPP = 8  # upsample_rates=[4,2] の積
SEED = 4321


def build():
    torch.manual_seed(SEED)
    enc_q = im.PosteriorEncoder(
        SPEC_BINS, INTER, HIDDEN, kernel_size=5, dilation_rate=1, n_layers=2,
        gin_channels=GIN,
    )
    flow = im.ResidualCouplingBlock(
        INTER, HIDDEN, 5, 1, 2, n_flows=2, gin_channels=GIN,
    )
    dec = im.GeneratorNSF(
        initial_channel=INTER,
        resblock="1",
        resblock_kernel_sizes=[3, 7],
        resblock_dilation_sizes=[[1, 3], [1, 3]],
        upsample_rates=[4, 2],
        upsample_initial_channel=16,
        upsample_kernel_sizes=[8, 4],
        gin_channels=GIN,
        sr=SR,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_q = enc_q
            self.flow = flow
            self.dec = dec

    return Model()


def main() -> None:
    out_dir = os.path.join(os.path.expanduser("~"), ".cache", "rcwx", "train", "golden")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "generator_golden.safetensors")

    model = build()
    g = torch.Generator().manual_seed(SEED + 1)
    spec = (torch.randn(B, SPEC_BINS, T, generator=g) * 0.5).float()
    lengths = torch.tensor([9, T])
    gf = torch.randn(B, GIN, 1, generator=g)  # g [B, gin, 1]
    randn = torch.randn(B, INTER, T, generator=g)
    f0 = torch.empty(B, T).uniform_(120.0, 400.0, generator=g)

    # har (励振) を固定シードで先に計算し、m_source を差し替え
    torch.manual_seed(SEED + 2)
    with torch.no_grad():
        har, _, _ = model.dec.m_source(f0, UPP)  # [B, T*upp, 1]
    har_t = har.transpose(1, 2)  # [B, 1, T*upp]

    class FixedSource(torch.nn.Module):
        def forward(self, x, upp=1):
            return har, None, None

    model.dec.m_source = FixedSource()

    x_mask = torch.unsqueeze(
        torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1), 1
    ).float()
    dbg = {}
    with torch.no_grad():
        x1 = model.enc_q.pre(spec) * x_mask
        dbg["pre"] = x1
        x2 = model.enc_q.enc(x1, x_mask, g=gf)
        dbg["wn"] = x2
        x3 = model.enc_q.proj(x2) * x_mask
        dbg["proj"] = x3
        cond = model.enc_q.enc.cond_layer(gf)
        dbg["cond"] = cond
    z, m_q, logs_q, _ = model.enc_q(spec, lengths, g=gf)
    z = (m_q + randn * torch.exp(logs_q)) * x_mask
    with torch.no_grad():
        fx = z
        for fi, fl in enumerate(model.flow.flows):
            fx, _ = fl(fx, x_mask, g=gf, reverse=False)
            dbg[f"flow{fi}"] = fx
    z_p = model.flow(z, x_mask, g=gf, reverse=False)
    with torch.no_grad():
        d = model.dec
        dx = d.conv_pre(z)
        dbg["d_pre"] = dx
        dx = dx + d.cond(gf)
        dbg["d_cond"] = dx
        for di in range(d.num_upsamples):
            dx = torch.nn.functional.leaky_relu(dx, 0.1)
            dx = d.ups[di](dx)
            dbg[f"d_up{di}"] = dx
            xsrc = d.noise_convs[di](har_t)
            dx = dx + xsrc
            dbg[f"d_noise{di}"] = dx
            xs = None
            for dj in range(d.num_kernels):
                rb = d.resblocks[di * d.num_kernels + dj](dx)
                xs = rb if xs is None else xs + rb
            dx = xs / d.num_kernels
            dbg[f"d_rb{di}"] = dx
            dbg["d_lrelu_final"] = torch.nn.functional.leaky_relu(dx, 0.1)
    y_hat = model.dec(z, f0, g=gf)

    loss = y_hat.sum() + z_p.sum() + m_q.sum() + logs_q.sum()
    loss.backward()

    def c(t: torch.Tensor) -> torch.Tensor:
        return t.detach().clone().contiguous()

    out: dict[str, torch.Tensor] = {}
    for k, v in dbg.items():
        out[f"dbg.{k}"] = c(v)
    out["in.spec"] = c(spec)
    out["in.mask"] = c(x_mask.squeeze(1))
    out["in.g"] = c(gf).reshape(B, GIN)
    out["in.randn"] = c(randn)
    out["in.f0"] = c(f0)
    out["in.har"] = c(har_t).reshape(-1)
    out["out.z"] = c(z)
    out["out.z_p"] = c(z_p)
    out["out.m_q"] = c(m_q)
    out["out.logs_q"] = c(logs_q)
    out["out.y_hat"] = c(y_hat)
    out["out.loss"] = torch.tensor([loss.item()])
    for name, p in model.named_parameters():
        out[f"w.{name}"] = c(p)
        out[f"g.{name}"] = c(p.grad if p.grad is not None else torch.zeros_like(p))

    from safetensors.torch import save_file

    save_file(out, out_path)
    print(f"saved {out_path}: {len(out)} tensors, loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
