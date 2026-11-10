"""M-T2 golden: enc_p (TextEncoder) の fwd+bwd 勾配 golden 生成 (Issue #8)。

tiny 設定で PyTorch 参照 (rcwx/models/infer_pack) を動かし、
重み・入力・出力 (m_p/logs_p)・全パラメータ勾配を safetensors に保存する。
Rust 側 (crates/rcwx-train/tests/module_parity.rs) と突き合わせる。

使い方: uv run python tools/train_ref.py
出力: ~/.cache/rcwx/train/golden/encp_golden.safetensors
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rcwx.models.infer_pack import attentions, commons  # noqa: E402

B, T, IN, H, INTER, FILT, HEADS, LAYERS, K, WS = 2, 13, 8, 16, 6, 32, 2, 2, 3, 10
LENGTH = 10  # 全バッチ同一 (T=13 のうち末尾 3 フレーム無効)
SEED = 1234


def build() -> torch.nn.Module:
    torch.manual_seed(SEED)

    class EncP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb_phone = torch.nn.Linear(IN, H)
            self.emb_pitch = torch.nn.Embedding(256, H)
            self.encoder = attentions.Encoder(
                H, FILT, HEADS, LAYERS, K, 0.0, window_size=WS
            )
            self.proj = torch.nn.Conv1d(H, INTER * 2, 1)

        dbg = {}

        def forward(self, phone, pitch, lengths):
            x = self.emb_phone(phone)
            self.dbg["xp"] = x
            x = x + self.emb_pitch(pitch)
            self.dbg["xpq"] = x
            x = x * math.sqrt(H)
            x = torch.nn.functional.leaky_relu(x, 0.1)
            x = x.transpose(1, 2)
            x_mask = torch.unsqueeze(
                commons.sequence_mask(lengths, x.size(2)), 1
            ).to(x.dtype)
            x = x * x_mask
            x.retain_grad()
            self.dbg["x0"] = x
            attn0 = self.encoder.attn_layers[0]
            attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
            self.dbg["q0"] = attn0.conv_q(x)
            # 分解 attention の内部 (PyTorch と同一式を明示)
            b_, d_, t_s, t_t = (*x.size(), x.size(2))
            q = attn0.conv_q(x)
            k = attn0.conv_k(x)
            v = attn0.conv_v(x)
            qc = q / math.sqrt(attn0.k_channels)
            qh = qc.view(b_, attn0.n_heads, attn0.k_channels, t_t).transpose(2, 3)
            kh = k.view(b_, attn0.n_heads, attn0.k_channels, t_s).transpose(2, 3)
            vh = v.view(b_, attn0.n_heads, attn0.k_channels, t_s).transpose(2, 3)
            scores = torch.matmul(qh, kh.transpose(-2, -1))
            relk = attn0._get_relative_embeddings(attn0.emb_rel_k, t_s)
            # attention() は reshape 済み query [b, h, t, dk] で rel matmul する
            rel_logits = torch.matmul(qh, relk.unsqueeze(0).transpose(-2, -1))
            scores_local = attn0._relative_position_to_absolute_position(rel_logits)
            self.dbg["qk0"] = scores
            self.dbg["rel0"] = scores_local
            scores = scores + scores_local
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, -1e4)
            p = torch.softmax(scores, dim=-1)
            self.dbg["p0"] = p
            self.dbg["pv0"] = torch.matmul(p, vh)
            y = attn0(x, x, attn_mask)
            self.dbg["att0"] = y
            x2 = self.encoder.norm_layers_1[0](x + y)
            self.dbg["ln1_0"] = x2
            y = self.encoder.ffn_layers[0](x2, x_mask)
            self.dbg["ffn0"] = y
            x3 = self.encoder.norm_layers_2[0](x2 + y)
            self.dbg["enc0"] = x3
            x = self.encoder(x, x_mask)
            self.dbg["enc"] = x
            stats = self.proj(x)
            stats = stats * x_mask
            m, logs = torch.split(stats, INTER, dim=1)
            return m, logs, x_mask

    return EncP()


def main() -> None:
    out_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "rcwx", "train", "golden"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "encp_golden.safetensors")

    model = build()
    g = torch.Generator().manual_seed(SEED + 1)
    phone = (torch.randn(B, T, IN, generator=g) * 0.5).float().requires_grad_(True)
    pitch = torch.randint(1, 255, (B, T), generator=g)
    lengths = torch.tensor([LENGTH, LENGTH])

    m_p, logs_p, _mask = model(phone, pitch, lengths)

    loss = m_p.sum() + logs_p.sum()
    loss.backward()

    out: dict[str, torch.Tensor] = {}
    def c(t: torch.Tensor) -> torch.Tensor:
        return t.detach().clone().contiguous()

    out["in.phone"] = c(phone)
    out["in.pitch"] = c(pitch).float()
    out["out.m_p"] = c(m_p)
    out["out.logs_p"] = c(logs_p)
    out["out.loss"] = torch.tensor([loss.item()])
    for k, v in model.dbg.items():
        out[f"dbg.{k}"] = c(v)
        out[f"dgrad.{k}"] = c(v.grad if v.grad is not None else torch.zeros_like(v))
    out["dgrad.phone"] = c(phone.grad if phone.grad is not None else torch.zeros_like(phone))
    for name, p in model.named_parameters():
        out[f"w.enc_p.{name}"] = c(p)
        out[f"g.enc_p.{name}"] = c(p.grad if p.grad is not None else torch.zeros_like(p))

    from safetensors.torch import save_file

    save_file(out, out_path)
    print(f"saved {out_path}: {len(out)} tensors, loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
