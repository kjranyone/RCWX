"""Exactness tests for the all-valid streaming TextEncoder fast path."""

from __future__ import annotations

import torch

from rcwx.models.infer_pack.models import TextEncoder


def test_all_frames_valid_matches_standard_mask_path() -> None:
    torch.manual_seed(7)
    encoder = TextEncoder(
        in_channels=6,
        out_channels=4,
        hidden_channels=8,
        filter_channels=16,
        n_heads=2,
        n_layers=2,
        kernel_size=3,
        p_dropout=0.0,
        f0=True,
    ).eval()
    phone = torch.randn(1, 12, 6)
    lengths = torch.tensor([12], dtype=torch.long)
    pitch = torch.randint(1, 255, (1, 12), dtype=torch.long)

    standard = encoder(phone, lengths, pitch, all_frames_valid=False)
    fast = encoder(phone, lengths, pitch, all_frames_valid=True)

    for expected, actual in zip(standard, fast):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(fast[3]) == fast[3].numel()
