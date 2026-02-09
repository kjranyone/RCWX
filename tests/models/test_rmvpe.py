"""Tests for RMVPE model architecture."""

import pytest
import torch


def test_rmvpe_architecture_matches_checkpoint():
    """Verify model architecture matches checkpoint structure."""
    from rcwx.models.rmvpe import E2E

    # Create model with same params as production
    model = E2E(
        n_blocks=4,
        n_gru=2,
        kernel_size=(2, 2),
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    )

    # Load checkpoint
    ckpt_path = r"C:\Users\kojiro\.cache\rcwx\models\rmvpe\rmvpe.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        pytest.skip("RMVPE checkpoint not found")

    model_state = model.state_dict()

    # Check all checkpoint keys exist in model
    missing_keys = []
    shape_mismatches = []

    for key, ckpt_tensor in ckpt.items():
        if key not in model_state:
            missing_keys.append(key)
        elif model_state[key].shape != ckpt_tensor.shape:
            shape_mismatches.append(
                f"{key}: model={model_state[key].shape}, ckpt={ckpt_tensor.shape}"
            )

    # Report all issues at once
    errors = []
    if missing_keys:
        errors.append(f"Missing keys in model:\n  " + "\n  ".join(missing_keys[:10]))
    if shape_mismatches:
        errors.append(f"Shape mismatches:\n  " + "\n  ".join(shape_mismatches))

    assert not errors, "\n".join(errors)


def test_rmvpe_forward_shapes():
    """Test that forward pass produces correct shapes."""
    from rcwx.models.rmvpe import E2E

    model = E2E(
        n_blocks=4,
        n_gru=2,
        kernel_size=(2, 2),
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    )
    model.eval()

    # Input: [batch, channels, mel_bands, time_frames]
    batch_size = 2
    mel_bands = 128
    time_frames = 100

    x = torch.randn(batch_size, 1, mel_bands, time_frames)

    with torch.no_grad():
        output = model(x)

    # Output should be [batch, time_frames, 360]
    assert output.shape == (batch_size, time_frames, 360), f"Got {output.shape}"


def test_rmvpe_encoder_shapes():
    """Test encoder layer output shapes."""
    from rcwx.models.rmvpe import DeepUnet

    unet = DeepUnet(
        kernel_size=(2, 2),
        n_blocks=4,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    )

    x = torch.randn(1, 1, 128, 64)  # [B, C, H, W]

    # Track encoder outputs
    skips = []
    for layer in unet.encoder.layers:
        x, skip = layer(x)
        skips.append(skip.shape)

    # Expected skip shapes: 16, 32, 64, 128, 256 channels
    expected_channels = [16, 32, 64, 128, 256]
    for i, (skip_shape, expected_ch) in enumerate(zip(skips, expected_channels)):
        assert skip_shape[1] == expected_ch, f"Layer {i}: expected {expected_ch} ch, got {skip_shape[1]}"


def test_rmvpe_inference():
    """Test full RMVPE inference pipeline."""
    from rcwx.models.rmvpe import RMVPE

    ckpt_path = r"C:\Users\kojiro\.cache\rcwx\models\rmvpe\rmvpe.pt"
    try:
        rmvpe = RMVPE(ckpt_path, device="cpu", dtype=torch.float32)
    except FileNotFoundError:
        pytest.skip("RMVPE checkpoint not found")

    # 1 second of audio at 16kHz
    audio = torch.randn(16000)

    with torch.no_grad():
        f0 = rmvpe.infer(audio)

    # Should produce ~100 F0 values (16000 / 160 hop_length)
    assert f0.shape[0] == 1  # batch
    assert 95 <= f0.shape[1] <= 105  # approximately 100 frames


if __name__ == "__main__":
    # Quick manual test
    print("Testing architecture match...")
    test_rmvpe_architecture_matches_checkpoint()
    print("OK")

    print("Testing forward shapes...")
    test_rmvpe_forward_shapes()
    print("OK")

    print("Testing encoder shapes...")
    test_rmvpe_encoder_shapes()
    print("OK")

    print("Testing inference...")
    test_rmvpe_inference()
    print("OK")

    print("\nAll tests passed!")
