"""Tests for RVC inference pipeline."""

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Disable network requests

import torch
import numpy as np
from pathlib import Path


def test_feature_dimension_reduction():
    """Test 768 -> 256 feature reduction for v1 models."""
    features = torch.randn(1, 100, 768)
    reduced = features.reshape(features.shape[0], features.shape[1], 256, 3).mean(-1)
    assert reduced.shape == (1, 100, 256)


def test_pipeline_infer_v1_model():
    """Test full inference with v1 model."""
    model_path = Path(r"C:\lib\github\grand2-products\RCWX\model\元気系アニメボイス Kana\voice\voice.pth")
    if not model_path.exists():
        print(f"Skipping: model not found at {model_path}")
        return

    from rcwx.pipeline.inference import RVCPipeline

    pipeline = RVCPipeline(str(model_path), device="cpu", dtype=torch.float32)
    pipeline.load()

    # Check v1 model detected
    assert pipeline.synthesizer.version == 1, f"Expected v1, got {pipeline.synthesizer.version}"

    # Test with 1 second of audio
    audio = np.random.randn(16000).astype(np.float32) * 0.1

    # Run inference
    output = pipeline.infer(audio, input_sr=16000, pitch_shift=0, f0_method="rmvpe")

    print(f"Input: {audio.shape}")
    print(f"Output: {output.shape}")

    # Output should be at model sample rate (40kHz for 1 second = 40000 samples)
    assert output.shape[0] > 30000, f"Output too short: {output.shape}"
    print("OK")


def test_hubert_feature_shape():
    """Test HuBERT feature extraction shape."""
    hubert_path = Path(r"C:\Users\kojiro\.cache\rcwx\models\hubert\pytorch_model.bin")
    if not hubert_path.exists():
        print("Skipping: HuBERT not found")
        return

    from rcwx.models.hubert_loader import HuBERTLoader

    hubert = HuBERTLoader(str(hubert_path), device="cpu")

    audio = torch.randn(1, 16000)  # 1 second at 16kHz
    features = hubert.extract(audio)

    print(f"Audio: {audio.shape}")
    print(f"Features: {features.shape}")

    # Features should be [B, T, 768] where T ~= audio_samples / 320
    assert features.shape[0] == 1
    assert features.shape[2] == 768
    assert 40 <= features.shape[1] <= 60  # ~50 frames for 1 second
    print("OK")


def test_synthesizer_v1():
    """Test synthesizer with 256-dim features (v1 model)."""
    model_path = Path(r"C:\lib\github\grand2-products\RCWX\model\元気系アニメボイス Kana\voice\voice.pth")
    if not model_path.exists():
        print(f"Skipping: model not found")
        return

    from rcwx.models.synthesizer import SynthesizerLoader

    synth = SynthesizerLoader(str(model_path), device="cpu", use_compile=False)
    synth.load()

    assert synth.version == 1, f"Expected v1, got v{synth.version}"
    assert synth.has_f0 == True

    # Test with 256-dim features
    features = torch.randn(1, 50, 256)
    lengths = torch.tensor([50])
    pitch = torch.randint(0, 255, (1, 50))
    pitchf = torch.randn(1, 50)

    output = synth.infer(features, lengths, pitch=pitch, pitchf=pitchf)
    print(f"Features: {features.shape} -> Output: {output.shape}")
    assert output.shape[0] == 1
    assert output.shape[1] > 10000  # Should produce audio
    print("OK")


if __name__ == "__main__":
    print("Test 1: Feature reduction...")
    test_feature_dimension_reduction()
    print("OK\n")

    print("Test 2: Synthesizer v1...")
    test_synthesizer_v1()
    print()

    # Skip network-dependent tests
    # print("Test 3: HuBERT features...")
    # test_hubert_feature_shape()
    # print("Test 4: Full pipeline...")
    # test_pipeline_infer_v1_model()
