"""Unit tests for chunking strategies."""

import numpy as np
import pytest

from rcwx.audio.chunking import (
    ChunkConfig,
    ChunkResult,
    ChunkingStrategy,
    HybridChunkingStrategy,
    RVCWebUIChunkingStrategy,
    WokadaChunkingStrategy,
    create_chunking_strategy,
)


class TestChunkConfig:
    """Tests for ChunkConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ChunkConfig()
        assert config.mic_sample_rate == 48000
        assert config.chunk_sec == 0.10
        assert config.context_sec == 0.10
        assert config.lookahead_sec == 0.0
        assert config.rvc_overlap_sec == 0.22

    def test_chunk_samples_calculation(self):
        """Test chunk_samples property."""
        config = ChunkConfig(mic_sample_rate=48000, chunk_sec=0.1)
        assert config.chunk_samples == 4800

    def test_context_samples_calculation(self):
        """Test context_samples property."""
        config = ChunkConfig(mic_sample_rate=48000, context_sec=0.05)
        assert config.context_samples == 2400

    def test_validation_negative_chunk_sec(self):
        """Test validation rejects negative chunk_sec."""
        with pytest.raises(ValueError, match="chunk_sec must be positive"):
            ChunkConfig(chunk_sec=-0.1)

    def test_validation_negative_context_sec(self):
        """Test validation rejects negative context_sec."""
        with pytest.raises(ValueError, match="context_sec must be non-negative"):
            ChunkConfig(context_sec=-0.1)


class TestWokadaChunkingStrategy:
    """Tests for WokadaChunkingStrategy."""

    def test_mode_name(self):
        """Test mode name is correct."""
        config = ChunkConfig()
        strategy = WokadaChunkingStrategy(config)
        assert strategy.mode_name == "wokada"

    def test_hop_equals_chunk(self):
        """Test hop equals chunk size (no overlap)."""
        config = ChunkConfig(mic_sample_rate=48000, chunk_sec=0.1)
        strategy = WokadaChunkingStrategy(config, align_to_hubert=False)
        assert strategy.hop_samples == strategy.chunk_samples

    def test_has_chunk_empty(self):
        """Test has_chunk returns False for empty buffer."""
        config = ChunkConfig()
        strategy = WokadaChunkingStrategy(config)
        assert strategy.has_chunk() is False

    def test_has_chunk_insufficient(self):
        """Test has_chunk returns False for insufficient samples."""
        config = ChunkConfig(mic_sample_rate=48000, chunk_sec=0.1)
        strategy = WokadaChunkingStrategy(config, align_to_hubert=False)

        # Add less than required
        audio = np.zeros(1000, dtype=np.float32)
        strategy.add_input(audio)
        assert strategy.has_chunk() is False

    def test_has_chunk_sufficient_first(self):
        """Test has_chunk returns True for sufficient samples (first chunk)."""
        config = ChunkConfig(
            mic_sample_rate=48000,
            chunk_sec=0.1,
            context_sec=0.05,
            lookahead_sec=0.0,
        )
        strategy = WokadaChunkingStrategy(config, align_to_hubert=False)

        # First chunk needs: chunk + lookahead = 4800 + 0 = 4800
        audio = np.zeros(5000, dtype=np.float32)
        strategy.add_input(audio)
        assert strategy.has_chunk() is True

    def test_get_chunk_first_has_reflection_padding(self):
        """Test first chunk includes reflection padding for context."""
        config = ChunkConfig(
            mic_sample_rate=48000,
            chunk_sec=0.1,
            context_sec=0.05,
            lookahead_sec=0.0,
        )
        strategy = WokadaChunkingStrategy(config, align_to_hubert=False)

        # Add enough samples
        audio = np.arange(6000, dtype=np.float32)
        strategy.add_input(audio)

        result = strategy.get_chunk()
        assert result is not None
        assert result.is_first_chunk is True
        # First chunk should be: context + main + lookahead
        expected_len = config.context_samples + config.chunk_samples + config.lookahead_samples
        assert len(result.chunk) == expected_len

    def test_get_chunk_subsequent(self):
        """Test subsequent chunks have correct size."""
        config = ChunkConfig(
            mic_sample_rate=48000,
            chunk_sec=0.1,
            context_sec=0.05,
            lookahead_sec=0.0,
        )
        strategy = WokadaChunkingStrategy(config, align_to_hubert=False)

        # Add enough for two chunks
        audio = np.arange(12000, dtype=np.float32)
        strategy.add_input(audio)

        # First chunk
        result1 = strategy.get_chunk()
        assert result1.is_first_chunk is True

        # Second chunk
        result2 = strategy.get_chunk()
        assert result2 is not None
        assert result2.is_first_chunk is False
        expected_len = config.context_samples + config.chunk_samples + config.lookahead_samples
        assert len(result2.chunk) == expected_len

    def test_clear_resets_state(self):
        """Test clear resets buffer and first chunk flag."""
        config = ChunkConfig()
        strategy = WokadaChunkingStrategy(config, align_to_hubert=False)

        # Add and process one chunk
        audio = np.zeros(10000, dtype=np.float32)
        strategy.add_input(audio)
        strategy.get_chunk()

        # Clear
        strategy.clear()

        # Should be back to initial state
        assert strategy.buffered_samples == 0
        assert strategy.has_chunk() is False


class TestRVCWebUIChunkingStrategy:
    """Tests for RVCWebUIChunkingStrategy."""

    def test_mode_name(self):
        """Test mode name is correct."""
        config = ChunkConfig()
        strategy = RVCWebUIChunkingStrategy(config)
        assert strategy.mode_name == "rvc_webui"

    def test_hop_less_than_chunk(self):
        """Test hop = chunk - overlap."""
        config = ChunkConfig(
            mic_sample_rate=48000,
            chunk_sec=0.1,  # 4800 samples
            rvc_overlap_sec=0.02,  # 960 samples
        )
        strategy = RVCWebUIChunkingStrategy(config, align_to_hubert=False)

        expected_hop = 4800 - 960  # 3840
        assert strategy.hop_samples == expected_hop

    def test_overlap_samples(self):
        """Test overlap samples property."""
        config = ChunkConfig(
            mic_sample_rate=48000,
            rvc_overlap_sec=0.02,
        )
        strategy = RVCWebUIChunkingStrategy(config)
        assert strategy.overlap_samples == 960

    def test_has_chunk_requires_chunk_samples(self):
        """Test has_chunk requires chunk_samples."""
        config = ChunkConfig(mic_sample_rate=48000, chunk_sec=0.1)
        strategy = RVCWebUIChunkingStrategy(config, align_to_hubert=False)

        # Less than chunk_samples
        audio = np.zeros(4000, dtype=np.float32)
        strategy.add_input(audio)
        assert strategy.has_chunk() is False

        # Exactly chunk_samples
        audio2 = np.zeros(800, dtype=np.float32)
        strategy.add_input(audio2)
        assert strategy.has_chunk() is True

    def test_get_chunk_returns_chunk_samples(self):
        """Test get_chunk returns exactly chunk_samples."""
        config = ChunkConfig(mic_sample_rate=48000, chunk_sec=0.1)
        strategy = RVCWebUIChunkingStrategy(config, align_to_hubert=False)

        audio = np.zeros(5000, dtype=np.float32)
        strategy.add_input(audio)

        result = strategy.get_chunk()
        assert result is not None
        assert len(result.chunk) == 4800  # chunk_samples

    def test_buffer_advances_by_hop(self):
        """Test buffer advances by hop_samples after get_chunk."""
        config = ChunkConfig(
            mic_sample_rate=48000,
            chunk_sec=0.1,  # 4800
            rvc_overlap_sec=0.02,  # 960
        )
        strategy = RVCWebUIChunkingStrategy(config, align_to_hubert=False)
        hop = 4800 - 960  # 3840

        audio = np.zeros(6000, dtype=np.float32)
        strategy.add_input(audio)

        initial_buffered = strategy.buffered_samples
        assert initial_buffered == 6000

        strategy.get_chunk()

        # Should have: 6000 - 3840 = 2160
        assert strategy.buffered_samples == 6000 - hop

    def test_invalid_overlap_raises(self):
        """Test overlap >= chunk raises ValueError."""
        config = ChunkConfig(
            mic_sample_rate=48000,
            chunk_sec=0.1,
            rvc_overlap_sec=0.1,  # Same as chunk
        )
        with pytest.raises(ValueError, match="Invalid overlap"):
            RVCWebUIChunkingStrategy(config, align_to_hubert=False)


class TestHybridChunkingStrategy:
    """Tests for HybridChunkingStrategy."""

    def test_mode_name(self):
        """Test mode name is correct."""
        config = ChunkConfig()
        strategy = HybridChunkingStrategy(config)
        assert strategy.mode_name == "hybrid"

    def test_hop_equals_chunk(self):
        """Test hop equals chunk size (like wokada)."""
        config = ChunkConfig(mic_sample_rate=48000, chunk_sec=0.1)
        strategy = HybridChunkingStrategy(config, align_to_hubert=False)
        assert strategy.hop_samples == strategy.chunk_samples

    def test_similar_to_wokada(self):
        """Test hybrid behaves similarly to wokada."""
        config = ChunkConfig(
            mic_sample_rate=48000,
            chunk_sec=0.1,
            context_sec=0.05,
        )
        wokada = WokadaChunkingStrategy(config, align_to_hubert=False)
        hybrid = HybridChunkingStrategy(config, align_to_hubert=False)

        # Same chunk and hop
        assert wokada.chunk_samples == hybrid.chunk_samples
        assert wokada.hop_samples == hybrid.hop_samples
        assert wokada.context_samples == hybrid.context_samples


class TestCreateChunkingStrategy:
    """Tests for create_chunking_strategy factory."""

    def test_create_wokada(self):
        """Test creating wokada strategy."""
        config = ChunkConfig()
        strategy = create_chunking_strategy("wokada", config)
        assert isinstance(strategy, WokadaChunkingStrategy)

    def test_create_rvc_webui(self):
        """Test creating rvc_webui strategy."""
        config = ChunkConfig()
        strategy = create_chunking_strategy("rvc_webui", config)
        assert isinstance(strategy, RVCWebUIChunkingStrategy)

    def test_create_hybrid(self):
        """Test creating hybrid strategy."""
        config = ChunkConfig()
        strategy = create_chunking_strategy("hybrid", config)
        assert isinstance(strategy, HybridChunkingStrategy)

    def test_case_insensitive(self):
        """Test mode name is case insensitive."""
        config = ChunkConfig()
        strategy = create_chunking_strategy("WOKADA", config)
        assert isinstance(strategy, WokadaChunkingStrategy)

    def test_invalid_mode_raises(self):
        """Test invalid mode raises ValueError."""
        config = ChunkConfig()
        with pytest.raises(ValueError, match="Unknown chunking mode"):
            create_chunking_strategy("invalid", config)


class TestChunkingStrategyProtocol:
    """Tests to verify all strategies implement the protocol correctly."""

    @pytest.fixture(params=["wokada", "rvc_webui", "hybrid"])
    def strategy(self, request):
        """Fixture to create each strategy type."""
        config = ChunkConfig(
            mic_sample_rate=48000,
            chunk_sec=0.1,
            context_sec=0.05,
            rvc_overlap_sec=0.02,
        )
        return create_chunking_strategy(request.param, config)

    def test_implements_protocol(self, strategy):
        """Test strategy implements ChunkingStrategy protocol."""
        assert isinstance(strategy, ChunkingStrategy)

    def test_has_required_properties(self, strategy):
        """Test strategy has all required properties."""
        assert hasattr(strategy, "mode_name")
        assert hasattr(strategy, "chunk_samples")
        assert hasattr(strategy, "hop_samples")
        assert hasattr(strategy, "buffered_samples")

    def test_has_required_methods(self, strategy):
        """Test strategy has all required methods."""
        assert callable(strategy.add_input)
        assert callable(strategy.has_chunk)
        assert callable(strategy.get_chunk)
        assert callable(strategy.clear)
        assert callable(strategy.get_expected_input_samples)

    def test_add_input_increases_buffer(self, strategy):
        """Test add_input increases buffered samples."""
        initial = strategy.buffered_samples
        audio = np.zeros(1000, dtype=np.float32)
        strategy.add_input(audio)
        assert strategy.buffered_samples == initial + 1000

    def test_clear_empties_buffer(self, strategy):
        """Test clear empties the buffer."""
        audio = np.zeros(1000, dtype=np.float32)
        strategy.add_input(audio)
        strategy.clear()
        assert strategy.buffered_samples == 0

    def test_get_chunk_returns_result_or_none(self, strategy):
        """Test get_chunk returns ChunkResult or None."""
        # Empty buffer should return None
        result = strategy.get_chunk()
        assert result is None

        # With enough data should return ChunkResult
        audio = np.zeros(20000, dtype=np.float32)
        strategy.add_input(audio)

        if strategy.has_chunk():
            result = strategy.get_chunk()
            assert isinstance(result, ChunkResult)


class TestHuBERTAlignment:
    """Tests for HuBERT hop alignment."""

    def test_wokada_aligns_by_default(self):
        """Test wokada aligns chunk to HuBERT hop."""
        config = ChunkConfig(
            mic_sample_rate=48000,
            chunk_sec=0.105,  # 5040 samples - not aligned
        )
        strategy = WokadaChunkingStrategy(config, align_to_hubert=True)

        # HuBERT hop at 48kHz = 960 samples
        # 5040 / 960 = 5.25, rounded up = 6 * 960 = 5760
        assert strategy.chunk_samples % 960 == 0

    def test_rvc_webui_no_align_by_default(self):
        """Test rvc_webui doesn't align by default."""
        config = ChunkConfig(
            mic_sample_rate=48000,
            chunk_sec=0.105,  # 5040 samples
        )
        strategy = RVCWebUIChunkingStrategy(config, align_to_hubert=False)

        # Should keep original size
        assert strategy.chunk_samples == 5040

    def test_can_disable_alignment(self):
        """Test alignment can be disabled."""
        config = ChunkConfig(
            mic_sample_rate=48000,
            chunk_sec=0.105,
        )
        strategy = WokadaChunkingStrategy(config, align_to_hubert=False)
        assert strategy.chunk_samples == 5040


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
