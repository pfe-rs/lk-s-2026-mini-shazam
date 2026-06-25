import numpy as np
from pipeline.config import RecognitionPipelineConfig
from pipeline.preprocessor import Preprocessor


def test_chunk_count():
    cfg = RecognitionPipelineConfig()
    pp = Preprocessor(cfg)
    audio = np.zeros(cfg.sample_rate * 12, dtype=np.float32)
    chunks = pp.to_chunks(audio, cfg.sample_rate)
    assert len(chunks) >= 2


def test_chunk_size():
    cfg = RecognitionPipelineConfig()
    pp = Preprocessor(cfg)
    audio = np.zeros(cfg.sample_rate * 12, dtype=np.float32)
    chunks = pp.to_chunks(audio, cfg.sample_rate)
    expected = int(cfg.sample_rate * cfg.chunk_seconds)
    for c in chunks:
        assert c.shape[0] == expected


def test_padding_short_audio():
    cfg = RecognitionPipelineConfig()
    pp = Preprocessor(cfg)
    audio = np.zeros(cfg.sample_rate * 2, dtype=np.float32)
    chunks = pp.to_chunks(audio, cfg.sample_rate)
    assert len(chunks) == 1
    expected = int(cfg.sample_rate * cfg.chunk_seconds)
    assert chunks[0].shape[0] == expected


def test_empty_audio_raises():
    cfg = RecognitionPipelineConfig()
    pp = Preprocessor(cfg)
    audio = np.array([], dtype=np.float32)
    try:
        pp.to_chunks(audio, cfg.sample_rate)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_degenerate_overlap_raises():
    cfg = RecognitionPipelineConfig()
    cfg.overlap_seconds = cfg.chunk_seconds
    try:
        Preprocessor(cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_overlap_exceeds_chunk_raises():
    cfg = RecognitionPipelineConfig()
    cfg.overlap_seconds = cfg.chunk_seconds + 1.0
    try:
        Preprocessor(cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass
