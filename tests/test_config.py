from pipeline.config import RecognitionPipelineConfig


def test_defaults():
    cfg = RecognitionPipelineConfig()
    assert cfg.sample_rate == 22050
    assert cfg.n_fft == 2048
    assert cfg.hop_length == 512
    assert cfg.n_mels == 128
    assert cfg.fmin == 0.0
    assert cfg.fmax == 8000.0
    assert cfg.chunk_seconds == 5.0
    assert cfg.overlap_seconds == 2.5
    assert cfg.embedding_dim == 128
    assert cfg.ffmpeg_bin == ""
    assert cfg.db_search_k == 5
