import numpy as np
from pipeline.config import RecognitionPipelineConfig
from pipeline.spectrogram import SpectrogramGenerator


def test_output_shape():
    cfg = RecognitionPipelineConfig()
    sg = SpectrogramGenerator(cfg)
    audio = np.random.randn(cfg.sample_rate * 5).astype(np.float32)
    spec = sg.generate(audio, cfg.sample_rate)
    assert spec.shape[0] == cfg.n_mels
    assert spec.dtype == np.float32


def test_deterministic():
    cfg = RecognitionPipelineConfig()
    sg = SpectrogramGenerator(cfg)
    audio = np.random.randn(cfg.sample_rate * 5).astype(np.float32)
    spec1 = sg.generate(audio, cfg.sample_rate)
    spec2 = sg.generate(audio, cfg.sample_rate)
    assert np.array_equal(spec1, spec2)
