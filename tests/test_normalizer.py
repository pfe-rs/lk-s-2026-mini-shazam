import os
from pathlib import Path

from pipeline.config import RecognitionPipelineConfig
from pipeline.normalizer import FFmpegNormalizer


def test_ffmpeg_binary_present():
    tools_dir = Path(__file__).parent.parent / "src" / "tools"
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    binary = tools_dir / exe
    assert binary.exists(), f"FFmpeg not found at {binary}"
    assert os.access(str(binary), os.X_OK), f"FFmpeg at {binary} is not executable"


def test_ffmpeg_normalizer_resolves_binary():
    cfg = RecognitionPipelineConfig()
    normalizer = FFmpegNormalizer(cfg)
    assert normalizer.ffmpeg_bin
    result = os.popen(f"{normalizer.ffmpeg_bin} -version 2>&1 | head -1").read().strip()
    assert "ffmpeg version" in result, f"Unexpected output: {result}"
