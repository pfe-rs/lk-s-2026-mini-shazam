import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

class AbstractNormalizer:
    def normalize(self, chunk: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError

def _resolve_ffmpeg(config) -> str:
    if config.ffmpeg_bin:
        return config.ffmpeg_bin
    tools_dir = Path(__file__).parent.parent / "tools"
    exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    candidate = tools_dir / exe
    if candidate.exists() and os.access(str(candidate), os.X_OK):
        return str(candidate)
    try:
        import imageio_ffmpeg
        fallback = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"WARNING: using imageio_ffmpeg fallback {fallback}", file=sys.stderr)
        return fallback
    except Exception:
        raise RuntimeError(
            "FFmpeg binary not found: set config.ffmpeg_bin, "
            "place binary in tools/, or install imageio-ffmpeg"
        )

class FFmpegNormalizer(AbstractNormalizer):
    def __init__(self, config):
        self.ffmpeg_bin = _resolve_ffmpeg(config)

    def normalize(self, chunk: np.ndarray, sr: int) -> np.ndarray:
        tmp_in = None
        tmp_out = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
                tmp_in = f_in.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                tmp_out = f_out.name

            sf.write(tmp_in, chunk, sr, subtype="PCM_16")

            measured = self._run_pass1(tmp_in)
            self._run_pass2(tmp_in, tmp_out, sr, measured)

            data, _ = sf.read(tmp_out, dtype="float32")
            return data
        finally:
            for p in (tmp_in, tmp_out):
                if p and os.path.exists(p):
                    os.unlink(p)

    def _run_pass1(self, path: str) -> dict:
        cmd = [
            self.ffmpeg_bin, "-y", "-i", path,
            "-af", "loudnorm=I=-23:TP=-1:LRA=11:print_format=json",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg pass1 failed:\n{result.stderr}")
        stderr = result.stderr
        idx = stderr.rfind("{")
        if idx == -1:
            raise RuntimeError(f"no JSON in pass1 output:\n{stderr}")
        try:
            return json.loads(stderr[idx:])
        except json.JSONDecodeError as e:
            raise RuntimeError(f"pass1 JSON parse error: {e}\n{stderr}")

    def _run_pass2(self, path_in: str, path_out: str, sr: int, measured: dict) -> None:
        cmd = [
            self.ffmpeg_bin, "-y", "-i", path_in,
            "-af",
            f"loudnorm=I=-23:TP=-1:LRA=11"
            f":measured_I={measured['input_i']}"
            f":measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}"
            f":measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}"
            ":linear=true",
            "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16",
            path_out,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg pass2 failed:\n{result.stderr}")
