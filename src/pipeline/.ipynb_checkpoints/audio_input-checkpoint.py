import numpy as np

import librosa


class AbstractAudioInput:
    def load(self, path: str, target_sr: int | None = None) -> np.ndarray:
        raise NotImplementedError


class FileInput(AbstractAudioInput):
    def load(self, path: str, target_sr: int | None = None) -> np.ndarray:
        data, sr = librosa.load(path, sr=None, mono=True)

        if target_sr is not None and sr != target_sr:
            # sve mora da bude u 22khz
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

        data = data.astype(np.float32)
        if len(data) == 0:
            raise ValueError(f"loaded zero-length audio from {path}")

        return data


class MicrophoneInput(AbstractAudioInput):
    # TODO: implementirati podrsku za mikrofon
    def load(self, path: str, target_sr: int | None = None) -> np.ndarray:
        raise NotImplementedError("MicrophoneInput not yet implemented")
