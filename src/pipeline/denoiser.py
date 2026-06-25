import numpy as np

class AbstractDenoiser:
    def process(self, spectrogram: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class NoDenoiser(AbstractDenoiser):
    def process(self, spectrogram: np.ndarray) -> np.ndarray:
        return spectrogram

class UNetDenoiser(AbstractDenoiser):
    def process(self, spectrogram: np.ndarray) -> np.ndarray:
        raise NotImplementedError("UNetDenoiser not yet implemented")
