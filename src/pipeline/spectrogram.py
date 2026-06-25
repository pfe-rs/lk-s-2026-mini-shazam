import librosa
import numpy as np

class SpectrogramGenerator:
    def __init__(self, config):
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.n_mels = config.n_mels
        self.fmin = config.fmin
        self.fmax = config.fmax

    def generate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            window="hann",
            power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db.astype(np.float32)
