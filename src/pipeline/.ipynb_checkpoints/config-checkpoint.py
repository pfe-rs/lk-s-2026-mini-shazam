from dataclasses import dataclass, field
import os

@dataclass
class RecognitionPipelineConfig:
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 8000.0
    chunk_seconds: float = 5.0
    overlap_seconds: float = 2.5
    embedding_dim: int = 128
    ffmpeg_bin: str = ""
    db_search_k: int = 5

@dataclass
class Datapaths:
    ROOT_DIR: str = "/home/jovyan/"
    SPECT_DIR: str = field(init=False)
    NOISE_DIR: str = field(init=False)
    MODEL_DIR: str = field(init=False)

    def __post_init__(self):
        # This runs automatically right after you do paths = Datapaths()
        self.SPECT_DIR = os.path.join(self.ROOT_DIR, "spectrograms-20260626T174438Z-3-001/spectrograms")
        self.NOISE_DIR = os.path.join(self.ROOT_DIR, "noiseMeowSpectrograms")
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, 'models')
        
        os.makedirs(self.MODEL_DIR, exist_ok=True)
