from dataclasses import dataclass

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
