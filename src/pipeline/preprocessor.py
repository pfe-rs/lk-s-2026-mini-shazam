import numpy as np

class Preprocessor:
    def __init__(self, config):
        self.chunk_samples = int(config.sample_rate * config.chunk_seconds)
        self.hop_samples = int(config.sample_rate * (config.chunk_seconds - config.overlap_seconds))
        if self.hop_samples <= 0:
            raise ValueError(
                "overlap_seconds must be less than chunk_seconds "
                f"(got {config.chunk_seconds=}, {config.overlap_seconds=})"
            )

    def to_chunks(self, audio: np.ndarray, sr: int) -> list[np.ndarray]:
        if len(audio) == 0:
            raise ValueError("audio is empty")

        chunks = []
        start = 0
        while start + self.chunk_samples <= len(audio):
            chunk = audio[start:start + self.chunk_samples]
            chunks.append(chunk)
            start += self.hop_samples
        if not chunks:
            padded = np.zeros(self.chunk_samples, dtype=np.float32)
            padded[:len(audio)] = audio
            chunks.append(padded)
        return chunks
