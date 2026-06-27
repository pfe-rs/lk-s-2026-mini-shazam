import numpy as np
from scipy.ndimage import maximum_filter


class AbstractFingerprinter:
    def fingerprint(self, spectrogram):
        raise NotImplementedError

    def merge_chunk_scores(self, existing: float, new: float) -> float:
        raise NotImplementedError

    def sort_key(self, song_id: str, score: float) -> tuple:
        raise NotImplementedError

    def confidence(self, top_k: list[tuple[str, float]]) -> float:
        raise NotImplementedError


class CNNFingerprinter(AbstractFingerprinter):
    from src.training_CNN import augmenter, dataset, trainer
    def fingerprint(self, spectrogram: np.ndarray) -> np.ndarray:
        raise NotImplementedError("CNNFingerprinter not yet implemented")

    @staticmethod
    def quantize(embedding: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(embedding)
        if n == 0:
            return np.zeros_like(embedding, dtype=np.int8)
        norm = embedding / n
        return np.round(norm * 127).clip(-127, 127).astype(np.int8)

    def merge_chunk_scores(self, existing: float, new: float) -> float:
        return min(existing, new)

    def sort_key(self, song_id: str, score: float) -> tuple:
        return (score, song_id)

    def confidence(self, top_k: list[tuple[str, float]]) -> float:
        if not top_k:
            return 0.0
        if len(top_k) == 1:
            return 1.0
        best, second = top_k[0][1], top_k[1][1]
        return 1.0 if second == 0 else max(0.0, 1.0 - (best / second))


class MathFingerprinter(AbstractFingerprinter):
    PEAK_NEIGHBORHOOD = 10
    FANOUT = 5
    TARGET_DT_BINS = (1, 50)
    TARGET_FREQ_RANGE = 64
    AMP_THRESHOLD_DB = -40

    def fingerprint(self, spectrogram: np.ndarray) -> list[tuple[int, float]]:
        landmarks = self._find_landmarks(spectrogram)
        return self._pair_landmarks(landmarks)

    def _find_landmarks(self, spec: np.ndarray) -> list[tuple[int, int]]:
        local_max = maximum_filter(spec, size=self.PEAK_NEIGHBORHOOD) == spec
        above_thresh = spec > self.AMP_THRESHOLD_DB
        freq_bins, time_bins = np.where(local_max & above_thresh)
        return sorted(zip(time_bins.tolist(), freq_bins.tolist()))

    def _pair_landmarks(self, landmarks: list) -> list[tuple[int, float]]:
        hashes = []
        for i, (t1, f1) in enumerate(landmarks):
            paired = 0
            for t2, f2 in landmarks[i + 1:]:
                dt = t2 - t1
                if dt < self.TARGET_DT_BINS[0]:
                    continue
                if dt > self.TARGET_DT_BINS[1]:
                    break
                if abs(f2 - f1) > self.TARGET_FREQ_RANGE:
                    continue
                h = self._hash(f1, f2, dt)
                hashes.append((h, float(t1)))
                paired += 1
                if paired >= self.FANOUT:
                    break
        return hashes

    @staticmethod
    def _hash(f1: int, f2: int, dt: int) -> int:
        return (f1 & 0x3FF) | ((f2 & 0x3FF) << 10) | ((dt & 0x3FF) << 20)

    def merge_chunk_scores(self, existing: float, new: float) -> float:
        return existing + new

    def sort_key(self, song_id: str, score: float) -> tuple:
        return (-score, song_id)

    def confidence(self, top_k: list[tuple[str, float]]) -> float:
        if not top_k:
            return 0.0
        if len(top_k) == 1:
            return 1.0
        best, second = top_k[0][1], top_k[1][1]
        if best == 0:
            return 0.0
        return max(0.0, 1.0 - (second / best))
