import json
import pickle
from pathlib import Path

import faiss
import numpy as np

class AbstractVectorDatabase:
    def add(self, embedding: np.ndarray, song_id: str, chunk_offset: float) -> None:
        raise NotImplementedError

    def search(self, embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

class LocalFaissDB(AbstractVectorDatabase):
    def __init__(self, config):
        self.index = faiss.IndexFlatL2(config.embedding_dim)
        self.song_ids: list[str] = []
        self.chunk_offsets: list[float] = []

    def add(self, embedding: np.ndarray, song_id: str, chunk_offset: float) -> None:
        vec = embedding.astype(np.float32).reshape(1, -1)
        self.index.add(vec)
        self.song_ids.append(song_id)
        self.chunk_offsets.append(chunk_offset)

    def search(self, embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        vec = embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(vec, k)
        results: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.song_ids[idx], float(dist)))
        results.sort(key=lambda x: (x[1], x[0]))
        return results

    def count(self) -> int:
        return self.index.ntotal

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path + ".faiss")
        meta = {"song_ids": self.song_ids, "chunk_offsets": self.chunk_offsets}
        Path(path + ".json").write_text(json.dumps(meta))

    def load(self, path: str) -> None:
        self.index = faiss.read_index(path + ".faiss")
        meta = json.loads(Path(path + ".json").read_text())
        self.song_ids = meta["song_ids"]
        self.chunk_offsets = meta["chunk_offsets"]
        if len(self.song_ids) != self.index.ntotal:
            raise RuntimeError(
                f"song_id count {len(self.song_ids)} != index size {self.index.ntotal}"
            )

class AbstractHashDatabase:
    def add(self, hashes: list[tuple[int, float]], song_id: str, chunk_offset: float = 0.0) -> None:
        raise NotImplementedError

    def search(self, hashes: list[tuple[int, float]], k: int) -> list[tuple[str, float]]:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

class ReverseIndexDB(AbstractHashDatabase):
    def __init__(self, config=None):
        self.index: dict[int, list[tuple[str, float]]] = {}

    def add(self, hashes: list[tuple[int, float]], song_id: str, chunk_offset: float = 0.0) -> None:
        for h, anchor_time in hashes:
            self.index.setdefault(h, []).append((song_id, anchor_time))

    def search(self, hashes: list[tuple[int, float]], k: int) -> list[tuple[str, float]]:
        counts: dict[str, float] = {}
        for h, _ in hashes:
            for song_id, _ in self.index.get(h, []):
                counts[song_id] = counts.get(song_id, 0) + 1.0
        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        return [(sid, float(c)) for sid, c in ranked[:k]]

    def save(self, path: str) -> None:
        with open(path + ".pkl", "wb") as f:
            pickle.dump(self.index, f)

    def load(self, path: str) -> None:
        with open(path + ".pkl", "rb") as f:
            self.index = pickle.load(f)

    def count(self) -> int:
        return len({sid for entries in self.index.values() for sid, _ in entries})
