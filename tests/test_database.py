import os
import tempfile

import numpy as np
from pipeline.config import RecognitionPipelineConfig
from pipeline.database import LocalFaissDB, ReverseIndexDB


def _make_emb():
    return np.random.randint(-127, 127, size=(128,)).astype(np.int8)


def _make_hashes(n=10):
    return [(i, float(i * 0.5)) for i in range(n)]


def test_add_and_count():
    cfg = RecognitionPipelineConfig()
    db = LocalFaissDB(cfg)
    assert db.count() == 0
    db.add(_make_emb(), "song_1", 0.0)
    assert db.count() == 1
    db.add(_make_emb(), "song_2", 5.0)
    assert db.count() == 2


def test_search_returns_results():
    cfg = RecognitionPipelineConfig()
    db = LocalFaissDB(cfg)
    emb = _make_emb()
    db.add(emb, "test_song", 0.0)
    results = db.search(emb, k=5)
    assert len(results) >= 1
    assert results[0][0] == "test_song"


def test_search_empty_db():
    cfg = RecognitionPipelineConfig()
    db = LocalFaissDB(cfg)
    results = db.search(_make_emb(), k=5)
    assert results == []


def test_save_and_load():
    cfg = RecognitionPipelineConfig()
    db = LocalFaissDB(cfg)
    db.add(_make_emb(), "song_a", 0.0)
    db.add(_make_emb(), "song_b", 2.5)
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "index")
        db.save(prefix)
        db2 = LocalFaissDB(cfg)
        db2.load(prefix)
        assert db2.count() == 2


def test_save_load_count_mismatch_raises():
    cfg = RecognitionPipelineConfig()
    db = LocalFaissDB(cfg)
    db.add(_make_emb(), "song_a", 0.0)
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "index")
        db.save(prefix)
        db2 = LocalFaissDB(cfg)
        meta_path = os.path.join(tmp, "index.json")
        import json
        meta = json.loads(open(meta_path).read())
        meta["song_ids"].append("stray_id")
        open(meta_path, "w").write(json.dumps(meta))
        try:
            db2.load(prefix)
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass


def test_reverse_index_add_and_count():
    db = ReverseIndexDB()
    assert db.count() == 0
    db.add(_make_hashes(5), "song_a")
    assert db.count() == 1
    db.add(_make_hashes(3), "song_b")
    assert db.count() == 2


def test_reverse_index_search_finds_match():
    db = ReverseIndexDB()
    db.add([(1, 0.0), (2, 0.5)], "song_a")
    result = db.search([(1, 0.0)], k=5)
    assert len(result) == 1
    assert result[0][0] == "song_a"
    assert result[0][1] == 1.0


def test_reverse_index_search_counts():
    db = ReverseIndexDB()
    db.add([(1, 0.0), (2, 0.5)], "song_a")
    db.add([(1, 1.0), (3, 1.5)], "song_b")
    result = db.search([(1, 0.0), (1, 2.0), (2, 3.0)], k=5)
    assert dict(result) == {"song_a": 3.0, "song_b": 2.0}


def test_reverse_index_search_unique_hashes():
    db = ReverseIndexDB()
    db.add([(1, 0.0), (2, 0.5)], "song_a")
    db.add([(1, 1.0), (3, 1.5)], "song_b")
    result = db.search([(1, 0.0), (2, 0.0)], k=5)  # svaki hash po jednom
    assert dict(result) == {"song_a": 2.0, "song_b": 1.0}


def test_reverse_index_search_empty():
    db = ReverseIndexDB()
    result = db.search([(1, 0.0)], k=5)
    assert result == []


def test_reverse_index_save_and_load():
    db = ReverseIndexDB()
    db.add([(10, 0.0)], "song_x")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "index")
        db.save(path)
        db2 = ReverseIndexDB()
        db2.load(path)
        assert db2.count() == 1
        assert db2.search([(10, 0.0)], k=5) == [("song_x", 1.0)]
