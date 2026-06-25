import numpy as np
from pipeline.fingerprinter import CNNFingerprinter, MathFingerprinter


def test_quantize_output_type():
    emb = np.random.randn(128).astype(np.float32)
    q = CNNFingerprinter.quantize(emb)
    assert q.dtype == np.int8
    assert q.shape == (128,)


def test_quantize_clamping():
    emb = np.ones(128, dtype=np.float32) * 1000
    q = CNNFingerprinter.quantize(emb)
    assert q.max() <= 127
    assert q.min() >= -127


def test_quantize_zero_embedding():
    emb = np.zeros(128, dtype=np.float32)
    q = CNNFingerprinter.quantize(emb)
    assert q.dtype == np.int8
    assert q.shape == (128,)
    assert np.all(q == 0)


def test_quantize_deterministic():
    emb = np.random.randn(128).astype(np.float32)
    q1 = CNNFingerprinter.quantize(emb)
    q2 = CNNFingerprinter.quantize(emb)
    assert np.array_equal(q1, q2)


def test_cnn_not_implemented():
    cnn = CNNFingerprinter()
    try:
        cnn.fingerprint(np.array([[1.0]]))
        assert False
    except NotImplementedError:
        pass


def test_cnn_merge_min():
    fp = CNNFingerprinter()
    assert fp.merge_chunk_scores(10.0, 3.0) == 3.0
    assert fp.merge_chunk_scores(1.0, 5.0) == 1.0


def test_cnn_sort_ascending():
    fp = CNNFingerprinter()
    items = [("b", 5.0), ("a", 3.0), ("c", 3.0)]
    sorted_items = sorted(items, key=lambda x: fp.sort_key(x[0], x[1]))
    assert sorted_items == [("a", 3.0), ("c", 3.0), ("b", 5.0)]


def test_cnn_confidence():
    fp = CNNFingerprinter()
    assert fp.confidence([]) == 0.0
    assert fp.confidence([("a", 0.0)]) == 1.0
    assert fp.confidence([("a", 5.0), ("b", 10.0)]) == 0.5
    assert fp.confidence([("a", 0.0), ("b", 10.0)]) == 1.0


def test_math_merge_sum():
    fp = MathFingerprinter()
    assert fp.merge_chunk_scores(5.0, 3.0) == 8.0
    assert fp.merge_chunk_scores(0.0, 7.0) == 7.0


def test_math_sort_descending():
    fp = MathFingerprinter()
    items = [("a", 5.0), ("b", 10.0), ("c", 5.0)]
    sorted_items = sorted(items, key=lambda x: fp.sort_key(x[0], x[1]))
    assert sorted_items == [("b", 10.0), ("a", 5.0), ("c", 5.0)]


def test_math_confidence():
    fp = MathFingerprinter()
    assert fp.confidence([]) == 0.0
    assert fp.confidence([("a", 10.0)]) == 1.0
    assert fp.confidence([("a", 10.0), ("b", 5.0)]) == 0.5
    assert fp.confidence([("a", 0.0), ("b", 5.0)]) == 0.0
    assert fp.confidence([("a", 10.0), ("b", 0.0)]) == 1.0


def test_math_landmark_tie_break():
    fp = MathFingerprinter()
    spec = np.full((128, 50), -80.0, dtype=np.float32)
    spec[50, 25] = 0.0
    spec[60, 25] = 0.0
    spec[50, 30] = 0.0
    hashes = fp.fingerprint(spec)
    assert len(hashes) > 0


def test_math_deterministic():
    fp = MathFingerprinter()
    spec = np.random.randn(128, 100).astype(np.float32)
    h1 = fp.fingerprint(spec)
    h2 = fp.fingerprint(spec)
    assert h1 == h2
