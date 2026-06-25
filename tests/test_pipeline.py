from pipeline import RecognitionPipeline, RecognitionPipelineConfig


def test_pipeline_constructs():
    cfg = RecognitionPipelineConfig()
    from pipeline.audio_input import FileInput
    from pipeline.preprocessor import Preprocessor
    from pipeline.normalizer import FFmpegNormalizer
    from pipeline.spectrogram import SpectrogramGenerator
    from pipeline.denoiser import NoDenoiser
    from pipeline.fingerprinter import CNNFingerprinter
    from pipeline.database import LocalFaissDB

    try:
        normalizer = FFmpegNormalizer(cfg)
        pipeline = RecognitionPipeline(
            config=cfg,
            audio_input=FileInput(),
            normalizer=normalizer,
            preprocessor=Preprocessor(cfg),
            spectrogram_generator=SpectrogramGenerator(cfg),
            denoiser=NoDenoiser(),
            fingerprinter=CNNFingerprinter(),
            database=LocalFaissDB(cfg),
        )
    except RuntimeError:
        class FakeNormalizer:
            def normalize(self, chunk, sr):
                return chunk

        pipeline = RecognitionPipeline(
            config=cfg,
            audio_input=FileInput(),
            normalizer=FakeNormalizer(),
            preprocessor=Preprocessor(cfg),
            spectrogram_generator=SpectrogramGenerator(cfg),
            denoiser=NoDenoiser(),
            fingerprinter=CNNFingerprinter(),
            database=LocalFaissDB(cfg),
        )

    assert pipeline.config.sample_rate == 22050


class _FakeDB:
    def __init__(self, results):
        self._results = results
        self.added = []

    def search(self, emb, k):
        return self._results

    def add(self, emb, song_id, chunk_offset):
        self.added.append((song_id, chunk_offset))

    def count(self):
        return len(self.added)


class _FakeNormalizer:
    def normalize(self, chunk, sr):
        return chunk


class _FakeAudioInput:
    def load(self, path, target_sr=None):
        import numpy as np
        return np.zeros(22050 * 12, dtype=np.float32)


class _FakeFingerprinter:
    def fingerprint(self, spectrogram):
        import numpy as np
        return np.zeros(128, dtype=np.int8)

    def merge_chunk_scores(self, existing, new):
        return min(existing, new)

    def sort_key(self, song_id, score):
        return (score, song_id)

    def confidence(self, top_k):
        if not top_k:
            return 0.0
        if len(top_k) == 1:
            return 1.0
        best, second = top_k[0][1], top_k[1][1]
        return 1.0 if second == 0 else max(0.0, 1.0 - (best / second))


def _make_pipeline(config, database):
    from pipeline.preprocessor import Preprocessor
    from pipeline.spectrogram import SpectrogramGenerator
    from pipeline.denoiser import NoDenoiser

    return RecognitionPipeline(
        config=config,
        audio_input=_FakeAudioInput(),
        normalizer=_FakeNormalizer(),
        preprocessor=Preprocessor(config),
        spectrogram_generator=SpectrogramGenerator(config),
        denoiser=NoDenoiser(),
        fingerprinter=_FakeFingerprinter(),
        database=database,
    )


def test_recognize_confidence_exact_match():
    cfg = RecognitionPipelineConfig()
    db = _FakeDB([("song_a", 0.0)])
    pipeline = _make_pipeline(cfg, db)
    result = pipeline.recognize("dummy.wav")
    assert result.song_id == "song_a"
    assert result.confidence == 1.0


def test_recognize_confidence_no_match():
    cfg = RecognitionPipelineConfig()
    db = _FakeDB([])
    pipeline = _make_pipeline(cfg, db)
    result = pipeline.recognize("dummy.wav")
    assert result.song_id == ""
    assert result.confidence == 0.0


def test_recognize_confidence_two_results():
    cfg = RecognitionPipelineConfig()
    db = _FakeDB([("song_a", 5.0), ("song_b", 10.0)])
    pipeline = _make_pipeline(cfg, db)
    result = pipeline.recognize("dummy.wav")
    assert result.song_id == "song_a"
    assert result.confidence == 0.5


def test_index_song_adds_all_chunks():
    cfg = RecognitionPipelineConfig()
    db = _FakeDB([])
    pipeline = _make_pipeline(cfg, db)
    count = pipeline.index_song("dummy.wav", "song_x")
    assert count >= 2
    assert all(sid == "song_x" for sid, _ in db.added)
    assert len(db.added) == count


class _AdversarialFingerprinter:
    def fingerprint(self, spectrogram):
        import numpy as np
        return np.zeros(128, dtype=np.int8)

    def merge_chunk_scores(self, existing, new):
        return max(existing, new)

    def sort_key(self, song_id, score):
        return (-score, song_id)

    def confidence(self, top_k):
        if not top_k:
            return 0.0
        if len(top_k) == 1:
            return 1.0
        best, second = top_k[0][1], top_k[1][1]
        if best == 0:
            return 0.0
        return max(0.0, 1.0 - (second / best))


class _ChunkedFakeDB:
    def __init__(self, per_chunk_results):
        self._results = list(per_chunk_results)
        self._call = 0
        self.added = []

    def search(self, emb, k):
        r = self._results[self._call % len(self._results)]
        self._call += 1
        return r

    def add(self, emb, song_id, chunk_offset):
        self.added.append((song_id, chunk_offset))

    def count(self):
        return len(self.added)


def test_pipeline_with_adversarial_fingerprinter():
    cfg = RecognitionPipelineConfig()
    db = _ChunkedFakeDB([
        [("song_a", 3.0), ("song_b", 5.0)],
        [("song_b", 1.0), ("song_c", 3.0)],
    ])
    from pipeline.preprocessor import Preprocessor
    from pipeline.spectrogram import SpectrogramGenerator
    from pipeline.denoiser import NoDenoiser

    pipeline = RecognitionPipeline(
        config=cfg,
        audio_input=_FakeAudioInput(),
        normalizer=_FakeNormalizer(),
        preprocessor=Preprocessor(cfg),
        spectrogram_generator=SpectrogramGenerator(cfg),
        denoiser=NoDenoiser(),
        fingerprinter=_AdversarialFingerprinter(),
        database=db,
    )
    result = pipeline.recognize("dummy.wav")
    assert result.song_id == "song_b"
    assert result.confidence == 0.4
    assert result.top_k[:2] == [("song_b", 5.0), ("song_a", 3.0)]
    assert db._call > 1
