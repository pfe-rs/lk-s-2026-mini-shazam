import time

from .denoiser import NoDenoiser
from .result import RecognitionResult


class RecognitionPipeline:
    def __init__(self, config, audio_input, normalizer, preprocessor,
                 spectrogram_generator, denoiser, fingerprinter, database):
        self.config = config
        self.audio_input = audio_input
        self.normalizer = normalizer
        self.preprocessor = preprocessor
        self.spectrogram_generator = spectrogram_generator
        self.denoiser = denoiser
        self.fingerprinter = fingerprinter
        self.database = database

    def recognize(self, path: str, k: int = 5) -> RecognitionResult:
        t0 = time.perf_counter()

        audio = self.audio_input.load(path, self.config.sample_rate)
        chunks = self.preprocessor.to_chunks(audio, self.config.sample_rate)

        all_scores: dict[str, float] = {}
        for chunk in chunks:
            norm = self.normalizer.normalize(chunk, self.config.sample_rate)
            spec = self.spectrogram_generator.generate(norm, self.config.sample_rate)
            spec = self.denoiser.process(spec)
            emb = self.fingerprinter.fingerprint(spec)

            results = self.database.search(emb, k)
            for song_id, score in results:
                if song_id not in all_scores:
                    all_scores[song_id] = score
                else:
                    all_scores[song_id] = self.fingerprinter.merge_chunk_scores(
                        all_scores[song_id], score
                    )

        sorted_results = sorted(
            all_scores.items(), key=lambda x: self.fingerprinter.sort_key(x[0], x[1])
        )
        top_k = sorted_results[:k]

        song_id = top_k[0][0] if top_k else ""
        confidence = self.fingerprinter.confidence(top_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return RecognitionResult(
            song_id=song_id,
            top_k=top_k,
            confidence=confidence,
            latency_ms=latency_ms,
            denoiser_used=not isinstance(self.denoiser, NoDenoiser),
            fingerprinter_used=type(self.fingerprinter).__name__,
        )

    def index_song(self, path: str, song_id: str) -> int:
        audio = self.audio_input.load(path, self.config.sample_rate)
        chunks = self.preprocessor.to_chunks(audio, self.config.sample_rate)

        for i, chunk in enumerate(chunks):
            norm = self.normalizer.normalize(chunk, self.config.sample_rate)
            spec = self.spectrogram_generator.generate(norm, self.config.sample_rate)
            spec = self.denoiser.process(spec)
            emb = self.fingerprinter.fingerprint(spec)

            chunk_offset = i * self.preprocessor.hop_samples / self.config.sample_rate
            self.database.add(emb, song_id, chunk_offset)

        return len(chunks)
