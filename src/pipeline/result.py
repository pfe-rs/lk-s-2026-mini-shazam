from dataclasses import dataclass

@dataclass
class RecognitionResult:
    song_id: str
    top_k: list[tuple[str, float]]
    confidence: float
    latency_ms: float
    denoiser_used: bool
    fingerprinter_used: str

@dataclass
class EvaluationResult:
    fingerprinter_name: str
    snr_db: float
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    mrr: float
    mean_latency_ms: float
    n_queries: int
