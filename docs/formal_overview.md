# Project Specification: Audio Fingerprinting vs. Deep Learning Latent Embeddings

## 1. Project Overview & Deliverables
This project is a machine learning and information retrieval research study evaluating deterministic mathematical audio matching against continuous latent space deep learning models.

* **Primary Objective:** Evaluate and compare methodologies for robust song identification under challenging acoustic and artistic variations.
* **Deliverable 1 (Codebase):** A Python framework implementing data scraping, deterministic audio preprocessing, a PyTorch-to-ONNX deep learning pipeline, and an end-to-end evaluation suite.
* **Deliverable 2 (Research Paper):** An academic paper documenting methodology, experimental setups, benchmarking results, and mathematical justifications.

---

## 2. Core Hypothesis
The research evaluates performance across two specialized degradation domains:

> **Mathematical Hash advantage in environmental noise:** When the original audio is contaminated by additive background noise (chatter, traffic), deterministic peak-pair hashing remains robust because the absolute highest time-frequency energy maxima survive masking.
>
> **Deep Learning advantage in structural mutation:** When evaluating structural/timbral mutations (covers, live performances), mathematical fingerprinting fails due to frequency shifts. A deep learning embedding model captures invariant melodic structures and semantic features in continuous latent space.

---

## 3. Data Engineering & Preprocessing Pipeline
To guarantee deterministic embeddings regardless of the source file's original length, the preprocessing pipeline dictates strict temporal chunking *prior* to normalization. 

* **Audio Slicing:** Raw audio is immediately sliced into 5.0-second chunks with a 2.5-second overlap.
* **Deterministic Normalization:** Each isolated chunk undergoes an independent two-pass FFmpeg `loudnorm` filter (EBU R128) to eliminate contextual gain variations and guarantee identical WAV outputs.
* **Mel-Spectrogram Generation:** The normalized chunks are processed into Mel-Spectrograms using fixed `librosa` parameters.

### Fixed Audio Parameters
| Parameter | Value | Justification |
| :--- | :--- | :--- |
| Sample Rate | 22050 Hz | Captures essential musical frequencies without redundant data overhead. |
| FFT Window (`n_fft`) | 2048 | Provides a ~93ms window for balanced time-frequency resolution. |
| Hop Length | 512 | Yields ~23ms spacing (87.5% overlap) for dense temporal tracking. |
| Mel Bins (`n_mels`) | 128 | Balances embedding dimensionality with structural information retention. |
| Chunk Size | 5.0 s | Long enough to capture melodic context, short enough for rapid search. |
| Overlap | 2.5 s | Effectively doubles available training chunks for the Siamese network. |

---

## 4. System Architecture

### Pre-processor: The Denoising Module
* **Architecture:** Convolutional Autoencoder (U-Net topology).
* **Function:** Maps noisy Mel-Spectrogram frames directly to clean target Mel-Spectrogram frames. 

### Method 1: Mathematical Fingerprinting
* **Mechanism:** Extracts time-frequency anchor points from prominent spectral peaks using `scipy.ndimage.maximum_filter`. 
* **Operation:** Pairs adjacent points to construct deterministic hashes, querying a `faiss-cpu` exact search index (IndexFlatL2) for offset alignment histograms.

### Method 2: Deep Learning Latent Embeddings
* **Development Phase:** Models (CNN Encoder and U-Net) are developed and trained using the PyTorch ecosystem, providing a familiar environment for building and tuning complex custom architectures like algorithmic music models.
* **Online Triplet Mining:** Training utilizes Siamese networks with Triplet Loss. The `pytorch-metric-learning` library employs `HardTripletMiner` to dynamically assemble challenging Anchor-Positive-Negative combinations directly within the active batch.

---

## 5. Commercialization & Inference Engine
To transition from the PyTorch research environment to a deterministic production deployment, the inference engine imposes strict execution constraints.

* **ONNX INT8 Quantization:** All inference models are exported to ONNX and statically quantized to INT8. This strips out floating-point variances between different CPU/GPU hardware vendors.
* **L2 Normalization:** Final vector embeddings undergo L2 normalization and `np.int8` clipping prior to database insertion.
* **Vector Search:** The `faiss-cpu` engine handles the nearest-neighbor retrieval on the discrete, reproducible integer embeddings.

---

## 6. Experimental Evaluation Framework
The evaluation suite measures performance across five specific Signal-to-Noise Ratio (SNR) levels (-5dB, 0dB, 5dB, 10dB, 20dB).

**Top-K Accuracy**
Measures how often the true song identity appears within the model's top $K$ retrieval candidates (evaluated at $K=1, 3, 5$).


**Mean Reciprocal Rank (MRR)**
Evaluates retrieval quality by penalizing systems when the correct match drops further down the ranking order.

