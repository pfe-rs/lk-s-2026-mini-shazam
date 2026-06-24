# Mini-Shazam: Dizajn dokument

## Pipeline

```
audio fajl ili mikrofon
    → AudioInput.load()
    ↓
  [chunkovanje]
    → Preprocessor.to_chunks() → list[audio_chunk] (raw)
    ↓
  [normalizacija]
    → FFmpeg two-pass loudnorm EBU R128 (nezavisno po chunk-u)
    ↓
  [spektrogram]
    → SpectrogramGenerator.generate() → float32 [128 × T]
    ↓
  [denojzer] (opciono, ONNX INT8)
    → PassthroughDenoiser ili UNetDenoiser
    ↓
  [encoder]
    → CNN enkoder (ONNX INT8) → int8 embedding [128]
    ↓
  [pretraga]
    → FAISS nearest-neighbor search
    → rezultat: pesma, confidence, top-k lista
```

FFmpeg normalizacija je obavezan korak. 
Svaki audio ulaz bez obzira na format, sample rate ili 
loudness mora biti sveden na identičan signal pre ulaska u 
spektrogram ili model. 
Ovo je uslov za determinizam, ista pesma snimljena na 
različitim mikrofonima ili preuzeta sa različitih izvora 
mora dati priblizno isti embedding (po mogucnosti skroz isti).

### Redosled chunk -> normalizacija (ključno za determinizam)

Chunkovanje se radi **pre** FFmpeg normalizacije. 
Svaki chunk se normalizuje nezavisno u dva prolaza. 
Jer ako normalizuješ ceo fajl pa ga onda chunkuješ, 
dva različita snimka iste pesme (npr. ceo fajl od 3min vs 5s sa mikrofona) 
dobijaju različitu normalizaciju jer `loudnorm` vidi različitu statističku 
distribuciju. 
Nezavisnom normalizacijom po chunku, 
isti 5s segment uvek daje identičan izlaz. 
Nema kontekstne zavisnosti.

### FFmpeg two-pass loudnorm

Single-pass `loudnorm` kontinuirano prilagođava gain tokom trajanja audia, 
što znači da prvih ~100ms može imati drugačiji gain od ostatka. 
Za determinizam do poslednjeg bita koristi se **two-pass** pristup:

Prvi prolaz meri tačne parametre (I, TP, LRA, offset). 
Drugi prolaz primenjuje fiksan gain bez adaptivnog ponašanja. 
Ovo garantuje da isti chunk uvek daje identičan WAV izlaz.

Parametri koji su uvek fiksni:

1. `-ac 1` za mono
2. `-ar 22050` za sample rate
3. `-sample_fmt s16` za 16-bit integer (eliminiše floating point varijacije)
4. `I=-23` za Integrated Loudness target (EBU R128 standard)
5. `TP=-1` za True Peak limit
6. `LRA=11` za Loudness Range

FFmpeg verzija mora biti pinovana u Docker image-u (ili venv-u)
jer loudnorm filter ima suptilne razlike u ponašanju između verzija.

## OOP arhitektura

Pogledaj diagram.

## Biblioteke i razlozi izbora

### Inference (salje se sa aplikacijom)

**onnxruntime**
Jedini deterministicki ML runtime. 
ONNX Runtime garantuje identicne rezultate na 
razlicitim CPU i GPU vendorima. 
PyTorch nema tu garanciju zbog razlika u SIMD 
implementacijama po hardveru. 
Quantized INT8 modeli u ONNX Runtime su i brzi i deterministicki 
jer nemaju razlike u floating point kalkulacijama.

**numpy**
Standardni numericki stack, nema alternative.

**librosa**
Jedina Python biblioteka za audio analizu. 
SpectrogramGenerator je jedini korisnik librosa-e u inference-u. 
Mel spektrogrami, resampling, amplitude-to-db konverzija, 
sve sa fiksnim parametrima iz RecognitionPipelineConfig. 
torchaudio radi slicno ali uvlaci PyTorch kao zavisnost 
u inference path sto ne zelimo.

**faiss-cpu**
Facebook-ov ANNS (approximate nearest neighbor search). 
IndexFlatL2 za exact search je dovoljan za bazu do 100k embeddings. 
Faiss je standardni izbor za ovaj problem, 
Facebook i Google ga koriste u produkciji. 
Alternativa je DiskANN od Microsofta ali je komplikovaniji za setup.

**sounddevice**
Najprostija biblioteka za snimanje sa mikrofona u Pythonu. 
Direktan pristup audio stream-u sa kontrolom nad sample rate-om.

**soundfile**
Citanje i pisanje audio fajlova, komplement sounddevice-u za wav fajlove.

**scipy**
Potreban za peak detection u MathFingerprinter-u (scipy.ndimage.maximum_filter).

**ffmpeg-python**
Python wrapper za FFmpeg. 
FFmpeg je jedini alat koji garantuje konzistentnu normalizaciju 
audio fajlova bez obzira na input format. 
loudnorm filter implementira EBU R128 standard.

**flet**
Flutter UI pisan u Pythonu. Flet pokrece RecognitionPipeline.run() 
i prikazuje RecognitionResult, ne zna nista o internim komponentama.

**IPC napomena:** Idealno bi UI i pipeline bili odvojeni procesi 
koji komuniciraju preko JSON poruka (stdin/stdout), 
ali to zahteva dodatno planiranje. 
Za sada CLI direktno poziva pipeline.

### Treniranje (ostaje na masinama za treniranje)

**torch, torchvision, torchaudio**
PyTorch ekosistem za treniranje. 
ResNet-18 (TODO: odrediti optimalan cnn) 
dolazi iz torchvision sa ImageNet tezinama za transfer learning. 
torchaudio se koristi samo u training pipeline-u, ne u inference-u.

**pytorch-metric-learning**
Biblioteka koja implementira TripletMarginLoss i HardTripletMiner. 
Andrej je vec koristio ovo u postojecem notebook-u i radi. 
Nema razloga da implementiramo triplet loss od nule.

**onnx**
Potreban za export PyTorch modela u ONNX format. 
torch.onnx.export generise ONNX graf.

**onnxruntime-gpu**
GPU verzija ONNX Runtime-a za brzu kvantizaciju. 
quantize_static iz onnxruntime.quantization konvertuje 
float32 ONNX model u INT8.

**pyroomacoustics**
Za sinteticku reverb augmentaciju tokom treniranja. 
Simulira akusticne prostorije sa kontrolisanim parametrima.

### Obrazloženje ključnih izbora

| Parametar | Zašto baš ova vrednost |
|---|---|
| `sample_rate = 22050` | Muzika na telefonu retko ima korisnog signala iznad 8kHz. 44100 bi duplirao podatke bez poboljšanja. |
| `n_fft = 2048` | Pri 22050Hz, 2048 uzoraka = ~93ms prozor. Dovoljno frekvencijske rezolucije da razlikuje tonove (~10.8Hz po bin-u), a opet dovoljno vremenske preciznosti za transient-e. |
| `hop_length = 512` | 512 / 22050 otp. 23ms. Četiri puta veća rezolucija od n_fft (87.5% overlap). Daje dobru vremensku dimenziju spektrograma za triplet učenje. |
| `n_mels = 128` | Balans između dimenzionalnosti i informativnosti. >128 ne poboljšava accuracy na ovom domenu, samo usporava. |
| `chunk_seconds = 5.0` | Dovoljno dugačak da sadrži refren ili prepoznatljiv deo pesme, a kratak za brzu pretragu. |
| `overlap_seconds = 2.5` | Za svaku pesmu dobijaš ~2× više chunkova nego bez overlapa. Više pozitivnih primera za triplet training. |

## Determinizam

```
FFmpeg two-pass loudnorm: identičan WAV iz istog chunk-a na svakoj mašini
librosa fiksni parametri: isti mel spektrogram iz istog WAV-a
ONNX Runtime:
    inter_op_num_threads = 1
    intra_op_num_threads = 1
    graph_optimization_level = ORT_ENABLE_ALL
INT8 kvantizacija: nema floating-point varijacija izmedju hardvera
L2: int8 embedding: finalni embedding je diskretan i ponovljiv
```

Finalni korak kvantizacije embeddinga pre cuvanja u bazu:

```python
embedding = embedding / np.linalg.norm(embedding)  # L2 normalizacija
quantized = np.round(embedding * 127).clip(-127, 127).astype(np.int8)
```

## Metrike evaluacije

### Protokol

Obe metode koriste identican set test pesama i identican preprocessing pipeline: AudioInput, chunkovanje, FFmpeg normalizacija, SpectrogramGenerator (librosa). Razlika je samo u fingerprintingu i pretrazi.

**CNNFingerprinter:** Svaki chunk se pretvara u int8 embedding[128] preko ONNX enkodera i cuva u FAISS indeksu. Query chunk se pretvori u isti embedding. FAISS vrati N najblizih vektora po L2 rastojanju. Za svaku pesmu se uzima najmanje L2 rastojanje (najslicniji chunk te pesme). Pesme se sortiraju od najmanjeg ka najvecem rastojanju.

**MathFingerprinter:** Svaki chunk se pretvara u hash parove (frekvencija1, frekvencija2, vremenski offset) i cuva u reverse indeksu. Query chunk se pretvori u iste hash parove. Za svaki hash se iz reverse indeksa dobiju pesme koje ga sadrze. Pesme se sortiraju po ukupnom broju poklopljenih hash parova, od najviseg ka najnizem.

Baza se pravi od clean chunkova. Query su isti chunkovi sa dodatim belim sumom (isti noise seed za obe metode). Nijedan query nije u bazi kao clean.

### Metrike

Top-k accuracy: tacno ako se prava pesma nalazi medju prvih k mesta na rang listi. Meri se za k = 1, 3 i 5.

MRR (Mean Reciprocal Rank): 1 / rang prave pesme. Prvo mesto je 1.0, drugo 0.5, trece 0.33. Prosek kroz sve queryje.

Tie break: ako dve ili vise pesama imaju identican skor (L2 rastojanje za CNN, broj hash poklapanja za Math), redosled se odredjuje leksikografski po song_id.

Sve metrike se mere na pet SNR nivoa: minus 5dB, 0dB, 5dB, 10dB, 20dB.

Poredjenje: CNNFingerprinter vs MathFingerprinter na svakom SNR nivou i svakom k.
