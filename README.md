# mini-shazam

Sistem za prepoznavanje muzike koji poredi deterministicko 
matematicko hesiranje (Shazam algoritam) sa deep learning embeddingima 
(ResNet triplet loss). Istraživacki projekat za LK/S 2026.

## Hipoteza

Matematicko hesiranje je otpornije na sum u okolini (galama, saobracaj) 
jer najvisi energetski pikovi u vremensko frekvencijskom prostoru preživljavaju 
maskiranje. 
Deep learning embeddingi prepoznaju strukturne mutacije (coveri, live izvedbe) 
koje lome hesiranje, jer model uci invarijantne melodijske strukture 
u kontinualnom latentnom prostoru.

Obe metode se porede na 5 SNR nivoa (minus 5, 0, 5, 10, 20 dB) 
na 750+ pesama iz FMA skupa.

## Arhitektura

Audio fajl prelazi kroz sledece faze. 
AudioInput.load() koristi librosa i resample na config.sample_rate. 
Preprocessor.to_chunks() deli audio na 5s chunkove sa 2.5s preklapanjem. 
FFmpegNormalizer radi two-pass EBU R128 loudnorm po chunku (deterministicki). 
SpectrogramGenerator pravi librosa mel spektrogram [128 x T]. 
Denoiser je NoDenoiser (prolaz) ili UNetDenoiser (stub). 
Fingerprinter je CNNFingerprinter ili MathFingerprinter. 
Database.search() je LocalFaissDB (L2) ili ReverseIndexDB (hes). 
Krajnji RecognitionResult je: song_id, confidence, top_k, latency_ms.

## Podesavanje

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### FFmpeg

FFmpegNormalizer zahteva ffmpeg binarni fajl sa loudnorm filterom.

Opcija A, bundlovan binary (preporuceno za determinizam). 
Pokreni bash src/tools/fetch_ffmpeg.sh za Linux ili powershell 
src/tools/fetch_ffmpeg.ps1 za Windows. 
Postavlja ffmpeg (ili ffmpeg.exe) u src/tools/. 
Verzija je fiksna FFmpeg n8.1.2 (autobuild-2026-06-24-13-41) i 
SHA256 se proverava pre raspakivanja.

Opcija B, sistemski FFmpeg. 
Postavi cfg = RecognitionPipelineConfig(ffmpeg_bin="/usr/bin/ffmpeg").
(ili put za tvoj OS)

Opcija C, imageio-ffmpeg fallback. 
Pokreni pip install imageio-ffmpeg, automatski se detektuje poslednji. 

## Brzi pocetak

```py
from pipeline import RecognitionPipeline, RecognitionPipelineConfig
from pipeline.audio_input import FileInput
from pipeline.preprocessor import Preprocessor
from pipeline.normalizer import FFmpegNormalizer
from pipeline.spectrogram import SpectrogramGenerator
from pipeline.denoiser import NoDenoiser
from pipeline.fingerprinter import CNNFingerprinter
from pipeline.database import LocalFaissDB

cfg = RecognitionPipelineConfig()

pipeline = RecognitionPipeline(
    config=cfg,
    audio_input=FileInput(),
    normalizer=FFmpegNormalizer(cfg),
    preprocessor=Preprocessor(cfg),
    spectrogram_generator=SpectrogramGenerator(cfg),
    denoiser=NoDenoiser(),
    fingerprinter=CNNFingerprinter(), #TODO: u toku
    database=LocalFaissDB(cfg),
)
```

Za indeksiranje pesme pozovi pipeline.index_song("pesma.mp3", "id_pesme"). 
Za prepoznavanje pozovi rezultat = pipeline.recognize("upit.mp3", k=5). 
Ispisi rezultat.song_id, rezultat.confidence, rezultat.top_k.

## Referenca modula

### config.py

RecognitionPipelineConfig ima sledeca polja. 
sample_rate je 22050, ciljni sample rate za sav audio. 
n_fft je 2048, velicina FFT prozora (~93ms na 22050Hz). 
hop_length je 512, korak STFT-a (~23ms, 87.5% preklapanje). 
n_mels je 128, broj mel filtera. 
fmin je 0.0, minimalna frekvencija za mel filter banku. 
fmax je 8000.0, maksimalna frekvencija. 
chunk_seconds je 5.0, duzina chunka. 
overlap_seconds je 2.5, preklapanje izmedju chunkova. 
embedding_dim je 128, dimenzija za CNN embeddinge i FAISS vektore. 
ffmpeg_bin je "", put do ffmpeg binarnog fajla. 
db_search_k je 5, podrazumevani top-K za pretragu baze.

### audio_input.py

FileInput cita audio iz bilo kog formata kroz 
librosa.load(path, sr=None, mono=True). 
Resampleuje na target_sr ako se sample rate fajla razlikuje. 
Dize ValueError na prazan audio. MicrophoneInput je stub (nije implementiran).

### preprocessor.py

Preprocessor deli audio na fiksne chunkove sa podesivim preklapanjem. 
chunk_samples = sr * chunk_seconds, hop_samples = sr * (chunk_seconds overlap_seconds). 

### normalizer.py

FFmpegNormalizer radi two-pass EBU R128 loudnorm po chunku. 
Pass 1 merenje: loudnorm=I=-23:TP=-1:LRA=11:print_format=json, cita chunk, 
izbacuje JSON sa input_i, input_tp, input_lra, input_thresh, target_offset. 
Pass 2 primena: isti filter sa measured_I/TP/LRA/thresh, offset, linear=true, 
primenjuje fiksan gain bez adaptivnog ponašanja.

Ulazni chunk se upisuje kao PCM_16 WAV u temp fajl. 
Izlaz se cita sa soundfile.read(dtype='float32'). 
Ako Pass 1 vrati nevalidne vrednosti (na primer tisina), 
vraca se originalni nepromenjeni chunk.

### spectrogram.py

SpectrogramGenerator koristi librosa.feature.melspectrogram sa fiksnim parametrima. 
Koristi ref=np.max u power_to_db jer FFmpegNormalizer vec resava apsolutnu glasnocu. 
Softmax po spektrogramu bi pojačao sum na tihim chunkovima.

### denoiser.py

NoDenoiser je prost prolaz. UNetDenoiser ceka ONNX model.

### fingerprinter.py

Za sada eksperimentalno i ocekuje promene.

### database.py

Implementacije su za sada samo prototipi i koriste
samo za dokaz rada, nisu skalabilne i ocekuju unapredjenja.

### registry.py

ModelRegistry je JSON manifest sa SHA256 checksumovima. Prati model fajlove za proveru integriteta.

```py
reg = ModelRegistry("models/manifest.json")
reg.register("cnn_v1", "models/cnn_v1.onnx")
reg.get("cnn_v1") vraca "models/cnn_v1.onnx"
reg.verify_checksum("cnn_v1") vraca True ili False (fajl izmenjen?)
reg.list_models() vraca ["cnn_v1"]
```

### pipeline.py

RecognitionPipeline orkestrira sve komponente.

recognize(path, k=5) ucitava, chunkuje, normalizuje, spektrogram, denojz, fingerprint, 
pretraga baze (svi chunkovi). 
Spaja skorove kroz fingerprinter.merge_chunk_scores(). 
Sortira kroz fingerprinter.sort_key(). 
Racuna confidence kroz fingerprinter.confidence(). 
Vraca RecognitionResult.

index_song(path, song_id) radi isti pipeline ali poziva 
database.add() umesto search(). Vraca broj dodatih chunkova.

### evaluator.py

Evaluator nije finalan za sad.

### result.py

RecognitionResult ima polja song_id (str), top_k (list[tuple[str, float]]), 
confidence (float), latency_ms (float), denoiser_used (bool), fingerprinter_used (str).

EvaluationResult ima polja fingerprinter_name (str), snr_db (float), 
top1_accuracy (float), top3_accuracy (float), top5_accuracy (float), 
mrr (float), mean_latency_ms (float), n_queries (int).

## Determinizam

Isti izvorni chunk prolazi kroz FFmpeg two-pass loudnorm 
(bit-egzaktan, pinovana verzija n8.1.2), librosa fiksni parametri 
(n_fft, hop, n_mels, fmin, fmax, window, power), ONNX INT8 kvantizovani enkoder 
(uklanja float varijacije), L2 normalizacija plus zaokruzi puta 127 plus ogranici 
+/-127 u int8 embedding. 
Rezultat je isti int8 embedding svaki put na bilo kojoj platformi.

## Dataset

1000 pesama iz FMA (Free Music Archive), stratifikovano po 15 žanrova (seed=42). 
Skinute sa YouTubea (481 pesma) i FMA ZIP arhiva (519 pesama). 
Kompletan manifest u dataset_downloader/manifest.json sa source URLovima, MD5 checksumovima 
i zanr labelama. Videti dataset_downloader/README.md za skripte za skidanje i verifikaciju.
Pronadjeno je oko 200 koruptiranih pesama.

## Pokretanje testova

Za sve ne-ffmpeg testove (podrazumevani CI) pokreni 
python -m pytest tests/ -v -m "not ffmpeg". 
Za sve testove ukljucujuci ffmpeg (zahteva ffmpeg binary) pokreni python -m pytest tests/ -v.

50 testova u 10 fajlova. 
Testovi koriste fake implementacije za audio ulaz, normalizator i bazu. 
Nisu potrebni pravi audio fajlovi za ne-ffmpeg testove. 
Adversarial test (test_pipeline_with_adversarial_fingerprinter) proverava da li 
_ChunkedFakeDB sa razlicitim rezultatima po chunku proizvodi drugaciji 
izlaz kad se merge_chunk_scores ponasanje promeni.

## CI/CD

ci.yml se okida na push na master/0.1-api-barebones i PR na master. 
Testira Python 3.11, 3.12, 3.13 sa -m "not ffmpeg". 
ffmpeg-weekly.yml se okida ponedeljkom 06:00 UTC i putem workflow_dispatch. 
Testira Python 3.12 sa -m ffmpeg (instalira ffmpeg kroz apt).

## Struktura projekta

`src/pipeline/` sadrzi pipeline za prepoznavanje (kompletno za API 0.1). 
`src/training/` sadrzi TripletDataset, AudioAugmenter, TrainingPipeline (za Andreja). 
`src/ui/` sadrzi IPC koncept (nije implementirano). 
`src/deploy/` sadrzi PyInstaller koncept (nije implementirano). 
`src/tools/` sadrzi precompute.py, visualize_spectrogram.py, ffmpeg fetch skripte. 
`tests/` sadrzi 50 testova (pytest). 
`dataset_downloader/` sadrzi FMA dataset skidanje, verifikaciju, popravke. 
`docs/` sadrzi design_doc.md, formal_overview.md, status.md, report_pipeline.md. 
`cache/spectrograms/` sadrzi prekomputovane .npy fajlove (gitignorovani).

## Sledeci deo rada

CNNFingerprinter.fingerprint() nije skroz implementiran, ceka ONNX model. 
Evaluator.evaluate() nije skroz implementiran, sva infrastruktura spremna. 
UNetDenoiser.process() nije implementiran. 
MicrophoneInput.load() nije implementiran, nije potrebno dok ne bude UI. 
MathFingerprinter podesavanje koristi pocetne parametre iz 2003 rada, 
treba tuning za mel spektrogram. 
LocalFaissDB ne pamti koji model je generisao embeddinge, 
zamena modela nevidljivo vraca pogresne rezultate. 
MathFingerprinter + ReverseIndexDB kroz recognize() nikad testirani zajedno. 
UI nije zapocet.
