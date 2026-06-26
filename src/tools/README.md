# Alati

Pomocne skripte za mini-shazam pipeline.

## precompute.py

Prekomputacija mel spektrograma. 
Batch generise mel spektrograme za sve pesme iz manifesta. 
Procesira svaku pesmu kroz ceo frontend pipeline 
(FileInput, Preprocessor, FFmpegNormalizer, SpectrogramGenerator
) i cuva svaki chunk kao .npy fajl u cache/spectrograms/<song_id>/<chunk_index>.npy. 
Ovo je potpuno isti pipeline koji koristi sistem za 
prepoznavanje, nema distribution shift-a izmedju 
prekomputovanih podataka i inferenca.

Primeri koriscenja.

```bash
python src/tools/precompute.py dataset_downloader/manifest.json
```

Sa custom cache direktorijumom.

```bash
python src/tools/precompute.py dataset_downloader/manifest.json --cache cache/spectrograms
```

Nastavi gde je stalo (preskace vec kesirane pesme).

```bash
python src/tools/precompute.py dataset_downloader/manifest.json --resume
```

Ako je dataset u drugom direktorijumu.

```bash
python src/tools/precompute.py dataset_downloader/manifest.json --dataset-dir data/fma
```

Ako --dataset-dir nije prosledjen, 
automatski pronalazi dataset/ pored manifest.json.

Izlazna struktura: cache/spectrograms/000001/0.npy 
je prvi 5s chunk, mel spektrogram [128 x T]. 
cache/spectrograms/000001/1.npy je drugi chunk 
(2.5s preklapanje). I tako dalje.

Ucitavanje pojedinacnog prekomputovanog chunka u Pythonu.

```python
import numpy as np
spec = np.load("cache/spectrograms/000001/0.npy")
print(spec.shape)
```

Ako corrupted_blacklist.json postoji pored manifesta, 
fajlovi na listi se preskacu. 
Blacklist generise `check_corrupted.py` (u `dataset_downloader/`).

## visualize_spectrogram.py

Pregled mel spektrograma. 
Generise matplotlib i librosa.display.specshow 
prikaz mel spektrograma.

Iz audio fajla (samo prvi chunk podrazumevano).

```bash
python src/tools/visualize_spectrogram.py pesma.mp3
```

Cela pesma u jednom spektrogramu (bez chunkovanja).

```bash
python src/tools/visualize_spectrogram.py pesma.mp3 --full
```

Iz prekomputovanog .npy fajla.

```bash
python src/tools/visualize_spectrogram.py cache/spectrograms/000001/0.npy
```

Sacuvaj kao PNG umesto prikaza.

```bash
python src/tools/visualize_spectrogram.py pesma.mp3 --save spektrogram.png
```

Prikazi koriste magma kolormapu sa mel y-osom 
i vremenskom x-osom. Korisno za rucnu proveru 
da li pipeline ispravno procesuira audio. 
Posebno kad se debaguje FFmpeg normalizacija 
ili neka pesma koja daje cudne rezultate prepoznavanja.

Funkcije from_audio() i from_cache() se mogu importovati.

```python
from tools.visualize_spectrogram import from_audio, from_cache
from pipeline.config import RecognitionPipelineConfig
cfg = RecognitionPipelineConfig()
from_audio("pesma.mp3", cfg, save="izlaz.png")
```

## fetch_ffmpeg.sh i fetch_ffmpeg.ps1

Skidaju FFmpeg n8.1.2 (autobuild-2026-06-24-13-41) 
sa `BtbN/FFmpeg-Builds` i postavljaju binarni fajl u `src/tools/`.

Za Linux pokreni `bash src/tools/fetch_ffmpeg.sh`. 
Za Windows pokreni `powershell src/tools/fetch_ffmpeg.ps1`.

Svaka skripta proverava SHA256 checksum arhive pre raspakivanja.
Ako se checksum ne poklapa, arhiva je izmenjena ili je 
ldownload bio neispravan. 
Skripta izlazi sa greskom i brise skinuti fajl.

Verzija je fiksna tako da FFmpegNormalizer 
proizvodi bit-identican izlaz na svakoj masini. 
Nemoj zamenjivati drugim FFmpeg build-om ako nisi spreman 
da izgubis garancije determinizma.

## check_corrupted.py

Rad sa korumpiranim fajlovima. 
Skripta se nalazi u dataset_downloader/. 
Skenira sve MP3 fajlove u datasetu kroz librosa.load() 
i belezi one koji ne prolaze (koruptirani headeri, prazan audio, fajlovi koji nedostaju). 
Izlaz je corrupted_blacklist.json, koji precompute.py cita 
da bi preskocio poznate lose fajlove.

```bash
python dataset_downloader/check_corrupted.py -j 8
```

Blacklist trenutno ima oko 225 unosa. 
Skoro sve su FMA Unknown zanr pesme sa NoBackendError 
(koruptirani MP3 headeri iz originalne FMA arhive), 
plus jedna trajno nestala YouTube pesma (120331, Wrecking Ball).

## Interakcije alata

`check_corrupted.py` generise `corrupted_blacklist.json`. 
`precompute.py` cita blacklist i preskace blacklistovane fajlove.

`fetch_ffmpeg.sh` i `fetch_ffmpeg.ps1` postavljaju ffmpeg 
binary u `src/tools/`. FFmpegNormalizer.init ga pronalazi.

`precompute.py` upisuje cache/spectrograms/<song_id>/<chunk_index>.npy. 
`visualize_spectrogram.py` cita ove fajlove sa --from-cache. 
Training pipeline cita ove fajlove da izbegne ponovno 
racunanje po epohi.

visualize_spectrogram.py je samostalni alat za pregled.

## Zahtevi za alate

Alati koji importuju iz `pipeline/` zahtevaju 
aktivirano virtuelno okruzenje.

```bash
python -m venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python src/tools/visualize_spectrogram.py pesma.mp3
```

Matplotlib je potreban samo za `visualize_spectrogram.py`. 

Fetch skripte koriste samo standardne alate. 
Curl i sha256sum za Linux. 
Invoke-WebRequest i Get-FileHash za Windows. 
