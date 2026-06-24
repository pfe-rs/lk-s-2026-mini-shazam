# Mini-Shazam: Dataset Downloader

Skripte za preuzimanje i verifikaciju FMA dataseta.

## Dataset

**FMA v1.0** (Jul 2024): Free Music Archive.

- Metadata: `fma_metadata.zip` — `https://os.unil.cloud.switch.ch/fma/fma_metadata.zip`
- Audio subsets: `fma_small.zip`, `fma_medium.zip`, `fma_full.zip`
- Izvor: https://os.unil.cloud.switch.ch/fma/

`fma_metadata.zip` MD5: `d3ebfd86e283345ee2366a5492495935`

Dataset se ne čuva u repozitorijumu. 
Pokretanjem `select_tracks.py` (opcionalno, vec postoji `selected_tracks.csv` i `manifest.json`) 
pa `download_dataset.py` reprodukuje se identičan skup od 1000 pesama.

## Setup

**Linux/macOS:**
```bash
./setup.sh
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

Zatim pokreni `download_dataset.py` da preuzmeš pesme.

## Skripte

| Skripta | Namena |
|---|---|
| `select_tracks.py` | Bira 1000 pesama iz FMA metapodataka stratifikovano po žanru. Output: `selected_tracks.csv`. |
| `download_dataset.py` | Preuzima pesme, prvo yt-dlp sa YouTube-a, fallback na FMA ZIP arhive. Ugrađuje ID3 tagove, beleži sve u `manifest.json`. |
| `verify_dataset.py` | Proverava MD5 heševe svih fajlova iz `manifest.json`. Opcioni `--check-urls` proverava i URL-ove. |
| `verify_and_replace.py` | Ponovo preuzima svaku pesmu sa URL-a, upoređuje MD5, zamenjuje ako se ne poklapa. |
| `recover.py` | Ponovo preuzima samo korumpirane/nedostajuće fajlove. Brži od `verify_and_replace.py`. |
| `fix_urls.py` | Vraća ispravne YouTube/FMA URL-ove u `manifest.json` na osnovu imena fajlova. |
| `restructure.py` | Organizuje MP3 fajlove u žanrovske poddirektorijume (`dataset/Blues/`, `dataset/Jazz/`, ...). |
