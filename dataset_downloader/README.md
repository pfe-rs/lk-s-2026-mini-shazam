# Mini-Shazam: Dataset Downloader

Skripte za preuzimanje i verifikaciju FMA dataseta.

## Dataset

FMA v1.0 (Jul 2024): Free Music Archive. 
Metapodaci su na https://os.unil.cloud.switch.ch/fma/fma_metadata.zip. 
Audio podskupovi su fma_small.zip, fma_medium.zip, fma_full.zip.
Izvor je https://os.unil.cloud.switch.ch/fma/. 
MD5 fma_metadata.zip je d3ebfd86e283345ee2366a5492495935.

Dataset se ne cuva u repozitorijumu. 
Pokretanjem select_tracks.py 
(opciono, vec postoji selected_tracks.csv i manifest.json) 
pa download_dataset.py reprodukuje se identican skup od 
1000 pesama.

## Podesavanje

Za Linux i macOS pokreni `./setup.sh` pa `source venv/bin/activate`. 
Za Windows PowerShell pokreni `.\setup.ps1` pa `.\venv\Scripts\Activate.ps1`. 
Zatim pokreni `download_dataset.py` da preuzmes pesme.

## Skripte

### select_tracks.py

Selekcija pesama. 
Bira 1000 pesama iz FMA metapodataka stratifikovano 
po zanru sa seed=42. Ulaz je fma_metadata/tracks.csv. 
Izlaz je selected_tracks.csv (1000 redova, 16 zanrova). 
Koristi se samo ako zelis da regenerises selekciju. 
Inace selected_tracks.csv vec postoji u repozitorijumu.

### download_dataset.py

Preuzima svih 1000 pesama. 
Redosled izvora je sledeci. 
Prvo YouTube preko yt-dlp 
(481 pesma, pune duzine, 4-12MB, 2-10min). 
Zatim FMA ZIP arhive 
(204 pesme, puni fajlovi iz FMA medium/full arhiva). 
Na kraju FMA URL preview (223 pesme, 30s isecci, 42KB).

Ubacuje ID3 tagove (ime izvodjaca, naziv pesme, zanr) 
preko mutagen biblioteke. 
Rezultat je manifest.json sa svim URLovima, 
MD5 checksumovima, filenameovima, zanrovima.

Struktura direktorijuma je 
dataset/Blues/000001_Artist_Song_title_hash.mp3, 
dataset/Classical/ i tako dalje.

### verify_dataset.py

Brza verifikacija. 
Proverava MD5 heseve svih fajlova iz manifest.json bez 
ponovnog skidanja. 
Pokreni python verify_dataset.py. 
Opciono dodaj --check-urls da proveri i URLove.

### verify_and_replace.py

Potpuna provera sa ponovnim skidanjem. 
Ponovo preuzima svaku pesmu sa URLa, uporedjuje MD5, 
zamenjuje ako se ne poklapa. Sadrzi 50% size guard. 
Ne zamenjuje fajl ako je download manji od 50% postojeceg fajla.
Ovo sprecava zamenu pune pesme sa 42KB FMA preview iseckom. 
Pokreni `python verify_and_replace.py`. 
Prati napredak u `progress/stats.txt`.

### recover.py

Popravka korumpiranih pesama. 
Ponovo preuzima samo korumpirane i nedostajuce fajlove. 
Mnogo brzi od verify_and_replace.py. 
Pokreni `python recover.py`.

### fix_urls.py

Popravka URLova. 
Vraca ispravne YouTube i FMA URLove u `manifest.json` 
na osnovu imena fajlova. 
YouTube video ID je deo filenamea (11 karaktera). 
Koristi se samo ako su URLovi u `manifest.json` 
iz bilo kog razloga postali netacni.

### restructure.py

Organizacija fajlova. 
Organizuje MP3 fajlove u zanrovske poddirektorijume 
(dataset/Blues/, dataset/Jazz/ i tako dalje). 
Pokreni `python restructure.py`.

### check_corrupted.py

Detekcija korumpiranih fajlova. 
Skenira sve MP3 fajlove u dataset/ kroz librosa.load() 
sa vise radnika (podrazumevano 4). 
Fajlovi koji ne mogu da se ucitaju 
(koruptirani headeri, prazan audio, nedostajuci fajlovi) 
se beleze u `corrupted_blacklist.json`.

Pokreni `python check_corrupted.py -j 8`. 
`precompute.py` (iz `src/tools/`) automatski cita blacklist 
i preskace navedene fajlove.

Trenutno stanje je oko 225 korumpiranih fajlova. 
Uglavnom su FMA Unknown zanr sa NoBackendError. 
Plus jedna trajno nestala YouTube pesma 
(120331, Wrecking Ball, nije ni na YouTubeu ni na FMA).

## Struktura manifesta

`manifest.json` je izvor istine. 
Svaki unos sadrzi filename 
(na primer Blues/000001_Artist_Song_title_hash.mp3), 
source_url (YouTube ili FMA URL), md5 (hash), 
genre (zanr), title (naziv pesme), artist (ime izvodjaca).

## Status dataseta

Ukupno pesama je 1000. MP3 fajlova na disku je 999. 
225 od njih su korumpirani.
Svi unosi u manifestu imaju URL, MD5 i zanr. 
MD5 potvrdjeno 999/999. 
Velicina dataseta je 5.5 GB. 
Trajno neuspelih je 1 (pesma 120331).

## Izvori pesama

YouTube pune pesme: 481, 4-12MB, 2-10min. 
FMA ZIP ekstrakti: 204, 1-10MB, puni fajlovi. 
FMA URL preview: 223, 42KB, 30s isecci. 
Ostali FMA izvori: 91, 1-5MB, varijabilno. 
Trajno neuspelih: 1.

## Distribucija zanrova

Blues 65. 
Classical 65. 
Country 65. 
Easy Listening 24. 
Electronic 65. 
Experimental 65. 
Folk 65. 
Hip-Hop 65. 
Instrumental 65. 
International 65. 
Jazz 65. 
Old-Time/Historic 65. 
Pop 65. 
Rock 66. 
Soul-RnB 65. 
Spoken 65.

## Ceste greske

yt-dlp: Unable to extract video data. YouTube JS challenge se promenio. Azuriraj yt-dlp sa `pip install -U yt-dlp`.

NoBackendError. MP3 fajl ima koruptiran header. Nije moguce lepo ucitati. Dodaj u blacklist ili ponovo skini sa drugog izvora.

MD5 mismatch. Fajl je zamenjen ili ostecen. Pokreni `recover.py` da ga ponovo skines.
