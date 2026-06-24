# Mini-Shazam: Status i istorija odluka

## Sta pravimo

Sistem za prepoznavanje muzike sličan Shazamu. 
Hipoteza je da CNN model treniran sa triplet metric 
learning-om može da generiše bolje audio fingerprintove od 
klasičnog matematičkog pristupa, posebno kada je signal 
degradiran šumom ili lošim snimanjem.

## Repozitorijumi

Postoje dva repozitorijuma koja nisu još spojena.

**mini-shazam-pfe** je ML pipeline prototip. 
Andrej ga je napravio. 
Sadrži kod.ipynb sa 1204 linije koji pokriva ceo 
pipeline od audio fajlova do predikcije. 
Model je ResNet-18 modifikovan za 1-kanalni ulaz sa 128-dim 
L2-normalized embeddingom na izlazu. 
Treniran je na 12 Tame Impala pesama (The Slow Rush). 
Rezultati su 98.68% standard accuracy i 86.03% na stress testu 
sa jakim distorzijama. Pretraga je brute-force torch.cdist. 
FAISS je u requirements.txt ali nije korišćen.

**mini-shazam (za sad lokalni dir)** je data pipeline. 
Lazar ga je napravio. 
Sadrži select_tracks.py koji bira 1000 pesama iz FMA metapodataka 
stratifikovano po žanru sa seed=42. 
Sadrži download_dataset.py koji preuzima pesme via yt-dlp 
sa fallback na FMA zip arhive direktno, ugrađuje ID3 tagove sa 
mutagenom (inace biblioteka a ne ono iz nindza kornjaca) 
i beleži sve u manifest.json. 
Treba da ode na google drive ako mi internet dozvoli danas.

Takodje trebamo da objedinimo jedan centralni repo.

## Dataset

FMA (Free Music Archive). Creative Commons licenca. 
Treniranje na licenciranoj muzici je dozvoljeno za 
istraživačke svrhe po EU DSM Direktivi clan 4 i 
srpskom zakonu o autorskim pravima koji je uskladjen sa EU regulativom.

Za razvoj koristimo podskup od 200 pesama zbog brzine iteracije. 
Za finalno treniranje i evaluaciju koristimo svih 1000.

Šum se generiše veštački tokom treniranja: beli šum, pozadinski govor, 
reverb, kompresija. Ovo nam daje kontrolu nad SNR nivoima za evaluaciju.

Dataset se ne redistribuira. Na GitHub-u dele select_tracks.py, 
download_dataset.py i manifest.json sa source URL-ovima tako da 
svako može da reprodukuje isti dataset pokretanjem skripte.

## Kljucne odluke i zasto

**Zasto CNN a ne matematicka metoda kao jedini pristup**
Matematicka metoda je deterministicka i fiksna. 
CNN sa triplet loss-om uci opstu funkciju slicnosti 
iz primera sa sumom, sto mu potencijalno daje prednost 
u realnim uslovima snimanja. 

**Zasto Flet za UI**
Flutter UI u Pythonu, brzo se dobija lep UI.

**Zasto i kako determinizam**
Objasnjeno u design_doc

## Podela rada

Nije fiksno, trebamo da se cujemo.

Za pocetak:

Lazar: OOP struktura i arhitektura, AudioInput, Preprocessor, 
Database, MathFingerprinter, i dataset priprema.

Andrej Vasiljevic: CNNFingerprinter (ResNet-18, ONNX INT8), 
TripletDataset, training loop, ONNX export i INT8 kvantizacija, 
AudioAugmenter?, UNetDenoiser.

## Sta sledece treba da se uradi (otprilike)

1. Spojiti dva repozitorijuma.
2. Refaktorisati preprocessing iz notebook-a u OOP strukturu.
3. Zameniti torch.cdist sa FAISS-om.
4. Implementirati evaluaciju: Top-k accuracy, MRR, po SNR nivoima.
5. Eksportovati model u ONNX i primeniti INT8 kvantizaciju.
6. Implementirati MathFingerprinter kao baseline.
7. Retrenirati na punom datasetu.
8. UNetDenoiser ako ostane vremena.

## Metrike

Top-k accuracy za k = 1, 3 i 5. 
MRR (Mean Reciprocal Rank) gde je 1. mesto 1.0, 2. mesto 0.5, 
3. mesto 0.33. 
Sve metrike se mere na SNR nivoima: minus 5dB, 0dB, 5dB, 10dB i 20dB. 
Poredimo CNNFingerprinter i MathFingerprinter na svakom nivou.
