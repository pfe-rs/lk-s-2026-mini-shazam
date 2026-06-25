# Progres na pipeline i barebones api 0.1

# Odluke

Moraćemo da bundle-ujemo svoj ffmpeg, da ne bi imali problema
sa različitim vrstama biblioteka na korisničkom sistemu, ovo
je sasvim i okej i vecinom je i standard.

# Directory Structure i kod

Pre svega dir structure u src/:
- pipeline/
- training/ (za Andreja, mogu da se reuse-uju klase i komponente iz pipeline)
- ui/ (kad dodjemo do toga, verovatno nesto Flutter based i hteo bi da imamo IPC)
- deploy/ (bilo bi fino da imamo pyinstaller da bi mogli da oformimo binary umesto da
          pokrecemo skripte svaki put, a relativno je jako mali i lak deo posla)
- tools/ (skripte za skidanje ffmpeg binarnih fajlova)

Takodje posto ima da imamo odvojen kod za aplikaciju i treniranje,
**preporucio bih** da se *NIKAKO* ne importuje training/ u nesto 
iz pipeline/ ili ui/
Ali mozete da importujete bilo šta iz pipeline-a u trening, normalno.

## Zavisnosti (i redosled za review)

### Layer 0
1. config.py: ne zavise ni od cega
2. result.py: ne zavisi ni od cega

### Layer 1
3. audio_input.py, zavisnost librosa za audio, tranzitivno ffmpeg
4. preprocessor.py: zavisnost numpy
5. spectrogram.py: zavisnost librosa za mel spektrograme
6. denoiser.py: NoDenoiser, UNetDenoiser (nije implementirano)
7. registry.py: ModelRegistry (hashlib/json)

### Layer 2, zavisnost na Layer 1
8. normalizer.py: FFmpegNormalizer (zahteva soundfile i subprocess podrsku)
9. fingerprinter.py: AbstractFingerprinter, CNNFingerprinter, MathFingerprinter, za sad numpy i scipy ali verovatno i onnx runtime
10. database.py: LocalFaissDB zavisno na faiss.cpu, ReverseIndexDB zavisno na stdlib pickle za cuvanje objekta

### Layer 3
11. pipeline.py: zavisno na sve ostalo
12. evaluator.py: zavisno na sve ostalo

## src/pipeline: 

### config.py

- [x] RecognitionPipelineConfig

Za sad samo parametri za ffmpeg normalizaciju, 
kao i put do ffmpeg-a i podaci za embedding i top-k.

### result.py

- [x] RecognitionResult
- [x] EvaluationResult

Podaci o uspehu evaluacije, i povratna
informacija prepoznavanja.

### audio_input.py

- [x] AbstractAudioInput:
    - [x] FileInput
    - [ ] MicrophoneInput

FileInput samo ima load metodu koja ucita
zvuk iz bilo kog formata i koristi neki 
backend da ga procita i konvertuje u odgovarajuci
tip PCM (odnosno datatype WAV fajlova).
Pošto ćemo da bundle-ujemo ffmpeg kao dependancy
ovo bi trebalo da radi sasvim okej.

### preprocessor.py

- [x] Preprocessor

Za sad preprocessor samo radi chunk-ovanje
gde iz PCM-a pravi chunko-ove od po 5 sekundi
sa 2.5 sekundi preklapanja.
U buducnosti ovde moze da se doda jos koraka, 
ali bitno je da ostanu pre normalizacije.
Funkcionise na principu sliding window-a.

### spectrogram.py

- [x] SpectrogramGenerator

Za sada smo našli da je librosa optimalan izbor
za ovakav projekat, ali u budućnosti možemo da
probamo i sa drugim backend-ovima za ovaj process.
hop_length je namešten za više preklapanja što je 
korisno za triplet loss.
Može se testirati sa većim n_mels parametrom ali će
to značajno povećati vreme treniranja.
Ostali parametri su namešteni tako da zaobiđu šum 
koliko mogu kao i frekvencije koje nisu korisne za
prepoznavanje i sluh.

### denoiser.py

- [x] AbstractDenoiser:
    - [x] NoDenoiser
    - [ ] UNetDenoiser

Ovo je samo passthrough kako bi došli do sledećeg
dela pipeline-a.

### registry.py

- [x] ModelRegistry

Sluzi kako bi cuvao integritet modela lokalno, 
i kako bi ih klasifikovao, cuva manifest svih modela.

Todo:
Za sada se ne koristi da bi pamtio koji model 
je generisao koje embeddinge u bazi i baze ne
podrzavaju pretrage za specifican model!!!!!

### normalizer.py

- [x] AbstractNormalizer:
    - [x] FFmpegNormalizer

Koristi ffmpeg da bi normalizovao zvuk
prositeći deterministički two-pass loudnorm
odnosno EBU R128, izlaz je isti PCM ali 
normalizovan.

Takođe uzima bundle-ovan ffmpeg i razresava ga
u runtime-u za Windows i za Linux.

### fingerprinter.py

- [x] AbstractFingerprinter:
    - [-] CNNFingerprinter
    - [-] MathFingerprinter

Ovde je samo implementiran apsolutni minimum koji 
definiše kako će se ovi modeli ponašati.
Za CNN je napravljena prosta funkcija da kvantizuje 
vektore na INT8 koji bi dali da je interna baza 
deterministička. (trenutno se ne koristi nigde)
Za neki veći training dataset bi trebali da povećamo
preciznost praktično svakog vektora.
merge_chunk_scores samo uzima najmanji L2 distance 
što je bolje.
I ako je najbolji chunk daleko bolji od drugog
confidence je veći.
Sve ovo je otvoreno promenama, verovatno će Andrej
da nadje bolja rešenja za sve ovo i izmeni.

Za MathFingerprinter su parametri 
(PEAK_NEIGHBORHOOD, FANOUT, prag, DT opseg) 
direktno prekopirani iz originalnog Shazam rada iz 2003. 
koji je radio na običnom spektrogramu, ne na mel skali. 
Ovi brojevi gotovo sigurno nisu optimalni za naš mel 
spektrogram i treba ih istuningovati eksperimentalno.
Takodje nema podršku za vremensko poklapanje hash parova
što je ključno.
Sve će se garantovano promeniti.

### database.py

Verovatno najgori i najeksperimentalniji deo
koji imamo za sada.

- [x] AbstractVectorDatabase:
    - [-] LocalFaissDB
- [x] AbstractHashDatabase:
    - [-] ReverseIndexDB

Obe baze nemaju proveru koji model je napravio koji 
embedding.
LocalFaissDB je samo tanak wrapper oko faiss.IndexFlatL2,
uopšte nije kompletno rešenje i nije skalabino za veće
baze. 
Koristi dva odvojena fajla za svoju bazu...
ReverseIndexDB je običan python dict (koji radi kao
hashset), koji serializujem i deserializujem pomoću
pickle-a, mislim da ne moram da objašnjavam zašto je
ovo užasno rešenje i da se treba doraditi na njemu.

### evaluator.py

- [ ] Evaluator

Ovde ništa ni nema.

## ui/

Samo moja zasadašnja ideja koncept za UI, nije trenutno
od značaja

## training/

Samo prazne klase, možete sve da promenite kako hoćete
i ako hoćete.

## deploy

Takodje samo ideja za kako bih package-ovao aplikaciju
na kraju, ali i to treba jos rada od moje strane.

# Pomoćne skripte

Napisao sam dve skripte u powershell i bash da bi
mogli da skinete iste ffmpeg lib binaries jer se takvi
fajlovi inače ne trebaju čuvati u git-u.

# Testovi i CI/CD

Oko 50 testova gde je svaki relativno prost ali
pokriva bazičnu funkcionalnost za svaki modul u pipeline-u.
Pokreću se sa pytest tests/ (i -m "not ffmpeg" ako niste skinuli).

CI za pull requestove i testira python 3.11, 3.12,
i 3.13 na ubuntu-latest.

CI pokrecem lokalno kroz [ACT projekat](https://github.com/nektos/act),

# Dodatni notes

Samo bi hteo da dodam još stvari iz typing biblioteke
za OOP ali to trenutno ništa ne radi i nema funkciju
već će samo ulepšati kod.
