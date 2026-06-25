import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from pipeline.audio_input import FileInput
from pipeline.preprocessor import Preprocessor
from pipeline.normalizer import FFmpegNormalizer
from pipeline.spectrogram import SpectrogramGenerator
from pipeline.config import RecognitionPipelineConfig


def precompute(config, manifest, cache_dir, dataset_dir="", resume=False, blacklist=None):
    audio_input = FileInput()
    preprocessor = Preprocessor(config)
    normalizer = FFmpegNormalizer(config)
    spectrogram_gen = SpectrogramGenerator(config)

    cache_dir = Path(cache_dir)
    dataset_dir = Path(dataset_dir) if dataset_dir else Path()
    total = len(manifest)
    skipped = 0
    done = 0
    failed = []

    for idx, (song_id, info) in enumerate(manifest.items()):
        rel_path = info.get("path") or info.get("filename", "")
        if not rel_path:
            continue
        path = str(dataset_dir / rel_path)

        if blacklist and rel_path in blacklist:
            skipped += 1
            print(f"[{idx+1}/{total}] {song_id} | {info.get('title', '?')} | crna lista")
            continue

        song_cache = cache_dir / song_id
        if resume and song_cache.exists():
            existing = list(song_cache.glob("*.npy"))
            if existing:
                skipped += 1
                print(f"[{idx+1}/{total}] {song_id} | {info.get('title', '?')} | kesiran")
                continue

        print(f"[{idx+1}/{total}] {song_id} | {info.get('title', '?')}")

        try:
            audio = audio_input.load(path, config.sample_rate)
        except Exception as e:
            print(f"  GRESKA: ne mogu da ucitam audio | {e}", file=sys.stderr)
            failed.append(song_id)
            continue

        chunks = preprocessor.to_chunks(audio, config.sample_rate)
        song_cache.mkdir(parents=True, exist_ok=True)

        for i, chunk in enumerate(chunks):
            cache_path = song_cache / f"{i}.npy"
            if cache_path.exists():
                continue

            try:
                norm = normalizer.normalize(chunk, config.sample_rate)
                spec = spectrogram_gen.generate(norm, config.sample_rate)
                np.save(str(cache_path), spec)
            except Exception as e:
                print(f"  GRESKA chunk {i}: {e}", file=sys.stderr)
                continue

        done += 1

    print(f"\nGotovo: {done} pesama obradjeno, {skipped} preskoceno, {len(failed)} neuspesno")
    if failed:
        print(f"Neuspesne: {', '.join(failed)}")


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
    warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
    parser = argparse.ArgumentParser(description="Precompute mel spectrograms for all songs")
    parser.add_argument("manifest", help="put do manifest.json")
    parser.add_argument("--cache", default="cache/spectrograms", help="dir za kes (default: cache/spectrograms)")
    parser.add_argument("--dataset-dir", default="", help="put do dataset direktorijuma (default: trazi dataset/ pored manifest.json)")
    parser.add_argument("--resume", action="store_true", help="nastavi gde je stalo (preskoci kesirane)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)

    if not args.dataset_dir:
        candidate = manifest_path.parent / "dataset"
        if candidate.is_dir():
            args.dataset_dir = str(candidate)
            print(f"Automatski pronadjen dataset: {candidate}")

    config = RecognitionPipelineConfig()

    blacklist_path = manifest_path.parent / "corrupted_blacklist.json"
    blacklist = None
    if blacklist_path.exists():
        blacklist = set(json.load(blacklist_path.open()))
        print(f"Ucitan blacklist: {len(blacklist)} korumpiranih fajlova")

    precompute(config, manifest, args.cache, dataset_dir=args.dataset_dir,
               resume=args.resume, blacklist=blacklist)


if __name__ == "__main__":
    main()
