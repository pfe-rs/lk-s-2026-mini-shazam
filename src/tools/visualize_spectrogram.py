import argparse
from pathlib import Path

import numpy as np

from pipeline.config import RecognitionPipelineConfig
from pipeline.audio_input import FileInput
from pipeline.preprocessor import Preprocessor
from pipeline.normalizer import FFmpegNormalizer
from pipeline.spectrogram import SpectrogramGenerator


def from_audio(audio_path, config, full=False, save=None):
    import librosa.display
    import matplotlib.pyplot as plt

    audio_input = FileInput()
    preprocessor = Preprocessor(config)
    normalizer = FFmpegNormalizer(config)
    spectrogram_gen = SpectrogramGenerator(config)
    hop_length = config.hop_length

    audio = audio_input.load(audio_path, config.sample_rate)

    if full:
        norm = normalizer.normalize(audio, config.sample_rate)
        spec = spectrogram_gen.generate(norm, config.sample_rate)
        chunks_data = [(spec, "full")]
    else:
        chunks = preprocessor.to_chunks(audio, config.sample_rate)
        chunks_data = []
        for i, chunk in enumerate(chunks):
            norm = normalizer.normalize(chunk, config.sample_rate)
            spec = spectrogram_gen.generate(norm, config.sample_rate)
            chunks_data.append((spec, str(i)))

    for spec, label in chunks_data:
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(spec, sr=config.sample_rate, hop_length=hop_length,
                                 x_axis="time", y_axis="mel", cmap="magma")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"{label} ({Path(audio_path).name})")
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150)
            plt.close()
            print(f"  saved: {save}")
        else:
            plt.show()


def from_cache(npy_path, save=None):
    import librosa.display
    import matplotlib.pyplot as plt

    spec = np.load(npy_path)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(spec, x_axis="time", y_axis="mel", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{Path(npy_path).name}")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
        plt.close()
        print(f"  saved: {save}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Prikazi mel spektrogram")
    parser.add_argument("path", help="put do audio fajla ili .npy fajla")
    parser.add_argument("--save", "-s", help="sacuvaj kao PNG umesto popup")
    parser.add_argument("--full", "-f", action="store_true", help="cela pesma u jednom spektrogramu (bez chunkovanja)")
    args = parser.parse_args()

    path = Path(args.path)
    if path.suffix == ".npy":
        from_cache(str(path), save=args.save)
    else:
        config = RecognitionPipelineConfig()
        from_audio(str(path), config, full=args.full, save=args.save)


if __name__ == "__main__":
    main()
