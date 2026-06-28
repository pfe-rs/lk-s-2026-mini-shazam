import torch
import numpy as np
import random

class AudioAugmenter:
    def add_white_noise(self) -> None:
        raise NotImplementedError("AudioAugmenter not yet implemented")

    def add_background_speech(
        self,
        clean_spec: torch.Tensor,
        noise_file_path: str,
        noise_reduction_db: float = 15.0,
    ) -> torch.Tensor:
        """
        Overlays a background noise spectrogram onto a clean spectrogram.

        clean_spec: Tensor of shape [1, M, T] or [M, T]
        """

        noise_spec = torch.from_numpy(np.load(noise_file_path)).float()

        squeeze_channel = False
        if clean_spec.ndim == 3:
            clean_spec = clean_spec.squeeze(0)
            squeeze_channel = True

        if noise_spec.ndim == 3:
            noise_spec = noise_spec.squeeze(0)

        target_time = clean_spec.shape[1]

        # Loop if necessary
        if noise_spec.shape[1] < target_time:
            repeats = (target_time + noise_spec.shape[1] - 1) // noise_spec.shape[1]
            noise_spec = noise_spec.repeat(1, repeats)

        # Trim
        random_start = random.randint(0,noise_spec.shape[1]-target_time)
        noise_spec = noise_spec[:, random_start:target_time+random_start]

        # Lower noise level
        noise_spec = noise_spec - noise_reduction_db

        # Mix
        # Mix (Physically accurate approach)
        # 1. Prebaci iz dB nazad u linearnu skalu energije
        clean_linear = 10 ** (clean_spec / 10)
        noise_linear = 10 ** (noise_spec / 10)
        
        # 2. Saberi prave energije
        mixed_linear = clean_linear + noise_linear
        
        # 3. Vrati miks nazad u decibele
        noisy_spec = 10 * torch.log10(mixed_linear)

        if squeeze_channel:
            noisy_spec = noisy_spec.unsqueeze(0)

        return noisy_spec

    def add_reverb(self) -> None:
        raise NotImplementedError("AudioAugmenter not yet implemented")
