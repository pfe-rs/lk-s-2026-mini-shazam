import torch
from torch.utils.data import Dataset
import random
from src.training.augmenter import AudioAugmenter
import os
import numpy as np


class TripletDataset(Dataset):
    def __init__(self, data_paths, labels, noise_paths, type_of_dataset):
        self.data_paths = data_paths
        self.labels = labels
        self.noise_paths = noise_paths
        self.type_of_dataset = type_of_dataset
        self.audioAugmenter = AudioAugmenter()

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        file_path = self.data_paths[index]
        label = self.labels[index]

        # Load the numpy array and convert to a PyTorch float tensor
        db_mel_spec = torch.from_numpy(np.load(file_path)).float()

        # --- FIX: Add explicit channel dimension [Height, Width] -> [1, Height, Width] ---
        if db_mel_spec.ndim == 2:
            db_mel_spec = db_mel_spec.unsqueeze(0)

        if self.type_of_dataset == "train":
            # 50% chance of adding background noise
            if random.random() < 0.5:
                db_mel_spec = self.audioAugmenter.add_background_speech(
                    db_mel_spec,
                    random.choice(self.noise_paths),
                    random.randint(-10, 5)  # attenuation in dB
                )
        elif self.type_of_dataset == "test":
            # Test gets noise every time to simulate real-world Shazam usage
            db_mel_spec = self.audioAugmenter.add_background_speech(
                    db_mel_spec,
                    random.choice(self.noise_paths),
                    random.randint(-10, 5)  
                )
        elif self.type_of_dataset == "original":
            pass
        else:
            print("Error: Unknown dataset type, returning original spectrogram.")
            
        return db_mel_spec, label

class UNetDataset(Dataset):
    def __init__(self, data_paths, train_labels, noise_paths, type_of_dataset):
        """
        Args:
            data_paths (list): Paths to the clean, original spectrogram numpy arrays.
            noise_paths (list): Paths to the noise audio/spectrograms.
        """
        self.data_paths = data_paths
        self.train_labels = train_labels
        self.noise_paths = noise_paths
        self.audioAugmenter = AudioAugmenter()
        self.type_of_dataset = type_of_dataset

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        # 1. Load the CLEAN ground-truth spectrogram
        file_path = self.data_paths[index]
        clean_spec = torch.from_numpy(np.load(file_path)).float()

        label = self.train_labels[index]

        # Add explicit channel dimension [Height, Width] -> [1, Height, Width]
        if clean_spec.ndim == 2:
            clean_spec = clean_spec.unsqueeze(0)

        # 2. Generate the NOISY input
        # We clone the clean tensor to ensure the original target is not mutated
        noisy_spec = clean_spec.clone()

        

        # Apply the noise augmentation. 
        if self.type_of_dataset == "train":
            # 90% chance of adding background noise
            if random.random() < 0.9:
                # FIX: Assign the result back to noisy_spec, NOT db_mel_spec!
                noisy_spec = self.audioAugmenter.add_background_speech(
                    noisy_spec,
                    random.choice(self.noise_paths),
                    random.randint(-10, 5)  # attenuation in dB
                )
        elif self.type_of_dataset == "test":
            # Test gets noise every time to simulate real-world Shazam usage
            # FIX: Assign the result back to noisy_spec
            noisy_spec = self.audioAugmenter.add_background_speech(
                    noisy_spec,
                    random.choice(self.noise_paths),
                    random.randint(-10, 5)  
                )
        elif self.type_of_dataset == "original":
            pass
        else:
            print("Error: Unknown dataset type, returning original spectrogram.")

        # 3. CRITICAL FIX: Min-Max Normalization to [0.0, 1.0]
        # We find the absolute min and max across BOTH tensors to ensure 
        # the relative volume difference between clean and noisy is preserved.
        min_val = min(clean_spec.min().item(), noisy_spec.min().item())
        max_val = max(clean_spec.max().item(), noisy_spec.max().item())
        
        # Prevent division by zero just in case the audio is pure dead silence
        if (max_val - min_val) > 1e-6:
            noisy_spec = (noisy_spec - min_val) / (max_val - min_val)
            clean_spec = (clean_spec - min_val) / (max_val - min_val)
            
        # Return the input (noisy), the target label (clean), and the class label
        return noisy_spec, clean_spec, label
class PathMaker:
    def __init__(self):
        pass
    def split_paths(self, spect_dir, test_size=0.2, random_state=42):
        """
        Odvaja zadati procenat celih pesama (foldera) isključivo za test skup,
        dok preostale pesme idu u trening skup.
        """
        train_filenames = []
        train_labels = []
        test_filenames = []
        test_labels = []

        # 1. Pokupi sve validne foldere (pesme)
        sve_pesme = []
        for song in sorted(os.listdir(spect_dir)):
            song_dir = os.path.join(spect_dir, song)
            if os.path.isdir(song_dir) and len(os.listdir(song_dir)) > 0:
                try:
                    # Provera da li je ime foldera validan int za labelu
                    int(song)
                    sve_pesme.append(song)
                except ValueError:
                    print(f"Preskačem folder '{song}' (ime nije int).")
                    continue

        # Postavljanje seed-a radi reproduktivnosti
        if random_state is not None:
            random.seed(random_state)

        # 2. Nasumično mešanje i split foldera (pesama)
        random.shuffle(sve_pesme)
        broj_test_pesama = int(len(sve_pesme) * test_size)
        
        # Ako je procenat premali pa ispadne 0, uzmi bar jednu pesmu za test
        if broj_test_pesama == 0 and len(sve_pesme) > 1:
            broj_test_pesama = 1

        test_pesme_skup = set(sve_pesme[:broj_test_pesama])
        train_pesme_skup = set(sve_pesme[broj_test_pesama:])

        print(f"Ukupno pesama: {len(sve_pesme)} | Za trening: {len(train_pesme_skup)} | Za test: {len(test_pesme_skup)}")

        # 3. Punjenje lista na osnovu splita
        for song in sorted(os.listdir(spect_dir)):
            song_dir = os.path.join(spect_dir, song)
            if not os.path.isdir(song_dir):
                continue
                
            if song in train_pesme_skup:
                for filename in sorted(os.listdir(song_dir)):
                    train_filenames.append(os.path.join(song_dir, filename))
                    train_labels.append(int(song))
                    
            elif song in test_pesme_skup:
                for filename in sorted(os.listdir(song_dir)):
                    test_filenames.append(os.path.join(song_dir, filename))
                    test_labels.append(int(song))

        # Change the return statement to this:
        return train_filenames, train_labels, test_filenames, test_labels, (train_filenames + test_filenames), (train_labels + test_labels) 
       
    def make_paths_from_folder(self, dir):
        noise_names =os.listdir(dir)
        return [os.path.join(dir, noise_name) for noise_name in noise_names]

