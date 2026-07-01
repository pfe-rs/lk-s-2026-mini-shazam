import torch
import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import torch.nn.functional as F
from src.training.evaluaator import Evaluator

import torch
import torch.nn.functional as F

import numpy as np
import scipy.ndimage as ndimage
import multiprocessing
import concurrent.futures


class AudioResNet(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        
        # 1. Load the pre-trained ResNet-18 (Transfer Learning)
        # Using pre-trained weights gives the model a massive head start in recognizing audio textures
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 2. MODIFY INPUT: Change the first layer to accept 1-Channel audio instead of 3-Channel RGB
        # We keep the exact same kernel size and stride as the original model
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )
        
        # 3. MODIFY OUTPUT: Replace the final Fully Connected (fc) layer
        # Standard ResNet outputs 1000 categories. We need it to output our 128-number embedding.
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, embedding_size)

    def forward(self, x):
        # Pass the 2D spectrogram tensor through the ResNet backbone
        embeddings = self.backbone(x)
        
        # CRITICAL FOR TRIPLET LOSS: L2 Normalize the embeddings
        # This forces all audio clips to sit on the surface of a perfect mathematical sphere,
        # making the distance calculations highly accurate.
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return normalized_embeddings
    
    def fit(
        self, 
        dataloader, 
        optimizer, 
        miner, 
        loss_func, 
        device, 
        model_dir, 
        num_epochs=20,
        eval_set = None
    ) -> None:
        """
        Pokreće trening petlju za Metric Learning model koristeći triplet miner i loss funkciju.
        Sve težine se čuvaju na kraju svake epohe u model_dir.
        """
        # Osiguraj da direktorijum za čuvanje modela postoji
        os.makedirs(model_dir, exist_ok=True)

        for epoch in range(num_epochs):
            print(f"\n=== EPOCH {epoch+1}/{num_epochs} ===")
            
            # BEST PRACTICE: Prebacivanje modela u train režim na početku epohe
            self.train()
            
            # Inicijalizacija tqdm trake napretka
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            
            for batch in loop:
                # 1. Prebacivanje podataka na odgovarajući uređaj (GPU/CPU)
                spectrograms, labels_batch = batch
                spectrograms, labels_batch = spectrograms.to(device), labels_batch.to(device)
                
                # Resetovanje gradijenata
                optimizer.zero_grad()

                # 2. Forward pass kroz model da se dobiju embedding-zi
                embeddings = self(spectrograms)

                # 3. NINE-LINE MAGIC: Miner pronalazi najteže triplete u ovom batch-u
                indices_tuple = miner(embeddings, labels_batch)

                # 4. Računanje loss-a koristeći samo te teške triplete
                loss = loss_func(embeddings, labels_batch, indices_tuple)

                # 5. Backpropagation i korak optimizatora
                loss.backward()
                optimizer.step()

                # Dinamičko ažuriranje parametara u tqdm ispisu
                loop.set_postfix(Loss=loss.item(), Triplets=miner.num_triplets)    

            if ((epoch+1)%10 == 0 or num_epochs<5) and eval_set is not None:
                print(f"Testing on epoch: {epoch+1}")
                self.eval()
                
                # Create a fresh evaluator to ensure no memory bleed from previous tests
                evaluator = Evaluator() 
                
                with torch.no_grad():
                    evaluator.evaluate(
                        model = self,
                        noisy_dataloader = eval_set[0],
                        original_dataloader = eval_set[1],
                        device = device
                    )
                print(evaluator.generate_report_card(k_values=[1, 5]))

            # BUG FIX: Čuvanje stanja modela u namenski direktorijum
            save_path = os.path.join(model_dir, f"audio_resnet_epoch_{epoch+1+self.model_num}.pth")
            torch.save(self.state_dict(), save_path)
            print(f"Model uspešno sačuvan na: {save_path}")
    def load_params(
        self,
        MODEL_DIR,
        model_num
    ):
        self.load_state_dict(
            torch.load(os.path.join(MODEL_DIR,f"audio_resnet_epoch_{model_num}.pth"),weights_only=True)
            )
        self.model_num = model_num
    def create_database_matrix(self, database_dataloader, device):
        self.eval()
        all_database_embeddings = []
        all_database_labels = []

        with torch.no_grad():
            for spectrograms, labels_batch in database_dataloader:
                spectrograms = spectrograms.to(device)
                
                # Generisanje embedding-a (npr. 128-dimenzionalni vektor)
                embeddings = self(spectrograms)
                
                all_database_embeddings.append(embeddings.cpu())
                all_database_labels.extend(labels_batch.tolist())

        # Pakovanje u jednu veliku matricu [Total_Chunks, 128]
        database_matrix = torch.cat(all_database_embeddings, dim=0)
        print(f"Baza uspešno izgrađena! Sačuvano {database_matrix.shape[0]} jedinstvenih otisaka.")
        
        return database_matrix, all_database_labels

import torch
import torch.nn as nn

class AudioUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Channel scaling optimized to hit ~11.5M total parameters
        c1, c2, c3, c4 = 64, 128, 256, 512
        
        # --- ENCODER (Downsampling Pathway) ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- CONVOLUTIONAL BOTTLENECK (Fully Parallel) ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # --- DECODER (Upsampling Pathway + Skip Connections) ---
        self.up1 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c4, c3, kernel_size=3, padding=1, bias=False),  # c3 (up) + c3 (skip) = c4
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c3, c2, kernel_size=3, padding=1, bias=False),  # c2 (up) + c2 (skip) = c2 * 2
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, kernel_size=3, padding=1, bias=False),  # c1 (up) + c1 (skip) = c1 * 2
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        
        # --- OUTPUT MASK LAYER ---
        self.mask_conv = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, noisy_spectrogram):
        
        # 1. Encoder Path (Save states for Skip Connections)
        s1 = self.enc1(noisy_spectrogram)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        
        # 2. Parallel Bottleneck
        b = self.bottleneck(self.pool3(s3))
        
        # 3. Decoder Path with Tensor Concatenation
        x = self.up1(b)
        x = torch.cat([x, s3], dim=1)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)
        
        x = self.up3(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec3(x)
        
        # 4. Mask Generation and Subtractive T-F Filtering
        mask = self.sigmoid(self.mask_conv(x))
        denoised_spectrogram = noisy_spectrogram * mask
        
        return denoised_spectrogram, mask
    def fit(
        self, 
        dataloader, 
        optimizer, 
        loss_func, 
        scheduler,
        device, 
        model_dir, 
        num_epochs=20,
        eval_set=None
    ) -> None:
        import os
        from tqdm import tqdm
        import torch

        # Osiguraj da direktorijum za čuvanje modela postoji
        os.makedirs(model_dir, exist_ok=True)

        for epoch in range(num_epochs):
            running_loss = 0.0

            print(f"\n=== EPOCH {epoch+1}/{num_epochs} ===")
            
            # BEST PRACTICE: Prebacivanje modela u train režim na početku epohe
            self.train()
            
            # Inicijalizacija tqdm trake napretka za trening
            train_loop = tqdm(dataloader, desc=f"Train Epoch {epoch+1}")
            
            for batch_idx, (noisy_spec, clean_spec, _) in enumerate(train_loop):
                # 1. Move tensors to GPU
                noisy_spec = noisy_spec.to(device)
                clean_spec = clean_spec.to(device)
                
                # 2. Zero the gradients
                optimizer.zero_grad()
                
                # 3. Forward pass
                denoised_spec, mask = self(noisy_spec)
                
                # 4. Calculate Loss
                loss = loss_func(denoised_spec, clean_spec)
                
                # 5. Backward pass and Optimization
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Update progress bar
                train_loop.set_postfix(Loss=f"{loss.item():.4f}")
                            
            scheduler.step()
    
            # Calculate average epoch train loss
            epoch_loss = running_loss / len(dataloader)
            print(f"--- Train Epoch {epoch+1} Completed | Average Loss: {epoch_loss:.4f} ---")

            # # ==========================================
            # # EVALUATION PHASE
            # # ==========================================
            # if eval_set is not None:
            #     # 1. Switch to evaluation mode (freezes Dropout/BatchNorm)
            #     self.eval()
            #     val_running_loss = 0.0
                
            #     # 2. Disable gradient engine to save GPU memory and speed up compute
            #     with torch.no_grad():
            #         val_loop = tqdm(eval_set, desc=f"Eval Epoch  {epoch+1}")
                    
            #         for noisy_spec, clean_spec, _ in val_loop:
            #             noisy_spec = noisy_spec.to(device)
            #             clean_spec = clean_spec.to(device)
                        
            #             # Forward pass ONLY
            #             denoised_spec, mask = self(noisy_spec)
            #             val_loss = loss_func(denoised_spec, clean_spec)
                        
            #             val_running_loss += val_loss.item()
            #             val_loop.set_postfix(Val_Loss=f"{val_loss.item():.4f}")
                
            #     # Calculate average epoch validation loss
            #     epoch_val_loss = val_running_loss / len(eval_set)
            #     print(f"--- Eval Epoch {epoch+1} Completed | Average Val Loss: {epoch_val_loss:.4f} ---")
            # # ==========================================

            # Save logic for U-Net
            save_path = os.path.join(model_dir, f"audio_unet_epoch_{epoch+1+self.model_num}.pth")
            torch.save(self.state_dict(), save_path)
            print(f"Model uspešno sačuvan na: {save_path}")

            
                # Dinamičko ažuriranje parametara u tqdm ispisu
            # if ((epoch+1)%10 == 0 or num_epochs<5) and eval_set is not None:
            #     print(f"Testing on epoch: {epoch+1}")
            #     self.eval()
                
            #     # Create a fresh evaluator to ensure no memory bleed from previous tests
            #     evaluator = Evaluator() 
                
            #     with torch.no_grad():
            #         evaluator.evaluate(
            #             model = self,
            #             noisy_dataloader = eval_set[0],
            #             original_dataloader = eval_set[1],
            #             device = device
            #         )
            #     print(evaluator.generate_report_card(k_values=[1, 5]))

            # BUG FIX: Čuvanje stanja modela u namenski direktorijum
            # save_path = os.path.join(model_dir, f"audio_resnet_epoch_{epoch+1+self.model_num}.pth")
            # torch.save(self.state_dict(), save_path)
            # print(f"Model uspešno sačuvan na: {save_path}")
    def load_params(
        self,
        MODEL_DIR,
        model_num
    ):
        self.load_state_dict(
            torch.load(os.path.join(MODEL_DIR,f"audio_unet_epoch_{model_num}.pth"),weights_only=True)
            )
        self.model_num = model_num
    
    def extract_peaks(self,batch_spectrograms, neighborhood_size=(16,16), threshold=0.25):
        """
        Runs the 2D sliding window entirely on CUDA using PyTorch.
        batch_spectrograms shape: (batch_size, 1, 128, 216)
        """
        # 1. Run a Max Pool with stride=1 to act as a sliding window
        # Padding ensures the output matrix is the exact same size as the input
        pad_h = neighborhood_size[0] // 2
        pad_w = neighborhood_size[1] // 2
        
        local_max = F.max_pool2d(
            batch_spectrograms, 
            kernel_size=neighborhood_size, 
            stride=1, 
            padding=(pad_h, pad_w)
        )
        
        # Because of padding, the dimensions might be off by 1 pixel depending on even/odd sizes,
        # so we explicitly slice it back to match the original size perfectly
        local_max = local_max[:, :, :batch_spectrograms.shape[2], :batch_spectrograms.shape[3]]
        
        # 2. Find where the original tensor matches the max pooled tensor
        peak_mask = (batch_spectrograms == local_max)
        threshold_mask = (batch_spectrograms > threshold)
        
        final_mask = peak_mask & threshold_mask
        
        # 3. Pull the sparse coordinates off the GPU to the CPU for hashing
        all_peaks = torch.nonzero(final_mask).cpu().numpy()
        
        # (From here, you group them by batch_idx exactly like the CPU version)
        batch_size = batch_spectrograms.shape[0]
        batch_peaks = []
        
        for b in range(batch_size):
            b_peaks = all_peaks[all_peaks[:, 0] == b]
            ft_peaks = b_peaks[:, 2:4]
            
            if len(ft_peaks) > 0:
                ft_peaks = ft_peaks[ft_peaks[:, 1].argsort()]
                
            batch_peaks.append(ft_peaks)
            
        return batch_peaks
    def generate_hashes(self, batch_peaks, fan_value=10, min_time_delta=1, max_time_delta=40, max_freq_delta=30):
        """
        Standard, single-threaded combinatorial hashing for a batch of peaks.
        """
        all_hashes = []
        
        if not batch_peaks:
            return all_hashes

        # Loop through every song in the batch sequentially
        for batch_idx, peaks in enumerate(batch_peaks):
            num_peaks = len(peaks)
            
            # Combinatorial pairing for this specific song
            for i in range(num_peaks):
                f_anchor, t_anchor = peaks[i]
                connections_made = 0
                
                for j in range(i + 1, num_peaks):
                    f_target, t_target = peaks[j]
                    t_delta = t_target - t_anchor
                    
                    if t_delta > max_time_delta:
                        break
                        
                    if t_delta >= min_time_delta and abs(f_target - f_anchor) <= max_freq_delta:
                        hash_signature = f"{f_anchor}|{f_target}|{t_delta}"
                        
                        all_hashes.append({
                            "batch_idx": batch_idx,
                            "hash": hash_signature,
                            "anchor_time": t_anchor
                        })
                        
                        connections_made += 1
                        if connections_made >= fan_value:
                            break
                            
        return all_hashes
        
    def create_database_hashes(
        self, 
        dataloader, 
        device, 
        neighborhood_size = (16,16), 
        threshold=0.25, 
        fan_value=10, 
        min_time_delta=1, 
        max_time_delta=40, 
        max_freq_delta=30
    ):
        self.eval()
        database_hashes = []
        original_hashes = []
        all_database_labels = []

        global_track_id = 0
        # spec_sopis = []
        # orig_sopis = []
        # recreated_sopis = []
        with torch.no_grad():
            for spectrograms, originals,labels_batch in tqdm(dataloader):
                spectrograms = spectrograms.to(device)
                # Generisanje embedding-a (npr. 128-dimenzionalni vektor)

                
                #SAMO RADI TESTA
                embeddings,masks = self(spectrograms)
                # embeddings = spectrograms
                
                noisy_batch_hashes = self.generate_hashes(
                    self.extract_peaks(
                        embeddings,
                        neighborhood_size = neighborhood_size,
                        threshold=threshold 
                    ),
                    fan_value=fan_value,
                    min_time_delta=min_time_delta,
                    max_time_delta=max_time_delta,
                    max_freq_delta=max_freq_delta
                )
                
                original_batch_hashes = self.generate_hashes(
                    self.extract_peaks(
                        originals,
                        neighborhood_size = neighborhood_size,
                        threshold=threshold 
                    ),
                    fan_value=fan_value,
                    min_time_delta=min_time_delta,
                    max_time_delta=max_time_delta,
                    max_freq_delta=max_freq_delta
                )

                
                # --- THE OVERWRITE FIX (Optimized) ---
                batch_size = spectrograms.size(0)
                
                # Iterate through the hashes exactly ONCE
                for h in noisy_batch_hashes:
                    h['batch_idx'] += global_track_id
                    
                for h in original_batch_hashes:
                    h['batch_idx'] += global_track_id
                        
                # Push the global ID forward for the next PyTorch batch
                global_track_id += batch_size

                # Append to main lists
                database_hashes.extend(noisy_batch_hashes)
                original_hashes.extend(original_batch_hashes)
                all_database_labels.extend(labels_batch.tolist())

                # spec_sopis.extend(spectrograms)
                # orig_sopis.extend(originals)
                # recreated_sopis.extend(embeddings)





        # # --- DEBUGGING: Analyze Track 0 ---
        
        # # 1. Filter the lists to grab only items where batch_idx == 0
        # track_0_noisy = [h for h in database_hashes if h['batch_idx'] == 0]
        # track_0_original = [h for h in original_hashes if h['batch_idx'] == 0]
        
        # # 2. Grab the corresponding label (Index 0)
        # track_0_label = all_database_labels[0]
        
        # # 3. Print the results clearly
        # print(f"\n========================================")
        # print(f"      DEBUG ANALYSIS FOR TRACK 0        ")
        # print(f"         Label/ID: {track_0_label}      ")
        # print(f"========================================")
        
        # print(f"\n--- Noisy Hashes from UNet (Total: {len(track_0_noisy)}) ---")
        # for h in track_0_noisy[:10]:
        #     print(h)
            
        # print(f"\n--- Original Clean Hashes (Total: {len(track_0_original)}) ---")
        # for h in track_0_original[:10]:
        #     print(h)

        # # --- PARSE HASHES INTO PLOTTABLE COORDINATES ---
        # def get_unique_peaks(hash_list):
        #     peaks = set()
        #     for h in hash_list:
        #         # Parse the string 'f_anchor|f_target|t_delta'
        #         f_anchor, f_target, t_delta = map(int, h['hash'].split('|'))
        #         t_anchor = int(h['anchor_time'])
        #         t_target = t_anchor + t_delta
                
        #         # Add both the anchor and the target to our set of points
        #         peaks.add((t_anchor, f_anchor))
        #         peaks.add((t_target, f_target))
        #     return peaks
            
        # noisy_peaks = get_unique_peaks(track_0_noisy)
        # original_peaks = get_unique_peaks(track_0_original)

        # # --- VISUALIZATION ---
        # import seaborn as sns
        # import matplotlib.pyplot as plt

        # fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # # --- Plot 1: Noisy Input ---
        # sns.heatmap(spec_sopis[0][0].cpu(), ax=axes[0], cmap="magma", cbar=True)
        # axes[0].set_title("Noisy Input Spectrogram", fontsize=14)
        # axes[0].invert_yaxis()  
        # axes[0].set_xlabel("Time Frames")
        # axes[0].set_ylabel("Frequency Bins")
        
        # # --- Plot 2: Clean Original (With Overlaid Hashes) ---
        # sns.heatmap(orig_sopis[0][0].cpu(), ax=axes[1], cmap="magma", cbar=True)
        
        # # Plot Original Peaks
        # if original_peaks:
        #     t_orig, f_orig = zip(*original_peaks)
        #     # We add 0.5 so the marker sits perfectly in the center of the seaborn cell
        #     axes[1].scatter([t + 0.5 for t in t_orig], [f + 0.5 for f in f_orig], 
        #                     color='cyan', marker='x', s=30, label='Extracted Peaks')
        #     axes[1].legend(loc="upper right")
            
        # axes[1].set_title("Clean Original Spectrogram", fontsize=14)
        # axes[1].invert_yaxis()
        # axes[1].set_xlabel("Time Frames")
        # axes[1].set_ylabel("Frequency Bins")
        
        # # --- Plot 3: Output Recreated (With Overlaid Hashes) ---
        # sns.heatmap(recreated_sopis[0][0].cpu(), ax=axes[2], cmap="magma", cbar=True)
        
        # # Plot Noisy/UNet Peaks
        # if noisy_peaks:
        #     t_noisy, f_noisy = zip(*noisy_peaks)
        #     axes[2].scatter([t + 0.5 for t in t_noisy], [f + 0.5 for f in f_noisy], 
        #                     color='cyan', marker='x', s=30, label='Extracted Peaks')
        #     axes[2].legend(loc="upper right")
            
        # axes[2].set_title("Output Recreated Spectrogram", fontsize=14) 
        # axes[2].invert_yaxis()                                         
        # axes[2].set_xlabel("Time Frames")                              
        # axes[2].set_ylabel("Frequency Bins")                           
        
        # plt.tight_layout()
        # plt.show()

        # print(f"========================================\n")
        # print(f"Baza uspešno izgrađena! Sačuvano {len(database_hashes)} jedinstvenih hashova.")

        # (Optional) Exit the script early so you can read the printout and see the plot
        # import sys; sys.exit()
        return database_hashes, original_hashes, all_database_labels
    
    def build_and_save_index(self, 
        original_dataloader, 
        device, 
        save_path="compressed_reverse_index.pkl",
        neighborhood_size = (16,16), 
        threshold=0.25, 
        fan_value=10, 
        min_time_delta=1, 
        max_time_delta=40, 
        max_freq_delta=30,
        MAX_SONGS_PER_HASH = 100
    ):
        import pickle
        import os
        from collections import defaultdict
        from tqdm import tqdm

        print("\n--- 1. Extracting Raw Hashes (This takes a while) ---")
        _, original_matrix, original_labels = self.create_database_hashes(
            original_dataloader, 
            device, 
            neighborhood_size = (16,16), 
            threshold=0.25, 
            fan_value=10, 
            min_time_delta=1, 
            max_time_delta=40, 
            max_freq_delta=30
        )

        print("\n--- 2. Building Reverse Index ---")
        print("\n--- 2. Building Reverse Index (With Time Anchors) ---")
        reverse_index = defaultdict(list)
        for orig in tqdm(original_matrix, desc="Indexing"):
            track_idx = orig['batch_idx']
            original_id = original_labels[track_idx]
            t_anchor = orig['anchor_time']
            
            # Umesto samo ID-ja, čuvamo torku (id_pesme, vreme_sidra)
            reverse_index[orig['hash']].append((original_id, t_anchor))

        print("\n--- 3. Applying Guillotine (MAX_SONGS_PER_HASH = 10) ---")
        keys_to_delete = [
            h for h, song_list in reverse_index.items() 
            if len(set(song_list)) > MAX_SONGS_PER_HASH
        ]
        
        for h in keys_to_delete:
            del reverse_index[h]
            
        print(f"Deleted {len(keys_to_delete)} noisy hashes.")

        print("\n--- 4. Saving Compressed Index to Disk ---")
        # Pro Tip: Convert defaultdict to a standard dict before pickling to avoid import errors later
        with open(save_path, "wb") as f:
            pickle.dump(dict(reverse_index), f)
            
        print(f"Success! Saved ultra-lightweight index to {save_path}. You can now run evaluation!")
class HashingDenoisingLoss(nn.Module):
    """
    Custom Loss for Audio Fingerprinting tasks.
    Heavily weights L1 Loss to preserve sharp, high-contrast structural peaks,
    while using a tiny bit of MSE to keep overall energy levels stable.
    """
    def __init__(self, l1_weight=1.0, mse_weight=0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight

    def forward(self, denoised_pred, clean_target):
        l1 = self.l1_loss(denoised_pred, clean_target)
        mse = self.mse_loss(denoised_pred, clean_target)
        
        return (self.l1_weight * l1) + (self.mse_weight * mse)

