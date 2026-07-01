import os
from src.pipeline.config import Datapaths

from pathlib import Path
import numpy as np
import random 
from tqdm import tqdm
import torch

from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import torch.optim as optim
from pytorch_metric_learning import losses, miners
import torch.nn as nn

from src.training.dataset import PathMaker
from src.training.evaluaator import Evaluator

from src.training.dataset import UNetDataset
from src.training.modelClass import AudioUNet, HashingDenoisingLoss
from src.training.evaluaator import EvaluatorUNetMath



class TrainingPipelineUNet:
    def __init__(self,batch_size=16):
        paths = Datapaths()
        self.ROOT_DIR=paths.ROOT_DIR
        self.SPECT_DIR=paths.SPECT_DIR
        self.NOISE_DIR=paths.NOISE_DIR
        self.MODEL_DIR=paths.MODEL_DIR

        ROOT_DIR=self.ROOT_DIR
        SPECT_DIR=self.SPECT_DIR
        NOISE_DIR=self.NOISE_DIR
        MODEL_DIR=self.MODEL_DIR
        
        torch.cuda.set_per_process_memory_fraction(0.18)  # ~4 GB out of 24 GB
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        spliter = PathMaker()
        train_filenames, train_labels, test_filenames, test_labels, original_filenames, original_labels = spliter.split_paths(
            SPECT_DIR, 
            test_size=0.2
            )
        noise_filenames = spliter.make_paths_from_folder(
            NOISE_DIR
        )

        train_dataset = UNetDataset(
            train_filenames,
            train_labels,
            noise_paths=noise_filenames,
            type_of_dataset='train'
        )
        test_dataset = UNetDataset(
            test_filenames,
            test_labels,
            noise_paths=noise_filenames,
            type_of_dataset='test'
        )
        original_dataset = UNetDataset(
            original_filenames,
            original_labels,
            noise_paths=noise_filenames,
            type_of_dataset='original'
        )
    

        

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size,       # Number of audio files to process simultaneously
            shuffle=True,        # Mixes up the order every epoch so the model learns features, not order
            drop_last=True       # Trims the last batch if it doesn't perfectly divide by 64
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size,       # Number of audio files to process simultaneously
            shuffle=False,        # Mixes up the order every epoch so the model learns features, not order
        )
        original_dataloader = DataLoader(
            original_dataset, 
            batch_size=batch_size,       # Number of audio files to process simultaneously
            shuffle=True,        # Mixes up the order every epoch so the model learns features, not order
        )
        
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.original_dataloader = original_dataloader

        self.original_filenames = original_filenames
        self.original_labels = original_labels
    def train_unet(self, train_epochs =0, model_num = 0) -> None:
        original_filenames = self.original_filenames
        original_labels = self.original_labels
        ROOT_DIR=self.ROOT_DIR
        SPECT_DIR=self.SPECT_DIR
        NOISE_DIR=self.NOISE_DIR
        MODEL_DIR=self.MODEL_DIR

        device = self.device
        
        train_dataloader = self.train_dataloader
        test_dataloader = self.test_dataloader

        
        train_model = AudioUNet().to(device)
        train_model.model_num = model_num

        if model_num>0:
            train_model.load_params(
                MODEL_DIR,
                model_num
            )
        
        dummy_spectrograms = torch.randn(2, 1, 128, 216).to(device)
            
        # Run the fake data through the model
        # What you need:
        denoised_output, mask_output = train_model(dummy_spectrograms)        
        
        print("--- Model Architecture Test ---")
        print(f"Input Spectrogram Shape:  {dummy_spectrograms.shape}")
        print("--- Model Architecture Test ---")
        print(f"Input Spectrogram Shape:  {dummy_spectrograms.shape}")
        print(f"Output Spectrogram Shape: {denoised_output.shape}") 
        print(f"Output Mask Shape:        {mask_output.shape}")
        print("Test Passed: The model successfully output denoised spectrogram normalized coordinates per audio clip!")
        print("----------------------------------------")
        # return

        # Initialize the Optimizer
        # AdamW is the standard for U-Nets (handles weight decay better than standard Adam)
        optimizer = optim.AdamW(train_model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Initialize a Learning Rate Scheduler 
        # Smoothly drops the learning rate as epochs progress for fine-tuning
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        # Initialize our custom loss
        criterion = HashingDenoisingLoss()

        train_model.fit(
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_func=criterion,
            scheduler = scheduler,
            device=device,
            model_dir=MODEL_DIR,
            num_epochs=train_epochs,  # Ili ručno stavi 20,
            eval_set = (self.test_dataloader, self.original_dataloader)
        )
    def eval_unet(self, model_num,epoch = 0):
        print(f"Doing evaluation on {len(self.original_filenames)} spectrograms and {len(self.original_labels)} labels.")
        test_model = AudioUNet()
        test_model.model_num = model_num
        
        test_model.load_params(
            self.MODEL_DIR,
            model_num
        )
        test_model.to(self.device)

        loss_func = HashingDenoisingLoss()

        # ==========================================
        # EVALUATION PHASE
        # ==========================================
        # 1. Switch to evaluation mode (freezes Dropout/BatchNorm)
        test_model.eval()
        val_running_loss = 0.0
        
        # 2. Disable gradient engine to save GPU memory and speed up compute
        with torch.no_grad():
            val_loop = tqdm(self.test_dataloader, desc=f"Eval Epoch  {epoch+1}")
            
            for noisy_spec, clean_spec, _ in val_loop:
                noisy_spec = noisy_spec.to(self.device)
                clean_spec = clean_spec.to(self.device)
                
                # Forward pass ONLY
                denoised_spec, mask = test_model(noisy_spec)
                val_loss = loss_func(denoised_spec, clean_spec)
                
                val_running_loss += val_loss.item()
                val_loop.set_postfix(Val_Loss=f"{val_loss.item():.4f}")
        
        # Calculate average epoch validation loss
        epoch_val_loss = val_running_loss / len(self.test_dataloader)
        return f"--- Eval Epoch {epoch+1} Completed | Average Val Loss: {epoch_val_loss:.4f} ---"
        # ==========================================
    
    def eval_model(self, model_num):
        print(f"Doing evaluation on {len(self.original_filenames)} spectrograms and {len(self.original_labels)} labels.")
        train_model = AudioUNet().to(self.device)
        train_model.model_num = model_num
        
        train_model.load_params(
            self.MODEL_DIR,
            model_num
        )
        
        # Initialize the evaluator 
        evaluator = EvaluatorUNetMath()
        evaluator.evaluate(
            model = train_model,
            noisy_dataloader = self.test_dataloader,
            original_dataloader= self.original_dataloader,
            device=self.device
        )
        return evaluator.generate_report_card(k_values=[1,2,3,4,5,6,7,8,9,10])

    


    def export_onnx(self) -> None:
        raise NotImplementedError("TrainingPipeline not yet implemented")

    def quantize_onnx(self) -> None:
        raise NotImplementedError("TrainingPipeline not yet implemented")
