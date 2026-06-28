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

from src.training.dataset import TripletDataset, PathMaker
from src.training.modelClass import AudioResNet
from src.training.evaluaator import Evaluator

class TrainingPipeline:
    def __init__(self):
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

        train_dataset = TripletDataset(
            train_filenames,
            train_labels,
            noise_paths=noise_filenames,
            type_of_dataset='train'
        )
        test_dataset = TripletDataset(
            test_filenames,
            test_labels,
            noise_paths=noise_filenames,
            type_of_dataset='test'
        )
        original_dataset = TripletDataset(
            original_filenames,
            original_labels,
            noise_paths=noise_filenames,
            type_of_dataset='original'
        )

        

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=64,       # Number of audio files to process simultaneously
            shuffle=True,        # Mixes up the order every epoch so the model learns features, not order
            drop_last=True       # Trims the last batch if it doesn't perfectly divide by 64
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=64,       # Number of audio files to process simultaneously
            shuffle=True,        # Mixes up the order every epoch so the model learns features, not order
        )
        
        original_dataloader = DataLoader(
            original_dataset, 
            batch_size=64,       # Number of audio files to process simultaneously
            shuffle=True,        # Mixes up the order every epoch so the model learns features, not order
        )

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.original_dataloader = original_dataloader

        self.original_filenames = original_filenames
        self.original_labels = original_labels
    def train_cnn(self, train_epochs =0, model_num = 0) -> None:
        original_filenames = self.original_filenames
        original_labels = self.original_labels
        ROOT_DIR=self.ROOT_DIR
        SPECT_DIR=self.SPECT_DIR
        NOISE_DIR=self.NOISE_DIR
        MODEL_DIR=self.MODEL_DIR

        device = self.device
        
        train_dataloader = self.train_dataloader
        test_dataloader = self.test_dataloader
        original_dataloader = self.original_dataloader

        
        train_model = AudioResNet(embedding_size=128).to(device)
        train_model.model_num = model_num

        train_model.load_params(
            MODEL_DIR,
            model_num
        )
        
        dummy_spectrograms = torch.randn(64, 1, 128, 216).to(device)
            
        # Run the fake data through the model
        output = train_model(dummy_spectrograms)
        
        
        print("--- Model Architecture Test ---")
        print(f"Input Spectrogram Shape:  {dummy_spectrograms.shape}")
        print(f"Output Embedding Shape:   {output.shape}")
        print("Test Passed: The model successfully output 128 normalized coordinates per audio clip!")

        miner = miners.TripletMarginMiner(margin=1.0, type_of_triplets="hard")
        loss_func = losses.TripletMarginLoss(margin=1.0)
        
        optimizer = optim.Adam(train_model.parameters(), lr=1e-4)

        train_model.fit(
            dataloader=train_dataloader,
            optimizer=optimizer,
            miner=miner,
            loss_func=loss_func,
            device=device,
            model_dir=MODEL_DIR,
            num_epochs=train_epochs,  # Ili ručno stavi 20,
            eval_set = (self.test_dataloader, self.original_dataloader)
        )
    def eval_model(self, model_num):
        print(f"Doing evaluation on {len(self.original_filenames)} spectrograms and {len(self.original_labels)} labels.")
        train_model = AudioResNet(embedding_size=128).to(self.device)
        train_model.model_num = model_num
        
        train_model.load_params(
            self.MODEL_DIR,
            model_num
        )
        
        # Initialize the evaluator 
        evaluator = Evaluator()
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
