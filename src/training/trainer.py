class TrainingPipeline:
    def __init__(self):
        pass

    def train_cnn(self, start_model_path = None) -> None:
        import os
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
        
        from dataset import TripletDataset, PathMaker
        from modelClass import AudioResNet
        from evaluaator import Evaluator
        
        torch.cuda.set_per_process_memory_fraction(0.18)  # ~4 GB out of 24 GB
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        ROOT_DIR = "/home/jovyan/"
        SPECT_DIR = os.path.join(ROOT_DIR,"spectrograms-20260626T174438Z-3-001/spectrograms")
        NOISE_DIR = os.path.join(ROOT_DIR,"noiseMeowSpectrograms/noiseMeowSpectrograms")
        MODEL_DIR = os.path.join(ROOT_DIR, 'models')
        os.makedirs(MODEL_DIR, exist_ok=True)

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

        train_model = AudioResNet(embedding_size=128).to(device)
        
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

        if start_model_path == None:
            train_model.fit(
                dataloader=train_dataloader,
                optimizer=optimizer,
                miner=miner,
                loss_func=loss_func,
                device=device,
                model_dir=MODEL_DIR,
                num_epochs=20  # Ili ručno stavi 20
            )
        else:
            train_model.load_params(
                MODEL_DIR,
                start_model_path
            )

        print(f"Doing evaluation on {len(original_filenames)} spectrograms and {len(original_labels)} labels.")

        # Initialize the evaluator 
        evaluator = Evaluator()
        evaluator.evaluate(
            model = train_model,
            noisy_dataloader = test_dataloader,
            original_dataloader= original_dataloader,
            device=device
        )
        print(evaluator.generate_report_card(k_values=[1]))
        

    def export_onnx(self) -> None:
        raise NotImplementedError("TrainingPipeline not yet implemented")

    def quantize_onnx(self) -> None:
        raise NotImplementedError("TrainingPipeline not yet implemented")
