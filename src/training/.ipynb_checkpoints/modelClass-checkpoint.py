import torch
import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import torch.nn.functional as F
from src.training.evaluaator import Evaluator

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

        if eval_set is not None:
            evaluator = Evaluator()

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
                evaluator.evaluate(
                    model = self,
                    noisy_dataloader = eval_set[0],
                    original_dataloader= eval_set[1],
                    device=device
                )
                print(evaluator.generate_report_card(k_values=[1]))

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
