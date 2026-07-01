import torch
from tqdm import tqdm
class Evaluator:
    def __init__(self):
        """
        Initializes the evaluator with the dataset.
        """
        pass

    def list_all_sorted(self, noisy_embedding, database_matrix, database_labels):
        """
        Returns a list of songs in the database compared to the noisy query, 
        sorted from most similar (smallest distance) to least similar.
        """
        # 1. Calculate the distance between the query and EVERY item in the database
        # Output shape is typically [1, N] where N is the number of database items
        distances = torch.cdist(noisy_embedding, database_matrix)
        
        # Flatten the distances to a 1D tensor for easier sorting
        distances_flat = distances.flatten()
        
        # 2. Get the indices that sort the distances in ascending order (smallest first)
        sorted_indices = torch.argsort(distances_flat).tolist()
        
        # 3. Build the sorted list of results
        sorted_results = []
        for idx in sorted_indices:
            predicted_int_label = database_labels[idx]
            predicted_song_name = predicted_int_label
            
            # Grab the actual distance value for this specific match
            distance_val = distances_flat[idx].item()
            
            # Append as a tuple: (Song Name, Distance)
            sorted_results.append((predicted_song_name, distance_val))
        
        return sorted_results
    def evaluate(
            self,
            model,
            noisy_dataloader,
            original_dataloader,
            device,
            test_chunks = None,
            noise_seed = None
        ):     

        # print("\n--- 1. Building Clean Database Matrix ---")
        self.database_matrix, self.database_labels = model.create_database_matrix(
            database_dataloader=original_dataloader,
            device=device
        )

        # print("\n--- 2. Building Noisy Query Matrix ---")
        # YOUR FIX: Reusing the function for the noisy data!
        # We grab the labels here too, so we don't have to track them manually.
        noisy_matrix, true_labels = model.create_database_matrix(
            database_dataloader=noisy_dataloader,
            device=device
        )

        # print("\n--- 3. Calculating Matches (CUDA Chunked) ---")
        
        database_matrix_gpu = self.database_matrix.to(device)
        sorted_indices_list = []
        
        # We will process 500 queries at a time to keep VRAM usage under 500 MB
        chunk_size = 500 
        
        for start_idx in tqdm(range(0, noisy_matrix.size(0), chunk_size), desc="GPU Math"):
            end_idx = min(start_idx + chunk_size, noisy_matrix.size(0))
            
            # 1. Push only a small chunk of queries to the GPU
            noisy_chunk_gpu = noisy_matrix[start_idx:end_idx].to(device)
            
            # 2. Calculate distances and sort just for this chunk
            distances_chunk = torch.cdist(noisy_chunk_gpu, database_matrix_gpu)
            sorted_chunk = torch.argsort(distances_chunk, dim=1)
            
            # 3. Pull the sorted integer results back to CPU RAM immediately
            sorted_indices_list.append(sorted_chunk.cpu().numpy())
            
            # 4. Clean up VRAM for the next loop
            del noisy_chunk_gpu, distances_chunk, sorted_chunk
            
        # Stitch all the chunks back together into one master array
        import numpy as np
        sorted_indices = np.vstack(sorted_indices_list)
        
        # Free the database from VRAM
        del database_matrix_gpu
        torch.cuda.empty_cache()

        # --- Proceed with Label Mapping ---
        db_labels_np = np.array(self.database_labels)
        prediction_matrix = []
        
        for i in tqdm(range(noisy_matrix.size(0)), desc="Mapping labels"):
            ordered_song_names = db_labels_np[sorted_indices[i]].tolist()
            prediction_matrix.append(ordered_song_names)
            
        # Store final metrics
        self.ground_truths = true_labels
        self.predictions = prediction_matrix
        self.total_queries = len(self.ground_truths)
    def calculate_top_k(self, k=3):
        """
        Calculates the Top-K Accuracy.
        """
        top_k_hits = 0
        for true_song, pred_list in zip(self.ground_truths, self.predictions):
            if true_song in pred_list[:k]:
                top_k_hits += 1
                
        return top_k_hits / self.total_queries

    def calculate_mrr_at_k(self, k=None):
        """
        Calculates the Mean Reciprocal Rank truncated at K (MRR@K).
        If k is None, evaluates the entire list.
        """
        sum_reciprocal_rank = 0.0
        for true_song, pred_list in zip(self.ground_truths, self.predictions):
            # Truncate the list to top K
            search_list = pred_list[:k] if k is not None else pred_list
            
            if true_song in search_list:
                rank = search_list.index(true_song) + 1  # 1-indexed rank
                sum_reciprocal_rank += 1.0 / rank
                
        return sum_reciprocal_rank / self.total_queries

    def calculate_rank_distribution(self):
        """
        Calculates the percentage distribution of ranks across all provided predictions.
        """
        rank_counts = {}
        misses = 0
        
        for true_song, pred_list in zip(self.ground_truths, self.predictions):
            if true_song in pred_list:
                rank = pred_list.index(true_song) + 1
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            else:
                misses += 1
                
        distribution = {}
        for rank in sorted(rank_counts.keys()):
            percentage = (rank_counts[rank] / self.total_queries) * 100
            distribution[f"Rank {rank}"] = f"{percentage:.1f}%"
            
        if misses > 0:
            miss_percentage = (misses / self.total_queries) * 100
            distribution["Not Found"] = f"{miss_percentage:.1f}%"
            
        return distribution

    def generate_report_card(self, k_values=[1, 3, 5]):
        """
        Returns a formatted report card evaluating Top-K and MRR@K as a text string.
        """
        report = []
        report.append(f"Total Queries Evaluated: {self.total_queries}\n")
        
        report.append("--- Performance at K Thresholds ---")
        # Clean table header
        report.append(f"{'Threshold':<12} | {'Top-K Accuracy':<15} | {'MRR@K'}")
        report.append("-" * 55)
        
        for k in k_values:
            acc = self.calculate_top_k(k)
            mrr = self.calculate_mrr_at_k(k)
            report.append(f"K = {k:<8} | {acc:>6.2%} ({acc:.3f})   | {mrr:.3f}")
            
        # report.append("\n--- Overall Rank Distribution ---")
        # distribution = self.calculate_rank_distribution()
            
        # Join all lines into a single master string separated by newlines
        return "\n".join(report)

class EvaluatorUNetMath:
    def __init__(self):
        """
        Initializes the evaluator with the dataset.
        """
        pass
    def extract_unet_hashes(self, model, dataloader, device):
        """
        Explicit extraction loop for UNet + Hashing pipeline.
        Replaces the generic 'model.create_database_hashes'.
        """
        model.eval()
        all_hashes = []
        all_labels = []

        with torch.no_grad():
            for inputs,_, labels in tqdm(dataloader, desc="Extracting Hashes"):
                inputs = inputs.to(device)

                # 1. UNet Feature Extraction / Denoising
                features = model(inputs)

                # 2. Hashing Layer
                continuous_hashes = model.generate_hashes(model.extract_peaks(features))
                all_hashes.append(continuous_hashes.cpu())
                all_labels.extend(labels)

        return torch.cat(all_hashes, dim=0), all_labels
    def evaluate(
            self,
            model,
            noisy_dataloader,      # TEST set (upiti)
            original_dataloader,   # FULL set (biblioteka)
            device,
            test_chunks=None,
            noise_seed=None
        ):     


        import pickle
        import os
        from collections import defaultdict
        from tqdm import tqdm
        from torch.utils.data import DataLoader, Subset

        # =====================================================================
        # 1. LOAD COMPRESSED INDEX
        # =====================================================================
        print("\n--- 1. Loading Compressed Database ---")
        index_path = "compressed_reverse_index.pkl"
        save_path=index_path
        neighborhood_size=(16, 16)               # Zlatni standard za čistoću
        threshold=0.25                           # Sečemo šum
        fan_value=15                             # Vraćeno na 15 za masovno glasanje
        min_time_delta=1                         # Uzimamo i prve komšije
        max_time_delta=60                        # Dugački heševi (do 60 frejmova)
        max_freq_delta=50                        # Širi spektar spajanja tonova
        MAX_SONGS_PER_HASH=200   
        
        if not os.path.exists(index_path):
            print("Index not found! Building it now...")
            index_path = "compressed_reverse_index.pkl"

            print("\n🚀 POKRETANJE GRAĐENJA BAZE (ITERACIJA 11 - Teška artiljerija)...")
            
            model.build_and_save_index(
                original_dataloader=original_dataloader,  # Tvoj DataLoader sa svih 7000 pesama
                device=device,                            # 'cuda' ili 'cpu'
                save_path=index_path,
                neighborhood_size=neighborhood_size,               # Zlatni standard za čistoću
                threshold=threshold,                           # Sečemo šum
                fan_value=fan_value,                             # Vraćeno na 15 za masovno glasanje
                min_time_delta=min_time_delta,                         # Uzimamo i prve komšije
                max_time_delta=max_time_delta,                        # Dugački heševi (do 60 frejmova)
                max_freq_delta=max_freq_delta,                        # Širi spektar spajanja tonova
                MAX_SONGS_PER_HASH=MAX_SONGS_PER_HASH                    # Praštamo ponavljanje do 200 pesama
            )
            
            print("✅ Baza je spremna! Sada možeš da pokreneš evaluate().")

        # Load the pre-built, pre-filtered dictionary directlyf into RAM
        with open(index_path, "rb") as f:
            reverse_index = pickle.load(f)
            
        print(f"Loaded compressed index with {len(reverse_index)} unique, valid hashes.")

        # =====================================================================
        # 2. EXTRACT NOISY QUERIES (TEST SET)
        # =====================================================================
        print("\n--- 2. Extracting Noisy Queries (TEST SET) ---")
        
        # Original dataset
        dataset = noisy_dataloader.dataset
        # Select specific indices (Ensure your test set actually has 5000+ items if you use this range!)
        indices = list(range(1000, 5000))  
        subset = Subset(dataset, indices)
        
        subset_dataloader = DataLoader(
            subset,
            batch_size=noisy_dataloader.batch_size,
            shuffle=False  # CRITICAL: Keep False for evaluation so labels align!
        )



        # --------------------------------------------------------------------------------------------------------------------------------------------
        
        noisy_hashes, _, test_labels = model.create_database_hashes(
            dataloader=noisy_dataloader,
            device=device,
            neighborhood_size=neighborhood_size,
            threshold=threshold,
            fan_value=fan_value,
            min_time_delta=min_time_delta,
            max_time_delta=max_time_delta,
            max_freq_delta=max_freq_delta
        )
        print(f"Testing {len(test_labels)} noisy songs (total {len(noisy_hashes)} hashes).")



        # --------------------------------------------------------------------------------------------------------------------------------------------
        

        # =====================================================================
        # 3. GLASANJE SA VREMENSKIM PORAVNANJEM (The Shazam Way)
        # =====================================================================
        print("\n--- 3. Glasanje sa vremenskim poravnanje ---")
        # Struktura: { noisy_track_id: { (original_song_id, time_offset): match_count } }
        match_tallies = defaultdict(lambda: defaultdict(int))
        
        for noisy in tqdm(noisy_hashes, desc="Voting on matches"):
            noisy_hash = noisy['hash']
            noisy_track_id = noisy['batch_idx'] 
            t_noisy = noisy['anchor_time']
            
            if noisy_hash in reverse_index:
                # Izvlačimo i ID pesme i vreme kada se taj heš desio u originalu
                for matched_original_song, t_orig in reverse_index[noisy_hash]:
                    # Ključni korak: Izračunavanje vremenskog ofseta
                    time_offset = t_orig - t_noisy
                    
                    # Glas ide za tačnu pesmu na tačnom ofsetu!
                    match_tallies[noisy_track_id][(matched_original_song, time_offset)] += 1
        
        # =====================================================================
        # 4. SORTIRANJE POBEDNIKA (Pronalaženje najvišeg vrha u histogramu)
        # =====================================================================
        print("\n--- 4. Sortiranje Matches (Histogram Peak) ---")
        prediction_matrix = []
        total_test_queries = len(test_labels) 
        
        for noisy_id in range(total_test_queries):
            # Rečnik ofseta za trenutni upit
            offset_votes = match_tallies.get(noisy_id, {})
            
            # Grupišemo glasove po pesmama, tražeći maksimalni ofset za svaku pesmu
            song_best_scores = defaultdict(int)
            for (song_id, offset), votes in offset_votes.items():
                if votes > song_best_scores[song_id]:
                    song_best_scores[song_id] = votes
            
            # Sortiramo pesme po njihovom najboljem ofset rezultatu u opadajućem redosledu
            sorted_matches = sorted(song_best_scores, key=song_best_scores.get, reverse=True)
            prediction_matrix.append(sorted_matches)
        self.ground_truths = test_labels
        self.predictions = prediction_matrix
        self.total_queries = len(self.ground_truths)
    def calculate_top_k(self, k=3):
        """
        Calculates the Top-K Accuracy.
        """
        top_k_hits = 0
        for true_song, pred_list in zip(self.ground_truths, self.predictions):
            if true_song in pred_list[:k]:
                top_k_hits += 1
                
        return top_k_hits / self.total_queries

    def calculate_mrr_at_k(self, k=None):
        """
        Calculates the Mean Reciprocal Rank truncated at K (MRR@K).
        If k is None, evaluates the entire list.
        """
        sum_reciprocal_rank = 0.0
        for true_song, pred_list in zip(self.ground_truths, self.predictions):
            # Truncate the list to top K
            search_list = pred_list[:k] if k is not None else pred_list
            
            if true_song in search_list:
                rank = search_list.index(true_song) + 1  # 1-indexed rank
                sum_reciprocal_rank += 1.0 / rank
                
        return sum_reciprocal_rank / self.total_queries

    def calculate_rank_distribution(self):
        """
        Calculates the percentage distribution of ranks across all provided predictions.
        """
        rank_counts = {}
        misses = 0
        
        for true_song, pred_list in zip(self.ground_truths, self.predictions):
            if true_song in pred_list:
                rank = pred_list.index(true_song) + 1
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            else:
                misses += 1
                
        distribution = {}
        for rank in sorted(rank_counts.keys()):
            percentage = (rank_counts[rank] / self.total_queries) * 100
            distribution[f"Rank {rank}"] = f"{percentage:.1f}%"
            
        if misses > 0:
            miss_percentage = (misses / self.total_queries) * 100
            distribution["Not Found"] = f"{miss_percentage:.1f}%"
            
        return distribution

    def generate_report_card(self, k_values=[1, 3, 5]):
        """
        Returns a formatted report card evaluating Top-K and MRR@K as a text string.
        """
        report = []
        report.append(f"Total Queries Evaluated: {self.total_queries}\n")
        
        report.append("--- Performance at K Thresholds ---")
        # Clean table header
        report.append(f"{'Threshold':<12} | {'Top-K Accuracy':<15} | {'MRR@K'}")
        report.append("-" * 55)
        
        for k in k_values:
            acc = self.calculate_top_k(k)
            mrr = self.calculate_mrr_at_k(k)
            report.append(f"K = {k:<8} | {acc:>6.2%} ({acc:.3f})   | {mrr:.3f}")
            
        # report.append("\n--- Overall Rank Distribution ---")
        # distribution = self.calculate_rank_distribution()
            
        # Join all lines into a single master string separated by newlines
        return "\n".join(report)