import numpy as np
import torch

class ConflictResolver:
   
    def __init__(self, sim_threshold=0.95):
        self.sim_threshold = sim_threshold

    def resolve(self, scans: np.ndarray, actions: np.ndarray, experts: np.ndarray, sigmas: np.ndarray, speeds: np.ndarray, return_experts=False, return_indices=False):
        
        N = len(scans)
        if N == 0:
            if return_indices:
                return scans, actions, experts, [] if return_experts else (scans, actions, [])
            if return_experts:
                return scans, actions, experts
            return scans, actions

        print(f"Resolving conflicts among {N} samples...")

        # 1. Normalize safety scores (sigmas) and speed scores
        def normalize(arr):
            ptp = np.ptp(arr)
            if ptp == 0:
                return np.zeros_like(arr)
            return (arr - np.min(arr)) / ptp

        norm_sigmas = normalize(sigmas)
        norm_speeds = normalize(speeds)
        
        # 2. Compute conflict resolution score
        scores = norm_sigmas + norm_speeds

        # 3. Sort by score descending
        sorted_indices = np.argsort(-scores)

        # 4. Use PyTorch for ultra-fast cosine similarity batching
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        scans_t = torch.tensor(scans, dtype=torch.float32, device=device)
        # L2 Normalize the scans to turn cosine similarity into a dot product
        norms = torch.norm(scans_t, dim=1, keepdim=True)
        scans_norm = scans_t / (norms + 1e-8)
        
        # Pre-allocate a tensor for kept scans to avoid slow concatenations
        kept_scans_tensor = torch.empty((N, scans.shape[1]), dtype=torch.float32, device=device)
        kept_count = 0
        keep_original_indices = []
        
        # 5. Greedy selection based on score priority
        for idx in sorted_indices:
            idx = int(idx)
            current_scan = scans_norm[idx]
            
            if kept_count > 0:
                # Compute cosine similarities with all previously kept scans
                # Shape: (kept_count, 360) @ (360,) -> (kept_count,)
                sims = torch.matmul(kept_scans_tensor[:kept_count], current_scan)
                
                # If the max similarity is above threshold, it's a conflict!
                # Since the dataset is sorted by score, the one ALREADY in kept_scans 
                # has a higher score. So we discard the current one.
                if torch.max(sims) > self.sim_threshold:
                    continue 
                    
            # Keep this sample
            kept_scans_tensor[kept_count] = current_scan
            keep_original_indices.append(idx)
            kept_count += 1

        print(f"Conflict Resolution Complete: Kept {kept_count} out of {N} samples.")
        
        # Return the filtered dataset
        if return_experts:
            if return_indices:
                return scans[keep_original_indices], actions[keep_original_indices], experts[keep_original_indices], keep_original_indices
            return scans[keep_original_indices], actions[keep_original_indices], experts[keep_original_indices]
        
        if return_indices:
            return scans[keep_original_indices], actions[keep_original_indices], keep_original_indices
        return scans[keep_original_indices], actions[keep_original_indices]
