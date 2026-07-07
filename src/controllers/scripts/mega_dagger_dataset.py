import os
import glob
import numpy as np
import torch
from torch.utils.data import TensorDataset

from scripts.conflict_resolution import ConflictResolver
from scripts.cbf import CBFdf

class DaggerDataset:
    """
    Dataset aggregator for MEGA-DAgger.
    - Loads all lap_*.npz files from expert directories.
    - Evaluates the CBF safety score for every sample.
    - Filters out unsafe samples.
    - Applies conflict resolution for multi-expert datasets.
    """
    def __init__(self, data_dir, dataset_folders=None, cbf_alpha=0.5, cbf_gamma=0.5, sim_threshold=0.95):
        self.data_dir = data_dir
        self.dataset_folders = dataset_folders
        self.cbf = CBFdf(alpha=cbf_alpha, gamma=cbf_gamma)
        self.resolver = ConflictResolver(sim_threshold=sim_threshold)

    def compute_cbf_sigmas_vectorized(self, scans: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Fast vectorized CBF sigma calculation for the entire dataset.
        scans: (N, 360)
        actions: (N, 2)
        """
        N = len(scans)
        if N == 0:
            return np.array([])
            
        speeds = actions[:, 0]
        omegas = actions[:, 1]
        
        dt = 0.2
        # Prevent divide by zero for R = v/w
        safe_omegas = np.where(np.abs(omegas) < 1e-6, 1e-6, omegas)
        R = speeds / safe_omegas
        
        x_next = np.where(np.abs(omegas) < 1e-6, speeds * dt, R * np.sin(omegas * dt))
        y_next = np.where(np.abs(omegas) < 1e-6, 0.0, R * (1.0 - np.cos(omegas * dt)))
        
        x_next = x_next.reshape(N, 1)
        y_next = y_next.reshape(N, 1)
        
        angle_min = -3.14159
        angle_max = 3.14159
        num_beams = scans.shape[1]
        angles = np.linspace(angle_min, angle_max, num_beams)
        
        obs_x = scans * np.cos(angles) # (N, 360)
        obs_y = scans * np.sin(angles) # (N, 360)
        
        hx_curr = obs_x**2 + obs_y**2 - self.cbf.alpha**2
        hx_next = (x_next - obs_x)**2 + (y_next - obs_y)**2 - self.cbf.alpha**2
        
        sigmas = hx_next - (1 - self.cbf.gamma) * hx_curr
        
        # We only care about obstacles within 1.0 meter
        mask = (scans <= 1.0) & np.isfinite(scans)
        sigmas[~mask] = np.inf
        
        min_sigmas = np.min(sigmas, axis=1)
        
        # If no points are within 1.0m, the state is safe
        min_sigmas[np.isinf(min_sigmas)] = 1.0
        
        return min_sigmas

    def load_data_with_metrics(self):
        all_scans = []
        all_actions = []
        all_experts = []

        if self.dataset_folders:
            search_dirs = [os.path.join(self.data_dir, folder) for folder in self.dataset_folders]
        else:
            search_dirs = [self.data_dir]

        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
                
            for root, dirs, files in os.walk(search_dir):
                for file in sorted(files):
                    if file.endswith('.npz'):
                        path = os.path.join(root, file)
                        data = np.load(path)
                        
                        scans = data['scans']
                        actions = data['actions']
                        expert_id = data.get('expert_id', 0)
                        
                        if isinstance(expert_id, np.ndarray):
                            expert_id = int(expert_id.item())

                        valid_mask = np.all(np.isfinite(scans), axis=1)
                        scans = scans[valid_mask]
                        actions = actions[valid_mask]

                        if len(scans) == 0:
                            continue

                        all_scans.append(scans)
                        all_actions.append(actions)
                        all_experts.extend([expert_id] * len(scans))

        if len(all_scans) == 0:
            return None, None, None

        scans_np = np.concatenate(all_scans, axis=0)
        actions_np = np.concatenate(all_actions, axis=0)
        experts_np = np.array(all_experts)
        
        initial_samples = len(scans_np)

        sigmas_np = self.compute_cbf_sigmas_vectorized(scans_np, actions_np)
        unsafe_count = int(np.sum(sigmas_np < 0.0))

        # We keep all samples for DAgger recovery, but we track metrics
        safe_scans = scans_np
        safe_actions = actions_np
        safe_experts = experts_np
        safe_sigmas = sigmas_np
        safe_speeds = safe_actions[:, 0]

        filtered_scans, filtered_actions, filtered_experts = self.resolver.resolve(
            safe_scans, safe_actions, safe_experts, safe_sigmas, safe_speeds, return_experts=True
        )

        tensor_x = torch.tensor(filtered_scans, dtype=torch.float32)
        tensor_y = torch.tensor(filtered_actions, dtype=torch.float32)
        
        cr_metrics = {
            "initial_samples": initial_samples,
            "unsafe_samples": unsafe_count,
            "final_samples": len(filtered_scans),
            "duplicate_labels_detected": initial_samples - len(filtered_scans),
            "safety_filter_percentage": (unsafe_count / initial_samples) * 100 if initial_samples > 0 else 0
        }
        
        return TensorDataset(tensor_x, tensor_y), filtered_experts, cr_metrics

    def load_data(self):
        ds, ex, _ = self.load_data_with_metrics()
        return ds, ex
