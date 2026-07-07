#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Adjust imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.mega_dagger_dataset import DaggerDataset

def filter_noise(px, py, min_neighbors=5, radius=0.8):
    """Filters out isolated noisy spatial points using KDTree."""
    if len(px) == 0:
        return np.array([], dtype=bool)
    points = np.column_stack((px, py))
    tree = KDTree(points)
    # Count neighbors within 'radius'
    counts = tree.query_ball_point(points, r=radius, return_length=True)
    # Keep points that have at least 'min_neighbors' (including itself)
    return counts >= min_neighbors

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.abspath(os.path.join(base_dir, 'data'))
    
    print(f"Loading datasets from {data_dir}...")
    dataset_builder = DaggerDataset(data_dir, dataset_folders=None, sim_threshold=0.999)
    
    # We load manually to also grab poses
    search_dirs = []
    if os.path.exists(data_dir):
        # We search inside all subdirectories of data_dir
        for root, dirs, files in os.walk(data_dir):
            if any(f.endswith('.npz') for f in files):
                search_dirs.append(root)
    
    # Remove duplicates
    search_dirs = list(set(search_dirs))
    
    all_scans, all_actions, all_experts, all_poses = [], [], [], []
    
    for search_dir in search_dirs:
        for file in sorted(os.listdir(search_dir)):
            if file.endswith('.npz'):
                path = os.path.join(search_dir, file)
                data = np.load(path)
                
                scans = data['scans']
                actions = data['actions']
                poses = data['poses']
                expert_id = data.get('expert_id', 0)
                
                if isinstance(expert_id, np.ndarray):
                    expert_id = int(expert_id.item())
                    
                valid_mask = np.all(np.isfinite(scans), axis=1)
                scans = scans[valid_mask]
                actions = actions[valid_mask]
                poses = poses[valid_mask]
                
                if len(scans) == 0:
                    continue
                    
                all_scans.append(scans)
                all_actions.append(actions)
                all_poses.append(poses)
                all_experts.extend([expert_id] * len(scans))
                    
    if len(all_scans) == 0:
        print("No data found!")
        return
        
    scans_np = np.concatenate(all_scans, axis=0)
    actions_np = np.concatenate(all_actions, axis=0)
    poses_np = np.concatenate(all_poses, axis=0)
    experts_np = np.array(all_experts)
    
    print(f"Aggregated {len(scans_np)} total samples. Applying safety filter...")
    sigmas_np = dataset_builder.compute_cbf_sigmas_vectorized(scans_np, actions_np)
    
    speeds = actions_np[:, 0]
    
    # Resolve conflicts and get indices back
    filtered_scans, filtered_actions, filtered_experts, keep_indices = dataset_builder.resolver.resolve(
        scans_np, actions_np, experts_np, sigmas_np, speeds, return_experts=True, return_indices=True
    )
    
    filtered_poses = poses_np[keep_indices]
    print(f"Retained {len(filtered_poses)} points after conflict resolution.")
    
    # ---------------------------------------------------------
    # FILTER EXPERTS AND NOISE
    # ---------------------------------------------------------
    
    # 1. Remove points where expert is 0 (lead waypoint), 999 or 9999
    valid_mask = ~np.isin(filtered_experts, [0, 999, 9999])
    px = filtered_poses[valid_mask, 0]
    py = filtered_poses[valid_mask, 1]
    experts = filtered_experts[valid_mask]
    
    print(f"Retained {len(px)} points from purely Experts 1-5.")
    
    # 2. Filter out noisy isolated spatial points
    noise_mask = filter_noise(px, py, min_neighbors=5, radius=0.8)
    px = px[noise_mask]
    py = py[noise_mask]
    experts = experts[noise_mask]
    
    print(f"Retained {len(px)} points after truncating noisy isolated data.")
    
    unique_experts = np.unique(experts)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_experts))))
    
    plt.figure(figsize=(10, 8))
    
    for i, exp in enumerate(unique_experts):
        mask = (experts == exp)
        plt.scatter(px[mask], py[mask], alpha=0.8, label=f"Expert {exp}", color=colors[i], s=15)
        
    plt.title("Combined MEGA-DAgger Trajectory (Noise Truncated)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    
    plot_path = os.path.join(base_dir, 'mega_dagger_expert_trajectory.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {plot_path}")
    plt.close()

    # Generate individual plots for each expert
    for i, exp in enumerate(unique_experts):
        plt.figure(figsize=(10, 8))
        mask = (experts == exp)
        plt.scatter(px[mask], py[mask], alpha=0.8, label=f"Expert {exp}", color=colors[i], s=15)
        plt.title(f"Expert {exp} Trajectory")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        indiv_path = os.path.join(base_dir, f'expert_{exp}_trajectory.png')
        plt.savefig(indiv_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization for Expert {exp} to {indiv_path}")
        plt.close()


    # ---------------------------------------------------------
    # PLOT FINAL EVALUATION TRAJECTORY (EXPERT 9999)
    # ---------------------------------------------------------
    raw_eval_mask = (experts_np == 9999)
    if np.any(raw_eval_mask):
        px_eval = poses_np[raw_eval_mask, 0]
        py_eval = poses_np[raw_eval_mask, 1]
        
        plt.figure(figsize=(10, 8))
        plt.plot(px_eval, py_eval, 'm-', alpha=0.8, label="Final Policy (MEGA-DAgger)", linewidth=2)
        plt.title("Final Novice Policy Trajectory")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        eval_plot_path = os.path.join(base_dir, 'final_expert_trajectory.png')
        plt.savefig(eval_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved final evaluation trajectory to {eval_plot_path}")
        plt.close()

if __name__ == '__main__':
    main()
