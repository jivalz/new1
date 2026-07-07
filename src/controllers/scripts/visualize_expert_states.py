import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Adjust imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.mega_dagger_dataset import DaggerDataset

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
    
    print(f"Plotting racetrack map with {len(filtered_poses)} points...")
    
    px = filtered_poses[:, 0]
    py = filtered_poses[:, 1]
    
    unique_experts, counts = np.unique(filtered_experts, return_counts=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_experts))))
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [2, 1]})
    
    # Plot 1: Racetrack mapped experts
    for i, exp in enumerate(unique_experts):
        mask = (filtered_experts == exp)
        label = f"Expert {exp}" if exp != 999 else "Base Expert (999)"
        axes[0].scatter(px[mask], py[mask], alpha=0.7, label=label, color=colors[i], s=10)
        
    axes[0].set_title("Best Expert Mapped on Racetrack")
    axes[0].set_xlabel("X Position (m)")
    axes[0].set_ylabel("Y Position (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_aspect('equal', 'box')
    
    # Plot 2: Histogram / Bar chart
    labels = [f"Expert {e}" if e != 999 else "Base Exp" for e in unique_experts]
    axes[1].bar(labels, counts, color=[colors[i] for i in range(len(unique_experts))])
    axes[1].set_title("Expert Utilization (Best Action Count)")
    axes[1].set_ylabel("Number of Samples")
    axes[1].set_xlabel("Experts")
    
    # Add counts on top of bars
    for i, v in enumerate(counts):
        axes[1].text(i, v + max(counts)*0.01, str(v), ha='center', fontweight='bold')
        
    plt.tight_layout()
    plot_path = os.path.join(base_dir, 'expert_racetrack_map.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {plot_path}")

if __name__ == '__main__':
    main()
