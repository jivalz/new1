#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# Adjust imports to work whether run from scripts/ or as a module
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.policy_network import PolicyNetwork as NovicePolicy
from scripts.mega_dagger_dataset import DaggerDataset

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description='Train MEGA-DAgger Novice Policy')
    parser.add_argument('--data-dir', default='data', help='Root data directory containing expert datasets')
    parser.add_argument('--datasets', nargs='+', default=['ego_data/bc_data/expert_0', 'ego_data/dagger/dagger_1', 'ego_data/dagger/dagger_2', 'ego_data/dagger/dagger_3', 'ego_data/dagger/dagger_4'], help='List of dataset folders to include')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--checkpoint', default=None, help='Load previous checkpoint weights')
    parser.add_argument('--iteration', type=int, default=0, help='Current DAgger iteration')
    parser.add_argument('--wandb-id', default=None, help='WandB Run ID to resume')
    args = parser.parse_args()

    # Expand relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.abspath(os.path.join(base_dir, args.data_dir))

    import wandb
    if args.wandb_id:
        wandb.init(project="mega_dagger", id=args.wandb_id, resume="allow", config=vars(args))
    else:
        wandb.init(project="mega_dagger", config=vars(args))

    # 1. Dataset Aggregation & Conflict Resolution
    print("=============================================")
    print(" MEGA-DAgger Multi-Expert Training Pipeline")
    print("=============================================")
    print(f"Aggregating datasets from {data_dir}: {args.datasets}")
    dataset_builder = DaggerDataset(data_dir, dataset_folders=args.datasets, sim_threshold=0.999)
    full_dataset, experts_arr, cr_metrics = dataset_builder.load_data_with_metrics()

    if full_dataset is None or len(full_dataset) == 0:
        print("Error: No valid data found to train on!")
        return

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Novice Policy Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = NovicePolicy(input_dim=1080, output_dim=2).to(device)

    # Support loading previous checkpoints for iterative DAgger training
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        print("Initializing new weights...")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 3. Training Loop
    print(f"Training on {train_size} samples, validating on {val_size} samples for {args.epochs} epochs...")
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    
    weights_dir = os.path.join(base_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    ckpt_path = os.path.join(weights_dir, 'novice_policy.pth')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_dataloader, criterion, device)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            
        wandb.log({
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "learning_rate": args.lr,
            "epoch": epoch
        }, step=epoch)
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{args.epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
            
            # Live Plotting Updates
            plt.figure(figsize=(8, 5))
            plt.plot(train_loss_history, label='Train Loss')
            plt.plot(val_loss_history, label='Val Loss')
            plt.title('MEGA-DAgger Live Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.legend()
            plot_path = os.path.join(base_dir, 'dagger_loss.png')
            plt.savefig(plot_path)
            plt.close()
            
    print(f"Saved best novice policy (Val MSE: {best_val_loss:.6f}) to {ckpt_path}")

    # Plot Expert Performance (Contribution Graph)
    if experts_arr is not None and len(experts_arr) > 0:
        unique_experts, counts = np.unique(experts_arr, return_counts=True)
        plt.figure(figsize=(8, 6))
        
        colors = ['skyblue' if e != 0 else 'lightgreen' for e in unique_experts]
        labels = [f"Expert {e}" if e != 0 else "Base Expert" for e in unique_experts]
        
        plt.bar(labels, counts, color=colors)
        plt.title('MEGA-DAgger: Expert Performance (Samples Retained)')
        plt.ylabel('Number of Safe Samples')
        plt.xlabel('Experts')
        
        for i, v in enumerate(counts):
            plt.text(i, v + max(counts)*0.01, str(v), ha='center')
            
        expert_plot_path = os.path.join(base_dir, 'expert_performance.png')
        plt.savefig(expert_plot_path)
        plt.close()
        print(f"Saved expert performance plot to {expert_plot_path}")
        
        expert_stats = {}
        for expert_id, count in zip(unique_experts, counts):
            expert_stats[f"expert_{expert_id}_samples"] = count
        wandb.log(expert_stats)
        
    wandb.log({
        "iteration_number": args.iteration,
        "dataset_size": cr_metrics["final_samples"],
        "unsafe_samples_removed": cr_metrics["unsafe_samples"],
        "duplicate_labels_detected": cr_metrics["duplicate_labels_detected"],
        "duplicate_labels_removed": cr_metrics["duplicate_labels_detected"],
        "safety_filter_percentage": cr_metrics["safety_filter_percentage"],
        "model_version": args.iteration,
        "total_parameters": sum(p.numel() for p in model.parameters())
    })
    
    # Save Model Weights as Artifact
    model_artifact = wandb.Artifact(f"novice_policy_iter_{args.iteration}", type="model")
    model_artifact.add_file(ckpt_path)
    wandb.log_artifact(model_artifact)

    # Save aggregated dataset as Artifact (just one npz file as representative, or the directory)
    # We will log the data_dir for reproducibility. 
    # Warning: Uploading large datasets directly might be slow, so we log the path or a sample.
    dataset_artifact = wandb.Artifact(f"dagger_dataset_iter_{args.iteration}", type="dataset")
    dataset_artifact.add_dir(data_dir, name="data")
    wandb.log_artifact(dataset_artifact)
    
    wandb.finish()

if __name__ == '__main__':
    main()
