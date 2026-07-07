import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=1080, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, scan_np):
        """Numpy convenience: (1080,) → (2,)."""
        import numpy as np
        with torch.no_grad():
            x = torch.tensor(scan_np, dtype=torch.float32).unsqueeze(0)
            out = self.net(x)
        return out.squeeze(0).numpy()
