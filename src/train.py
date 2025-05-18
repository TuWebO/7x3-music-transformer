import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from models import NextNoteTransformer

class GridDataset(Dataset):
    def __init__(self, data_dir, octaves, key, device):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.octaves = octaves
        self.key = key
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz = np.load(self.files[idx])
        grid = npz['grid']  # shape (T,7,3,O)
        T, r, c, O = grid.shape
        x = torch.tensor(grid.reshape(T, -1), dtype=torch.float32)
        # target is next grid-frame
        y = torch.roll(x, -1, dims=0)
        return x.to(self.device), y.to(self.device)


def train(config):
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    # Data
    train_ds = GridDataset(config['data']['train_dir'], config['data']['octaves'], config['data']['key'], device)
    loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)

    # Model
    feature_dim = 7 * 3 * len(config['data']['octaves'])
    model = NextNoteTransformer(
        feature_dim=feature_dim,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        max_len=config['training']['max_len'],
        grid_dims=feature_dim,
        batch_first=True
    ).to(device)

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    # Training loop
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0.0
        for X, y in loader:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{config['training']['epochs']} - Loss: {avg_loss:.4f}")
        # Save checkpoint
        ckpt_dir = config['training']['ckpt_dir']
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_epoch{epoch}.pt"))


def main():
    parser = argparse.ArgumentParser(description="Train 7x3 Music Transformer")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config)

if __name__ == '__main__':
    main()
