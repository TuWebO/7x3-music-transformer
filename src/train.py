import os
import argparse
import json
import yaml
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from models import NextNoteTransformer


def collate_fn(batch):
    """
    Pads variable-length (T_i, F) sequences to (T_max, F) and returns a mask.
    batch: list of (X, y) tensors
    Returns:
      Xp: (B, T_max, F), yp: (B, T_max, F), mask: (B, T_max)
    """
    Xs, ys = zip(*batch)
    lengths = [x.size(0) for x in Xs]
    T_max = max(lengths)
    B = len(batch)
    F = Xs[0].size(1)
    device = Xs[0].device

    Xp = torch.zeros(B, T_max, F, device=device)
    yp = torch.zeros(B, T_max, F, device=device)
    mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)

    for i, (x, y) in enumerate(batch):
        t = x.size(0)
        Xp[i, :t] = x
        yp[i, :t] = y
        mask[i, :t] = 1

    return Xp, yp, mask


class GridDataset(Dataset):
    """
    Loads preprocessed 7×3×O grid .npz files and aligns them to a global octave set.
    Expects data_dir/metadata.json describing each file's key and octave list.
    """
    def __init__(self, data_dir, device):
        self.data_dir = data_dir
        self.device = device
        # Load metadata
        meta_path = os.path.join(data_dir, 'metadata.json')
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
        # File order
        self.files = list(self.metadata.keys())
        # Determine global octave set
        all_octaves = set()
        for info in self.metadata.values():
            all_octaves.update(info['octaves'])
        self.global_octaves = sorted(all_octaves)
        # Feature dimension: 7 rows × 3 cols × global octaves
        self.feature_dim = 7 * 3 * len(self.global_octaves)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        info = self.metadata[fname]
        # Load grid: shape (T,7,3,O_file)
        arr = np.load(os.path.join(self.data_dir, info['file']))['grid']
        T, R, C, O_file = arr.shape
        # Expand to global octaves
        G = len(self.global_octaves)
        expanded = np.zeros((T, R, C, G), dtype=arr.dtype)
        for i_file, octv in enumerate(info['octaves']):
            g_idx = self.global_octaves.index(octv)
            expanded[:, :, :, g_idx] = arr[:, :, :, i_file]
        # Flatten to (T, F)
        X = torch.tensor(expanded.reshape(T, -1), dtype=torch.float32, device=self.device)
        # Next-frame target
        y = torch.roll(X, shifts=-1, dims=0)
        return X, y


def train(config):
    # Device
    device = torch.device(config.get('device', 'cpu'))
    if config.get('device') == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')

    # Dataset & DataLoader
    ds = GridDataset(config['data']['train_dir'], device)
    loader = DataLoader(
        ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    # Model instantiation
    feature_dim = ds.feature_dim
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
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    # Training loop
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0.0
        total_elems = 0
        for X, y, mask in loader:
            optimizer.zero_grad()
            logits = model(X)  # (B, T_max, F)
            loss_mat = criterion(logits, y)  # per-element loss
            # Apply mask
            masked = loss_mat * mask.unsqueeze(-1)
            # Sum of losses
            sum_loss = masked.sum()
            # Count of valid elements
            num_elems = mask.sum().item() * logits.size(-1)
            # Normalize
            loss = sum_loss / num_elems
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()

            total_loss += sum_loss.item()
            total_elems += num_elems

        avg_loss = total_loss / total_elems
        print(f"Epoch {epoch}/{config['training']['epochs']} - Loss: {avg_loss:.6f}")
        # Save checkpoint
        ckpt_dir = config['training']['ckpt_dir']
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_epoch{epoch}.pt"))


def main():
    parser = argparse.ArgumentParser(description="Train 7×3 Music Transformer")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == '__main__':
    main()
