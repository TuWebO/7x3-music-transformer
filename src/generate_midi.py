import os
import argparse
import json
import numpy as np
import torch
import pretty_midi
import yaml
import matplotlib.pyplot as plt
from models import NextNoteTransformer

DIATONIC_BASE = [0, 2, 4, 5, 7, 9, 11]
STEP = 0.125


def load_grid_and_metadata(data_dir, fname):
    with open(os.path.join(data_dir, 'metadata.json')) as f:
        metadata = json.load(f)
    info = metadata[fname]
    npz = np.load(os.path.join(data_dir, info['file']))
    grid = npz['grid']  # (T, 7, 3, O)
    return grid, info, metadata


def expand_grid_to_global(grid, local_octaves, global_octaves):
    T, R, C, O = grid.shape
    G = len(global_octaves)
    expanded = np.zeros((T, R, C, G), dtype=grid.dtype)
    for i, octv in enumerate(local_octaves):
        g = global_octaves.index(octv)
        expanded[:, :, :, g] = grid[:, :, :, i]
    return expanded


def predict_grid(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(x.unsqueeze(0))  # (1, T, F)
        probs = torch.sigmoid(logits)[0]  # (T, F)
        return (probs > 0.5).int().numpy()  # binary grid (T, F)


def grid_to_note_events(grid_flat, octaves):
    T = grid_flat.shape[0]
    G = len(octaves)
    frame_grid = grid_flat.reshape(T, 7, 3, G)
    notes = []
    for t in range(T):
        for i in range(7):
            for j in range(3):
                for g in range(G):
                    if frame_grid[t, i, j, g]:
                        j_rel = j - 1
                        midi = 12 * (octaves[g] + 1) + DIATONIC_BASE[i] + j_rel
                        start = t * STEP
                        end = (t + 1) * STEP
                        notes.append((start, end, midi))
    return notes


def write_midi(notes, out_path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for start, end, pitch in notes:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    pm.write(out_path)


def plot_comparison(pred_notes, orig_notes):
    def plot(notes, label, color):
        if not notes: return
        starts, pitches = zip(*[(n[0], n[2]) for n in notes])
        plt.scatter(starts, pitches, s=5, label=label, alpha=0.6, color=color)

    plt.figure(figsize=(12, 5))
    plot(orig_notes, 'Original', 'blue')
    plot(pred_notes, 'Predicted', 'red')
    plt.xlabel("Time (s)")
    plt.ylabel("MIDI Pitch")
    plt.title("Prediction vs Ground Truth (Piano Roll)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate MIDI from trained model and grid file")
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--fname', help='Original .midi filename from metadata.json')
    parser.add_argument('--out_midi', default='predicted.mid')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load grid and metadata
    fname = args.fname
    grid, info, metadata = load_grid_and_metadata(args.data_dir, fname)

    # Build global octave list
    all_octaves = sorted(set(o for v in metadata.values() for o in v['octaves']))

    # Expand grid to global octave layout
    expanded = expand_grid_to_global(grid, info['octaves'], all_octaves)
    x = torch.tensor(expanded.reshape(grid.shape[0], -1), dtype=torch.float32)

    # Load model
    model = NextNoteTransformer(
        feature_dim=x.shape[1],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        max_len=config['training']['max_len'],
        grid_dims=x.shape[1],
        batch_first=True
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    # Predict
    pred = predict_grid(model, x)
    pred_notes = grid_to_note_events(pred, all_octaves)
    orig_notes = grid_to_note_events(x.numpy().round().astype(int), all_octaves)

    # Write MIDI
    write_midi(pred_notes, args.out_midi)
    print(f"✅ Saved predicted MIDI to {args.out_midi}")

    if args.plot:
        plot_comparison(pred_notes, orig_notes)


if __name__ == '__main__':
    main()
