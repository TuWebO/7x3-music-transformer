import os
import argparse
import numpy as np
import pretty_midi
import json

# Base semitone intervals for natural notes C, D, E, F, G, A, B in 12-EDO
DIATONIC_BASE = [0, 2, 4, 5, 7, 9, 11]

# Major key signature shifts: +1 = sharp, -1 = flat, 0 = natural
KEY_SHIFTS = {
    "C":  [0,0,0,0,0,0,0],
    "G":  [0,0,0,1,0,0,0],
    "D":  [1,0,0,1,0,0,0],
    "A":  [1,0,0,1,1,0,0],
    "E":  [1,1,0,1,1,0,0],
    "B":  [1,1,0,1,1,1,0],
    "F#": [1,1,1,1,1,1,0],
    "C#": [1,1,1,1,1,1,1],
    "F":  [0,0,0,0,0,0,-1],
    "Bb": [0,0,-1,0,0,0,-1],
    "Eb": [0,0,-1,0,0,-1,-1],
    "Ab": [0,-1,-1,0,0,-1,-1],
    "Db": [0,0,-1,-1,-1,-1,-1],
    "Gb": [0,-1,-1,-1,-1,-1,-1],
    "Cb": [-1,-1,-1,-1,-1,-1,-1]
}


def pitch_to_grid(pitch, key_shift, octaves, o0=0):
    """
    Convert a MIDI pitch number to grid indices (i, j_rel, o_idx).
    - pitch: MIDI pitch number (0-127)
    - key_shift: list of length 7, KEY_SHIFTS for selected key
    - octaves: list of integer octave numbers to include
    Returns: (diatonic index i, relative accidental j_rel, octave index o_idx)
    """
    # Determine octave number and index
    octave = pitch // 12 - 1
    if octave not in octaves:
        return None
    o_idx = octaves.index(octave)

    # Compute semitone within octave
    semitone = pitch % 12

    # Find diatonic index i and absolute accidental j_abs
    # Search for nearest DIATONIC_BASE + k (k in -1, 0, 1)
    i = None
    j_abs = None
    for idx, base in enumerate(DIATONIC_BASE):
        for j in (-1, 0, 1):
            if (base + j) % 12 == semitone:
                i = idx
                j_abs = j
                break
        if i is not None:
            break
    if i is None:
        # pitch not accounted by single accidental
        return None

    # Compute relative accidental by subtracting key shift
    j_rel = j_abs - key_shift[i]
    if j_rel < -1 or j_rel > 1:
        # out of grid (double accidental)
        return None

    return i, j_rel, o_idx


def midi_to_grid(midi_file, octaves, key):
    """
    Parse a MIDI file and convert to a time-series of grid frames.
    Returns a numpy array of shape (T, 7, 3, len(octaves)).
    """
    key_shift = KEY_SHIFTS.get(key)
    if key_shift is None:
        raise ValueError(f"Unsupported key: {key}")

    pm = pretty_midi.PrettyMIDI(midi_file)
    end_time = pm.get_end_time()
    step = 0.125  # quantize to 16th notes
    times = np.arange(0, end_time, step)

    frames = []
    for t in times:
        grid = np.zeros((7, 3, len(octaves)), dtype=np.uint8)
        for inst in pm.instruments:
            for note in inst.notes:
                if note.start <= t < note.end:
                    res = pitch_to_grid(note.pitch, key_shift, octaves)
                    if res is not None:
                        i, j_rel, o_idx = res
                        grid[i, j_rel + 1, o_idx] = 1
        frames.append(grid)

    return np.stack(frames)


def main(args):
    os.makedirs(args.output, exist_ok=True)
    metadata = {}
    for fname in os.listdir(args.input):
        if not fname.lower().endswith(('.mid', '.midi')):
            continue
        in_path = os.path.join(args.input, fname)
        print(f"Processing {fname}...")
        grid_seq = midi_to_grid(in_path, args.octaves, args.key)
        # save as compressed numpy
        out_fname = os.path.splitext(fname)[0] + '.npz'
        out_path = os.path.join(args.output, out_fname)
        np.savez_compressed(out_path, grid=grid_seq)
        metadata[fname] = out_fname

    # dump metadata mapping
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert MIDI files to 7x3xO grid sequences."
    )
    parser.add_argument('--input', required=True, help='Input directory of MIDI files')
    parser.add_argument('--output', required=True, help='Output directory for grid .npz files')
    parser.add_argument('--octaves', nargs='+', type=int, default=[3,4,5],
                        help='List of octave numbers to include')
    parser.add_argument('--key', default='C', help='Major key signature (e.g. C, G, D, F#)')
    args = parser.parse_args()
    main(args)
