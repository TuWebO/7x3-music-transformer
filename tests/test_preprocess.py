import os
import numpy as np
import pretty_midi
import pytest

from preprocess import pitch_to_grid, midi_to_grid, KEY_SHIFTS

# MIDI note numbers for reference
# C4 = 60, D4 = 62, E4 = 64, F4 = 65, G4 = 67, A4 = 69, B4 = 71, C#4 = 61, Bb4 = 70

@pytest.mark.parametrize("pitch,expected", [
    # Natural notes in C major, octave 4
    (60, (0, 0, 0)),  # C4 -> i=0 (C), j_rel=0, o_idx=0
    (62, (1, 0, 0)),  # D4 -> i=1 (D)
    (64, (2, 0, 0)),  # E4 -> i=2 (E)
    (65, (3, 0, 0)),  # F4 -> i=3 (F)
    (67, (4, 0, 0)),  # G4 -> i=4 (G)
    (69, (5, 0, 0)),  # A4 -> i=5 (A)
    (71, (6, 0, 0)),  # B4 -> i=6 (B)
    # Sharps and flats
    (61, (0, 1, 0)),  # C#4 -> C row, j_abs=+1, j_rel=1
    (70, (6, -1, 0)), # Bb4 -> B row, j_abs=-1, j_rel=-1
])
def test_pitch_to_grid_c_major(pitch, expected):
    key_shift = KEY_SHIFTS['C']
    octaves = [4]
    result = pitch_to_grid(pitch, key_shift, octaves)
    assert result == expected


def create_test_midi(path, pitch=60, start=0.0, end=1.0):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
    inst.notes.append(note)
    pm.instruments.append(inst)
    pm.write(path)


@pytest.mark.parametrize("pitch", [60])
def test_midi_to_grid(tmp_path, pitch):
    # Create a temporary MIDI file with a single note lasting 1 second
    midi_file = tmp_path / "test.mid"
    create_test_midi(str(midi_file), pitch=pitch, start=0.0, end=1.0)

    # Convert to grid with octaves [4] and C major
    grid_seq = midi_to_grid(str(midi_file), octaves=[4], key='C')

    # Assert type and shape: 1 second / 0.125 step = 8 frames, grid dims 7x3x1
    assert isinstance(grid_seq, np.ndarray)
    assert grid_seq.shape == (8, 7, 3, 1)

    # All frames should have a 1 at (i=0, j_rel+1=1, o_idx=0)
    for frame in grid_seq:
        assert frame[0, 1, 0] == 1
        # all other positions must be 0
        tmp = frame.copy()
        tmp[0, 1, 0] = 0
        assert tmp.sum() == 0


def test_pitch_out_of_octave_returns_none():
    # Pitch outside specified octaves
    key_shift = KEY_SHIFTS['C']
    result = pitch_to_grid(60, key_shift, octaves=[3,5])
    assert result is None


def test_unsupported_key_raises(tmp_path):
    midi_file = tmp_path / "dummy.mid"
    with pytest.raises(ValueError):
        midi_to_grid(str(midi_file), octaves=[4], key='Z')
