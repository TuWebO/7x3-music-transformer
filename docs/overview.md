# 7×3 Music Representation and AI Pipeline

This document consolidates the core concepts, mathematical foundations, visualization strategies, and algorithmic/AI integration for a unified, key-aware 7×3 grid music representation.

---

## 1. 7×3 Grid Representation (Structural Layer)

* **Rows** (`i = 1..7`): natural-note classes in order: C, D, E, F, G, A, B
* **Columns** (`j ∈ {–1, 0, +1}`): relative accidentals

  * `j = –1`: flat (♭)
  * `j = 0`: default (no written accidental) in current key
  * `j = +1`: sharp (♯)
* **Octave index** (`o_idx ∈ 0..O–1`): index into the chosen octave list

A single note maps to a grid coordinate:

```
g(n) = (i, j_rel, o_idx)
  where i = diatonic row,
        j_rel = relative accidental in key (–1,0,+1),
        o_idx = index of octave layer
```

---

## 2. Acoustic Scalar Mapping (Pitch Height)

We map each note to a monotonic integer `s(n)` to capture true pitch distance:

1. **Equal divisions per octave**:

   ```
   D = 12  # (12-tone equal temperament by default)
   ```
2. **Diatonic bases** (12-EDO naturals):

   ```
   base = [0,2,4,5,7,9,11]
   ```
3. **Index within octave**:

   ```
   idx(i,j) = base[i] * (D/12) + j * (D/12)
   ```
4. **Full scalar**:

   ```
   s(n) = D*(octave - o0) + idx(i,j)
   ```

This produces MIDI-style pitch numbers and generalizes to any `D`-EDO tuning.

---

## 3. Dual Metrics

* **Grid-metric** `d_g(g1,g2)`: Manhattan distance on the 7×3×O lattice:

  ```
  d_g = |i2 - i1| + |j2 - j1| + |o2 - o1|
  ```
* **Acoustic-metric** `d_s(n1,n2)`: true semitone distance:

  ```
  d_s = |s(n2) - s(n1)|
  ```

Use `d_g` for visualization and combinatorial tasks; use `d_s` for harmonic and AI analysis.

---

## 4. Key Signature Shifts

Each major key `k` has a **shift vector** `m_k[1..7] ∈ {–1,0,+1}` for {C,D,E,F,G,A,B}:

```text
C  : [0,0,0,0,0,0,0]
G  : [0,0,0,1,0,0,0]
D  : [1,0,0,1,0,0,0]
...    (F: [0,0,0,0,0,0,-1])
```

The **relative column** becomes:

```
j_rel = j_abs – m_k[i]
```

Default (j\_abs=0) notes land at column `m_k[i]`, maintaining a uniform 7×3×O lattice across keys.

---

## 5. Visual Examples

**C Major** (all defaults in center column)

```
C♭ | **C** | C♯
D♭ | **D** | D♯
... etc.
```

**G Major** (F shifts right):

```
F♮ | **F♯** | F𝄪
```

Read columns as (flat | default | sharp) relative to the key.

---

## 6. Metric Comparison

| Interval    | Grid-metric `d_g` | Acoustic-metric `d_s` |
| ----------- | ----------------- | --------------------- |
| Minor 2nd   | 1                 | 1                     |
| Major 2nd   | 1                 | 2                     |
| Perfect 5th | 4                 | 7                     |
| Octave leap | 1                 | 12                    |

Visual adjacency ≠ true pitch proximity, motivating dual metrics.

---

## 7. Algorithmic & AI Integration

### Data Pipeline

1. Parse MIDI/MusicXML → quantized time frames
2. Convert active notes → 7×3×O binary grid
3. (Optional) Append `s(n)` scalar arrays and key one-hot
4. Flatten to `(T, F)` feature sequences

### Model Architectures

* RNNs / Transformers for sequence modeling
* CNNs on grid-frames (2D/3D) for pattern detection
* VAEs / Diffusion for generative tasks

### Example (PyTorch)

```python
# Input: X shape (T, B, F)
# TransformerEncoder(..., batch_first=True)
```

---

## 8. Implementation Roadmap

1. **Dataset**: e.g. Bach Chorales, MAESTRO
2. **Preprocessing**: `src/preprocess.py` with `pretty_midi`
3. **Positional Encodings**: add to `d_model`
4. **Model**: `src/models.py` Transformer
5. **Train / Eval**: `src/train.py`, `src/evaluate.py` with `BCEWithLogitsLoss`
6. **Tests**: `tests/test_preprocess.py` etc.
7. **Demo**: Streamlit / Gradio in `demos/`

---

## 9. Platforms & Deployment

* **Code & Env**: Python, Jupyter, VS Code
* **Libraries**: `pretty_midi`, `torch`, `transformers`
* **Visualization**: Matplotlib, Plotly, VexFlow
* **Deployment**: Hugging Face Hub, Replicate, Streamlit/Gradio

---

*This file serves as the canonical reference and can be checked into the repository under `docs/overview.md`.*
