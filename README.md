# VCP-machine-learning-pattern
### VCP Detector with Self-Organizing Map

An **unsupervised learning pipeline** that detects **Volatility Contraction Patterns (VCP)** in daily OHLCV financial data. The detector combines a **rule-based scoring system** with a **Self-Organizing Map (SOM)** to learn shape-based similarities from price/volume structures, without ever seeing labels during training. Labels are used only to interpret the SOM's neurons.

This project is **research‑oriented** – it uses synthetic data with known ground truth to validate the logic and is **not** a production‑ready trading signal.

---

## Pipeline Overview

```
Daily OHLCV Series
        │
        ▼
┌───────────────────┐
│  Sliding Window   │  (lookback = 160 bars, step = 5)
└────────┬──────────┘
         ▼
┌───────────────────┐
│  Feature Extract  │  (10 shape features + rule sub‑scores)
└────────┬──────────┘
         ▼
┌───────────────────┐       ┌───────────────────┐
│   Standardize     │ ──▶   │   Train SOM       │  (unsupervised)
└───────────────────┘       └────────┬──────────┘
                                      ▼
                      ┌───────────────────────────┐
                      │  Label Neurons with       │
                      │  Affinity to True VCPs    │  (uses synthetic ground truth)
                      └───────────┬───────────────┘
                                  ▼
                      ┌───────────────────────────┐
                      │  Score New Window =       │
                      │  0.5×rule_score +         │
                      │  0.5×neuron_affinity      │
                      └───────────────────────────┘
```

---

## Key Features

- **Unsupervised shape learning** – SOM organizes windows purely by vector similarity; no labels influence training.
- **Interpretable rule score** – built from named sub‑scores (contraction, volume dry‑up, prior uptrend, etc.) – provides a grounded baseline.
- **Synthetic data generator** – creates VCP and challenging negative series with known pivot locations, enabling honest evaluation.
- **Quantitative evaluation** – reports recall vs. false‑positive rate on a held‑out test set.
- **Visualization** – affinity map, series charts with detected windows, and match scores over time.

---

## Project Structure

```
.
├── detector.py         # Main detector class (VCPDetector)
├── features.py         # Window feature extraction & rule scoring
├── som.py              # Simple 2D Self‑Organizing Map implementation
├── synthetic_data.py   # Generate synthetic VCP/negative series with ground truth
├── demo.py             # End‑to‑end training + evaluation + chart generation
└── output/             # Created by demo.py – charts and affinity map
```

---

## Installation

```bash
git clone https://github.com/yourusername/vcp-detector-som.git
cd vcp-detector-som
pip install numpy pandas matplotlib
```

All dependencies are pure Python with no heavy ML frameworks.

---

## Usage

### Run the Demo

```bash
python demo.py
```

This will:
- Generate a training universe (VCP + negative series) with known labels.
- Train the SOM and label neurons.
- Evaluate on a separate held‑out universe.
- Print recall/FPR at several thresholds.
- Produce charts in the `output/` directory:
  - `som_affinity_map.png` – neuron VCP‑affinity heatmap.
  - `example_VCP*.png` – three VCP series with top matches.
  - `example_*_false_positive_case.png` – a negative series with the highest false‑positive score.

### Use the Detector in Your Own Code

```python
from detector import VCPDetector
from synthetic_data import generate_universe

# Generate or load your data (dict of DataFrames, dict of Meta)
data, meta = generate_universe(n_vcp=30, n_negative_each=12)

det = VCPDetector(grid_x=8, grid_y=8, lookback=160, scan_step=5)
det.fit(data, meta, n_epochs=100)

# Score a single window
result = det.score_window(data["VCP000"], end_idx=250)
print(result["match_score"])

# Scan an entire series
results = det.scan_series(data["VCP000"])
top_hits = det.top_matches(data["VCP000"], top_k=3)
```

### Required Input Format

- `data`: `dict[symbol] = pandas.DataFrame` with columns `['Open','High','Low','Close','Volume']` and a **DatetimeIndex**.
- `meta`: `dict[symbol] = SeriesMeta` (from `synthetic_data.py`) or an object with at least:
  - `is_vcp`: `bool`
  - `base_end_idx`: index of the known pivot/pattern end (used for labeling neurons).

---

## Components in Detail

### `features.py`

- **Zigzag pivots** – percentage‑threshold swing detection (standard approach).
- **Pullback legs** – extracted from zigzag as `H→L` sequences.
- **10‑dimensional feature vector**:
  - number of legs, mean depth ratio, contraction monotonicity, last leg depth, volume trend slope, volume dry‑up ratio, ATR contraction, prior advance %, proximity to pivot, above‑MA score.
- **Rule‑based score** – weighted combination of 8 sub‑scores, each [0‑1], designed to reflect VCP characteristics.

### `som.py`

- Pure‑NumPy 2D SOM with Gaussian neighbourhood and exponential decay.
- Training is unsupervised – weights updated to match input vectors.
- Provides BMU lookup, quantization error, and U‑matrix.

### `detector.py`

- **`collect_training_samples`** – slides a window over all training series, extracts features, and marks whether the window is near a true VCP pivot (ground truth).
- **`fit`** – standardizes features, trains the SOM, then computes for each neuron the fraction of its assigned windows that were near a true pivot → `neuron_affinity_`.
- **`score_window`** – normalises the vector, finds BMU, blends rule score and neuron affinity.
- **`top_matches`** – returns top‑K windows with non‑overlapping de‑duplication.

### `synthetic_data.py`

Generates two families:

- **VCP** – prior uptrend + `N` contracting pullbacks (each shallower than the last) + volume dry‑up + optional breakout.
- **Negatives** – `random_walk`, `strong_uptrend_no_pullback`, `choppy_equal_swings`, `downtrend`, `expanding_volatility_top`.

Each series returns a `SeriesMeta` with known `base_end_idx` (the pivot) and `is_vcp` flag, enabling honest evaluation.

---

## Evaluation (Example from Demo)

On a held‑out test universe (20 VCP, 50 negative series), the demo reports:

```
threshold 0.30:  recall=0.91   false-positive-rate=0.12
threshold 0.35:  recall=0.85   false-positive-rate=0.08
threshold 0.40:  recall=0.75   false-positive-rate=0.04
threshold 0.45:  recall=0.61   false-positive-rate=0.02
threshold 0.50:  recall=0.48   false-positive-rate=0.01
```

These numbers are **illustrative** – they depend on the synthetic generation parameters and can be tuned.

---

## Visualisation Outputs

- **Affinity Map** – shows which SOM neurons are most associated with true VCP windows (warmer = higher affinity).
- **Series Charts** – each chart shows:
  - Price with ground‑truth pivot and breakout (if any).
  - Highlighted top‑matching windows (with score).
  - Volume bars.
  - Match score over time (with a 0.35 threshold line).

---

## Limitations & Future Work

- **Synthetic data only** – performance on real market data may differ significantly; the pipeline should be validated on historical data before any trading use.
- **No regime handling** – the detector treats all windows equally; market regime changes can affect feature distributions.
- **Parameter sensitivity** – `lookback`, `pct_threshold` for zigzag, SOM grid size, and blending weight all affect results.
- **Feature engineering** – additional features (e.g., volatility, relative strength) could improve discrimination.
- **Real‑time scoring** – the current implementation is batch‑oriented; with minor modifications it can be used for online scoring.

---

![example_RAND004_false_positive_case](/example_RAND004_false_positive_case.png)
![example_VCP000](/example_VCP000.png)
![som_affinity_map](/som_affinity_map.png)


---

## License

This project is provided under the **MIT License**. Feel free to use, modify, and distribute it for research and educational purposes.

---

## Acknowledgements

Built with Python, NumPy, pandas, and Matplotlib.

**Disclaimer**: This is a research prototype. It is not financial advice and should not be used to make real trading decisions.
```
