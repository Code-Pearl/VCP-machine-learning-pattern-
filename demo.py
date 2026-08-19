"""
End-to-end demo: generate synthetic data, train the SOM-based VCP detector,
evaluate it against known ground truth, and render example charts.

Run: python3 demo.py
Outputs go to ./output/
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from synthetic_data import generate_universe
from detector import VCPDetector

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_series_with_detections(df, meta, results, top_hits, sym, path, lookback):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    dates = df.index
    ax1.plot(dates, df["Close"], color="#1f4e8c", linewidth=1.1, label="Close")
    if getattr(meta, "is_vcp", False):
        pivot_date = dates[meta.base_end_idx]
        ax1.axvline(pivot_date, color="#888888", linestyle="--", linewidth=1, alpha=0.7,
                    label="true pattern (ground truth)")
        if meta.breakout_idx is not None:
            ax1.axvline(dates[meta.breakout_idx], color="#2e8b57", linestyle=":", linewidth=1.2,
                        label="breakout (ground truth)")

    for i, hit in enumerate(top_hits):
        end_idx = hit["end_idx"]
        start_idx = max(0, end_idx - lookback + 1)
        color = "#d1495b" if i == 0 else "#e0a458"
        ax1.axvspan(dates[start_idx], dates[end_idx], color=color, alpha=0.15)
        ax1.annotate(
            f"match {hit['match_score']:.2f}",
            xy=(dates[end_idx], df["Close"].iloc[end_idx]),
            xytext=(0, 14), textcoords="offset points",
            fontsize=8, color=color, ha="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    ax1.set_ylabel("Price")
    ax1.set_title(f"{sym}  ({meta.pattern})")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    ax2.bar(dates, df["Volume"], color="#7f9fc9", width=1.2)
    ax2.set_ylabel("Volume")

    if results:
        xs = [dates[r["end_idx"]] for r in results]
        ys = [r["match_score"] for r in results]
        ax3.plot(xs, ys, color="#333333", linewidth=1.0)
        ax3.axhline(0.35, color="#d1495b", linestyle="--", linewidth=0.8, label="threshold 0.35")
        ax3.legend(loc="upper left", fontsize=8)
    ax3.set_ylabel("match_score")
    ax3.set_ylim(0, 1)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_som_affinity(det: VCPDetector, path: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(det.neuron_affinity_.T, origin="lower", cmap="RdYlGn_r" if False else "viridis",
                    vmin=0, vmax=max(0.05, det.neuron_affinity_.max()))
    ax.set_title("SOM neuron VCP-affinity\n(fraction of assigned windows near a true VCP pivot)")
    ax.set_xlabel("SOM grid x")
    ax.set_ylabel("SOM grid y")
    fig.colorbar(im, ax=ax, label="affinity")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    print("Generating synthetic training universe...")
    train_data, train_meta = generate_universe(n_vcp=30, n_negative_each=12, seed=1, total_days=500)
    print("Generating synthetic held-out test universe...")
    test_data, test_meta = generate_universe(n_vcp=20, n_negative_each=10, seed=999, total_days=500)

    det = VCPDetector(grid_x=8, grid_y=8, lookback=160, scan_step=5)
    print(f"Training SOM on {len(det.collect_training_samples(train_data, train_meta))} windows...")
    det.fit(train_data, train_meta, n_epochs=100, seed=0)
    print(f"  SOM quantization error: {det.som.quantization_error(det._fit_Xs):.4f}")

    plot_som_affinity(det, os.path.join(OUT_DIR, "som_affinity_map.png"))

    # ---- held-out evaluation ----
    TOL = 15
    pos_scores, neg_scores = [], []
    for sym, m in test_meta.items():
        df = test_data[sym]
        results = det.scan_series(df)
        if m.is_vcp:
            near = [r["match_score"] for r in results if abs(r["end_idx"] - m.base_end_idx) <= TOL]
            pos_scores.append(max(near) if near else 0.0)
        else:
            neg_scores.append(max((r["match_score"] for r in results), default=0.0))

    print("\n=== Held-out evaluation (seed=999, unseen during training) ===")
    for thr in (0.30, 0.35, 0.40, 0.45, 0.50):
        recall = np.mean([s >= thr for s in pos_scores])
        fpr = np.mean([s >= thr for s in neg_scores])
        print(f"  threshold {thr:.2f}:  recall={recall:.2f}   false-positive-rate={fpr:.2f}")

    # ---- example charts: a couple of clean detections + a false-positive case ----
    print("\nRendering example charts...")
    vcp_syms = sorted([s for s, m in test_meta.items() if m.is_vcp])
    for sym in vcp_syms[:3]:
        df, m = test_data[sym], test_meta[sym]
        results = det.scan_series(df)
        top_hits = det.top_matches(df, top_k=2)
        plot_series_with_detections(df, m, results, top_hits, sym,
                                     os.path.join(OUT_DIR, f"example_{sym}.png"), det.lookback)

    neg_syms = sorted([s for s, m in test_meta.items() if not m.is_vcp])
    # show the worst false-positive-prone negative for honesty
    worst_neg, worst_score = None, -1
    for sym in neg_syms:
        df = test_data[sym]
        results = det.scan_series(df)
        best = max((r["match_score"] for r in results), default=0.0)
        if best > worst_score:
            worst_score, worst_neg = best, sym
    df, m = test_data[worst_neg], test_meta[worst_neg]
    results = det.scan_series(df)
    top_hits = det.top_matches(df, top_k=2)
    plot_series_with_detections(df, m, results, top_hits, worst_neg,
                                 os.path.join(OUT_DIR, f"example_{worst_neg}_false_positive_case.png"),
                                 det.lookback)

    print(f"\nDone. Charts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
