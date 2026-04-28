"""
compare_runoff_models.py

Trains GRU, LSTM, and Transformer on both stations and produces a full suite
of comparison charts in runoff/comparison/.

Charts produced
---------------
1. comparison_rmse_by_lead.png        - RMSE by lead hour (line, all models)
2. comparison_mae_by_lead.png         - MAE by lead hour (line, all models)
3. comparison_training_loss.png       - Train/val loss curves
4. comparison_overall_metrics.png     - Grouped bar: mean RMSE & MAE per model
5. comparison_scatter.png             - 3×2 scatter: predicted vs true error
6. comparison_corrected_flow_<id>.png - Time-series corrected streamflow
7. comparison_rmse_reduction.png      - % RMSE reduction vs NWM baseline
8. comparison_rmse_heatmap.png        - Heatmap: RMSE (model × lead hour)
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import keras
keras.utils.set_random_seed(812)

# ---------------------------------------------------------------------------
# Path setup — insert each sub-package so its imports resolve
# ---------------------------------------------------------------------------
_RUNOFF_DIR      = os.path.dirname(os.path.abspath(__file__))
_GRU_DIR         = os.path.join(_RUNOFF_DIR, "gru")
_LSTM_DIR        = os.path.join(_RUNOFF_DIR, "lstm")
_TRANSFORMER_DIR = os.path.join(_RUNOFF_DIR, "transformer")
_RESULTS_DIR     = os.path.join(_RUNOFF_DIR, "comparison")
os.makedirs(_RESULTS_DIR, exist_ok=True)

for _p in [_RUNOFF_DIR, _GRU_DIR, _LSTM_DIR, _TRANSFORMER_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runoff_data_process import (
    process_station,
    STATION_A_NWM_PATH, STATION_A_USGS_PATH,
    STATION_B_NWM_PATH, STATION_B_USGS_PATH,
)
from gru_runoff_model         import build_runoff_gru
from lstm_runoff_model        import build_runoff_lstm
from transformer_runoff_model import build_runoff_transformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LEAD_HOURS    = list(range(1, 19))
BATCH_SIZE    = 64
EPOCHS        = 100
LR            = 1e-3
PATIENCE      = 10

MODEL_COLORS  = {"GRU": "steelblue", "LSTM": "tomato", "Transformer": "mediumseagreen"}
MODEL_MARKERS = {"GRU": "o",         "LSTM": "s",       "Transformer": "^"}
STATION_NAMES = ["20380357 (09520500)", "21609641 (11266500)"]

_USGS_LEAD_COLS = [f"USGS_at_lead_{h}h" for h in range(1, 19)]


# ---------------------------------------------------------------------------
# Weighted MSE loss (Transformer)
# ---------------------------------------------------------------------------
def high_flow_weighted_mse(alpha=5.0, flow_scale=100.0):
    def loss(y_true, y_pred):
        true_errors = y_true[:, :18]
        usgs_obs    = y_true[:, 18:]
        weights     = 1.0 + alpha * (usgs_obs / flow_scale)
        sq_err      = keras.ops.square(y_pred - true_errors)
        return keras.ops.mean(weights * sq_err)
    return loss


# ---------------------------------------------------------------------------
# Shared callback factory
# ---------------------------------------------------------------------------
def _callbacks(save_path):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=0
        ),
        keras.callbacks.ModelCheckpoint(
            save_path, monitor="val_loss", save_best_only=True, verbose=0
        ),
    ]


# ---------------------------------------------------------------------------
# Per-model training functions
# ---------------------------------------------------------------------------
def _concat(data, split):
    return np.concatenate([data[f"X_nwm_{split}"], data[f"X_usgs_{split}"]], axis=1)


def train_gru(data, save_path):
    model = build_runoff_gru()
    model.compile(optimizer=keras.optimizers.Adam(LR),
                  loss="MeanSquaredError", metrics=["mse"])
    t0 = time.time()
    history = model.fit(
        _concat(data, "train"), data["y_train"],
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_data=(_concat(data, "val"), data["y_val"]),
        callbacks=_callbacks(save_path), verbose=0,
    )
    elapsed = time.time() - t0
    preds = model.predict(_concat(data, "test"), verbose=0)
    return model, history, preds, elapsed


def train_lstm(data, save_path):
    model = build_runoff_lstm()
    model.compile(optimizer=keras.optimizers.Adam(LR),
                  loss="MeanSquaredError", metrics=["mse"])
    t0 = time.time()
    history = model.fit(
        _concat(data, "train"), data["y_train"],
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_data=(_concat(data, "val"), data["y_val"]),
        callbacks=_callbacks(save_path), verbose=0,
    )
    elapsed = time.time() - t0
    preds = model.predict(_concat(data, "test"), verbose=0)
    return model, history, preds, elapsed


def train_transformer(data, save_path):
    y_train_aug = np.concatenate(
        [data["y_train"], data["train_df"][_USGS_LEAD_COLS].values], axis=1
    )
    y_val_aug = np.concatenate(
        [data["y_val"], data["val_df"][_USGS_LEAD_COLS].values], axis=1
    )
    model = build_runoff_transformer()
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=high_flow_weighted_mse(alpha=5.0, flow_scale=100.0),
    )
    t0 = time.time()
    history = model.fit(
        [data["X_nwm_train"], data["X_usgs_train"]], y_train_aug,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_data=([data["X_nwm_val"], data["X_usgs_val"]], y_val_aug),
        callbacks=_callbacks(save_path), verbose=0,
    )
    elapsed = time.time() - t0
    preds = model.predict([data["X_nwm_test"], data["X_usgs_test"]], verbose=0)
    return model, history, preds, elapsed


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def compute_metrics(preds, y_test):
    rmse = np.sqrt(np.mean((preds - y_test) ** 2, axis=0))
    mae  = np.mean(np.abs(preds - y_test), axis=0)
    return rmse, mae


def nwm_baseline_metrics(data):
    """NWM with no correction — predicting zero error — gives RMSE = std(y_test)."""
    rmse = np.sqrt(np.mean(data["y_test"] ** 2, axis=0))
    mae  = np.mean(np.abs(data["y_test"]), axis=0)
    return rmse, mae


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _save(fig, name):
    path = os.path.join(_RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Chart 1 — RMSE by lead hour (line plot)
# ---------------------------------------------------------------------------
def plot_rmse_by_lead(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)

    for ax, sname in zip(axes, STATION_NAMES):
        ax.plot(LEAD_HOURS, results[sname]["baseline_rmse"],
                color="gray", linestyle="--", linewidth=1.5,
                label="NWM Baseline", zorder=1)

        for m in ("GRU", "LSTM", "Transformer"):
            ax.plot(LEAD_HOURS, results[sname][m]["rmse"],
                    color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                    markersize=5, linewidth=1.8, label=m, zorder=2)

        sid = sname.split()[0]
        ax.set_title(f"Station {sid} — RMSE by Lead Hour")
        ax.set_xlabel("Lead Hour")
        ax.set_ylabel("RMSE  (m³/s)")
        ax.set_xticks(LEAD_HOURS)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Error-Correction RMSE by Lead Hour — All Models", fontsize=13)
    plt.tight_layout()
    _save(fig, "comparison_rmse_by_lead.png")


# ---------------------------------------------------------------------------
# Chart 2 — MAE by lead hour (line plot)
# ---------------------------------------------------------------------------
def plot_mae_by_lead(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)

    for ax, sname in zip(axes, STATION_NAMES):
        ax.plot(LEAD_HOURS, results[sname]["baseline_mae"],
                color="gray", linestyle="--", linewidth=1.5,
                label="NWM Baseline", zorder=1)

        for m in ("GRU", "LSTM", "Transformer"):
            ax.plot(LEAD_HOURS, results[sname][m]["mae"],
                    color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                    markersize=5, linewidth=1.8, label=m, zorder=2)

        sid = sname.split()[0]
        ax.set_title(f"Station {sid} — MAE by Lead Hour")
        ax.set_xlabel("Lead Hour")
        ax.set_ylabel("MAE  (m³/s)")
        ax.set_xticks(LEAD_HOURS)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Error-Correction MAE by Lead Hour — All Models", fontsize=13)
    plt.tight_layout()
    _save(fig, "comparison_mae_by_lead.png")


# ---------------------------------------------------------------------------
# Chart 3 — Training & validation loss curves
# ---------------------------------------------------------------------------
def plot_training_loss(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, sname in zip(axes, STATION_NAMES):
        for m in ("GRU", "LSTM", "Transformer"):
            hist = results[sname][m]["history"].history
            ep   = range(1, len(hist["loss"]) + 1)
            ax.plot(ep, hist["loss"],
                    color=MODEL_COLORS[m], linewidth=1.5, label=f"{m} Train")
            ax.plot(ep, hist["val_loss"],
                    color=MODEL_COLORS[m], linewidth=1.5, linestyle=":",
                    label=f"{m} Val")

        sid = sname.split()[0]
        ax.set_title(f"Station {sid} — Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Training & Validation Loss  (note: Transformer uses weighted MSE)",
        fontsize=12
    )
    plt.tight_layout()
    _save(fig, "comparison_training_loss.png")


# ---------------------------------------------------------------------------
# Chart 4 — Overall mean metrics: grouped bar chart
# ---------------------------------------------------------------------------
def plot_overall_metrics(results):
    model_names_full = ["NWM Baseline", "GRU", "LSTM", "Transformer"]
    all_colors       = ["gray"] + [MODEL_COLORS[m] for m in ("GRU", "LSTM", "Transformer")]

    x       = np.arange(2)
    width   = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric, ax, ylabel, title_suffix in [
        ("rmse", axes[0], "Mean RMSE  (m³/s)", "Mean RMSE"),
        ("mae",  axes[1], "Mean MAE  (m³/s)",  "Mean MAE"),
    ]:
        for i, (mname, color) in enumerate(zip(model_names_full, all_colors)):
            if mname == "NWM Baseline":
                vals = [results[s][f"baseline_{metric}"].mean() for s in STATION_NAMES]
            else:
                vals = [results[s][mname][metric].mean() for s in STATION_NAMES]

            bars = ax.bar(x + offsets[i], vals, width,
                          label=mname, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.15,
                        f"{val:.1f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(title_suffix)
        ax.set_xticks(x)
        ax.set_xticklabels([s.split()[0] for s in STATION_NAMES])
        ax.set_xlabel("Station")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Overall Mean Metrics — Averaged Across All 18 Lead Hours", fontsize=13)
    plt.tight_layout()
    _save(fig, "comparison_overall_metrics.png")


# ---------------------------------------------------------------------------
# Chart 5 — Scatter: predicted vs true NWM error (3 models × 2 stations)
# ---------------------------------------------------------------------------
def plot_scatter(results):
    model_names = ("GRU", "LSTM", "Transformer")
    cmap        = cm.plasma
    fig, axes   = plt.subplots(3, 2, figsize=(14, 18))

    for row, m in enumerate(model_names):
        for col, sname in enumerate(STATION_NAMES):
            ax     = axes[row][col]
            preds  = results[sname][m]["preds"]
            y_test = results[sname]["y_test"]

            for h in range(18):
                color = cmap(h / 17)
                label = f"{h+1}h" if h % 6 == 0 else None
                ax.scatter(y_test[:, h], preds[:, h],
                           s=3, alpha=0.2, color=color, label=label)

            lim = max(np.abs(y_test).max(), np.abs(preds).max())
            ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, label="Perfect")
            sid = sname.split()[0]
            ax.set_title(f"{m}  —  Station {sid}")
            ax.set_xlabel("True NWM Error  (m³/s)")
            ax.set_ylabel("Predicted Error  (m³/s)")
            ax.legend(markerscale=3, fontsize=7, title="Lead")
            ax.grid(True, alpha=0.3)

    plt.suptitle("Predicted vs True NWM Error — All Models × Both Stations", fontsize=14)
    plt.tight_layout()
    _save(fig, "comparison_scatter.png")


# ---------------------------------------------------------------------------
# Chart 6 — Corrected streamflow time series (one file per station)
# ---------------------------------------------------------------------------
def plot_corrected_flow(results):
    LEADS       = [1, 6, 12]
    model_names = ("GRU", "LSTM", "Transformer")
    ls_map      = {"GRU": "-", "LSTM": "--", "Transformer": ":"}

    for sname in STATION_NAMES:
        test_df = results[sname]["test_df"].reset_index(drop=True)
        times   = test_df["init_time"].values

        fig, axes = plt.subplots(len(LEADS), 1,
                                 figsize=(18, 4 * len(LEADS)), sharex=True)

        for ax, h in zip(axes, LEADS):
            raw_nwm  = test_df[f"NWM_lead_{h}h"].values
            observed = test_df[f"USGS_at_lead_{h}h"].values

            ax.plot(times, observed, color="black",
                    linewidth=1.5, label="USGS Observed", zorder=5)
            ax.plot(times, raw_nwm,  color="silver",
                    linewidth=1.0, label="Raw NWM", zorder=4)

            for m in model_names:
                corrected = raw_nwm - results[sname][m]["preds"][:, h - 1]
                ax.plot(times, corrected,
                        color=MODEL_COLORS[m], linestyle=ls_map[m],
                        linewidth=1.3, alpha=0.9, label=f"{m} Corrected", zorder=3)

            ax.set_ylabel("Streamflow  (m³/s)")
            ax.set_title(f"Lead {h}h")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date")
        fig.autofmt_xdate()
        sid = sname.split()[0]
        plt.suptitle(f"Station {sid} — Corrected Streamflow: All Models", fontsize=13)
        plt.tight_layout()
        _save(fig, f"comparison_corrected_flow_{sid}.png")


# ---------------------------------------------------------------------------
# Chart 7 — % RMSE reduction vs NWM baseline
# ---------------------------------------------------------------------------
def plot_rmse_reduction(results):
    model_names = ("GRU", "LSTM", "Transformer")
    x = np.arange(len(model_names))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, sname in zip(axes, STATION_NAMES):
        baseline_mean = results[sname]["baseline_rmse"].mean()
        reductions    = [
            100 * (baseline_mean - results[sname][m]["rmse"].mean()) / baseline_mean
            for m in model_names
        ]

        bars = ax.bar(x, reductions,
                      color=[MODEL_COLORS[m] for m in model_names],
                      alpha=0.85, edgecolor="white")

        for bar, val in zip(bars, reductions):
            ypos = bar.get_height() + (0.3 if val >= 0 else -1.5)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:+.1f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        sid = sname.split()[0]
        ax.set_title(f"Station {sid}")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylabel("Mean RMSE Reduction  (%)")
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Mean RMSE Reduction vs Raw NWM Baseline", fontsize=13)
    plt.tight_layout()
    _save(fig, "comparison_rmse_reduction.png")


# ---------------------------------------------------------------------------
# Chart 8 — RMSE heatmap (model × lead hour)
# ---------------------------------------------------------------------------
def plot_rmse_heatmap(results):
    model_names = ["NWM Baseline", "GRU", "LSTM", "Transformer"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 4))

    for ax, sname in zip(axes, STATION_NAMES):
        matrix = np.array([
            results[sname]["baseline_rmse"],
            results[sname]["GRU"]["rmse"],
            results[sname]["LSTM"]["rmse"],
            results[sname]["Transformer"]["rmse"],
        ])

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        fig.colorbar(im, ax=ax, label="RMSE  (m³/s)")

        ax.set_xticks(range(18))
        ax.set_xticklabels([f"{h}h" for h in LEAD_HOURS], fontsize=8)
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=9)
        sid = sname.split()[0]
        ax.set_title(f"Station {sid} — RMSE Heatmap")
        ax.set_xlabel("Lead Hour")

        for i in range(len(model_names)):
            for j in range(18):
                ax.text(j, i, f"{matrix[i, j]:.1f}",
                        ha="center", va="center", fontsize=6.5)

    plt.suptitle("RMSE Heatmap — Model × Lead Hour", fontsize=13)
    plt.tight_layout()
    _save(fig, "comparison_rmse_heatmap.png")


# ---------------------------------------------------------------------------
# Chart 9 — Training time & parameter count summary
# ---------------------------------------------------------------------------
def plot_model_complexity(results):
    model_names = ("GRU", "LSTM", "Transformer")
    sname       = STATION_NAMES[0]

    params  = [results[sname][m]["params"] for m in model_names]
    times_a = [results[s][m]["train_time"] for s in STATION_NAMES for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parameter counts
    ax = axes[0]
    bars = ax.bar(model_names, params,
                  color=[MODEL_COLORS[m] for m in model_names], alpha=0.85)
    for bar, val in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Trainable Parameter Count")
    ax.set_ylabel("Parameters")
    ax.grid(True, axis="y", alpha=0.3)

    # Training time per station
    ax = axes[1]
    x      = np.arange(2)
    width  = 0.25
    off    = np.array([-1, 0, 1]) * width
    for i, m in enumerate(model_names):
        t_vals = [results[s][m]["train_time"] for s in STATION_NAMES]
        ax.bar(x + off[i], t_vals, width,
               label=m, color=MODEL_COLORS[m], alpha=0.85)
    ax.set_title("Training Time per Station")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels([s.split()[0] for s in STATION_NAMES])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Model Complexity & Training Cost", fontsize=13)
    plt.tight_layout()
    _save(fig, "comparison_model_complexity.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    results = {}

    for sname, nwm_path, usgs_path in [
        (STATION_NAMES[0], STATION_A_NWM_PATH, STATION_A_USGS_PATH),
        (STATION_NAMES[1], STATION_B_NWM_PATH, STATION_B_USGS_PATH),
    ]:
        print(f"\n{'='*70}")
        print(f"  STATION: {sname}")
        print("="*70)

        data = process_station(sname, nwm_path, usgs_path)
        sid  = sname.split()[0]

        b_rmse, b_mae = nwm_baseline_metrics(data)
        results[sname] = {
            "y_test":       data["y_test"],
            "test_df":      data["test_df"],
            "baseline_rmse": b_rmse,
            "baseline_mae":  b_mae,
        }

        for mname, train_fn in [
            ("GRU",         train_gru),
            ("LSTM",        train_lstm),
            ("Transformer", train_transformer),
        ]:
            print(f"\n  --- Training {mname} ---")
            save_path = os.path.join(
                _RESULTS_DIR, f"{mname.lower()}_runoff_{sid}.keras"
            )
            model, history, preds, elapsed = train_fn(data, save_path)
            rmse, mae = compute_metrics(preds, data["y_test"])

            results[sname][mname] = {
                "model":      model,
                "history":    history,
                "preds":      preds,
                "rmse":       rmse,
                "mae":        mae,
                "params":     model.count_params(),
                "train_time": elapsed,
            }
            print(
                f"  Params: {model.count_params():,}  |  "
                f"Train time: {elapsed:.0f}s  |  "
                f"Mean RMSE: {rmse.mean():.4f}  |  Mean MAE: {mae.mean():.4f}"
            )

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    header = f"{'Model':<14}  {'Station':<25}  {'Mean RMSE':>10}  {'Mean MAE':>10}  {'Params':>10}"
    print(header)
    print("-" * len(header))
    for sname in STATION_NAMES:
        sid = sname.split()[0]
        print(f"{'NWM Baseline':<14}  {sid:<25}  "
              f"{results[sname]['baseline_rmse'].mean():>10.4f}  "
              f"{results[sname]['baseline_mae'].mean():>10.4f}  {'—':>10}")
        for m in ("GRU", "LSTM", "Transformer"):
            r = results[sname][m]
            print(f"{m:<14}  {sid:<25}  "
                  f"{r['rmse'].mean():>10.4f}  "
                  f"{r['mae'].mean():>10.4f}  "
                  f"{r['params']:>10,}")

    # ------------------------------------------------------------------
    # Generate all charts
    # ------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("  GENERATING CHARTS")
    print("="*70)

    plot_rmse_by_lead(results)
    plot_mae_by_lead(results)
    plot_training_loss(results)
    plot_overall_metrics(results)
    plot_scatter(results)
    plot_corrected_flow(results)
    plot_rmse_reduction(results)
    plot_rmse_heatmap(results)
    plot_model_complexity(results)

    print(f"\nAll charts saved to: {_RESULTS_DIR}")


if __name__ == "__main__":
    main()
