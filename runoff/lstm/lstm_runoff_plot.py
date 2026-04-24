import os
import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from runoff_data_process import (
    process_station,
    STATION_A_NWM_PATH,
    STATION_A_USGS_PATH,
    STATION_B_NWM_PATH,
    STATION_B_USGS_PATH,
)


LEAD_HOURS = list(range(1, 19))
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)


def _rmse_by_lead(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))


def plot_rmse_by_lead(
    rmse_list,
    station_names,
    save_path=os.path.join(_RESULTS_DIR, "lstm_runoff_rmse_by_lead.png"),
):
    x = np.arange(18)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["steelblue", "tomato"]

    for i, (rmse, name) in enumerate(zip(rmse_list, station_names)):
        ax.bar(x + i * width, rmse, width, label=name, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xlabel("Lead Hour")
    ax.set_ylabel("RMSE (m³/s)")
    ax.set_title("LSTM Error Correction RMSE by Lead Hour")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"{h}h" for h in LEAD_HOURS], fontsize=8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_corrected_vs_observed(
    data,
    preds,
    station_name,
    lead_hours=(1, 6, 12),
    save_path=None,
):
    test_df = data["test_df"].reset_index(drop=True)
    times = test_df["init_time"]

    fig, axes = plt.subplots(
        len(lead_hours),
        1,
        figsize=(16, 4 * len(lead_hours)),
        sharex=True,
    )
    if len(lead_hours) == 1:
        axes = [axes]

    for ax, h in zip(axes, lead_hours):
        raw_nwm = test_df[f"NWM_lead_{h}h"].values
        observed = test_df[f"USGS_at_lead_{h}h"].values
        corrected = raw_nwm - preds[:, h - 1]

        ax.plot(times, observed, label="USGS Observed", color="black", linewidth=1.2)
        ax.plot(times, raw_nwm, label="Raw NWM", color="steelblue", linewidth=1.0, alpha=0.8)
        ax.plot(times, corrected, label="Corrected NWM", color="tomato", linewidth=1.0, alpha=0.8)

        ax.set_ylabel("Streamflow (m³/s)")
        ax.set_title(f"{station_name} — Lead {h}h Forecast (LSTM correction)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(
            _RESULTS_DIR, f"lstm_runoff_corrected_flow_{station_name.split()[0]}.png"
        )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_scatter(
    data_list,
    preds_list,
    station_names,
    save_path=os.path.join(_RESULTS_DIR, "lstm_runoff_scatter.png"),
):
    fig, axes = plt.subplots(1, len(data_list), figsize=(14, 6))
    if len(data_list) == 1:
        axes = [axes]

    cmap = cm.plasma

    for ax, data, preds, name in zip(axes, data_list, preds_list, station_names):
        true_errors = data["y_test"]

        for h in range(18):
            color = cmap(h / 17)
            label = f"{h + 1}h" if h % 6 == 0 else None
            ax.scatter(true_errors[:, h], preds[:, h], s=4, alpha=0.25, color=color, label=label)

        lim_min = min(true_errors.min(), preds.min())
        lim_max = max(true_errors.max(), preds.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, label="Perfect")

        ax.set_xlabel("True NWM Error (m³/s)")
        ax.set_ylabel("Predicted NWM Error (m³/s)")
        ax.set_title(f"{name} — Predicted vs True Error (LSTM)")
        ax.legend(markerscale=3, title="Lead hour", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def evaluate_and_plot(station_name, nwm_path, usgs_path, model_path):
    print(f"\n{'='*60}")
    print(f"Evaluating + plotting: {station_name}")
    print('='*60)
    print("Loading model:", model_path)

    data = process_station(station_name, nwm_path, usgs_path)
    model = keras.models.load_model(model_path)

    preds = model.predict(data["X_test"], verbose=0)
    rmse = _rmse_by_lead(data["y_test"], preds)

    plot_corrected_vs_observed(data, preds, station_name)
    return data, preds, rmse


if __name__ == "__main__":
    name_a = "20380357 (09520500)"
    name_b = "21609641 (11266500)"

    model_a_path = os.path.join(_RESULTS_DIR, "lstm_runoff_model_20380357.keras")
    model_b_path = os.path.join(_RESULTS_DIR, "lstm_runoff_model_21609641.keras")

    if not os.path.exists(model_a_path) or not os.path.exists(model_b_path):
        raise FileNotFoundError(
            "Missing saved LSTM model(s). Expected:\n"
            f"  {model_a_path}\n"
            f"  {model_b_path}\n"
            "Run lstm_runoff_train.py first."
        )

    data_a, preds_a, rmse_a = evaluate_and_plot(name_a, STATION_A_NWM_PATH, STATION_A_USGS_PATH, model_a_path)
    data_b, preds_b, rmse_b = evaluate_and_plot(name_b, STATION_B_NWM_PATH, STATION_B_USGS_PATH, model_b_path)

    plot_rmse_by_lead([rmse_a, rmse_b], [name_a, name_b])
    plot_scatter([data_a, data_b], [preds_a, preds_b], [name_a, name_b])

    print("\nPlots saved in:", _RESULTS_DIR)
