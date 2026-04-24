import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

LEAD_HOURS   = list(range(1, 19))
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def plot_training_history(histories, station_names,
                          save_path=os.path.join(_RESULTS_DIR, 'transformer_runoff_training_history.png')):
    fig, axes = plt.subplots(1, len(histories), figsize=(14, 5))
    if len(histories) == 1:
        axes = [axes]

    for ax, hist, name in zip(axes, histories, station_names):
        epochs = range(1, len(hist.history['loss']) + 1)
        ax.plot(epochs, hist.history['loss'],     label='Train', color='steelblue')
        ax.plot(epochs, hist.history['val_loss'], label='Val',   color='tomato')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(f'Training History — {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_rmse_by_lead(rmse_list, station_names,
                      save_path=os.path.join(_RESULTS_DIR, 'transformer_runoff_rmse_by_lead.png')):
    x     = np.arange(18)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['steelblue', 'tomato']

    for i, (rmse, name) in enumerate(zip(rmse_list, station_names)):
        ax.bar(x + i * width, rmse, width, label=name, color=colors[i], alpha=0.85)

    ax.set_xlabel('Lead Hour')
    ax.set_ylabel('RMSE (m³/s)')
    ax.set_title('Error Correction RMSE by Lead Hour')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f'{h}h' for h in LEAD_HOURS], fontsize=8)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_corrected_vs_observed(data, preds, station_name,
                               lead_hours=(1, 6, 12),
                               save_path=None):
    test_df = data['test_df'].reset_index(drop=True)
    times   = test_df['init_time']

    fig, axes = plt.subplots(len(lead_hours), 1,
                             figsize=(16, 4 * len(lead_hours)),
                             sharex=True)
    if len(lead_hours) == 1:
        axes = [axes]

    for ax, h in zip(axes, lead_hours):
        raw_nwm   = test_df[f'NWM_lead_{h}h'].values
        observed  = test_df[f'USGS_at_lead_{h}h'].values
        corrected = raw_nwm - preds[:, h - 1]

        ax.plot(times, observed,  label='USGS Observed', color='black',     linewidth=1.2)
        ax.plot(times, raw_nwm,   label='Raw NWM',       color='steelblue', linewidth=1.0, alpha=0.8)
        ax.plot(times, corrected, label='Corrected NWM', color='tomato',    linewidth=1.0, alpha=0.8)

        ax.set_ylabel('Streamflow (m³/s)')
        ax.set_title(f'{station_name} — Lead {h}h Forecast')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(_RESULTS_DIR, f"transformer_runoff_corrected_flow_{station_name.split()[0]}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_scatter(data_list, preds_list, station_names,
                 save_path=os.path.join(_RESULTS_DIR, 'transformer_runoff_scatter.png')):
    fig, axes = plt.subplots(1, len(data_list), figsize=(14, 6))
    if len(data_list) == 1:
        axes = [axes]

    cmap = cm.plasma

    for ax, data, preds, name in zip(axes, data_list, preds_list, station_names):
        true_errors = data['y_test']

        for h in range(18):
            color = cmap(h / 17)
            label = f'{h + 1}h' if h % 6 == 0 else None
            ax.scatter(true_errors[:, h], preds[:, h],
                       s=4, alpha=0.25, color=color, label=label)

        lim_min = min(true_errors.min(), preds.min())
        lim_max = max(true_errors.max(), preds.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                'k--', linewidth=1, label='Perfect')

        ax.set_xlabel('True NWM Error (m³/s)')
        ax.set_ylabel('Predicted NWM Error (m³/s)')
        ax.set_title(f'{name} — Predicted vs True Error')
        ax.legend(markerscale=3, title='Lead hour', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_all(results_a, results_b):
    model_a, data_a, history_a, preds_a, rmse_a, name_a = results_a
    model_b, data_b, history_b, preds_b, rmse_b, name_b = results_b

    print("\nGenerating plots...")

    plot_training_history(
        [history_a, history_b],
        [name_a, name_b],
    )

    plot_rmse_by_lead(
        [rmse_a, rmse_b],
        [name_a, name_b],
    )

    plot_corrected_vs_observed(data_a, preds_a, name_a)
    plot_corrected_vs_observed(data_b, preds_b, name_b)

    plot_scatter(
        [data_a, data_b],
        [preds_a, preds_b],
        [name_a, name_b],
    )
