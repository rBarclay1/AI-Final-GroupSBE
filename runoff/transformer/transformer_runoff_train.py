import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import keras
keras.utils.set_random_seed(812)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from runoff_data_process import (
    process_station,
    STATION_A_NWM_PATH, STATION_A_USGS_PATH,
    STATION_B_NWM_PATH, STATION_B_USGS_PATH,
)
from transformer_runoff_model import build_runoff_transformer
from transformer_runoff_visualize import plot_all

_RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
_USGS_LEAD_COLS = [f'USGS_at_lead_{h}h' for h in range(1, 19)]

BATCH_SIZE = 64
EPOCHS     = 100
LR         = 1e-3
PATIENCE   = 10


def high_flow_weighted_mse(alpha=5.0, flow_scale=100.0):
    """
    Weighted MSE that penalises errors during high-flow events more heavily.

    y_true layout expected by this loss:
      columns  0:18  — true NWM errors (m³/s), what the model is trained to predict
      columns 18:36  — raw USGS observed flow at each lead hour (m³/s), used only
                       to compute per-lead weights; not a model target

    Weight per lead = 1 + alpha * (usgs_obs / flow_scale)
      - At  0 m³/s : weight = 1.0  (baseline)
      - At 100 m³/s: weight = 6.0  (alpha=5, flow_scale=100)
      - At 400 m³/s: weight = 21.0

    The augmented y layout avoids adding a second output head to the model and
    keeps the loss self-contained.
    """
    def loss(y_true, y_pred):
        true_errors = y_true[:, :18]
        usgs_obs    = y_true[:, 18:]
        weights     = 1.0 + alpha * (usgs_obs / flow_scale)
        sq_err      = keras.ops.square(y_pred - true_errors)
        return keras.ops.mean(weights * sq_err)
    return loss


def run(station_name, nwm_path, usgs_path):
    print(f"\n{'='*60}")
    print(f"Training: {station_name}")
    print('='*60)

    data = process_station(station_name, nwm_path, usgs_path)

    model = build_runoff_transformer()
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=high_flow_weighted_mse(alpha=5.0, flow_scale=100.0),
    )

    # Augment targets with raw USGS-at-lead values so the loss function can
    # compute per-lead flow weights without a separate model input.
    y_train_aug = np.concatenate(
        [data['y_train'], data['train_df'][_USGS_LEAD_COLS].values], axis=1
    )
    y_val_aug = np.concatenate(
        [data['y_val'], data['val_df'][_USGS_LEAD_COLS].values], axis=1
    )

    save_path = os.path.join(_RESULTS_DIR, f"transformer_runoff_model_{station_name.split()[0]}.keras")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            save_path, monitor="val_loss", save_best_only=True, verbose=0
        ),
    ]

    history = model.fit(
        [data['X_nwm_train'], data['X_usgs_train']], y_train_aug,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([data['X_nwm_val'], data['X_usgs_val']], y_val_aug),
        callbacks=callbacks,
        verbose=1,
    )

    preds         = model.predict([data['X_nwm_test'], data['X_usgs_test']], verbose=0)
    rmse_per_lead = np.sqrt(np.mean((preds - data['y_test']) ** 2, axis=0))

    print("\n  RMSE by lead hour (m³/s):")
    for h, rmse in enumerate(rmse_per_lead, start=1):
        print(f"    lead {h:2d}h : {rmse:.4f}")

    return model, data, history, preds, rmse_per_lead


if __name__ == "__main__":
    name_a = "20380357 (09520500)"
    name_b = "21609641 (11266500)"

    model_a, data_a, history_a, preds_a, rmse_a = run(name_a, STATION_A_NWM_PATH, STATION_A_USGS_PATH)
    model_b, data_b, history_b, preds_b, rmse_b = run(name_b, STATION_B_NWM_PATH, STATION_B_USGS_PATH)

    plot_all(
        (model_a, data_a, history_a, preds_a, rmse_a, name_a),
        (model_b, data_b, history_b, preds_b, rmse_b, name_b),
    )
