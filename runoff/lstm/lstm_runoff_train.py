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

from lstm_runoff_model import build_runoff_lstm
from lstm_runoff_plot import plot_training_history, plot_rmse_by_lead, plot_scatter, plot_corrected_vs_observed

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
PATIENCE = 10


def run(station_name, nwm_path, usgs_path):
    print(f"\n{'='*60}")
    print(f"Training: {station_name}")
    print('='*60)

    data = process_station(station_name, nwm_path, usgs_path)

    model = build_runoff_lstm()
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="MeanSquaredError",
        metrics=["mse"],
    )

    save_path = os.path.join(_RESULTS_DIR, f"lstm_runoff_model_{station_name.split()[0]}.keras")

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

    X_train = np.concatenate([data['X_nwm_train'], data['X_usgs_train']], axis=1)
    X_val   = np.concatenate([data['X_nwm_val'],   data['X_usgs_val']],   axis=1)
    X_test  = np.concatenate([data['X_nwm_test'],  data['X_usgs_test']],  axis=1)

    history = model.fit(
        X_train, data['y_train'],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, data['y_val']),
        callbacks=callbacks,
        verbose=1,
    )

    test_mse = model.evaluate(X_test, data['y_test'], verbose=0)[0]
    print(f"\n  Test MSE : {test_mse:.6f}")

    preds = model.predict(X_test, verbose=0)
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

    plot_training_history([history_a, history_b], [name_a, name_b])
    plot_rmse_by_lead([rmse_a, rmse_b], [name_a, name_b])
    plot_scatter([data_a, data_b], [preds_a, preds_b], [name_a, name_b])
    plot_corrected_vs_observed(data_a, preds_a, name_a)
    plot_corrected_vs_observed(data_b, preds_b, name_b)

    print("\nSaved results in:", _RESULTS_DIR)
