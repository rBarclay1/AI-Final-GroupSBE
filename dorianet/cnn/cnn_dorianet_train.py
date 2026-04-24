import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import keras
keras.utils.set_random_seed(812)

from cnn_dorianet_model import build_dorianet_cnn
from cnn_dorianet_data import make_datasets
from cnn_dorianet_visualize import plot_all

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_RESULTS_DIR, exist_ok=True)

EPOCHS        = 50
EPOCHS_FINETUNE = 30
LR            = 1e-3
LR_FINETUNE   = 1e-5
PATIENCE      = 8


def run():
    print("Loading data...")
    train_ds, val_ds, test_ds, y_test, class_weight = make_datasets()

    model = build_dorianet_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    save_path = os.path.join(_RESULTS_DIR, "cnn_dorianet_model.keras")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            save_path, monitor="val_accuracy", save_best_only=True, verbose=0
        ),
    ]

    print("\n--- Phase 1: Frozen base ---")
    history1 = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n--- Phase 2: Fine-tuning top 20 layers ---")
    base = model.layers[2]
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(LR_FINETUNE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_ds,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nEvaluating on test set...")
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"  Test accuracy : {acc:.4f}")
    print(f"  Test loss     : {loss:.4f}")

    preds = model.predict(test_ds, verbose=0).argmax(axis=1)

    import numpy as np
    combined_history = {
        'loss':         history1.history['loss']         + history2.history['loss'],
        'val_loss':     history1.history['val_loss']     + history2.history['val_loss'],
        'accuracy':     history1.history['accuracy']     + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    }
    history1.history = combined_history

    plot_all(history1, y_test, preds)

    return model, history1, y_test, preds


if __name__ == "__main__":
    run()
