import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import keras
keras.utils.set_random_seed(812)


def build_runoff_gru(
    seq_len=19,
    units=64,
    dropout=0.1,
    num_layers=2,
):
    inputs = keras.layers.Input(shape=(seq_len,))

    x = keras.layers.Reshape((seq_len, 1))(inputs)

    for i in range(num_layers):
        x = keras.layers.GRU(
            units,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=0.0,
            name=f"gru_{i + 1}",
        )(x)

    x = x[:, :18, :]

    x = keras.layers.Dense(1)(x)
    outputs = keras.layers.Reshape((18,))(x)

    return keras.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    model = build_runoff_gru()
    print(model.summary(line_length=80))
