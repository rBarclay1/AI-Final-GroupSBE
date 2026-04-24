import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import keras
keras.utils.set_random_seed(812)


class PositionalEmbedding(keras.layers.Layer):

    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embedding = keras.layers.Embedding(seq_len, d_model)

    def call(self, x):
        positions = keras.ops.arange(self.seq_len)
        return x + self.embedding(positions)


def _encoder_block(x, d_model, nhead, dim_feedforward, dropout):
    normed = keras.layers.LayerNormalization()(x)
    attn   = keras.layers.MultiHeadAttention(
        num_heads=nhead, key_dim=d_model // nhead, dropout=dropout
    )(normed, normed)
    attn = keras.layers.Dropout(dropout)(attn)
    x    = keras.layers.Add()([x, attn])

    normed  = keras.layers.LayerNormalization()(x)
    ffn_out = keras.layers.Dense(dim_feedforward, activation="relu")(normed)
    ffn_out = keras.layers.Dense(d_model)(ffn_out)
    ffn_out = keras.layers.Dropout(dropout)(ffn_out)
    x       = keras.layers.Add()([x, ffn_out])

    return x


def build_runoff_transformer(
    seq_len=19,
    d_model=64,
    nhead=4,
    dim_feedforward=128,
    dropout=0.1,
    num_layers=2,
):
    inputs = keras.layers.Input(shape=(seq_len,))

    x = keras.layers.Reshape((seq_len, 1))(inputs)
    x = keras.layers.Dense(d_model)(x)

    x = PositionalEmbedding(seq_len, d_model)(x)

    for _ in range(num_layers):
        x = _encoder_block(x, d_model, nhead, dim_feedforward, dropout)

    x = x[:, :18, :]

    x       = keras.layers.Dense(1)(x)
    outputs = keras.layers.Reshape((18,))(x)

    return keras.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    model = build_runoff_transformer()
    print(model.summary(line_length=80))
