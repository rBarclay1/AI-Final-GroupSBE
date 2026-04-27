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
    seq_len=18,           # NWM leads only — USGS obs is a separate conditioning input
    d_model=128,          # up from 64
    nhead=8,              # up from 4; key_dim = d_model // nhead = 16
    dim_feedforward=256,  # up from 128 (kept at 2 × d_model)
    dropout=0.1,
    num_layers=4,         # up from 2
):
    nwm_input  = keras.layers.Input(shape=(seq_len,), name="nwm_leads")
    usgs_input = keras.layers.Input(shape=(1,),       name="usgs_obs")

    # Project each NWM lead scalar to d_model
    x = keras.layers.Reshape((seq_len, 1))(nwm_input)
    x = keras.layers.Dense(d_model)(x)                           # (batch, 18, d_model)

    # Project the USGS observation to d_model, then tile it across all 18 positions
    # so every encoder layer can attend to the current observed flow as a global bias.
    usgs_ctx = keras.layers.Dense(d_model)(usgs_input)           # (batch, d_model)
    usgs_ctx = keras.layers.RepeatVector(seq_len)(usgs_ctx)      # (batch, 18, d_model)
    x = keras.layers.Add()([x, usgs_ctx])

    x = PositionalEmbedding(seq_len, d_model)(x)

    for _ in range(num_layers):
        x = _encoder_block(x, d_model, nhead, dim_feedforward, dropout)

    x       = keras.layers.Dense(1)(x)
    outputs = keras.layers.Reshape((seq_len,))(x)

    return keras.Model(inputs=[nwm_input, usgs_input], outputs=outputs)


if __name__ == "__main__":
    model = build_runoff_transformer()
    print(model.summary(line_length=80))
