import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import keras
keras.utils.set_random_seed(812)

IMG_SIZE   = 224
NUM_CLASSES = 6


def build_dorianet_cnn(dropout=0.4):
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False

    inputs = keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    model = build_dorianet_cnn()
    model.summary(line_length=80)
