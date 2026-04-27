import os
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf

_MASK_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', 'MASK')
IMG_SIZE   = 224
BATCH_SIZE = 32


def _get_paths_and_labels(mask_dir=_MASK_DIR):
    paths, labels = [], []
    for fname in sorted(os.listdir(mask_dir)):
        m = re.search(r'Level(\d)', fname)
        if m:
            paths.append(os.path.join(mask_dir, fname))
            labels.append(int(m.group(1)))
    return np.array(paths), np.array(labels)


def _load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.cast(img, tf.float32), label


_rotation = tf.keras.layers.RandomRotation(factor=15 / 360, fill_mode='reflect')
_zoom     = tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1)


def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.clip_by_value(img, 0.0, 255.0)
    img = _rotation(img, training=True)
    img = _zoom(img, training=True)
    return img, label


def make_datasets(val_size=0.15, test_size=0.15, seed=42):
    paths, labels = _get_paths_and_labels()

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=seed
    )

    counts = Counter(y_train)
    total  = len(y_train)
    class_weight = {c: total / (6 * counts[c]) for c in range(6)}

    def _make_ds(x, y, augment=False, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y.astype(np.int32)))
        if shuffle:
            ds = ds.shuffle(len(x), seed=seed)
        ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"  Class weights: { {k: round(v, 3) for k, v in class_weight.items()} }")

    return (
        _make_ds(X_train, y_train, augment=True, shuffle=True),
        _make_ds(X_val,   y_val),
        _make_ds(X_test,  y_test),
        y_test,
        class_weight,
    )
