"""
Outputs will be written to dorianet/yolo/results/

yolo_dorianet/weights/best.pt
yolo_dorianet_confusion_matrix.png  — test-set matrix
yolo_dorianet_per_class_accuracy.png
"""

import os
import re
import math
import shutil
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from ultralytics import YOLO

#Paths
_SCRIPT_DIR  = Path(__file__).resolve().parent
_MASK_DIR    = (_SCRIPT_DIR / '..' / 'data' / 'raw' / 'MASK').resolve()
_RESULTS_DIR = _SCRIPT_DIR / 'results'

#Hyperparameters 
MODEL_CKPT  = 'yolov8m-cls.pt'
CLASS_NAMES = ['Level0', 'Level1', 'Level2', 'Level3', 'Level4', 'Level5']
IMG_SIZE    = 256
BATCH       = 32    # identical to CNN
EPOCHS      = 80
PATIENCE    = 25
SEED        = 42    # identical to CNN split seed


# Load paths & labels
# Mirrors cnn_dorianet_data._get_paths_and_labels exactly.
def _get_paths_and_labels(mask_dir: Path = _MASK_DIR):
    paths, labels = [], []
    for fname in sorted(os.listdir(mask_dir)):
        m = re.search(r'Level(\d)', fname)
        if m:
            paths.append(str(mask_dir / fname))
            labels.append(int(m.group(1)))
    return np.array(paths), np.array(labels)


#Split
def _split(paths, labels, val_size=0.15, test_size=0.15, seed=SEED):
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=seed
    )
    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


#Build ImageNet-style directory 
def _build_yolo_dataset(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    dest: Path,
) -> Path:
    if dest.exists():
        shutil.rmtree(dest)

    train_counts = Counter(y_train)
    target = max(train_counts.values())

    for split_name, paths, labels in [
        ('train', X_train, y_train),
        ('val',   X_val,   y_val),
        ('test',  X_test,  y_test),
    ]:
        for src, lbl in zip(paths, labels):
            dst_dir = dest / split_name / CLASS_NAMES[lbl]
            dst_dir.mkdir(parents=True, exist_ok=True)
            if split_name == 'train':
                repeat = math.ceil(target / train_counts[lbl])
                for i in range(repeat):
                    shutil.copy2(src, dst_dir / f"{i}_{Path(src).name}")
            else:
                shutil.copy2(src, dst_dir / Path(src).name)

    print(f"  Dataset staged at: {dest}")
    print(f"  Train class counts (before): { {CLASS_NAMES[k]: v for k, v in sorted(train_counts.items())} }")
    print(f"  Oversampled to target: {target} per class")
    return dest


#Train
def train_model(dataset_dir: Path, results_dir: Path) -> YOLO:
    model = YOLO(MODEL_CKPT)
    model.train(
        data=str(dataset_dir),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        seed=SEED,
        workers=0,
        dropout=0.5,
        label_smoothing=0.1,
        flipud=0.5,
        degrees=15.0,
        scale=0.1,
        mixup=0.1,
        project=str(results_dir),
        name='yolo_dorianet_v2',
        exist_ok=True,
    )
    best = results_dir / 'yolo_dorianet_v2' / 'weights' / 'best.pt'
    print(f"\nLoading best weights: {best}")
    return YOLO(str(best))


#Evaluate 
def evaluate(model: YOLO, dataset_dir: Path, results_dir: Path):
    test_dir = dataset_dir / 'test'

    # Collect images/ground truth lables
    test_images: list[str] = []
    y_true: list[int]       = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = test_dir / class_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.iterdir()):
            test_images.append(str(img_path))
            y_true.append(class_idx)

    preds   = model.predict(test_images, imgsz=IMG_SIZE, verbose=False)
    y_pred  = [int(r.probs.top1) for r in preds]

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    acc     = (y_true == y_pred).mean()
    n_right = int((y_true == y_pred).sum())
    print(f"\nTest Accuracy: {acc:.4f}  ({n_right}/{len(y_true)})")

    _save_confusion_matrix(y_true, y_pred, results_dir)
    _save_per_class_accuracy(y_true, y_pred, results_dir)

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=['L0', 'L1', 'L2', 'L3', 'L4', 'L5'],
    ))


def _save_confusion_matrix(y_true, y_pred, results_dir: Path):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['L0', 'L1', 'L2', 'L3', 'L4', 'L5'],
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix — Test Set (YOLOv8)')
    plt.tight_layout()
    path = results_dir / 'yolo_dorianet_confusion_matrix.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _save_per_class_accuracy(y_true, y_pred, results_dir: Path):
    cm        = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)
    labels    = ['L0\n(None)', 'L1\n(Minor)', 'L2\n(Moderate)',
                 'L3\n(Major)', 'L4\n(Destroyed)', 'L5\n(Other)']
    colors    = ['steelblue', 'mediumseagreen', 'gold', 'tomato', 'mediumpurple', 'sienna']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, per_class, color=colors, alpha=0.85)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy — Test Set (YOLOv8)')
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.1%}',
                ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    path = results_dir / 'yolo_dorianet_per_class_accuracy.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


#Main
if __name__ == '__main__':
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading paths and labels...")
    paths, labels = _get_paths_and_labels()

    print("Splitting dataset (same logic as CNN baseline)...")
    X_train, X_val, X_test, y_train, y_val, y_test = _split(paths, labels)

    print("Building YOLOv8 dataset directory...")
    dataset_dir = _RESULTS_DIR / '_yolo_dataset'
    _build_yolo_dataset(X_train, X_val, X_test, y_train, y_val, y_test, dataset_dir)

    print(f"\nTraining YOLOv8 ({MODEL_CKPT})...")
    model = train_model(dataset_dir, _RESULTS_DIR)

    print("\nEvaluating on test set...")
    evaluate(model, dataset_dir, _RESULTS_DIR)
