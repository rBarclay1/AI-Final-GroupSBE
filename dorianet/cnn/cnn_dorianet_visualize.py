import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
_CLASS_NAMES  = ['L0\n(None)', 'L1\n(Minor)', 'L2\n(Moderate)', 'L3\n(Major)', 'L4\n(Destroyed)', 'L5\n(Other)']
_CLASS_COLORS = ['steelblue', 'mediumseagreen', 'gold', 'tomato', 'mediumpurple', 'sienna']


def plot_training_history(history, save_path=None):
    if save_path is None:
        save_path = os.path.join(_RESULTS_DIR, 'cnn_dorianet_training_history.png')

    epochs = range(1, len(history.history['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history.history['loss'],     label='Train', color='steelblue')
    axes[0].plot(epochs, history.history['val_loss'], label='Val',   color='tomato')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History — Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.history['accuracy'],     label='Train', color='steelblue')
    axes[1].plot(epochs, history.history['val_accuracy'], label='Val',   color='tomato')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training History — Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    if save_path is None:
        save_path = os.path.join(_RESULTS_DIR, 'cnn_dorianet_confusion_matrix.png')

    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['L0', 'L1', 'L2', 'L3', 'L4', 'L5'])

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix — Test Set')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_class_accuracy(y_true, y_pred, save_path=None):
    if save_path is None:
        save_path = os.path.join(_RESULTS_DIR, 'cnn_dorianet_per_class_accuracy.png')

    cm        = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(_CLASS_NAMES, per_class, color=_CLASS_COLORS, alpha=0.85)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy — Test Set')
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.1%}',
                ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_all(history, y_true, y_pred):
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    print("\nGenerating plots...")
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred)
    plot_per_class_accuracy(y_true, y_pred)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['L0', 'L1', 'L2', 'L3', 'L4', 'L5']))
