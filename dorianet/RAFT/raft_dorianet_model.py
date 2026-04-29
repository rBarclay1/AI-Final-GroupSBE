"""
RAFT model inference and visualization for DORIANET optical flow.
Loads a trained model and generates flow visualizations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except Exception as e:
    raise RuntimeError(
        "This script requires Pillow and matplotlib. Install with: "
        "pip install pillow matplotlib"
    ) from e


def _repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).resolve().parents[2]


def _load_image_rgb(path: Path) -> torch.Tensor:
    """Load an image as RGB tensor."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = torch.from_numpy(np.array(im))
    # arr: (H, W, 3) uint8
    x = arr.permute(2, 0, 1).float() / 255.0
    return x


def _resize_to_multiple(x: torch.Tensor, *, multiple: int = 8) -> torch.Tensor:
    """Resize image to be a multiple of `multiple`."""
    _, h, w = x.shape
    new_h = int(math.ceil(h / multiple) * multiple)
    new_w = int(math.ceil(w / multiple) * multiple)
    if (new_h, new_w) == (h, w):
        return x
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return x.squeeze(0)


def _get_raft(model_name: str, *, pretrained: bool) -> nn.Module:
    """Load RAFT model."""
    try:
        import torchvision
        from torchvision.models.optical_flow import raft_large, raft_small
        from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
    except Exception as e:
        raise RuntimeError(
            "torchvision with optical_flow RAFT is required. "
            "Install/upgrade: pip install -U torchvision"
        ) from e

    name = model_name.lower().strip()
    if name in ("small", "raft_small"):
        weights = Raft_Small_Weights.DEFAULT if pretrained else None
        return raft_small(weights=weights, progress=True)
    if name in ("large", "raft_large"):
        weights = Raft_Large_Weights.DEFAULT if pretrained else None
        return raft_large(weights=weights, progress=True)
    raise ValueError("model_name must be one of: small, large")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def plot_training_history(history_path: Path, out_dir: Path) -> None:
    """Plot training and validation metrics from a JSONL history file."""
    history = _read_jsonl(history_path)
    if len(history) == 0:
        raise RuntimeError(f"Training history is empty: {history_path}")

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train"]["loss"] for item in history]
    train_photo = [item["train"]["photo"] for item in history]
    train_smooth = [item["train"]["smooth"] for item in history]

    val_loss = [item.get("val", {}).get("loss") for item in history]
    val_photo = [item.get("val", {}).get("photo") for item in history]
    val_smooth = [item.get("val", {}).get("smooth") for item in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("RAFT Training Performance", fontsize=16)

    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    if any(v is not None for v in val_loss):
        axes[0].plot(epochs, val_loss, marker="o", label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_photo, marker="o", label="Train Photo")
    if any(v is not None for v in val_photo):
        axes[1].plot(epochs, val_photo, marker="o", label="Val Photo")
    axes[1].set_title("Photometric Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Photometric Loss")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, train_smooth, marker="o", label="Train Smooth")
    if any(v is not None for v in val_smooth):
        axes[2].plot(epochs, val_smooth, marker="o", label="Val Smooth")
    axes[2].set_title("Smoothness Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Smoothness Loss")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output = out_dir / "training_performance.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved training performance plot to {output}")


def _predict_flow(model: nn.Module, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Predict optical flow between two images.
    Returns upsampled flow (B, 2, H, W).
    """
    out = model(img1, img2)

    # torchvision RAFT returns either a list/tuple of flows or an object with attribute
    if isinstance(out, (list, tuple)) and len(out) > 0:
        flow = out[-1]
    elif hasattr(out, "flow_up"):
        flow = out.flow_up
    else:
        flow = out

    if not torch.is_tensor(flow):
        raise RuntimeError(f"Unexpected RAFT output type: {type(out)}")
    return flow


def _flow_to_rgb(flow: torch.Tensor) -> np.ndarray:
    """
    Convert optical flow to RGB visualization using HSV color space.
    
    Args:
        flow: Tensor of shape (2, H, W) or (B, 2, H, W) in range [-max_flow, max_flow]
    
    Returns:
        RGB image as uint8 array of shape (H, W, 3) or (B, H, W, 3)
    """
    if flow.dim() == 3:
        flow = flow.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    b, _, h, w = flow.shape
    flow_np = flow.detach().cpu().numpy()
    
    # Compute magnitude and angle
    u = flow_np[:, 0, :, :]  # (B, H, W)
    v = flow_np[:, 1, :, :]  # (B, H, W)
    
    rad = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi  # angle in [-1, 1]
    
    fk = rad.max()  # maximum flow magnitude
    if fk > 0:
        rad = rad / fk
    
    # Convert to HSV
    hsv = np.zeros((b, h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ((a + 1) / 2 * 255).astype(np.uint8)  # Hue
    hsv[..., 1] = (rad * 255).astype(np.uint8)  # Saturation
    hsv[..., 2] = 255  # Value
    
    # Convert HSV to RGB
    rgb = np.zeros((b, h, w, 3), dtype=np.uint8)
    for i in range(b):
        rgb[i] = (mcolors.hsv_to_rgb(hsv[i].astype(np.float32) / 255) * 255).astype(np.uint8)
    
    if squeeze_output:
        rgb = rgb[0]
    
    return rgb


def infer_and_visualize(
    img1_path: Path,
    img2_path: Path,
    checkpoint_path: Path,
    out_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Infer optical flow between two images and create visualizations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", "small")
    
    # Load model
    model = _get_raft(model_name, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded {model_name} model")
    
    # Load and preprocess images
    print(f"Loading images: {img1_path}, {img2_path}")
    img1 = _load_image_rgb(img1_path)
    img2 = _load_image_rgb(img2_path)
    
    orig_h, orig_w = img1.shape[1], img1.shape[2]
    
    img1 = _resize_to_multiple(img1, multiple=8)
    img2 = _resize_to_multiple(img2, multiple=8)
    
    # Add batch dimension and move to device
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    
    # Predict flow
    print("Predicting optical flow...")
    with torch.no_grad():
        flow = _predict_flow(model, img1, img2)
    
    flow = flow.squeeze(0)  # Remove batch dimension
    
    # Visualize flow as RGB
    print("Creating flow visualization...")
    flow_rgb = _flow_to_rgb(flow)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RAFT Optical Flow Visualization", fontsize=16)
    
    # Plot first image
    img1_np = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0, 0].imshow(img1_np)
    axes[0, 0].set_title("Image 1")
    axes[0, 0].axis("off")
    
    # Plot second image
    img2_np = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0, 1].imshow(img2_np)
    axes[0, 1].set_title("Image 2")
    axes[0, 1].axis("off")
    
    # Plot flow as RGB
    axes[1, 0].imshow(flow_rgb)
    axes[1, 0].set_title("Optical Flow (HSV)")
    axes[1, 0].axis("off")
    
    # Plot flow magnitude
    flow_mag = torch.sqrt(flow[0]**2 + flow[1]**2).cpu().numpy()
    im = axes[1, 1].imshow(flow_mag, cmap="jet")
    axes[1, 1].set_title("Flow Magnitude")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # Save figure
    output_fig = out_dir / "flow_visualization.png"
    print(f"Saving visualization to {output_fig}")
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Save flow RGB
    flow_img = Image.fromarray(flow_rgb)
    output_flow = out_dir / "flow_rgb.png"
    flow_img.save(output_flow)
    print(f"Saved flow RGB to {output_flow}")
    
    # Save flow magnitude
    mag_normalized = (flow_mag / flow_mag.max() * 255).astype(np.uint8)
    mag_img = Image.fromarray(mag_normalized, mode="L")
    output_mag = out_dir / "flow_magnitude.png"
    mag_img.save(output_mag)
    print(f"Saved flow magnitude to {output_mag}")
    
    # Save flow as numpy file
    flow_np = flow.cpu().numpy()
    output_npy = out_dir / "flow.npy"
    np.save(output_npy, flow_np)
    print(f"Saved flow to {output_npy}")
    
    # Print statistics
    print("\n--- Flow Statistics ---")
    print(f"Flow shape: {flow.shape}")
    print(f"Flow magnitude range: [{flow_mag.min():.4f}, {flow_mag.max():.4f}]")
    print(f"Flow magnitude mean: {flow_mag.mean():.4f}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="RAFT model inference and optical flow visualization for DORIANET."
    )
    p.add_argument(
        "--img1",
        type=str,
        default=str(Path("dorianet/data/raw/FRAME/2_0001.jpg")),
        help="Path to first image",
    )
    p.add_argument(
        "--img2",
        type=str,
        default=str(Path("dorianet/data/raw/FRAME/2_0004.jpg")),
        help="Path to second image",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path("dorianet/RAFT/results/best.pt")),
        help="Path to model checkpoint",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("dorianet/RAFT/results")),
        help="Output directory for visualizations",
    )
    p.add_argument(
        "--history",
        type=str,
        default=str(Path("dorianet/RAFT/results/train_history.jsonl")),
        help="Path to training history JSONL file for performance plots",
    )
    p.add_argument(
        "--plot-history-only",
        action="store_true",
        help="Only plot training history without running inference",
    )
    p.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    args = p.parse_args(argv)
    
    repo_root = _repo_root()
    
    img1_path = Path(args.img1)
    if not img1_path.is_absolute():
        img1_path = repo_root / img1_path
    
    img2_path = Path(args.img2)
    if not img2_path.is_absolute():
        img2_path = repo_root / img2_path
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path
    
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    
    if args.plot_history_only:
        history_path = Path(args.history)
        if not history_path.is_absolute():
            history_path = repo_root / history_path
        if history_path.exists():
            plot_training_history(history_path=history_path, out_dir=out_dir)
        else:
            raise FileNotFoundError(
                f"Training history not found at {history_path}. Run training first."
            )
        return

    infer_and_visualize(
        img1_path=img1_path,
        img2_path=img2_path,
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
        device=str(args.device),
    )

    history_path = Path(args.history)
    if not history_path.is_absolute():
        history_path = repo_root / history_path
    if history_path.exists():
        plot_training_history(history_path=history_path, out_dir=out_dir)
    else:
        print(f"Training history not found at {history_path}; skipping performance plots.")


if __name__ == "__main__":
    main()
