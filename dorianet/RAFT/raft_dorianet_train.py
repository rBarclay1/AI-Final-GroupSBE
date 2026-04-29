from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires Pillow. Install with: pip install pillow") from e


def _repo_root() -> Path:
    # file: dorianet/RAFT/raft_dorianet_train.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _load_image_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = torch.from_numpy(np.array(im))
    # arr: (H, W, 3) uint8
    x = arr.permute(2, 0, 1).float() / 255.0
    return x


def _resize_to_multiple(x: torch.Tensor, *, multiple: int = 8) -> torch.Tensor:
    _, h, w = x.shape
    new_h = int(math.ceil(h / multiple) * multiple)
    new_w = int(math.ceil(w / multiple) * multiple)
    if (new_h, new_w) == (h, w):
        return x
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return x.squeeze(0)


def _random_crop_pair(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    crop_hw: Tuple[int, int],
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, h, w = a.shape
    ch, cw = crop_hw
    if ch > h or cw > w:
        return a, b
    top = int(torch.randint(0, h - ch + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, w - cw + 1, (1,), generator=generator).item())
    return a[:, top : top + ch, left : left + cw], b[:, top : top + ch, left : left + cw]


def _augment_pair(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # horizontal flip
    if bool(torch.randint(0, 2, (1,), generator=generator).item()):
        a = torch.flip(a, dims=[2])
        b = torch.flip(b, dims=[2])
    # vertical flip
    if bool(torch.randint(0, 2, (1,), generator=generator).item()):
        a = torch.flip(a, dims=[1])
        b = torch.flip(b, dims=[1])
    # mild color jitter (brightness/contrast)
    def jitter(x: torch.Tensor) -> torch.Tensor:
        br = float(torch.empty(1).uniform_(0.9, 1.1, generator=generator).item())
        ct = float(torch.empty(1).uniform_(0.9, 1.1, generator=generator).item())
        mean = x.mean(dim=(1, 2), keepdim=True)
        y = (x - mean) * ct + mean
        y = y * br
        return y.clamp(0.0, 1.0)

    a = jitter(a)
    b = jitter(b)
    return a, b


@dataclass(frozen=True)
class PairRow:
    image_1: str
    image_2: str
    video_id: int
    frame_id_1: int
    frame_id_2: int


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _save_training_performance_plot(history_path: Path, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping training performance plot.")
        return

    history = _read_jsonl(history_path)
    if len(history) == 0:
        print(f"No training history found in {history_path}; skipping plot.")
        return

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
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "training_performance.png"
    plt.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved training performance plot to {output}")


class RaftPairsDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        *,
        crop_hw: Optional[Tuple[int, int]] = None,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        self.repo_root = _repo_root()
        # Resolve manifest path relative to repo root if it's relative
        if not manifest_path.is_absolute():
            manifest_path = self.repo_root / manifest_path
        self.rows = _read_jsonl(manifest_path)
        self.crop_hw = crop_hw
        self.augment = augment
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        # Convert backslashes to forward slashes for cross-platform compatibility
        img1_path = r["image_1"].replace("\\", "/")
        img2_path = r["image_2"].replace("\\", "/")
        p1 = (self.repo_root / img1_path).resolve()
        p2 = (self.repo_root / img2_path).resolve()
        a = _load_image_rgb(p1)
        b = _load_image_rgb(p2)

        # Deterministic per-sample generator (so multi-worker is stable)
        g = torch.Generator()
        g.manual_seed((self.seed * 1_000_003 + idx) & 0xFFFFFFFF)

        a = _resize_to_multiple(a, multiple=8)
        b = _resize_to_multiple(b, multiple=8)

        if self.crop_hw is not None:
            a, b = _random_crop_pair(a, b, crop_hw=self.crop_hw, generator=g)

        if self.augment:
            a, b = _augment_pair(a, b, generator=g)

        return {
            "img1": a,
            "img2": b,
            "meta": {
                "video_id": int(r.get("video_id", -1)),
                "frame_id_1": int(r.get("frame_id_1", -1)),
                "frame_id_2": int(r.get("frame_id_2", -1)),
            },
        }


def _flow_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    b, _, h, w = img.shape
    # base grid in pixel coords
    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device),
        torch.arange(w, device=img.device),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=0).float()  # (2, H, W)
    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # (B, 2, H, W)
    coords = grid + flow

    # normalize to [-1, 1]
    x = 2.0 * (coords[:, 0] / (w - 1.0)) - 1.0
    y = 2.0 * (coords[:, 1] / (h - 1.0)) - 1.0
    grid_norm = torch.stack((x, y), dim=-1)  # (B, H, W, 2)

    return F.grid_sample(img, grid_norm, mode="bilinear", padding_mode="border", align_corners=True)


def _charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def photometric_loss(img1: torch.Tensor, img2_warp: torch.Tensor) -> torch.Tensor:
    return _charbonnier(img1 - img2_warp).mean()


def smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    dx = _charbonnier(flow[:, :, :, 1:] - flow[:, :, :, :-1]).mean()
    dy = _charbonnier(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
    return dx + dy


def _get_raft(model_name: str, *, pretrained: bool) -> nn.Module:
    try:
        import torchvision
        from torchvision.models.optical_flow import raft_large, raft_small
        from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
    except Exception as e:  # pragma: no cover
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


def _predict_flow(model: nn.Module, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
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


def train(
    *,
    train_manifest: Path,
    val_manifest: Optional[Path],
    out_dir: Path,
    model_name: str,
    pretrained: bool,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    crop: Optional[str],
    num_workers: int,
    seed: int,
    lambda_smooth: float,
    grad_clip: Optional[float],
    log_every: int,
    max_steps_per_epoch: Optional[int],
) -> None:
    repo_root = _repo_root()
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_hw: Optional[Tuple[int, int]] = None
    if crop:
        w, h = crop.lower().split("x")
        crop_hw = (int(h), int(w))

    train_ds = RaftPairsDataset(train_manifest, crop_hw=crop_hw, augment=True, seed=seed)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=("cuda" in device),
        drop_last=True,
    )

    val_dl: Optional[DataLoader] = None
    if val_manifest is not None and val_manifest.exists():
        val_ds = RaftPairsDataset(val_manifest, crop_hw=crop_hw, augment=False, seed=seed + 1)
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(0, num_workers // 2),
            pin_memory=("cuda" in device),
            drop_last=False,
        )

    model = _get_raft(model_name, pretrained=pretrained).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_cuda_amp = ("cuda" in device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    def run_epoch(dl: DataLoader, *, train_mode: bool) -> Dict[str, float]:
        if train_mode:
            model.train()
        else:
            model.eval()

        total = 0
        loss_sum = 0.0
        photo_sum = 0.0
        smooth_sum = 0.0

        for step, batch in enumerate(dl, start=1):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                flow = _predict_flow(model, img1, img2)
                img2_w = _flow_warp(img2, flow)
                l_photo = photometric_loss(img1, img2_w)
                l_smooth = smoothness_loss(flow)
                loss = l_photo + (lambda_smooth * l_smooth)

            if train_mode:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                scaler.step(opt)
                scaler.update()

            bs = int(img1.shape[0])
            total += bs
            loss_sum += float(loss.detach().item()) * bs
            photo_sum += float(l_photo.detach().item()) * bs
            smooth_sum += float(l_smooth.detach().item()) * bs

            if log_every > 0 and (step % log_every == 0):
                mode = "train" if train_mode else "val"
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "step": step,
                            "samples": total,
                            "avg_loss": loss_sum / max(1, total),
                            "avg_photo": photo_sum / max(1, total),
                            "avg_smooth": smooth_sum / max(1, total),
                        }
                    )
                )

            if max_steps_per_epoch is not None and step >= int(max_steps_per_epoch):
                break

        return {
            "loss": loss_sum / max(1, total),
            "photo": photo_sum / max(1, total),
            "smooth": smooth_sum / max(1, total),
        }

    best_val = float("inf")
    history: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        print(json.dumps({"event": "epoch_start", "epoch": epoch}))
        tr = run_epoch(train_dl, train_mode=True)
        log = {"epoch": epoch, "train": tr}

        if val_dl is not None:
            with torch.no_grad():
                va = run_epoch(val_dl, train_mode=False)
            log["val"] = va
            val_loss = va["loss"]
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
        else:
            is_best = False

        print(json.dumps(log, indent=2))

        history.append(log)
        history_path = out_dir / "train_history.jsonl"
        with history_path.open("w", encoding="utf-8") as f:
            for item in history:
                f.write(json.dumps(item) + "\n")

        ckpt = {
            "epoch": epoch,
            "model_name": model_name,
            "pretrained": pretrained,
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "best_val": best_val,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if is_best:
            torch.save(ckpt, out_dir / "best.pt")

    _save_training_performance_plot(out_dir / "train_history.jsonl", out_dir)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Train/fine-tune RAFT on DORIANET pairs (self-supervised).")
    p.add_argument(
        "--train-manifest",
        type=str,
        default=str(Path("dorianet/RAFT/data/manifests/train.jsonl")),
    )
    p.add_argument(
        "--val-manifest",
        type=str,
        default=str(Path("dorianet/RAFT/data/manifests/val.jsonl")),
    )
    p.add_argument("--out-dir", type=str, default=str(Path("dorianet/RAFT/results")))
    p.add_argument("--model", type=str, default="small", choices=["small", "large"])
    p.add_argument("--pretrained", action="store_true", help="Start from torchvision pretrained weights.")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--crop", type=str, default="512x384", help="WxH crop, e.g. 512x384. Set empty to disable.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-smooth", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=5, help="Print running averages every N steps.")
    p.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=20,
        help="Cap steps per epoch (useful on CPU). Set <=0 to disable.",
    )

    args = p.parse_args(argv)

    crop = args.crop.strip() if isinstance(args.crop, str) else None
    if crop == "":
        crop = None

    train(
        train_manifest=Path(args.train_manifest),
        val_manifest=Path(args.val_manifest) if args.val_manifest else None,
        out_dir=Path(args.out_dir),
        model_name=str(args.model),
        pretrained=bool(args.pretrained),
        device=str(args.device),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        crop=crop,
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        lambda_smooth=float(args.lambda_smooth),
        grad_clip=(float(args.grad_clip) if args.grad_clip is not None else None),
        log_every=int(args.log_every),
        max_steps_per_epoch=(int(args.max_steps_per_epoch) if int(args.max_steps_per_epoch) > 0 else None),
    )


if __name__ == "__main__":
    main()
