from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_FRAME_RE = re.compile(r"^(?P<vid>\d+)_(?P<fid>\d+)\.(?P<ext>jpg|jpeg|png)$", re.IGNORECASE)
_MASK_RE = re.compile(
    r"^(?P<vid>\d+)_(?P<fid>\d+)_B0XX_(?P<bid>\d+)_Level(?P<level>\d)\.(?P<ext>jpg|jpeg|png)$",
    re.IGNORECASE,
)


def _repo_relative_to_this_file(*parts: str) -> Path:
    """
    Resolve paths relative to the *repo root*.

    This file lives at `dorianet/RAFT/raft_dorianet_data.py`, so:
    - DORIANET root = parents[1]  -> `dorianet/`
    - Repo root     = parents[2]  -> `<repo>/`
    """
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.joinpath(*parts).resolve()


def _dorianet_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _try_import_pil():
    try:
        from PIL import Image  # type: ignore

        return Image
    except Exception:
        return None


def _parse_loose_json(path: Path) -> Dict[str, Any]:
    """
    DORIANET JSON files contain the token `NaN`, which is not valid JSON.
    We sanitize it to `null` so Python's json.loads can parse it.
    """
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return {}
    raw = re.sub(r"\bNaN\b", "null", raw)
    return json.loads(raw)


@dataclass(frozen=True)
class FrameItem:
    vid: int
    fid: int
    frame_path: Path
    json_path: Optional[Path]
    masks: Tuple[Path, ...]
    meta: Dict[str, Any]


def _list_frames(frame_dir: Path) -> List[Tuple[int, int, Path]]:
    items: List[Tuple[int, int, Path]] = []
    for p in frame_dir.iterdir():
        if not p.is_file():
            continue
        m = _FRAME_RE.match(p.name)
        if not m:
            continue
        items.append((int(m.group("vid")), int(m.group("fid")), p))
    items.sort(key=lambda t: (t[0], t[1]))
    return items


def _index_masks(mask_dir: Path) -> Dict[Tuple[int, int], List[Path]]:
    out: Dict[Tuple[int, int], List[Path]] = {}
    if not mask_dir.exists():
        return out
    for p in mask_dir.iterdir():
        if not p.is_file():
            continue
        m = _MASK_RE.match(p.name)
        if not m:
            continue
        k = (int(m.group("vid")), int(m.group("fid")))
        out.setdefault(k, []).append(p)
    for k in out:
        out[k].sort()
    return out


def _summarize_frame_meta(meta: Dict[str, Any], masks: Sequence[Path]) -> Dict[str, Any]:
    buildings = meta.get("Buildings") or []
    # Expected format (from data samples):
    # ["B014", "lat, lon", "mask_filename.jpg", damage_level, ..., ..., NaN]
    levels: List[int] = []
    for b in buildings:
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            lvl = b[3]
            if isinstance(lvl, (int, float)) and int(lvl) == lvl:
                levels.append(int(lvl))
    summary = {
        "capture_date": meta.get("Capture date"),
        "region": meta.get("Region"),
        "original_video_link": meta.get("Original video link"),
        "num_buildings": int(len(buildings)) if isinstance(buildings, list) else 0,
        "damage_level_max": max(levels) if levels else None,
        "damage_level_mean": (sum(levels) / len(levels)) if levels else None,
        "mask_count": len(masks),
    }
    return summary


def _build_frame_items(
    frame_dir: Path,
    json_dir: Path,
    mask_dir: Path,
    *,
    require_json: bool,
) -> List[FrameItem]:
    mask_index = _index_masks(mask_dir)
    frames = _list_frames(frame_dir)
    out: List[FrameItem] = []
    for vid, fid, frame_path in frames:
        json_path = json_dir / f"{vid}_{fid:04d}.json"
        if not json_path.exists():
            # Many JSON files appear to be named without fixed-width padding too.
            json_path2 = json_dir / f"{vid}_{fid}.json"
            json_path = json_path2 if json_path2.exists() else json_path

        if require_json and not json_path.exists():
            continue

        meta: Dict[str, Any] = {}
        jp: Optional[Path] = json_path if json_path.exists() else None
        if jp is not None:
            try:
                meta = _parse_loose_json(jp)
            except Exception:
                meta = {}

        masks = tuple(mask_index.get((vid, fid), []))
        meta_summary = _summarize_frame_meta(meta, masks)
        out.append(
            FrameItem(
                vid=vid,
                fid=fid,
                frame_path=frame_path,
                json_path=jp,
                masks=masks,
                meta=meta_summary,
            )
        )
    return out


def _group_by_video(items: Sequence[FrameItem]) -> Dict[int, List[FrameItem]]:
    vids: Dict[int, List[FrameItem]] = {}
    for it in items:
        vids.setdefault(it.vid, []).append(it)
    for v in vids:
        vids[v].sort(key=lambda x: x.fid)
    return vids


def _make_pairs(
    by_video: Dict[int, List[FrameItem]],
    *,
    stride: int,
    max_pairs_per_video: Optional[int],
    min_frame_gap: int,
) -> List[Tuple[FrameItem, FrameItem]]:
    pairs: List[Tuple[FrameItem, FrameItem]] = []
    for vid, frames in sorted(by_video.items(), key=lambda kv: kv[0]):
        local: List[Tuple[FrameItem, FrameItem]] = []
        for i in range(0, len(frames) - stride):
            a = frames[i]
            b = frames[i + stride]
            if (b.fid - a.fid) < min_frame_gap:
                continue
            local.append((a, b))
        if max_pairs_per_video is not None:
            local = local[: max_pairs_per_video]
        pairs.extend(local)
    return pairs


def _split_by_video_ids(
    video_ids: Sequence[int],
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[set, set, set]:
    """
    Split by *video id* to avoid leakage between consecutive frames.
    Deterministic shuffle using a simple LCG to avoid extra deps.
    """
    vids = list(video_ids)
    if not vids:
        return set(), set(), set()

    # deterministic shuffle (Fisher-Yates) using a basic PRNG
    rng = seed & 0xFFFFFFFF

    def rand32() -> int:
        nonlocal rng
        rng = (1664525 * rng + 1013904223) & 0xFFFFFFFF
        return rng

    for i in range(len(vids) - 1, 0, -1):
        j = rand32() % (i + 1)
        vids[i], vids[j] = vids[j], vids[i]

    n = len(vids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = set(vids[:n_test])
    val = set(vids[n_test : n_test + n_val])
    train = set(vids[n_test + n_val :])
    return train, val, test


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    _safe_mkdir(path.parent)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _copy_or_resize_image(
    src: Path,
    dst: Path,
    *,
    resize_wh: Optional[Tuple[int, int]],
) -> None:
    _safe_mkdir(dst.parent)
    if resize_wh is None:
        shutil.copy2(src, dst)
        return

    Image = _try_import_pil()
    if Image is None:
        raise RuntimeError(
            "Pillow is required for --resize. Install it with: pip install pillow"
        )
    with Image.open(src) as im:
        im = im.convert("RGB")
        im = im.resize(resize_wh, resample=Image.BILINEAR)
        im.save(dst, format="JPEG", quality=95)


def prepare(
    *,
    raw_root: Path,
    out_root: Path,
    stride: int,
    min_frame_gap: int,
    max_pairs_per_video: Optional[int],
    val_frac: float,
    test_frac: float,
    seed: int,
    require_json: bool,
    write_images: bool,
    resize_wh: Optional[Tuple[int, int]],
) -> Dict[str, Any]:
    frame_dir = raw_root / "FRAME"
    json_dir = raw_root / "JSON"
    mask_dir = raw_root / "MASK"

    if not frame_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frame_dir}")

    items = _build_frame_items(frame_dir, json_dir, mask_dir, require_json=require_json)
    by_video = _group_by_video(items)
    pairs = _make_pairs(
        by_video,
        stride=stride,
        max_pairs_per_video=max_pairs_per_video,
        min_frame_gap=min_frame_gap,
    )

    video_ids = sorted(by_video.keys())
    train_vids, val_vids, test_vids = _split_by_video_ids(
        video_ids, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    def rel(p: Path) -> str:
        # Manifests store paths relative to repo root (best for portability)
        try:
            repo_root = _repo_relative_to_this_file()
            return str(p.resolve().relative_to(repo_root))
        except Exception:
            return str(p)

    prepared_images_dir = out_root / "images" if write_images else None

    def dst_img_path(src: Path) -> str:
        if prepared_images_dir is None:
            return rel(src)
        return rel(prepared_images_dir / src.name)

    if write_images:
        assert prepared_images_dir is not None
        for it in items:
            dst = prepared_images_dir / it.frame_path.name
            if not dst.exists():
                _copy_or_resize_image(it.frame_path, dst, resize_wh=resize_wh)

    def row_for(a: FrameItem, b: FrameItem) -> Dict[str, Any]:
        return {
            "video_id": a.vid,
            "frame_id_1": a.fid,
            "frame_id_2": b.fid,
            "image_1": dst_img_path(a.frame_path),
            "image_2": dst_img_path(b.frame_path),
            # These are often useful for conditioning / filtering, even if RAFT ignores them
            "meta_1": a.meta,
            "meta_2": b.meta,
            "masks_1": [rel(p) for p in a.masks],
            "masks_2": [rel(p) for p in b.masks],
        }

    train_rows = (row_for(a, b) for (a, b) in pairs if a.vid in train_vids)
    val_rows = (row_for(a, b) for (a, b) in pairs if a.vid in val_vids)
    test_rows = (row_for(a, b) for (a, b) in pairs if a.vid in test_vids)

    manifests_dir = out_root / "manifests"
    n_train = _write_jsonl(manifests_dir / "train.jsonl", train_rows)
    n_val = _write_jsonl(manifests_dir / "val.jsonl", val_rows)
    n_test = _write_jsonl(manifests_dir / "test.jsonl", test_rows)

    summary = {
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "videos": len(video_ids),
        "frames_indexed": len(items),
        "pairs_total": len(pairs),
        "pairs_train": n_train,
        "pairs_val": n_val,
        "pairs_test": n_test,
        "write_images": bool(write_images),
        "resize_wh": resize_wh,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _parse_resize(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    m = re.match(r"^(?P<w>\d+)[xX](?P<h>\d+)$", s.strip())
    if not m:
        raise argparse.ArgumentTypeError("resize must look like: 512x384")
    return int(m.group("w")), int(m.group("h"))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare DORIANET for RAFT (pair manifest).")
    parser.add_argument(
        "--raw-root",
        type=str,
        default=str(_dorianet_root() / "data" / "raw"),
        help="Path to DORIANET raw root (contains FRAME/ JSON/ MASK/).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(_dorianet_root() / "data" / "raft"),
        help="Output dataset root (will write manifests/ and optional images/).",
    )
    parser.add_argument("--stride", type=int, default=1, help="Pair frames i and i+stride.")
    parser.add_argument(
        "--min-frame-gap",
        type=int,
        default=1,
        help="Skip pairs where (fid2 - fid1) is smaller than this.",
    )
    parser.add_argument(
        "--max-pairs-per-video",
        type=int,
        default=None,
        help="Optional cap on number of pairs per video id.",
    )
    parser.add_argument("--val-frac", type=float, default=0.2, help="Val split fraction (by video).")
    parser.add_argument("--test-frac", type=float, default=0.2, help="Test split fraction (by video).")
    parser.add_argument("--seed", type=int, default=42, help="Split shuffle seed.")
    parser.add_argument(
        "--require-json",
        action="store_true",
        help="Only include frames that have JSON metadata present.",
    )
    parser.add_argument(
        "--write-images",
        action="store_true",
        help="Copy (and optionally resize) frames into out_root/images/ for a self-contained dataset.",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Optional resize for --write-images, e.g. 512x384 (requires pillow).",
    )

    args = parser.parse_args(argv)

    resize_wh = _parse_resize(args.resize)
    summary = prepare(
        raw_root=Path(args.raw_root),
        out_root=Path(args.out_root),
        stride=int(args.stride),
        min_frame_gap=int(args.min_frame_gap),
        max_pairs_per_video=(int(args.max_pairs_per_video) if args.max_pairs_per_video else None),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
        require_json=bool(args.require_json),
        write_images=bool(args.write_images),
        resize_wh=resize_wh,
    )

    print("Wrote RAFT manifests:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
