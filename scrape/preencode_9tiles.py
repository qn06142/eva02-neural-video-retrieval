#!/usr/bin/env python3
"""High-Performance 9-Tile Pre-Encoding for 147k Curated Videos.

Optimizations:
  - Native model.half() pure FP16 execution (removes dynamic autocast overhead).
  - torch.inference_mode() for zero-overhead tensor operations.
  - allow_tf32 & cudnn.benchmark for Ada Lovelace Tensor Core saturation.
  - Optimized DataLoader with pinned memory, persistent workers, and prefetch.
  - Fast Bilinear interpolation on image tiles.
  - Rolling window instantaneous speed & ETA calculation.
  - Progressive memmap streaming with auto-resume from existing checkpoint.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import sqlite3
import sys
import time
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(PROJECT_DIR, "scrape", "thumbs.db")
DEFAULT_WEIGHTS = os.path.join(PROJECT_DIR, "scrape", "siglip_runs", "eva02_curated_v2", "best.pt")
DEFAULT_OUT = os.path.join(PROJECT_DIR, "scrape", "embeddings_v2")
DEFAULT_MODEL = "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class NineTileDataset(Dataset):
    """Fast extraction of all 9 tiles from 3x3 video contact sheets."""

    def __init__(self, records: List[Tuple], transform=None):
        # records: (idx, eid, clean_title, clothing_tags, position_tags, asset_path)
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.records[idx]
        asset_path = item[5]

        tiles = []
        try:
            with Image.open(asset_path) as im:
                w, h = im.size
                for r in range(3):
                    for c in range(3):
                        box = (c * w // 3, r * h // 3, (c + 1) * w // 3, (r + 1) * h // 3)
                        tiles.append(im.crop(box).convert("RGB"))
        except Exception:
            tiles = [Image.new("RGB", (336, 336), (0, 0, 0)) for _ in range(9)]

        if self.transform is not None:
            tensors = [self.transform(t) for t in tiles]
            batch_tensor = torch.stack(tensors, dim=0) # (9, 3, 336, 336)
        else:
            batch_tensor = torch.zeros(9, 3, 336, 336)

        return batch_tensor, idx


def main():
    parser = argparse.ArgumentParser(description="Optimized 9-tile pre-encoding on RTX 4060")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to thumbs.db")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to checkpoint")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Model architecture")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size in videos (12 vids = 108 tiles)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--limit", type=int, default=0, help="Limit items (0 for all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    embeddings_file = os.path.join(args.out_dir, "keyframes_9tiles_147k.npy")
    state_file = os.path.join(args.out_dir, "keyframes_9tiles_state.json")

    # 1. Fetch records
    log(f"Querying records from {args.db}...")
    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    cur.execute("SELECT eid, clean_title, clothing_tags, position_tags, asset_path FROM videos_curated ORDER BY rowid ASC")
    raw_rows = cur.fetchall()
    conn.close()

    total_available = len(raw_rows)
    if args.limit > 0:
        raw_rows = raw_rows[:args.limit]
    total_items = len(raw_rows)
    log(f"Found {total_available:,} total records. Selected {total_items:,} items for 9-tile encoding.")

    records = [
        (i, row[0], row[1], row[2], row[3], row[4])
        for i, row in enumerate(raw_rows)
    ]

    # 2. Check resume state
    start_idx = 0
    if os.path.exists(state_file) and os.path.exists(embeddings_file):
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
                start_idx = state.get("completed_items", 0)
                if start_idx < total_items:
                    log(f"Resuming 9-tile encoding from item {start_idx:,} / {total_items:,}...")
                else:
                    log(f"All {total_items:,} items are already encoded with 9 tiles!")
                    return
        except Exception:
            start_idx = 0

    # 3. Setup Model in Pure FP16
    log(f"Loading EVA-02 model: {args.model_name}...")
    model, _, _ = open_clip.create_model_and_transforms(args.model_name)

    if os.path.exists(args.weights):
        log(f"Loading weights from: {args.weights}...")
        ckpt = torch.load(args.weights, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    else:
        log(f"WARNING: Weights file {args.weights} not found! Using zero-shot weights.")

    if args.device == "cuda":
        log("Enabling pure native FP16 (model.half()) on Ada Lovelace Tensor Cores...")
        model = model.half().to(args.device).eval()
    else:
        model = model.to(args.device).eval()

    val_transform = transforms.Compose([
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    # 4. Open Memory-Mapped Array: (N, 9, 768)
    dim = 768
    mode = "r+" if os.path.exists(embeddings_file) else "w+"
    mmap_arr = np.memmap(embeddings_file, dtype="float16", mode=mode, shape=(total_items, 9, dim))

    remaining_records = records[start_idx:]
    dataset = NineTileDataset(remaining_records, transform=val_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(args.device == "cuda"),
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
        drop_last=False
    )

    interrupted = False
    def sig_handler(sig, frame):
        nonlocal interrupted
        log("\nInterrupt signal received! Flushing progress to disk...")
        interrupted = True

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # 5. Optimized Encoding Loop
    tiles_per_batch = args.batch_size * 9
    log(f"Beginning optimized 9-tile encoding: {len(remaining_records):,} items (Batch: {args.batch_size} vids / {tiles_per_batch} tiles)...")
    t0 = time.time()
    items_done = start_idx
    last_flush = time.time()

    # Rolling window for instantaneous speed (last 10 batches)
    rolling_times = collections.deque(maxlen=10)
    rolling_counts = collections.deque(maxlen=10)
    batch_idx = 0

    try:
        with torch.inference_mode():
            for batch_tensors, batch_indices in loader:
                if interrupted:
                    break

                batch_idx += 1
                b_start_time = time.time()
                b, k, c, h, w = batch_tensors.shape # (B, 9, 3, 336, 336)
                flat_tensors = batch_tensors.view(b * k, c, h, w)

                if args.device == "cuda":
                    flat_tensors = flat_tensors.to(args.device, dtype=torch.float16, non_blocking=True)
                else:
                    flat_tensors = flat_tensors.to(args.device, non_blocking=True)

                feats = model.encode_image(flat_tensors)
                feats = F.normalize(feats, dim=-1).view(b, k, dim)

                feats_fp16 = feats.cpu().numpy()

                target_start = items_done
                target_end = items_done + b
                mmap_arr[target_start:target_end] = feats_fp16
                items_done += b

                b_elapsed = time.time() - b_start_time
                rolling_times.append(b_elapsed)
                rolling_counts.append(b)

                # Periodic progress logging (every 10 batches = 120 videos)
                if batch_idx % 10 == 0 or items_done >= total_items:
                    window_time = sum(rolling_times)
                    window_count = sum(rolling_counts)
                    vids_per_sec = window_count / window_time if window_time > 0 else 0
                    fps = vids_per_sec * 9

                    rem = total_items - items_done
                    eta_sec = rem / vids_per_sec if vids_per_sec > 0 else 0
                    eta_m, eta_s = divmod(int(eta_sec), 60)
                    eta_h, eta_m = divmod(eta_m, 60)

                    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if args.device == "cuda" else 0
                    pct = (items_done / total_items) * 100
                    log(f"[{items_done:,}/{total_items:,}] ({pct:.1f}%) | Speed: {vids_per_sec:.2f} vids/s ({fps:.1f} fps) | ETA: {eta_h}h {eta_m:02d}m | VRAM: {vram_gb:.2f} GB")

                # Flush memmap every 60 seconds
                if time.time() - last_flush > 60:
                    mmap_arr.flush()
                    with open(state_file, "w") as f:
                        json.dump({"completed_items": items_done, "total_items": total_items, "dim": dim, "num_tiles": 9}, f)
                    last_flush = time.time()

    finally:
        log("Flushing 9-tile memory-mapped embeddings to disk...")
        mmap_arr.flush()
        del mmap_arr
        with open(state_file, "w") as f:
            json.dump({"completed_items": items_done, "total_items": total_items, "dim": dim, "num_tiles": 9}, f)

    total_time = time.time() - t0
    log(f"Finished! Processed {items_done - start_idx:,} items in {total_time:.1f}s ({items_done:,}/{total_items:,} total).")
    log(f"Embeddings saved to: {embeddings_file} ({os.path.getsize(embeddings_file) / (1024**2):.1f} MB)")


if __name__ == "__main__":
    main()
