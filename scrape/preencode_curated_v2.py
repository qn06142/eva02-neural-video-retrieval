#!/usr/bin/env python3
"""High-throughput Pre-Encoding of Curated Video Storyboards using EVA-02-L (v2).

Features:
  - Extracts the core center-action tile (Tile 4) from the 3x3 contact sheets.
  - Generates normalized 768-d FP16 embeddings for sub-millisecond MIPS search.
  - Progressive streaming to np.memmap (safe against interruptions/crashes).
  - Checkpointed state & auto-resume capability.
  - Accompanied by a lightweight SQLite metadata index for instant O(1) query lookups.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(PROJECT_DIR, "scrape", "thumbs.db")
DEFAULT_WEIGHTS = os.path.join(PROJECT_DIR, "scrape", "siglip_runs", "eva02_curated_v2", "best.pt")
DEFAULT_OUT = os.path.join(PROJECT_DIR, "scrape", "embeddings_v2")
DEFAULT_MODEL = "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class FastCenterTileDataset(Dataset):
    """Zero-overhead dataset extracting the center action tile (Tile 4) from 3x3 contact sheets."""

    def __init__(self, records: List[Tuple], transform=None):
        # records: (idx, eid, clean_title, clothing_tags, position_tags, asset_path)
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.records[idx]
        asset_path = item[5]

        try:
            with Image.open(asset_path) as im:
                w, h = im.size
                # Slice center tile 4: row 1, col 1
                crop_box = (w // 3, h // 3, 2 * w // 3, 2 * h // 3)
                tile = im.crop(crop_box).convert("RGB")
        except Exception:
            # Fallback black tile
            tile = Image.new("RGB", (336, 336), (0, 0, 0))

        if self.transform is not None:
            tensor = self.transform(tile)
        else:
            tensor = torch.zeros(3, 336, 336)

        return tensor, idx


def build_metadata_db(records: List[Tuple], db_path: str) -> None:
    """Stores metadata into a compact SQLite index for O(1) row lookups."""
    log(f"Building metadata index at: {db_path}...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS video_metadata")
    cur.execute("""
        CREATE TABLE video_metadata (
            idx INTEGER PRIMARY KEY,
            eid TEXT UNIQUE,
            title TEXT,
            clothing_tags TEXT,
            position_tags TEXT,
            asset_path TEXT
        )
    """)
    cur.executemany("""
        INSERT INTO video_metadata (idx, eid, title, clothing_tags, position_tags, asset_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
    conn.close()
    log(f"Metadata index built with {len(records):,} records.")


def main():
    parser = argparse.ArgumentParser(description="Pre-encode 147k curated videos using EVA-02-L v2")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to thumbs.db containing videos_curated")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to trained checkpoint (best.pt)")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Model architecture")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="Directory to save embeddings & metadata")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size (default: 64)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items (0 for all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    embeddings_file = os.path.join(args.out_dir, "globals_147k.npy")
    metadata_file = os.path.join(args.out_dir, "metadata_147k.sqlite3")
    state_file = os.path.join(args.out_dir, "encoding_state.json")

    # 1. Fetch valid items from SQLite
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
    log(f"Found {total_available:,} total records. Selected {total_items:,} items for encoding.")

    records = [
        (i, row[0], row[1], row[2], row[3], row[4])
        for i, row in enumerate(raw_rows)
    ]

    # Save metadata index once if not present or size differs
    if not os.path.exists(metadata_file) or args.limit > 0:
        build_metadata_db(records, metadata_file)

    # 2. Check for resume state
    start_idx = 0
    if os.path.exists(state_file) and os.path.exists(embeddings_file) and args.limit == 0:
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
                start_idx = state.get("completed_items", 0)
                if start_idx < total_items:
                    log(f"Resuming encoding from item {start_idx:,} / {total_items:,}...")
                else:
                    log(f"All {total_items:,} items are already encoded!")
                    return
        except Exception:
            start_idx = 0

    # 3. Setup Model & Transform
    log(f"Loading EVA-02 model: {args.model_name}...")
    model, _, _ = open_clip.create_model_and_transforms(args.model_name)

    if os.path.exists(args.weights):
        log(f"Loading weights from: {args.weights}...")
        ckpt = torch.load(args.weights, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    else:
        log(f"WARNING: Weights file {args.weights} not found! Using zero-shot weights.")

    model = model.to(args.device).eval()

    val_transform = transforms.Compose([
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    # 4. Allocate / Open Memory-Mapped Array (768-d FP16)
    dim = 768
    mode = "r+" if os.path.exists(embeddings_file) and start_idx > 0 else "w+"
    mmap_arr = np.memmap(embeddings_file, dtype="float16", mode=mode, shape=(total_items, dim))

    # Slice remaining records if resuming
    remaining_records = records[start_idx:]
    dataset = FastCenterTileDataset(remaining_records, transform=val_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(args.device == "cuda"),
        drop_last=False
    )

    # Graceful exit handler
    interrupted = False
    def sig_handler(sig, frame):
        nonlocal interrupted
        log("\nInterrupt signal received! Flushing progress to disk...")
        interrupted = True

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # 5. Encoding Loop
    log(f"Beginning encoding of {len(remaining_records):,} items (Batch size: {args.batch_size}, Device: {args.device})...")
    t0 = time.time()
    items_done = start_idx
    last_flush = time.time()

    try:
        with torch.no_grad():
            for batch_tensors, batch_indices in loader:
                if interrupted:
                    break

                batch_tensors = batch_tensors.to(args.device, non_blocking=True)
                with torch.amp.autocast(args.device):
                    feats = model.encode_image(batch_tensors)
                    feats = F.normalize(feats, dim=-1)

                feats_fp16 = feats.cpu().to(torch.float16).numpy()

                # Map global index
                b_size = len(batch_indices)
                target_start = items_done
                target_end = items_done + b_size
                mmap_arr[target_start:target_end] = feats_fp16
                items_done += b_size

                # Periodic progress logging
                if items_done % 1280 == 0 or items_done == total_items:
                    elapsed = time.time() - t0
                    processed = items_done - start_idx
                    vids_per_sec = processed / elapsed if elapsed > 0 else 0
                    rem = total_items - items_done
                    eta_sec = rem / vids_per_sec if vids_per_sec > 0 else 0
                    eta_m, eta_s = divmod(int(eta_sec), 60)
                    eta_h, eta_m = divmod(eta_m, 60)

                    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if args.device == "cuda" else 0
                    pct = (items_done / total_items) * 100
                    log(f"[{items_done:,}/{total_items:,}] ({pct:.1f}%) | Speed: {vids_per_sec:.1f} vids/s | ETA: {eta_h}h {eta_m:02d}m | VRAM: {vram_gb:.2f} GB")

                # Flush memmap every 60 seconds
                if time.time() - last_flush > 60:
                    mmap_arr.flush()
                    with open(state_file, "w") as f:
                        json.dump({"completed_items": items_done, "total_items": total_items, "dim": dim}, f)
                    last_flush = time.time()

    finally:
        log("Flushing memory-mapped embeddings to disk...")
        mmap_arr.flush()
        del mmap_arr
        with open(state_file, "w") as f:
            json.dump({"completed_items": items_done, "total_items": total_items, "dim": dim}, f)

    total_time = time.time() - t0
    log(f"Finished! Processed {items_done - start_idx:,} items in {total_time:.1f}s ({items_done:,}/{total_items:,} total).")
    log(f"Embeddings saved to: {embeddings_file} ({os.path.getsize(embeddings_file) / (1024**2):.1f} MB)")


if __name__ == "__main__":
    main()
