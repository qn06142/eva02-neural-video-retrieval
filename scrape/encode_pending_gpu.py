#!/usr/bin/env python3
"""Dedicated GPU 9-Tile Batch Encoder.

Pulls unprocessed downloaded mosaics from SQLite (`videos_diverse` / `videos`)
and encodes them in memory-safe PyTorch FP16 batches on GPU:
  - Multi-process DataLoader for fast parallel cropping of 3x3 tiles
  - Native FP16 Tensor Core execution
  - Appends to live binary embeddings & updates video_metadata in SQLite
  - Can run continuously or in standalone batches alongside the scraper
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

SCRAPE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRAPE_DIR)
DEFAULT_DB = os.path.join(SCRAPE_DIR, "thumbs.db")
DEFAULT_EMB_DIR = os.path.join(SCRAPE_DIR, "embeddings_v2")
DEFAULT_WEIGHTS = os.path.join(SCRAPE_DIR, "siglip_runs", "eva02_curated_v2", "best.pt")
DEFAULT_MODEL = "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(336, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])


class MosaicDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str, str, str]]):
        # items: (eid, title, asset_path, preview_url)
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        eid, title, path, preview = self.items[idx]
        tiles = []
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                w, h = im.size
                tw, th = w // 3, h // 3
                for r in range(3):
                    for c in range(3):
                        box = (c * tw, r * th, (c + 1) * tw, (r + 1) * th)
                        tiles.append(IMAGE_TRANSFORM(im.crop(box)))
                batch_tensor = torch.stack(tiles, dim=0)  # [9, 3, 336, 336]
        except Exception:
            batch_tensor = torch.zeros(9, 3, 336, 336)
        return batch_tensor, idx


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Dedicated GPU 9-Tile Batch Encoder")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to thumbs.db")
    parser.add_argument("--emb-dir", default=DEFAULT_EMB_DIR, help="Embeddings directory")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to fine-tuned weights")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size in videos (12 vids = 108 tiles)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos to encode (0 = all pending)")
    parser.add_argument("--continuous", action="store_true", help="Keep running and polling for newly downloaded videos")
    parser.add_argument("--poll-interval", type=int, default=5, help="Seconds to wait between polls in continuous mode")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    meta_db = os.path.join(args.emb_dir, "metadata_147k.sqlite3")
    raw_9tiles_file = os.path.join(args.emb_dir, "live_keyframes_9tiles.bin")
    raw_1tile_file = os.path.join(args.emb_dir, "live_globals.bin")
    state_file = os.path.join(args.emb_dir, "live_encoding_state.json")

    log(f"Loading EVA-02 model on {args.device}...")
    model, _, _ = open_clip.create_model_and_transforms(args.model_name)
    if os.path.exists(args.weights):
        log(f"Loading fine-tuned weights from {args.weights}...")
        ckpt = torch.load(args.weights, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    if args.device == "cuda":
        model = model.half().to(args.device).eval()
    else:
        model = model.to(args.device).eval()

    log(f"Model ready on {args.device} (Native FP16).")

    stop_requested = False

    def handle_sigint(sig, frame):
        nonlocal stop_requested
        log("Stop signal received, completing current batch and exiting...")
        stop_requested = True

    signal.signal(signal.SIGINT, handle_sigint)

    while not stop_requested:
        # 1. Query unprocessed items from thumbs.db
        conn = sqlite3.connect(args.db, timeout=60)
        cur = conn.cursor()
        cur.execute(f"ATTACH DATABASE '{meta_db}' AS mdb")

        # Prioritize videos_diverse, fallback to any done video in videos table
        query = """
            SELECT d.eid, d.title, d.asset_path, d.preview_url
            FROM videos_diverse d
            LEFT JOIN mdb.video_metadata m ON d.eid = m.eid
            WHERE m.eid IS NULL AND d.asset_path IS NOT NULL
        """
        if args.limit > 0:
            query += f" LIMIT {args.limit}"

        cur.execute(query)
        pending_items = cur.fetchall()
        conn.close()

        # Filter for items that actually exist on disk
        valid_items = [
            item for item in pending_items
            if os.path.exists(item[2]) and os.path.getsize(item[2]) > 1000
        ]

        if not valid_items:
            if args.continuous:
                time.sleep(args.poll_interval)
                continue
            else:
                log("No pending videos found to encode! All caught up.")
                break

        log(f"Found {len(valid_items):,} pending videos to encode on GPU. Processing...")

        dataset = MosaicDataset(valid_items)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=(args.device == "cuda"),
            prefetch_factor=2 if args.workers > 0 else None,
        )

        meta_conn = sqlite3.connect(meta_db, timeout=60)
        meta_cur = meta_conn.cursor()
        meta_cur.execute("SELECT MAX(idx), COUNT(*) FROM video_metadata")
        row = meta_cur.fetchone()
        next_idx = (row[0] + 1) if row and row[0] is not None else 0
        total_meta_count = row[1] if row else 0

        t0 = time.time()
        encoded_in_run = 0

        for batch_tensor, batch_indices in loader:
            if stop_requested:
                break

            # batch_tensor: [B, 9, 3, 336, 336]
            batch_tensor = batch_tensor.to(args.device)
            if args.device == "cuda":
                batch_tensor = batch_tensor.half()

            B = batch_tensor.size(0)
            flat_tiles = batch_tensor.view(B * 9, 3, 336, 336)

            with torch.inference_mode():
                feats = model.encode_image(flat_tiles)  # [B*9, 768]
                feats = F.normalize(feats.float(), dim=-1).half()
                all_9tiles_np = feats.view(B, 9, 768).cpu().numpy()
                center_1tile_np = all_9tiles_np[:, 4, :].copy()

            # Append binary embeddings
            with open(raw_9tiles_file, "ab") as f9:
                f9.write(all_9tiles_np.tobytes())
            with open(raw_1tile_file, "ab") as f1:
                f1.write(center_1tile_np.tobytes())

            # Update SQLite metadata
            meta_rows = []
            for idx_in_batch, original_idx in enumerate(batch_indices.tolist()):
                eid, title, path, preview = valid_items[original_idx]
                meta_rows.append((next_idx, eid, title, "[]", "[]", path, preview))
                next_idx += 1

            meta_cur.executemany("""
                INSERT OR REPLACE INTO video_metadata (idx, eid, title, clothing_tags, position_tags, asset_path, preview_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, meta_rows)
            meta_conn.commit()

            encoded_in_run += B
            total_meta_count += B

            elapsed = max(0.1, time.time() - t0)
            vids_per_sec = encoded_in_run / elapsed
            frames_per_sec = vids_per_sec * 9

            if encoded_in_run % (args.batch_size * 2) == 0 or encoded_in_run == len(valid_items):
                log(f"GPU Encoded: {encoded_in_run:,} / {len(valid_items):,} ({vids_per_sec:.1f} vids/s, {frames_per_sec:.1f} frames/s) | Total Corpus: {total_meta_count:,}")

        meta_conn.close()

        # Update state json
        try:
            with open(state_file, "w") as sf:
                json.dump({
                    "total_items": total_meta_count,
                    "completed_items": total_meta_count,
                    "last_idx": next_idx - 1,
                    "last_update": time.time(),
                }, sf)
        except Exception:
            pass

        log(f"Batch completed! Encoded {encoded_in_run:,} videos ({encoded_in_run * 9:,} keyframes).")
        if not args.continuous:
            break


if __name__ == "__main__":
    main()
