#!/usr/bin/env python3
"""Merges Colab-generated embeddings and metadata into local search index.

Usage:
    python scrape/merge_colab_results.py \
        --npy-9tiles path/to/encoded_keyframes_9tiles.npy \
        --npy-1tile path/to/encoded_globals.npy \
        --manifest path/to/encoded_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

import numpy as np

SCRAPE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRAPE_DIR)
DEFAULT_EMB = os.path.join(SCRAPE_DIR, "embeddings_v2")
DEFAULT_META = os.path.join(DEFAULT_EMB, "metadata_147k.sqlite3")


def main():
    parser = argparse.ArgumentParser(description="Merge Colab output embeddings into local search engine")
    parser.add_argument("--emb-dir", default=DEFAULT_EMB, help="Local embeddings directory")
    parser.add_argument("--meta-db", default=DEFAULT_META, help="Local metadata_147k.sqlite3 path")
    parser.add_argument("--npy-9tiles", required=True, help="Path to Colab encoded_keyframes_9tiles.npy")
    parser.add_argument("--npy-1tile", default=None, help="Path to Colab encoded_globals.npy")
    parser.add_argument("--manifest", required=True, help="Path to Colab encoded_manifest.json")
    args = parser.parse_args()

    print(f"[*] Loading Colab results from {args.npy_9tiles} and {args.manifest}...")
    emb_9 = np.load(args.npy_9tiles)
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    num_items = len(manifest)
    if emb_9.shape[0] != num_items:
        raise ValueError(f"Mismatch: embeddings have {emb_9.shape[0]} rows but manifest has {num_items} items!")

    print(f"[*] Validated {num_items:,} new encoded videos ({num_items * 9:,} keyframes).")

    # 1. Append to local binary files
    raw_9tiles_file = os.path.join(args.emb_dir, "live_keyframes_9tiles.bin")
    raw_1tile_file = os.path.join(args.emb_dir, "live_globals.bin")

    emb_1 = emb_9[:, 4, :].copy() if args.npy_1tile is None else np.load(args.npy_1tile)

    print(f"[*] Appending embeddings to {raw_9tiles_file} and {raw_1tile_file}...")
    with open(raw_9tiles_file, "ab") as f9:
        f9.write(emb_9.astype(np.float16).tobytes())
    with open(raw_1tile_file, "ab") as f1:
        f1.write(emb_1.astype(np.float16).tobytes())

    # 2. Update metadata SQLite database
    print(f"[*] Updating {args.meta_db}...")
    conn = sqlite3.connect(args.meta_db)
    cur = conn.cursor()
    cur.execute("SELECT MAX(idx), COUNT(*) FROM video_metadata")
    row = cur.fetchone()
    next_idx = (row[0] + 1) if row and row[0] is not None else 0

    rows = []
    for item in manifest:
        rows.append((
            next_idx,
            item["eid"],
            item["title"],
            "[]",
            "[]",
            item["asset_path"],
            item.get("preview_url", "")
        ))
        next_idx += 1

    cur.executemany("""
        INSERT OR REPLACE INTO video_metadata (idx, eid, title, clothing_tags, position_tags, asset_path, preview_url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM video_metadata")
    total_count = cur.fetchone()[0]
    conn.close()

    # 3. Update state file
    state_file = os.path.join(args.emb_dir, "live_encoding_state.json")
    try:
        with open(state_file, "w") as sf:
            json.dump({
                "total_items": total_count,
                "completed_items": total_count,
                "last_idx": next_idx - 1,
                "last_update": time.time(),
            }, sf)
    except Exception:
        pass

    print(f"\n[+] Successfully merged Colab batch into local search engine!")
    print(f"    Added Videos : {num_items:,} ({num_items * 9:,} keyframes)")
    print(f"    Total Corpus : {total_count:,} searchable videos")


if __name__ == "__main__":
    main()
