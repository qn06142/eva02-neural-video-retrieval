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
import time

import numpy as np

SCRAPE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRAPE_DIR)
DEFAULT_EMB = os.path.join(SCRAPE_DIR, "embeddings_v2")
DEFAULT_META = os.path.join(DEFAULT_EMB, "metadata_147k.sqlite3")
EMBED_DIM = 768
NUM_TILES = 9
BYTES_PER_1TILE = EMBED_DIM * 2
BYTES_PER_9TILES = NUM_TILES * EMBED_DIM * 2


def get_base_count(emb_dir: str) -> int:
    """Return number of items stored in the base globals_147k.npy memmap."""
    base_1 = os.path.join(emb_dir, "globals_147k.npy")
    if os.path.exists(base_1):
        return os.path.getsize(base_1) // BYTES_PER_1TILE
    state_1 = os.path.join(emb_dir, "encoding_state.json")
    if os.path.exists(state_1):
        with open(state_1, "r", encoding="utf-8") as f:
            return int(json.load(f).get("completed_items", 0))
    return 0


def sync_live_binaries_to_sql(emb_dir: str, meta_db: str, reset_live: bool = False) -> tuple[int, int]:
    """Ensure live_*.bin row counts strictly align with metadata_147k rows >= base_count.

    Returns (base_count, next_idx).
    """
    base_count = get_base_count(emb_dir)
    raw_9 = os.path.join(emb_dir, "live_keyframes_9tiles.bin")
    raw_1 = os.path.join(emb_dir, "live_globals.bin")

    conn = sqlite3.connect(meta_db, timeout=60)
    cur = conn.cursor()

    if reset_live:
        cur.execute("DELETE FROM video_metadata WHERE idx >= ?", (base_count,))
        conn.commit()
        for p in (raw_9, raw_1):
            if os.path.exists(p):
                os.remove(p)
        print(f"[*] Reset live residual rows (>= {base_count:,}) and cleared old live_*.bin files.")

    row = cur.execute("SELECT MAX(idx), COUNT(*) FROM video_metadata").fetchone()
    max_idx = row[0] if row and row[0] is not None else -1
    total_sql = row[1] if row else 0
    conn.close()

    if max_idx + 1 != total_sql:
        raise RuntimeError(
            f"SQLite video_metadata has index gaps/duplicates (MAX(idx)+1={max_idx + 1:,} vs COUNT(*)={total_sql:,})."
        )

    expected_live = max(0, total_sql - base_count)
    rows_9 = (os.path.getsize(raw_9) // BYTES_PER_9TILES) if os.path.exists(raw_9) else 0
    rows_1 = (os.path.getsize(raw_1) // BYTES_PER_1TILE) if os.path.exists(raw_1) else 0

    if rows_9 != expected_live or rows_1 != expected_live:
        # Auto-trim only uncommitted trailing rows if binary > expected SQL rows
        if rows_9 >= expected_live and rows_1 >= expected_live and expected_live == 0:
            for p in (raw_9, raw_1):
                if os.path.exists(p):
                    os.remove(p)
        elif rows_9 > expected_live or rows_1 > expected_live:
            print(
                f"[!] Trimming uncommitted trailing rows in live_*.bin "
                f"(9tile={rows_9}, 1tile={rows_1} -> aligned={expected_live})..."
            )
            if os.path.exists(raw_9) and rows_9 > expected_live:
                with open(raw_9, "r+b") as f:
                    f.truncate(expected_live * BYTES_PER_9TILES)
            if os.path.exists(raw_1) and rows_1 > expected_live:
                with open(raw_1, "r+b") as f:
                    f.truncate(expected_live * BYTES_PER_1TILE)
        else:
            raise RuntimeError(
                f"Live binary files have fewer rows than SQLite (9tile={rows_9}, 1tile={rows_1}, sql_live={expected_live}). "
                f"Re-run with --reset-live to start live index cleanly from base_count={base_count:,}."
            )

    return base_count, max_idx + 1


def merge_colab_batch(
    emb_dir: str,
    meta_db: str,
    npy_9tiles: str,
    manifest_path: str,
    npy_1tile: str | None = None,
    reset_live: bool = False,
) -> dict:
    """Validate, deduplicate against existing SQLite eids, and atomically merge Colab results."""
    base_count, next_idx = sync_live_binaries_to_sql(emb_dir, meta_db, reset_live=reset_live)

    print(f"[*] Loading Colab results from {npy_9tiles} and {manifest_path}...")
    emb_9 = np.load(npy_9tiles)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if emb_9.shape[0] != len(manifest):
        raise ValueError(f"Mismatch: embeddings have {emb_9.shape[0]} rows but manifest has {len(manifest)} items!")
    if emb_9.ndim != 3 or emb_9.shape[1:] != (NUM_TILES, EMBED_DIM):
        raise ValueError(f"Expected 9-tile embeddings of shape (N, 9, 768), got {emb_9.shape}")

    emb_1 = emb_9[:, 4, :].copy() if npy_1tile is None else np.load(npy_1tile)
    if emb_1.shape != (len(manifest), EMBED_DIM):
        raise ValueError(f"Expected 1-tile embeddings of shape ({len(manifest)}, 768), got {emb_1.shape}")

    # Filter out any EID already indexed in SQLite or duplicated within manifest
    conn = sqlite3.connect(meta_db, timeout=60)
    cur = conn.cursor()
    existing_eids = {r[0] for r in cur.execute("SELECT eid FROM video_metadata").fetchall()}

    keep_indices = []
    seen_new = set()
    rows = []
    cur_idx = next_idx
    for i, item in enumerate(manifest):
        eid = item["eid"]
        if eid in existing_eids or eid in seen_new:
            continue
        seen_new.add(eid)
        keep_indices.append(i)
        rows.append((
            cur_idx,
            eid,
            item.get("title", ""),
            "[]",
            "[]",
            item.get("asset_path", ""),
            item.get("preview_url", ""),
        ))
        cur_idx += 1

    skipped = len(manifest) - len(keep_indices)
    if skipped > 0:
        print(f"[*] Skipped {skipped:,} duplicate/already-indexed EIDs to preserve strict 1-to-1 index alignment.")

    if not keep_indices:
        conn.close()
        print("[*] All manifest items are already in metadata_147k.sqlite3. Nothing to merge.")
        return {"added": 0, "skipped": skipped, "total": next_idx}

    emb_9_filtered = emb_9[keep_indices].astype(np.float16)
    emb_1_filtered = emb_1[keep_indices].astype(np.float16)

    raw_9tiles_file = os.path.join(emb_dir, "live_keyframes_9tiles.bin")
    raw_1tile_file = os.path.join(emb_dir, "live_globals.bin")

    print(f"[*] Appending {len(keep_indices):,} aligned videos ({len(keep_indices) * 9:,} keyframes) to live_*.bin...")
    with open(raw_9tiles_file, "ab") as f9:
        f9.write(emb_9_filtered.tobytes())
    with open(raw_1tile_file, "ab") as f1:
        f1.write(emb_1_filtered.tobytes())

    print(f"[*] Committing {len(rows):,} metadata rows (idx {next_idx:,} .. {cur_idx - 1:,}) to {meta_db}...")
    cur.executemany("""
        INSERT INTO video_metadata (idx, eid, title, clothing_tags, position_tags, asset_path, preview_url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()

    total_count = cur.execute("SELECT COUNT(*) FROM video_metadata").fetchone()[0]
    conn.close()

    state_file = os.path.join(emb_dir, "live_encoding_state.json")
    with open(state_file, "w", encoding="utf-8") as sf:
        json.dump({
            "total_items": total_count,
            "completed_items": total_count,
            "base_items": base_count,
            "live_items": total_count - base_count,
            "last_idx": cur_idx - 1,
            "last_update": time.time(),
        }, sf, indent=2)

    print(f"\n[+] Successfully merged Colab batch into local search engine!")
    print(f"    Added Videos : {len(keep_indices):,} ({len(keep_indices) * 9:,} keyframes)")
    print(f"    Total Corpus : {total_count:,} searchable videos (0 .. {cur_idx - 1:,})")
    return {"added": len(keep_indices), "skipped": skipped, "total": total_count}


def main():
    parser = argparse.ArgumentParser(description="Merge Colab output embeddings into local search engine")
    parser.add_argument("--emb-dir", default=DEFAULT_EMB, help="Local embeddings directory")
    parser.add_argument("--meta-db", default=DEFAULT_META, help="Local metadata_147k.sqlite3 path")
    parser.add_argument("--npy-9tiles", required=True, help="Path to Colab encoded_keyframes_9tiles.npy")
    parser.add_argument("--npy-1tile", default=None, help="Path to Colab encoded_globals.npy")
    parser.add_argument("--manifest", required=True, help="Path to Colab encoded_manifest.json")
    parser.add_argument(
        "--reset-live",
        action="store_true",
        help="Discard existing live_*.bin and SQLite rows >= base_count before merging cleanly",
    )
    args = parser.parse_args()

    merge_colab_batch(
        emb_dir=args.emb_dir,
        meta_db=args.meta_db,
        npy_9tiles=args.npy_9tiles,
        manifest_path=args.manifest,
        npy_1tile=args.npy_1tile,
        reset_live=args.reset_live,
    )


if __name__ == "__main__":
    main()
