#!/usr/bin/env python3
"""Packs unencoded downloaded mosaics into a portable Colab batch tarball.

Usage:
    python scrape/pack_colab_batch.py --limit 20000 --out mosaics_batch_01.tar.gz
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tarfile
import time

SCRAPE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRAPE_DIR)
DEFAULT_DB = os.path.join(SCRAPE_DIR, "thumbs.db")
DEFAULT_META = os.path.join(SCRAPE_DIR, "embeddings_v2", "metadata_147k.sqlite3")
DEFAULT_MOSAICS = os.path.join(PROJECT_DIR, "mosaics")
DEFAULT_OUT = os.path.join(PROJECT_DIR, "colab_batch.tar.gz")


def main():
    parser = argparse.ArgumentParser(description="Pack unencoded mosaics for Colab G4 encoding")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to thumbs.db")
    parser.add_argument("--meta-db", default=DEFAULT_META, help="Path to metadata_147k.sqlite3")
    parser.add_argument("--mosaic-dir", default=DEFAULT_MOSAICS, help="Path to mosaics folder")
    parser.add_argument("--limit", type=int, default=20000, help="Number of videos to pack (default: 20,000)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output tar.gz archive path")
    args = parser.parse_args()

    print(f"[*] Querying unencoded videos from database...")
    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    cur.execute(f"ATTACH DATABASE '{args.meta_db}' AS mdb")

    query = """
        SELECT d.eid, d.title, d.asset_path, d.preview_url
        FROM videos_diverse d
        LEFT JOIN mdb.video_metadata m ON d.eid = m.eid
        WHERE m.eid IS NULL AND d.asset_path IS NOT NULL
        LIMIT ?
    """
    cur.execute(query, (args.limit,))
    candidates = cur.fetchall()
    conn.close()

    valid_items = []
    for eid, title, asset_path, preview_url in candidates:
        if os.path.exists(asset_path) and os.path.getsize(asset_path) > 1000:
            valid_items.append({
                "eid": eid,
                "title": title or "",
                "filename": os.path.basename(asset_path),
                "asset_path": asset_path,
                "preview_url": preview_url or "",
            })

    if not valid_items:
        print("[!] No pending unencoded mosaics found on disk to pack!")
        return

    print(f"[*] Found {len(valid_items):,} valid mosaics ready for Colab.")
    print(f"[*] Packing into {args.out}...")

    t0 = time.time()
    manifest_path = os.path.join(PROJECT_DIR, "colab_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(valid_items, f, indent=2)

    with tarfile.open(args.out, "w:gz") as tar:
        # Add manifest
        tar.add(manifest_path, arcname="manifest.json")
        for i, item in enumerate(valid_items):
            tar.add(item["asset_path"], arcname=f"mosaics/{item['filename']}")
            if (i + 1) % 2500 == 0 or (i + 1) == len(valid_items):
                print(f"    Archived {i + 1:,} / {len(valid_items):,} mosaics...", flush=True)

    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    tar_size_mb = os.path.getsize(args.out) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"\n[+] Success! Created Colab package:")
    print(f"    Archive : {args.out}")
    print(f"    Videos  : {len(valid_items):,} ({len(valid_items)*9:,} keyframes)")
    print(f"    Size    : {tar_size_mb:.2f} MB")
    print(f"    Time    : {elapsed:.1f} s")
    print(f"\nUpload this file to Google Drive and open the Colab G4 notebook!")


if __name__ == "__main__":
    main()
