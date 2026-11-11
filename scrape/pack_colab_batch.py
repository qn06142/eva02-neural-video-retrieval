#!/usr/bin/env python3
"""Packs unencoded downloaded mosaics into portable Colab batch archives.

Usage:
    # Pack ALL unencoded mosaics into a single high-speed .tar archive:
    python scrape/pack_colab_batch.py --all --out colab_batch_all.tar

    # Pack ALL unencoded mosaics in chunks of 50,000 (~2 GB each):
    python scrape/pack_colab_batch.py --all --chunk-size 50000

    # Pack a quick subset of 20,000 mosaics:
    python scrape/pack_colab_batch.py --limit 20000 --out colab_batch_20k.tar
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time

SCRAPE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRAPE_DIR)
DEFAULT_DB = os.path.join(SCRAPE_DIR, "thumbs.db")
DEFAULT_META = os.path.join(SCRAPE_DIR, "embeddings_v2", "metadata_147k.sqlite3")
DEFAULT_MOSAICS = os.path.join(PROJECT_DIR, "mosaics")
DEFAULT_OUT = os.path.join(PROJECT_DIR, "colab_batch_all.tar")


def pack_single_archive(
    items: list[dict],
    out_archive: str,
    project_dir: str,
    use_gzip: bool = False,
) -> None:
    """Packs items and manifest into a single tar or tar.gz archive using fast bsdtar if available."""
    t0 = time.time()
    out_dir = os.path.dirname(os.path.abspath(out_archive))
    os.makedirs(out_dir, exist_ok=True)

    # Create temporary manifest and file list
    with tempfile.TemporaryDirectory(dir=project_dir, prefix="pack_temp_") as tmp_dir:
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)

        file_list_path = os.path.join(tmp_dir, "archive_list.txt")
        # Relative paths from project root: manifest.json and mosaics/<file>
        with open(file_list_path, "w", encoding="utf-8") as fl:
            fl.write(f"manifest.json\n")
            for item in items:
                # Relative path from project_dir
                rel_path = os.path.relpath(item["asset_path"], project_dir).replace("\\", "/")
                fl.write(f"{rel_path}\n")

        # Copy manifest into project_dir temporarily so tar sees both in relative root
        temp_manifest_project = os.path.join(project_dir, "manifest.json")
        shutil.copyfile(manifest_path, temp_manifest_project)

        has_tar_cli = shutil.which("tar") is not None
        tar_success = False

        if has_tar_cli:
            try:
                cmd = ["tar", "-czf" if use_gzip else "-cf", out_archive, "-C", project_dir, "-T", file_list_path]
                print(f"[*] Running native bsdtar ({'gzip' if use_gzip else 'uncompressed'})...")
                subprocess.run(cmd, check=True)
                tar_success = True
            except Exception as e:
                print(f"[!] Native tar encountered an error ({e}), falling back to Python tarfile...")

        if not tar_success:
            mode = "w:gz" if use_gzip else "w"
            print(f"[*] Packing via Python tarfile ({mode})...")
            with tarfile.open(out_archive, mode) as tar:
                tar.add(temp_manifest_project, arcname="manifest.json")
                for i, item in enumerate(items):
                    tar.add(item["asset_path"], arcname=f"mosaics/{item['filename']}")
                    if (i + 1) % 5000 == 0 or (i + 1) == len(items):
                        print(f"    Archived {i + 1:,} / {len(items):,} mosaics...", flush=True)

        # Cleanup temporary manifest at project root
        if os.path.exists(temp_manifest_project):
            os.remove(temp_manifest_project)

    elapsed = max(0.1, time.time() - t0)
    size_mb = os.path.getsize(out_archive) / (1024 * 1024)
    print(f"[+] Archive ready: {out_archive}")
    print(f"    Videos : {len(items):,} ({len(items)*9:,} keyframes)")
    print(f"    Size   : {size_mb:.2f} MB ({size_mb/1024:.2f} GB)")
    print(f"    Time   : {elapsed:.1f} s ({len(items)/elapsed:.1f} items/s)\n")


def main():
    parser = argparse.ArgumentParser(description="Pack unencoded mosaics for Colab G4 encoding")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to thumbs.db")
    parser.add_argument("--meta-db", default=DEFAULT_META, help="Path to metadata_147k.sqlite3")
    parser.add_argument("--mosaic-dir", default=DEFAULT_MOSAICS, help="Path to mosaics folder")
    parser.add_argument("--limit", type=int, default=0, help="Number of videos to pack (0 = all unencoded)")
    parser.add_argument("--all", action="store_true", help="Pack ALL unencoded mosaics")
    parser.add_argument("--chunk-size", type=int, default=0, help="Split into chunks of N videos (0 = single archive)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output archive path")
    parser.add_argument("--gz", action="store_true", help="Use gzip compression (default: uncompressed .tar for 12x speed)")
    args = parser.parse_args()

    print("[*] Connecting to SQLite databases to query unencoded videos...")
    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    cur.execute(f"ATTACH DATABASE '{args.meta_db}' AS mdb")

    query = """
        SELECT u.eid, MAX(u.title), u.asset_path, MAX(u.preview_url)
        FROM (
            SELECT eid, title, asset_path, preview_url FROM videos_diverse WHERE asset_path IS NOT NULL
            UNION ALL
            SELECT eid, title, asset_path, '' AS preview_url FROM videos WHERE asset_path IS NOT NULL
        ) u
        LEFT JOIN mdb.video_metadata m ON u.eid = m.eid
        WHERE m.eid IS NULL
        GROUP BY u.eid
    """
    if args.limit > 0 and not args.all:
        query += f" LIMIT {args.limit}"

    cur.execute(query)
    candidates = cur.fetchall()
    conn.close()

    print(f"[*] Found {len(candidates):,} candidate unencoded videos. Validating files on disk...")

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

    print(f"[*] Validated {len(valid_items):,} existing mosaics on disk ready for Colab.")

    # Determine chunking
    if args.chunk_size > 0 and len(valid_items) > args.chunk_size:
        total_chunks = (len(valid_items) + args.chunk_size - 1) // args.chunk_size
        print(f"[*] Splitting {len(valid_items):,} items into {total_chunks} chunks of {args.chunk_size:,} videos each...\n")

        base_out, ext = os.path.splitext(args.out)
        if ext.lower() == ".gz":
            base_out, _ = os.path.splitext(base_out)

        out_ext = ".tar.gz" if args.gz else ".tar"

        for chunk_idx in range(total_chunks):
            chunk_items = valid_items[chunk_idx * args.chunk_size : (chunk_idx + 1) * args.chunk_size]
            chunk_archive = f"{base_out}_part{chunk_idx + 1:02d}{out_ext}"
            print(f"--- Packaging Chunk {chunk_idx + 1} / {total_chunks} ({len(chunk_items):,} items) ---")
            pack_single_archive(chunk_items, chunk_archive, PROJECT_DIR, use_gzip=args.gz)
    else:
        out_path = args.out
        if args.gz and not (out_path.endswith(".tar.gz") or out_path.endswith(".tgz")):
            out_path = out_path + ".gz" if out_path.endswith(".tar") else out_path + ".tar.gz"
        print(f"[*] Packaging all {len(valid_items):,} items into single archive: {out_path}...")
        pack_single_archive(valid_items, out_path, PROJECT_DIR, use_gzip=args.gz)

    print("[+] Packaging completed successfully!")


if __name__ == "__main__":
    main()
