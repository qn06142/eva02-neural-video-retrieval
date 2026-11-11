#!/usr/bin/env python3
"""Populates preview_url column in metadata_147k.sqlite3 from thumbs.db."""

import os
import re
import sqlite3
import time

SCRAPE_DIR = os.path.dirname(os.path.abspath(__file__))
META_DB = os.path.join(SCRAPE_DIR, "embeddings_v2", "metadata_147k.sqlite3")
THUMBS_DB = os.path.join(SCRAPE_DIR, "thumbs.db")


def main():
    t0 = time.perf_counter()
    conn = sqlite3.connect(META_DB)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(video_metadata)")
    cols = [c[1] for c in cur.fetchall()]
    if "preview_url" not in cols:
        cur.execute("ALTER TABLE video_metadata ADD COLUMN preview_url TEXT")
        print("[*] Added 'preview_url' column to video_metadata.")

    print("[*] Attaching thumbs.db and updating preview URLs...")
    cur.execute(f"ATTACH DATABASE '{THUMBS_DB}' AS tdb")
    cur.execute("""
        UPDATE video_metadata 
        SET preview_url = (
            SELECT asset_url FROM tdb.videos 
            WHERE tdb.videos.eid = video_metadata.eid
        )
        WHERE preview_url IS NULL
    """)
    conn.commit()

    # Now convert .jpg -> preview.mp4 directly in SQL
    print("[*] Converting asset URLs to canonical preview.mp4 URLs...")
    cur.execute("""
        UPDATE video_metadata
        SET preview_url = substr(preview_url, 1, instr(preview_url, '/xv_') - 1) || '/preview.mp4'
        WHERE preview_url LIKE '%/xv_%'
    """)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM video_metadata WHERE preview_url IS NOT NULL")
    count = cur.fetchone()[0]
    dur = time.perf_counter() - t0
    print(f"[*] Done! {count:,} / 147,412 videos have direct preview.mp4 URLs in {dur:.2f}s.")

    cur.execute("SELECT eid, preview_url FROM video_metadata WHERE preview_url IS NOT NULL LIMIT 3")
    for eid, purl in cur.fetchall():
        print(f"  {eid} -> {purl}")

    conn.close()


if __name__ == "__main__":
    main()
