#!/usr/bin/env python3
"""Targeted XVideos search harvester with live curation filtering and mosaic streaming.

Features:
  - Automated Danbooru Taxonomy Ingestion: Dynamically loads queries from
    `scrape/danbooru_taxonomy_extracted.json` (zero reliance on dirty XVideos tags or manual lists).
  - Dynamic Deficit Ordering: Queries are sorted by deficit in `thumbs.db` (starved positions
    like lotus, prone bone, mating press, spooning, standing split are fetched first).
  - Live Pre-Download Filtering: Uncurated / low-signal videos are marked 'skipped_uncurated'
    in SQLite and never saved to disk.
  - Live Curated Table Sync: Downloaded mosaics are immediately inserted into `videos_curated`
    with clean titles and structured posture/clothing labels.
  - High-Throughput Mosaic Streaming: 16 background threads stream 3x3 mosaics concurrently.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import os
import queue
import random
import re
import sqlite3
import sys
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_DIR, "scrape", "thumbs.db")
MOSAICS_DIR = os.path.join(PROJECT_DIR, "mosaics")
TAXONOMY_JSON = os.path.join(PROJECT_DIR, "scrape", "danbooru_taxonomy_extracted.json")

sys.path.append(os.path.join(PROJECT_DIR, "scrape"))
from filter_curated_pairs import clean_title, extract_clothing, extract_position

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

EID_RE = re.compile(r"/video\.([A-Za-z0-9]+)")


def mosaic_url_from_frame(frame_url: str) -> str:
    """Convert a single frame CDN URL to the 3x3 contact sheet URL."""
    return re.sub(r"[^/]+\.(?:jpg|jpeg|png|webp|avif)$", "mozaique_listing.jpg", frame_url or "")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Dynamic Deficit Query Generator (from Danbooru Mass Analysis)
# --------------------------------------------------------------------------- #

def get_deficit_ordered_queries(db_path: str, taxonomy_json_path: str) -> List[Tuple[str, str, int, int]]:
    """Generate search queries strictly from Danbooru taxonomy, ordered by deficit in SQLite.

    Returns:
        List of tuples: (query_text, category_type, current_db_count, danbooru_popularity)
    """
    if not os.path.exists(taxonomy_json_path):
        raise FileNotFoundError(f"Missing taxonomy file: {taxonomy_json_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Ensure table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS videos_curated (
            eid TEXT PRIMARY KEY,
            clean_title TEXT,
            clothing_tags TEXT,
            position_tags TEXT,
            has_both INTEGER,
            asset_path TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_curated_both ON videos_curated (has_both)")

    # Tally current database frequencies
    pos_counts: Dict[str, int] = collections.defaultdict(int)
    cloth_counts: Dict[str, int] = collections.defaultdict(int)
    for p_raw, c_raw in cur.execute("SELECT position_tags, clothing_tags FROM videos_curated").fetchall():
        if p_raw:
            for t in json.loads(p_raw):
                pos_counts[t] += 1
        if c_raw:
            for t in json.loads(c_raw):
                cloth_counts[t] += 1
    conn.close()

    with open(taxonomy_json_path, encoding="utf-8") as f:
        tax = json.load(f)

    seen_queries: Set[str] = set()
    query_candidates = []

    # 1. Postures from Danbooru Mass Analysis
    for p in tax.get("postures", []):
        q = p["search_query"].strip()
        matched = extract_position(q)
        if not matched:
            continue
        if q in seen_queries:
            continue
        seen_queries.add(q)
        db_cnt = min(pos_counts[t] for t in matched)
        query_candidates.append((q, "posture", db_cnt, p["popularity"]))

    # 2. Clothing from Danbooru Mass Analysis (top 300)
    for c in tax.get("clothing", [])[:300]:
        q = c["search_query"].strip()
        matched = extract_clothing(q)
        if not matched:
            continue
        if q in seen_queries:
            continue
        seen_queries.add(q)
        db_cnt = min(cloth_counts[t] for t in matched)
        query_candidates.append((q, "clothing", db_cnt, c["popularity"]))

    # Sort primarily by lowest count in database (deficit), secondarily by highest Danbooru popularity
    query_candidates.sort(key=lambda x: (x[2], -x[3]))
    return query_candidates


# --------------------------------------------------------------------------- #
# Database Manager
# --------------------------------------------------------------------------- #

class HarvesterDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=60)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.lock = threading.Lock()
        with self.lock:
            self.known_eids: Set[str] = set(
                row[0] for row in self.conn.execute("SELECT eid FROM videos").fetchall()
            )
        log(f"Harvester loaded {len(self.known_eids):,} known video IDs from database.")

    def add_curated_video(self, eid: str, url: str, title: str, asset_url: str) -> bool:
        with self.lock:
            if eid in self.known_eids:
                return False
            self.known_eids.add(eid)
            self.conn.execute(
                """INSERT OR IGNORE INTO videos (eid, url, title, asset_url, page_state, asset_state, discovered)
                   VALUES (?, ?, ?, ?, 'fetched', 'pending', ?)""",
                (eid, url, title, asset_url, time.time())
            )
            self.conn.commit()
            return True

    def add_uncurated_video(self, eid: str, url: str, title: str) -> bool:
        with self.lock:
            if eid in self.known_eids:
                return False
            self.known_eids.add(eid)
            self.conn.execute(
                """INSERT OR IGNORE INTO videos (eid, url, title, page_state, asset_state, error, discovered)
                   VALUES (?, ?, ?, 'fetched', 'skipped_uncurated', 'no_posture_attire_signal', ?)""",
                (eid, url, title, time.time())
            )
            self.conn.commit()
            return True

    def mark_download_done(self, eid: str, asset_path: str, clean_t: str, c_tags: List[str], p_tags: List[str]) -> None:
        with self.lock:
            self.conn.execute(
                """UPDATE videos SET asset_state = 'done', asset_path = ?, downloaded = ? WHERE eid = ?""",
                (asset_path, time.time(), eid)
            )
            has_both = int(bool(c_tags and p_tags))
            self.conn.execute(
                """INSERT OR REPLACE INTO videos_curated (eid, clean_title, clothing_tags, position_tags, has_both, asset_path)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (eid, clean_t, json.dumps(c_tags), json.dumps(p_tags), has_both, asset_path)
            )
            self.conn.commit()

    def mark_download_failed(self, eid: str, error: str) -> None:
        with self.lock:
            self.conn.execute(
                """UPDATE videos SET asset_state = 'failed', error = ? WHERE eid = ?""",
                (error, eid)
            )
            self.conn.commit()


# --------------------------------------------------------------------------- #
# Harvester & Downloader
# --------------------------------------------------------------------------- #

class TargetedHarvester:
    def __init__(self, db: HarvesterDB, max_pages_per_query: int = 40,
                 download_workers: int = 16, target_new_curated: int = 50000):
        self.db = db
        self.max_pages_per_query = max_pages_per_query
        self.target_new_curated = target_new_curated
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        # Queue item: (eid, mosaic_url, clean_title, clothing_tags, position_tags)
        self.download_queue: queue.Queue[Tuple[str, str, str, List[str], List[str]]] = queue.Queue(maxsize=2000)
        self.stop_event = threading.Event()
        self.stats_lock = threading.Lock()
        self.discovered_total = 0
        self.curated_queued = 0
        self.downloaded_count = 0
        self.download_workers = download_workers

    def start_download_workers(self):
        os.makedirs(MOSAICS_DIR, exist_ok=True)
        for i in range(self.download_workers):
            t = threading.Thread(target=self._download_worker_loop, name=f"Downloader-{i}", daemon=True)
            t.start()

    def _download_worker_loop(self):
        worker_session = requests.Session()
        worker_session.headers.update(HEADERS)
        while not self.stop_event.is_set():
            try:
                eid, mosaic_url, c_title, c_tags, p_tags = self.download_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            target_path = os.path.join(MOSAICS_DIR, f"{eid}.jpg")
            try:
                resp = worker_session.get(mosaic_url, timeout=12)
                if resp.status_code == 200 and len(resp.content) > 5000:
                    with open(target_path, "wb") as f:
                        f.write(resp.content)
                    self.db.mark_download_done(eid, target_path, c_title, c_tags, p_tags)
                    with self.stats_lock:
                        self.downloaded_count += 1
                else:
                    self.db.mark_download_failed(eid, f"HTTP {resp.status_code} len={len(resp.content)}")
            except Exception as e:
                self.db.mark_download_failed(eid, str(e))
            finally:
                self.download_queue.task_done()

    def harvest_query(self, query: str, cat_type: str, current_db_count: int, pop: int):
        log(f"--- Harvesting [{cat_type.upper()}] '{query}' (current DB: {current_db_count}, Danbooru: {pop:,}) ---")
        consecutive_empty = 0

        for page in range(1, self.max_pages_per_query + 1):
            if self.stop_event.is_set():
                break

            url = f"https://www.xvideos.com/?k={requests.utils.quote(query)}&p={page}"
            try:
                resp = self.session.get(url, timeout=10)
                if resp.status_code != 200:
                    log(f"  [{query}] p={page}: HTTP {resp.status_code}")
                    break

                soup = BeautifulSoup(resp.text, "html.parser")
                blocks = soup.select(".thumb-block")
                if not blocks:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        break
                    continue
                consecutive_empty = 0

                page_discovered = 0
                page_curated = 0

                for b in blocks:
                    title_a = b.select_one("p.title a")
                    if not title_a:
                        continue
                    raw_title = title_a.get_text(strip=True)
                    rel_url = title_a.get("href", "")
                    eid_match = EID_RE.search(rel_url)
                    if not eid_match:
                        continue
                    eid = eid_match.group(1)

                    # Thumbnail image
                    img = b.select_one("img")
                    frame_url = img.get("data-src") or img.get("src") if img else ""
                    if not frame_url or "lightbox-blank" in frame_url:
                        continue

                    m_url = mosaic_url_from_frame(frame_url)
                    c_title = clean_title(raw_title)

                    c_tags = extract_clothing(c_title)
                    p_tags = extract_position(c_title)
                    is_high_signal = bool(c_tags or p_tags) and len(c_title) >= 10 and len(c_title.split()) >= 3

                    page_discovered += 1
                    if is_high_signal:
                        full_url = f"https://www.xvideos.com{rel_url}" if rel_url.startswith("/") else rel_url
                        if self.db.add_curated_video(eid, full_url, c_title, m_url):
                            self.download_queue.put((eid, m_url, c_title, c_tags, p_tags))
                            page_curated += 1
                    else:
                        self.db.add_uncurated_video(eid, rel_url, c_title)

                with self.stats_lock:
                    self.discovered_total += page_discovered
                    self.curated_queued += page_curated

                log(f"  [{query}] Page {page}: found {page_discovered} | +{page_curated} curated | "
                    f"Total Curated: {self.curated_queued:,} | Downloaded: {self.downloaded_count:,}")

                time.sleep(random.uniform(0.6, 1.2))

                if self.curated_queued >= self.target_new_curated:
                    log(f"Reached target curated count: {self.target_new_curated:,}!")
                    self.stop_event.set()
                    break

            except Exception as e:
                log(f"Error harvesting '{query}' page {page}: {e}")
                time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted XVideos search harvester driven by Danbooru mass ontology")
    parser.add_argument("--max-pages", type=int, default=30, help="Max search pages per keyword query")
    parser.add_argument("--target", type=int, default=50000, help="Target new curated samples to harvest")
    parser.add_argument("--workers", type=int, default=16, help="Mosaic download worker threads")
    parser.add_argument("--mode", type=str, default="deficit", choices=["deficit", "all", "postures", "clothing"],
                        help="Harvesting mode: deficit (highest database deficit first), postures, clothing, all")
    args = parser.parse_args()

    db = HarvesterDB(DB_PATH)
    harvester = TargetedHarvester(
        db=db,
        max_pages_per_query=args.max_pages,
        download_workers=args.workers,
        target_new_curated=args.target
    )
    harvester.start_download_workers()

    log(f"Deriving search queries dynamically from Danbooru taxonomy ({TAXONOMY_JSON})...")
    candidates = get_deficit_ordered_queries(DB_PATH, TAXONOMY_JSON)

    if args.mode == "postures":
        candidates = [c for c in candidates if c[1] == "posture"]
    elif args.mode == "clothing":
        candidates = [c for c in candidates if c[1] == "clothing"]

    log(f"Generated {len(candidates)} prioritized taxonomy queries (mode={args.mode}, target={args.target:,}).")

    try:
        for q, cat, db_cnt, pop in candidates:
            if harvester.stop_event.is_set():
                break
            harvester.harvest_query(q, cat, db_cnt, pop)
    except KeyboardInterrupt:
        log("Interrupted by user. Shutting down gracefully...")
        harvester.stop_event.set()

    log("Waiting for pending mosaic downloads to finish...")
    harvester.download_queue.join()
    log(f"Harvest complete! Discovered: {harvester.discovered_total:,} | Curated: {harvester.curated_queued:,} | Downloaded: {harvester.downloaded_count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
