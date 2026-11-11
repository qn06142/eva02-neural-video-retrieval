#!/usr/bin/env python3
"""High-performance parallel XVideos search harvester with batch SQLite commits.

Architecture:
  - Concurrent Search Workers (16 threads): Scrapes search pages in parallel.
  - Ultra-fast Compiled Regex Parsing: ~1 ms per 27-video page (bypasses BeautifulSoup).
  - Dedicated Downloader Pool (32 threads): High-concurrency mosaic streaming.
  - Dedicated Batch DB Writer (1 thread): Batches SQLite writes to eliminate lock contention.
  - In-memory EID Set: Instant O(1) deduplication without disk locks.
  - Dynamic Deficit Prioritization: Driven strictly by Danbooru mass taxonomy.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import os
import queue
import re
import sqlite3
import sys
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

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

BLOCK_RE = re.compile(
    r'<div\s+id="video_([A-Za-z0-9]+)".*?'
    r'data-mzl="([^"]+)".*?'
    r'<p\s+class="title"><a\s+href="([^"]+)"\s+title="([^"]+)"',
    re.DOTALL
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def create_session(pool_size: int = 32) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


# --------------------------------------------------------------------------- #
# Batch Database Writer
# --------------------------------------------------------------------------- #

class BatchDBWriter(threading.Thread):
    def __init__(self, db_path: str, batch_size: int = 100, flush_interval: float = 0.5):
        super().__init__(name="BatchDBWriter", daemon=True)
        self.db_path = db_path
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.conn = None
        self.curated_queue: queue.Queue[Tuple] = queue.Queue()
        self.uncurated_queue: queue.Queue[Tuple] = queue.Queue()
        self.stop_event = threading.Event()

    def run(self):
        self.conn = sqlite3.connect(self.db_path, timeout=60, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS videos_curated (
                eid TEXT PRIMARY KEY,
                clean_title TEXT,
                clothing_tags TEXT,
                position_tags TEXT,
                has_both INTEGER,
                asset_path TEXT
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_curated_both ON videos_curated (has_both)")

        last_flush = time.time()
        curated_batch = []
        uncurated_batch = []

        while not self.stop_event.is_set() or not self.curated_queue.empty() or not self.uncurated_queue.empty():
            try:
                # Drain curated
                while len(curated_batch) < self.batch_size:
                    item = self.curated_queue.get_nowait()
                    curated_batch.append(item)
                    self.curated_queue.task_done()
            except queue.Empty:
                pass

            try:
                # Drain uncurated
                while len(uncurated_batch) < self.batch_size:
                    item = self.uncurated_queue.get_nowait()
                    uncurated_batch.append(item)
                    self.uncurated_queue.task_done()
            except queue.Empty:
                pass

            now = time.time()
            if (len(curated_batch) >= self.batch_size or len(uncurated_batch) >= self.batch_size or
                    (now - last_flush) >= self.flush_interval):
                self._flush(curated_batch, uncurated_batch)
                curated_batch.clear()
                uncurated_batch.clear()
                last_flush = now

            time.sleep(0.05)

        # Final flush
        self._flush(curated_batch, uncurated_batch)
        self.conn.close()

    def _flush(self, curated_batch: List[Tuple], uncurated_batch: List[Tuple]):
        if not curated_batch and not uncurated_batch:
            return

        cur = self.conn.cursor()
        now_ts = time.time()

        if uncurated_batch:
            # (eid, url, title)
            cur.executemany(
                """INSERT OR IGNORE INTO videos (eid, url, title, page_state, asset_state, attempts, error, discovered)
                   VALUES (?, ?, ?, 'fetched', 'skipped_uncurated', 0, 'no_posture_attire_signal', ?)""",
                [(b[0], b[1], b[2], now_ts) for b in uncurated_batch]
            )

        if curated_batch:
            # (eid, url, clean_t, asset_path, c_tags, p_tags)
            cur.executemany(
                """INSERT OR REPLACE INTO videos (eid, url, title, asset_path, page_state, asset_state, attempts, discovered, downloaded)
                   VALUES (?, ?, ?, ?, 'fetched', 'done', 1, ?, ?)""",
                [(b[0], b[1], b[2], b[3], now_ts, now_ts) for b in curated_batch]
            )
            cur.executemany(
                """INSERT OR REPLACE INTO videos_curated (eid, clean_title, clothing_tags, position_tags, has_both, asset_path)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [(b[0], b[2], json.dumps(b[4]), json.dumps(b[5]), int(bool(b[4] and b[5])), b[3]) for b in curated_batch]
            )

        self.conn.commit()


# --------------------------------------------------------------------------- #
# Fast Harvester Engine
# --------------------------------------------------------------------------- #

class FastHarvester:
    def __init__(self, db_path: str, search_workers: int = 16, download_workers: int = 32,
                 max_pages_per_query: int = 30, target_curated: int = 50000):
        self.db_path = db_path
        self.search_workers = search_workers
        self.download_workers = download_workers
        self.max_pages_per_query = max_pages_per_query
        self.target_curated = target_curated

        # Load known EIDs in memory
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        self.known_eids: Set[str] = set(r[0] for r in cur.execute("SELECT eid FROM videos").fetchall())
        conn.close()
        log(f"FastHarvester pre-loaded {len(self.known_eids):,} known video IDs into memory.")

        self.eid_lock = threading.Lock()
        self.active_lock = threading.Lock()
        self.active_search_workers = 0

        self.db_writer = BatchDBWriter(db_path)
        self.page_task_queue: queue.Queue[Tuple[str, int, str]] = queue.Queue()
        self.download_queue: queue.Queue[Tuple[str, str, str, str, List[str], List[str]]] = queue.Queue(maxsize=10000)

        self.stop_event = threading.Event()
        self.stats_lock = threading.Lock()
        self.discovered_count = 0
        self.curated_queued_count = 0
        self.downloaded_count = 0
        self.pages_fetched = 0
        self.total_seen = 0
        self.t_start = time.time()

    def run(self, candidates: List[Tuple[str, str, int, int]]):
        os.makedirs(MOSAICS_DIR, exist_ok=True)
        self.db_writer.start()

        # Seed ONLY page 1 for each query (dynamic pagination discovery)
        for q, cat, db_cnt, pop in candidates:
            self.page_task_queue.put((q, 1, cat))

        log(f"Seeded {self.page_task_queue.qsize():,} query entrypoints into dynamic pagination queue.")

        # Start Download Workers
        download_threads = []
        for i in range(self.download_workers):
            t = threading.Thread(target=self._download_worker_loop, name=f"Downloader-{i}", daemon=True)
            t.start()
            download_threads.append(t)

        # Start Search Workers
        search_threads = []
        for i in range(self.search_workers):
            t = threading.Thread(target=self._search_worker_loop, name=f"Searcher-{i}", daemon=True)
            t.start()
            search_threads.append(t)

        # Monitor Thread
        monitor_thread = threading.Thread(target=self._monitor_loop, name="Monitor", daemon=True)
        monitor_thread.start()

        log(f"FastHarvester running: {self.search_workers} search threads | {self.download_workers} download threads...")

        try:
            # Wait for search tasks to complete or target reached
            while not self.stop_event.is_set():
                with self.active_lock:
                    if self.page_task_queue.empty() and self.active_search_workers == 0:
                        break
                time.sleep(0.5)

            if not self.stop_event.is_set():
                log("All search pages and query pagination completed! Waiting for active downloads to finish...")
                self.download_queue.join()
        except KeyboardInterrupt:
            log("Interrupted by user. Shutting down gracefully...")
            self.stop_event.set()

        self.stop_event.set()
        self.db_writer.stop_event.set()
        self.db_writer.join(timeout=10)

        elapsed = time.time() - self.t_start
        log(f"Harvester Finished in {elapsed:.1f}s!")
        log(f"Final Stats: {self.total_seen:,} Scanned | {self.discovered_count:,} New Discovered | {self.curated_queued_count:,} Curated | {self.downloaded_count:,} Downloaded")

    def _monitor_loop(self):
        last_seen = 0
        last_down = 0
        last_pages = 0
        last_t = time.time()

        while not self.stop_event.is_set():
            time.sleep(3.0)
            now = time.time()
            dt = max(0.001, now - last_t)
            with self.stats_lock:
                d_cnt = self.discovered_count
                dl_cnt = self.downloaded_count
                q_cur = self.curated_queued_count
                seen_cnt = self.total_seen
                pages_cnt = self.pages_fetched
                remaining_tasks = self.page_task_queue.qsize()

            scan_rate = (seen_cnt - last_seen) / dt
            down_rate = (dl_cnt - last_down) / dt
            page_rate = (pages_cnt - last_pages) / dt
            last_seen = seen_cnt
            last_down = dl_cnt
            last_pages = pages_cnt
            last_t = now

            log(f"[SPEED] {scan_rate:,.1f} vids/s ({page_rate:.1f} pages/s) | {down_rate:,.1f} mosaics/s | "
                f"Scanned: {seen_cnt:,} | New Disc: {d_cnt:,} | Curated: {q_cur:,} | Downloaded: {dl_cnt:,} | Queue: {remaining_tasks:,}")

            if q_cur >= self.target_curated:
                log(f"Target of {self.target_curated:,} curated samples reached!")
                self.stop_event.set()
                break

    def _search_worker_loop(self):
        session = create_session(pool_size=16)

        while not self.stop_event.is_set():
            try:
                query, page, cat = self.page_task_queue.get(timeout=1.0)
            except queue.Empty:
                with self.active_lock:
                    if self.active_search_workers == 0 and self.page_task_queue.empty():
                        break
                continue

            with self.active_lock:
                self.active_search_workers += 1

            url = f"https://www.xvideos.com/?k={requests.utils.quote(query)}&p={page}"
            try:
                resp = session.get(url, timeout=8)
                if resp.status_code == 429:
                    # Rate limit burst: re-queue task and back off
                    self.page_task_queue.put((query, page, cat))
                    time.sleep(random.uniform(1.2, 1.8))
                    continue
                elif resp.status_code != 200:
                    continue

                matches = BLOCK_RE.findall(resp.text)
                page_seen = len(matches)
                page_disc = 0
                page_cur = 0

                # Dynamic pagination: if this page had full results and hasn't hit max_pages, queue next page
                if page_seen >= 15 and page < self.max_pages_per_query and not self.stop_event.is_set():
                    self.page_task_queue.put((query, page + 1, cat))

                for eid, m_url, rel_url, raw_title in matches:
                    with self.eid_lock:
                        if eid in self.known_eids:
                            continue
                        self.known_eids.add(eid)

                    c_title = clean_title(raw_title)
                    c_tags = extract_clothing(c_title)
                    p_tags = extract_position(c_title)
                    is_high_signal = bool(c_tags or p_tags) and len(c_title) >= 10 and len(c_title.split()) >= 3

                    page_disc += 1
                    full_url = f"https://www.xvideos.com{rel_url}" if rel_url.startswith("/") else rel_url

                    if is_high_signal:
                        self.download_queue.put((eid, m_url, full_url, c_title, c_tags, p_tags))
                        page_cur += 1
                    else:
                        self.db_writer.uncurated_queue.put((eid, full_url, c_title))

                with self.stats_lock:
                    self.pages_fetched += 1
                    self.total_seen += page_seen
                    self.discovered_count += page_disc
                    self.curated_queued_count += page_cur

                # Origin search pacing: ~1-1.5 requests/sec per worker to prevent 429 locks
                time.sleep(random.uniform(0.7, 0.9))

            except Exception as e:
                pass
            finally:
                with self.active_lock:
                    self.active_search_workers -= 1
                self.page_task_queue.task_done()

    def _download_worker_loop(self):
        session = create_session(pool_size=16)

        while not self.stop_event.is_set():
            try:
                eid, m_url, full_url, c_title, c_tags, p_tags = self.download_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            target_path = os.path.join(MOSAICS_DIR, f"{eid}.jpg")
            try:
                resp = session.get(m_url, timeout=10)
                if resp.status_code == 200 and len(resp.content) > 5000:
                    with open(target_path, "wb") as f:
                        f.write(resp.content)
                    self.db_writer.curated_queue.put((eid, full_url, c_title, target_path, c_tags, p_tags))
                    with self.stats_lock:
                        self.downloaded_count += 1
            except Exception:
                pass
            finally:
                self.download_queue.task_done()


HIGH_YIELD_ATTIRES = [
    "maid", "lingerie", "bikini", "stockings", "latex", "dress",
    "skirt", "schoolgirl", "nurse", "bunny", "apron", "leather",
    "thigh highs", "fishnets", "sheer", "monokini", "swimsuit"
]

HIGH_YIELD_POSTURES = [
    "prone bone", "mating press", "spooning", "standing split", "lotus position",
    "squatting", "doggystyle", "cowgirl", "missionary", "kneeling",
    "standing", "bent over", "on all fours", "spread legs", "reverse cowgirl"
]

BROAD_SCALING_TERMS = [
    "amateur homemade", "verified amateur", "real couple", "college couple",
    "POV compilation", "sensual couple", "romantic", "bedroom sex",
    "brunette babe", "blonde babe", "asian cutie", "latina babe",
    "japanese uncensored", "korean babe", "hardcore compilation"
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ultra-fast concurrent XVideos search harvester")
    parser.add_argument("--search-workers", type=int, default=2, help="Concurrent search scraping threads (paced for origin search)")
    parser.add_argument("--download-workers", type=int, default=32, help="Concurrent mosaic download threads (high-speed CDN77)")
    parser.add_argument("--max-pages", type=int, default=40, help="Max search pages per keyword query")
    parser.add_argument("--target", type=int, default=100000, help="Target new curated samples to harvest")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "composite", "deficit", "all", "postures", "clothing"],
                        help="Query mode: full (deficit + high-yield composite + broad scaling), composite, deficit, postures, clothing")
    args = parser.parse_args()

    from harvest_xvideos_targeted import get_deficit_ordered_queries

    log(f"Deriving prioritized search queries dynamically from Danbooru taxonomy ({TAXONOMY_JSON})...")
    candidates = get_deficit_ordered_queries(DB_PATH, TAXONOMY_JSON)

    if args.mode in ("composite", "full"):
        # High-yield composite pairs (Attire x Posture)
        composite_queries = []
        for p in HIGH_YIELD_POSTURES:
            for cl in HIGH_YIELD_ATTIRES:
                composite_queries.append((f"{cl} {p}", "composite", 0, 10000))

        if args.mode == "full":
            broad_queries = [(b, "broad", 0, 5000) for b in BROAD_SCALING_TERMS]
            candidates = candidates + composite_queries + broad_queries
            log(f"Loaded FULL matrix: {len(candidates)} queries (deficit + {len(composite_queries)} composite + {len(broad_queries)} broad).")
        else:
            candidates = candidates + composite_queries
            log(f"Loaded COMPOSITE matrix: {len(candidates)} queries.")

    elif args.mode == "postures":
        candidates = [c for c in candidates if c[1] == "posture"]
    elif args.mode == "clothing":
        candidates = [c for c in candidates if c[1] == "clothing"]

    log(f"Starting FastHarvester (workers: {args.search_workers} search / {args.download_workers} download, target={args.target:,}, max_pages={args.max_pages})...")

    harvester = FastHarvester(
        db_path=DB_PATH,
        search_workers=args.search_workers,
        download_workers=args.download_workers,
        max_pages_per_query=args.max_pages,
        target_curated=args.target
    )

    harvester.run(candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
