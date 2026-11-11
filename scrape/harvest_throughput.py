#!/usr/bin/env python3
"""High-Throughput Unfiltered Video & Mosaic Scraper.

Designed strictly for throughput and corpus diversity (no training filters):
  - Ingests all videos without Danbooru/posture/clothing curation filters.
  - Multi-threaded asynchronous mosaic streamer (32-64 workers) direct from CDN.
  - Dedicated Batch SQLite Writer to eliminate disk lock contention.
  - Auto-extracts canonical preview.mp4 hover URLs on-the-fly.
  - Immediately expands searchable catalog diversity.
"""

from __future__ import annotations

import argparse
import html
import os
import queue
import re
import signal
import sqlite3
import sys
import threading
import time
from typing import Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SCRAPE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRAPE_DIR)
DEFAULT_DB = os.path.join(SCRAPE_DIR, "thumbs.db")
DEFAULT_MOSAIC_DIR = os.path.join(PROJECT_DIR, "mosaics")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


def mosaic_url(frame_url: str) -> str:
    """Derive 3x3 contact sheet URL from any individual thumbnail frame URL."""
    return re.sub(r"[^/]+\.(?:jpg|jpeg|png|webp|avif)$", "mozaique_listing.jpg", frame_url or "")


def preview_mp4_url(asset_url: str) -> Optional[str]:
    """Derive canonical hover preview video URL from thumbnail URL."""
    if not asset_url:
        return None
    idx = asset_url.find("/xv_")
    if idx != -1:
        return asset_url[:idx] + "/preview.mp4"
    return None


def clean_text_basic(raw_title: Optional[str]) -> str:
    """Minimal text unescape and whitespace normalization (no filtering)."""
    if not raw_title:
        return ""
    text = html.unescape(raw_title)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_session(pool_size: int = 64) -> requests.Session:
    sess = requests.Session()
    sess.headers.update(HEADERS)
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=retries)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


# --------------------------------------------------------------------------- #
# Batch Database Writer
# --------------------------------------------------------------------------- #

class BatchDBWriter(threading.Thread):
    def __init__(self, db_path: str, batch_size: int = 150, flush_interval: float = 0.5):
        super().__init__(name="BatchDBWriter", daemon=True)
        self.db_path = db_path
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.queue: queue.Queue[Tuple] = queue.Queue()
        self.stop_event = threading.Event()
        self.total_written = 0

    def run(self):
        conn = sqlite3.connect(self.db_path, timeout=60, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -64000")  # 64MB cache

        # Ensure table for diverse search indexing exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS videos_diverse (
                eid TEXT PRIMARY KEY,
                title TEXT,
                asset_path TEXT,
                preview_url TEXT,
                downloaded REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_diverse_path ON videos_diverse (asset_path)")
        conn.commit()

        buffer = []
        last_flush = time.time()

        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.2)
                buffer.append(item)
            except queue.Empty:
                pass

            now = time.time()
            if len(buffer) >= self.batch_size or (buffer and now - last_flush >= self.flush_interval):
                try:
                    # Update both the raw crawler videos table and diverse search table in 1 transaction
                    cur = conn.cursor()
                    cur.executemany("""
                        INSERT OR REPLACE INTO videos_diverse (eid, title, asset_path, preview_url, downloaded)
                        VALUES (?, ?, ?, ?, ?)
                    """, buffer)

                    # Also reflect 'done' in the main videos tracking table
                    status_updates = [(b[2], b[4], b[0]) for b in buffer]
                    cur.executemany("""
                        UPDATE videos
                        SET asset_state = 'done', asset_path = ?, downloaded = ?
                        WHERE eid = ?
                    """, status_updates)

                    conn.commit()
                    self.total_written += len(buffer)
                except Exception as e:
                    print(f"[Writer Error] {e}", flush=True)
                finally:
                    buffer.clear()
                    last_flush = time.time()

        # Final flush on exit
        if buffer:
            try:
                cur = conn.cursor()
                cur.executemany("""
                    INSERT OR REPLACE INTO videos_diverse (eid, title, asset_path, preview_url, downloaded)
                    VALUES (?, ?, ?, ?, ?)
                """, buffer)
                status_updates = [(b[2], b[4], b[0]) for b in buffer]
                cur.executemany("""
                    UPDATE videos
                    SET asset_state = 'done', asset_path = ?, downloaded = ?
                    WHERE eid = ?
                """, status_updates)
                conn.commit()
                self.total_written += len(buffer)
            except Exception as e:
                print(f"[Writer Final Flush Error] {e}", flush=True)

        conn.close()


# --------------------------------------------------------------------------- #
# High-Throughput Unfiltered Streamer
# --------------------------------------------------------------------------- #

class HighThroughputScraper:
    def __init__(self, db_path: str, mosaic_dir: str, num_workers: int = 48, limit: Optional[int] = None):
        self.db_path = db_path
        self.mosaic_dir = mosaic_dir
        self.num_workers = num_workers
        self.limit = limit
        os.makedirs(mosaic_dir, exist_ok=True)

        self.db_writer = BatchDBWriter(db_path=self.db_path)
        self.task_queue: queue.Queue[Tuple[str, str, str]] = queue.Queue(maxsize=100000)
        self.stop_event = threading.Event()
        self.local_state = threading.local()

        self.stats_lock = threading.Lock()
        self.downloaded_count = 0
        self.cached_count = 0
        self.failed_count = 0
        self.total_bytes = 0
        self.t_start = time.time()

    def _get_session(self) -> requests.Session:
        s = getattr(self.local_state, "session", None)
        if s is None:
            s = create_session(pool_size=16)
            self.local_state.session = s
        return s

    def worker_loop(self):
        while not self.stop_event.is_set():
            try:
                item = self.task_queue.get(timeout=0.5)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue

            eid, title, asset_url = item
            target_path = os.path.join(self.mosaic_dir, f"{eid}.jpg")
            clean_title = clean_text_basic(title)
            preview_url = preview_mp4_url(asset_url)

            # Check if already present on disk
            if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
                self.db_writer.queue.put((eid, clean_title, target_path, preview_url, time.time()))
                with self.stats_lock:
                    self.cached_count += 1
                self.task_queue.task_done()
                continue

            url = mosaic_url(asset_url)
            tmp_path = f"{target_path}.{threading.get_ident()}.part"
            sess = self._get_session()
            success = False

            try:
                r = sess.get(url, timeout=15)
                if r.status_code == 200 and len(r.content) > 1000:
                    with open(tmp_path, "wb") as fh:
                        fh.write(r.content)
                    os.replace(tmp_path, target_path)
                    success = True
                    b_len = len(r.content)
                    self.db_writer.queue.put((eid, clean_title, target_path, preview_url, time.time()))
                    with self.stats_lock:
                        self.downloaded_count += 1
                        self.total_bytes += b_len
                else:
                    with self.stats_lock:
                        self.failed_count += 1
            except Exception:
                with self.stats_lock:
                    self.failed_count += 1
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                self.task_queue.task_done()

    def reporter_loop(self):
        while not self.stop_event.is_set():
            time.sleep(2.0)
            elapsed = max(0.1, time.time() - self.t_start)
            with self.stats_lock:
                dl = self.downloaded_count
                cached = self.cached_count
                fail = self.failed_count
                mb = self.total_bytes / (1024 * 1024)
            q_size = self.task_queue.qsize()
            written = self.db_writer.total_written
            rate = dl / elapsed
            mb_rate = mb / elapsed

            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"Done: {dl:,} ({rate:.1f} imgs/s, {mb_rate:.2f} MB/s) | "
                f"Cached: {cached:,} | Fail: {fail:,} | Queue: {q_size:,} | DB Committed: {written:,}",
                flush=True
            )

    def run(self):
        print(f"[*] Starting High-Throughput Unfiltered Scraper...")
        print(f"[*] Target Mosaic Directory: {self.mosaic_dir}")
        print(f"[*] Database: {self.db_path}")
        print(f"[*] Worker Threads: {self.num_workers}")

        self.db_writer.start()

        # Start workers
        workers = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self.worker_loop, name=f"MosaicWorker-{i}", daemon=True)
            t.start()
            workers.append(t)

        reporter = threading.Thread(target=self.reporter_loop, name="Reporter", daemon=True)
        reporter.start()

        # Stream candidate tasks from database in chunks
        conn = sqlite3.connect(self.db_path, timeout=60)
        cur = conn.cursor()

        query = """
            SELECT eid, title, asset_url
            FROM videos
            WHERE asset_url IS NOT NULL 
              AND asset_state != 'done'
        """
        if self.limit:
            query += f" LIMIT {self.limit}"

        cur.execute(query)

        pushed = 0
        while True:
            rows = cur.fetchmany(5000)
            if not rows:
                break
            for row in rows:
                self.task_queue.put(row)
                pushed += 1

        conn.close()
        print(f"[*] Pushed {pushed:,} candidate download tasks to queue. Streaming from CDN...")

        # Wait for queue completion
        self.task_queue.join()
        self.stop_event.set()
        self.db_writer.stop_event.set()
        self.db_writer.join()

        elapsed = time.time() - self.t_start
        print(f"\n[+] High-Throughput Scrape Complete!")
        print(f"    Total Downloaded : {self.downloaded_count:,}")
        print(f"    Total Disk Cached: {self.cached_count:,}")
        print(f"    Failed           : {self.failed_count:,}")
        print(f"    Total Volume     : {self.total_bytes / (1024*1024):.2f} MB")
        print(f"    Time Elapsed     : {elapsed:.1f} s ({self.downloaded_count / max(0.1, elapsed):.1f} imgs/s)")


def main():
    parser = argparse.ArgumentParser(description="High-Throughput Unfiltered XVideos Scraper")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite database")
    parser.add_argument("--mosaic-dir", default=DEFAULT_MOSAIC_DIR, help="Destination directory for mosaics")
    parser.add_argument("--workers", type=int, default=48, help="Parallel CDN download threads (default: 48)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of items to scrape (default: all)")
    args = parser.parse_args()

    scraper = HighThroughputScraper(
        db_path=args.db,
        mosaic_dir=args.mosaic_dir,
        num_workers=args.workers,
        limit=args.limit
    )

    def handle_sigint(sig, frame):
        print("\n[*] Stopping workers cleanly...")
        scraper.stop_event.set()
        scraper.db_writer.stop_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    scraper.run()


if __name__ == "__main__":
    main()
