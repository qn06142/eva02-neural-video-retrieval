#!/usr/bin/env python3
"""High-performance curated BFS crawler for XVideos recommendation graph.

Features:
  - Graph BFS Traversal: Crawls video watch pages (`video_related` arrays, ~40 recommended videos per page).
  - Completely Bypasses Search Rate Limits: Watch pages do not have search burst rate limits.
  - Live In-Memory Curation Filter: Evaluates titles against Danbooru physical postures and attire.
  - Selective Mosaic Streaming: ONLY downloads mosaics for verified curated videos; uncurated
    videos are marked 'skipped_uncurated' in SQLite to protect disk space.
  - Live Curated Table Sync: Downloaded mosaics are instantly written to `videos_curated`.
  - Rate-Limited Origin Pacing: Paces watch page fetches smoothly with Token Bucket (--rps 10.0).
  - High Concurrency CDN Streamer: 32 parallel threads download 3x3 mosaics from CDN77.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import queue
import random
import re
import signal
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

SCRAPE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRAPE_DIR)
MOSAIC_DIR = os.path.join(PROJECT_DIR, "mosaics")
DEFAULT_DB = os.path.join(SCRAPE_DIR, "thumbs.db")
DEFAULT_SEEDS = os.path.join(SCRAPE_DIR, "seeds.txt")

sys.path.append(SCRAPE_DIR)
from filter_curated_pairs import clean_title, extract_clothing, extract_position

BASE_URL = "https://www.xvideos.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

EID_RE = re.compile(r"/video\.([A-Za-z0-9]+)")
JSONLD_RE = re.compile(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', re.S)

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    eid         TEXT PRIMARY KEY,
    url         TEXT,
    title       TEXT,
    asset_url   TEXT,
    page_state  TEXT NOT NULL DEFAULT 'queued',
    asset_state TEXT NOT NULL DEFAULT 'pending',
    asset_path  TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    discovered  REAL NOT NULL,
    fetched     REAL,
    downloaded  REAL
);
CREATE INDEX IF NOT EXISTS idx_videos_page_state  ON videos(page_state);
CREATE INDEX IF NOT EXISTS idx_videos_asset_state ON videos(asset_state);
"""

_LOG_LOCK = threading.Lock()


def log(msg: str) -> None:
    with _LOG_LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mosaic_url(frame_url: str) -> str:
    return re.sub(r"[^/]+\.(?:jpg|jpeg|png|webp|avif)$", "mozaique_listing.jpg", frame_url or "")


def extract_eid(url: str) -> Optional[str]:
    m = EID_RE.search(url or "")
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Database Wrapper
# --------------------------------------------------------------------------- #

class Database:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=60)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)
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

    def add_video(self, eid: str, url: str = None, title: str = None,
                  asset_url: str = None, is_curated: bool = False) -> bool:
        now = time.time()
        initial_asset_state = "pending" if is_curated else "skipped_uncurated"
        c_t = title.strip() if (isinstance(title, str) and title.strip()) else None

        with self.lock:
            row = self.conn.execute("SELECT asset_state FROM videos WHERE eid = ?", (eid,)).fetchone()
            if row:
                upgrade = ""
                if is_curated and row[0] == "skipped_uncurated":
                    upgrade = ", asset_state = 'pending'"
                self.conn.execute(
                    f"""UPDATE videos SET
                           url       = COALESCE(?, url),
                           title     = COALESCE(?, title),
                           asset_url = COALESCE(?, asset_url)
                           {upgrade}
                       WHERE eid = ?""",
                    (url, c_t, asset_url, eid),
                )
                return False

            self.conn.execute(
                """INSERT INTO videos (eid, url, title, asset_url, asset_state, discovered)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (eid, url, c_t, asset_url, initial_asset_state, now),
            )
            return True

    def mark_curated_done(self, eid: str, asset_path: str, clean_t: str, c_tags: list, p_tags: list):
        now = time.time()
        has_both = int(bool(c_tags and p_tags))
        with self.lock:
            self.conn.execute(
                """UPDATE videos
                   SET asset_state = 'done', asset_path = ?, downloaded = ?
                   WHERE eid = ?""",
                (asset_path, now, eid),
            )
            self.conn.execute(
                """INSERT OR REPLACE INTO videos_curated (eid, clean_title, clothing_tags, position_tags, has_both, asset_path)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (eid, clean_t, json.dumps(c_tags), json.dumps(p_tags), has_both, asset_path),
            )

    def set_page_state(self, eid: str, state: str, error: Optional[str] = None):
        with self.lock:
            self.conn.execute(
                "UPDATE videos SET page_state = ?, error = ?, "
                "fetched = CASE WHEN ? = 'fetched' THEN ? ELSE fetched END "
                "WHERE eid = ?",
                (state, error, state, time.time(), eid),
            )

    def set_asset_state(self, eid: str, state: str, asset_path: Optional[str] = None, error: Optional[str] = None):
        with self.lock:
            self.conn.execute(
                """UPDATE videos
                   SET asset_state = ?, asset_path = ?, error = ?,
                       attempts = attempts + 1,
                       downloaded = CASE WHEN ? = 'done' THEN ? ELSE downloaded END
                   WHERE eid = ?""",
                (state, asset_path, error, state, time.time(), eid),
            )

    def get_video(self, eid: str) -> Optional[sqlite3.Row]:
        with self.lock:
            return self.conn.execute("SELECT * FROM videos WHERE eid = ?", (eid,)).fetchone()

    def pending_frontier(self, limit: int = 50000) -> List[str]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT eid FROM videos WHERE page_state = 'queued' LIMIT ?", (limit,)
            ).fetchall()
            return [r[0] for r in rows]

    def pending_assets(self, limit: int = 10000) -> List[str]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT eid FROM videos WHERE asset_state = 'pending' AND asset_url IS NOT NULL LIMIT ?", (limit,)
            ).fetchall()
            return [r[0] for r in rows]


# --------------------------------------------------------------------------- #
# Rate Limiter
# --------------------------------------------------------------------------- #

class RateLimiter:
    def __init__(self, rps: float):
        self.rps = max(0.1, rps)
        self.interval = 1.0 / self.rps
        self.last = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            gap = now - self.last
            if gap < self.interval:
                time.sleep(self.interval - gap)
            self.last = time.time()


# --------------------------------------------------------------------------- #
# Curated BFS Crawler
# --------------------------------------------------------------------------- #

class CuratedBFSCrawler:
    def __init__(self, args):
        self.args = args
        self.db = Database(args.db)
        self.rate_limiter = RateLimiter(args.rps)
        self.page_queue: queue.Queue[str] = queue.Queue(maxsize=100000)
        self.scheduled: Set[str] = set()
        self.seen_eids: Set[str] = set()
        self.sched_lock = threading.Lock()
        self.pending = 0
        self.stop_event = threading.Event()
        self._local = threading.local()

        self.stats_lock = threading.Lock()
        self.pages_fetched = 0
        self.discovered_total = 0
        self.curated_discovered = 0
        self.assets_done = 0
        self.t_start = time.time()

        self.download_pool = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="Downloader")

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update(HEADERS)
            retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
            adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=retries)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            self._local.session = s
        return s

    def fetch_page(self, url: str) -> Optional[str]:
        sess = self._session()
        for attempt in range(1, self.args.retries + 1):
            self.rate_limiter.wait()
            try:
                r = sess.get(url, timeout=20, allow_redirects=True)
                if r.status_code == 200:
                    r.encoding = "utf-8"
                    return r.text
                if r.status_code in (403, 429, 503):
                    time.sleep(2.0 + random.uniform(0, 1))
                    continue
                if r.status_code == 404:
                    return None
            except Exception:
                time.sleep(1.0)
        return None

    def parse_related(self, html: str) -> list:
        idx = html.find("video_related")
        if idx == -1:
            return []
        eq = html.find("=", idx)
        if eq == -1:
            return []
        start = eq + 1
        while start < len(html) and html[start] in " \t\r\n":
            start += 1
        if start >= len(html) or html[start] != "[":
            return []
        try:
            data, _ = json.JSONDecoder().raw_decode(html[start:])
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []

        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            u = item.get("u") or ""
            eid = extract_eid(u)
            if not eid:
                continue
            thumb = None
            for key in ("i", "il", "if", "ip"):
                v = item.get(key)
                if isinstance(v, str) and v.startswith("http"):
                    thumb = v
                    break
            out.append({
                "eid": eid,
                "url": u if u.startswith("http") else BASE_URL + u,
                "title": item.get("tf") or item.get("t") or "",
                "asset_url": thumb,
            })
        return out

    def parse_main(self, html: str, url: str) -> dict:
        eid = extract_eid(url)
        title = None
        thumb = None

        m = re.search(r'<meta property="og:title" content="([^"]*)"', html)
        if m:
            title = m.group(1)

        for m in JSONLD_RE.finditer(html):
            try:
                ld = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            tu = ld.get("thumbnailUrl")
            if isinstance(tu, list) and tu and isinstance(tu[0], str):
                thumb = tu[0]
            elif isinstance(tu, str):
                thumb = tu
            if thumb:
                break

        if not thumb:
            m = re.search(r"setThumbSlideBig\('([^']+)'", html) or re.search(r"setThumbSlide\('([^']+)'", html)
            if m:
                thumb = m.group(1)

        return {"eid": eid, "url": url, "title": title, "asset_url": thumb}

    def download_asset(self, eid: str) -> None:
        row = self.db.get_video(eid)
        if row is None or row["asset_state"] == "done":
            return
        raw_title = row["title"] or ""
        c_title = clean_title(raw_title)
        c_tags = extract_clothing(c_title)
        p_tags = extract_position(c_title)
        if not c_tags and not p_tags:
            self.db.set_asset_state(eid, "skipped_uncurated", error="no_posture_attire_signal")
            return

        asset_url = row["asset_url"]
        if not asset_url:
            self.db.set_asset_state(eid, "failed", error="no thumb url")
            return

        url = mosaic_url(asset_url)
        path = os.path.join(MOSAIC_DIR, f"{eid}.jpg")
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            self.db.mark_curated_done(eid, path, c_title, c_tags, p_tags)
            return

        tmp = f"{path}.{threading.get_ident()}.part"
        try:
            sess = self._session()
            r = sess.get(url, timeout=20, headers={"Referer": BASE_URL + "/"})
            if r.status_code == 200 and len(r.content) > 1000:
                with open(tmp, "wb") as fh:
                    fh.write(r.content)
                os.replace(tmp, path)
                self.db.mark_curated_done(eid, path, c_title, c_tags, p_tags)
                with self.stats_lock:
                    self.assets_done += 1
            else:
                self.db.set_asset_state(eid, "failed", error=f"http {r.status_code}")
        except Exception as exc:
            self.db.set_asset_state(eid, "failed", error=str(exc)[:200])
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _schedule(self, eid: str) -> bool:
        with self.sched_lock:
            if eid in self.scheduled:
                return False
            try:
                self.page_queue.put_nowait(eid)
                self.scheduled.add(eid)
                self.pending += 1
                return True
            except queue.Full:
                return False

    def _submit_download(self, eid: str) -> None:
        if not self.stop_event.is_set():
            try:
                self.download_pool.submit(self.download_asset, eid)
            except RuntimeError:
                pass

    def _process_page(self, eid: str) -> None:
        row = self.db.get_video(eid)
        url = row["url"] if (row and row["url"]) else f"{BASE_URL}/video.{eid}/"

        html = self.fetch_page(url)
        if html is None:
            self.db.set_page_state(eid, "failed", error="fetch failed")
            return
        if len(html) < 5000 or "video_related" not in html:
            self.db.set_page_state(eid, "failed", error="suspicious page")
            return

        self.db.set_page_state(eid, "fetched")
        with self.stats_lock:
            self.pages_fetched += 1

        main = self.parse_main(html, url)
        if main["eid"]:
            c_title = clean_title(main["title"] or "")
            c_tags = extract_clothing(c_title)
            p_tags = extract_position(c_title)
            is_cur = bool(c_tags or p_tags) and len(c_title) >= 10 and len(c_title.split()) >= 3
            self.db.add_video(main["eid"], url=url, title=main["title"],
                              asset_url=main["asset_url"], is_curated=is_cur)
            if is_cur:
                self._submit_download(main["eid"])

        related = self.parse_related(html)
        new_cnt = 0
        cur_cnt = 0
        for item in related:
            eid = item["eid"]
            c_title = clean_title(item["title"] or "")
            c_tags = extract_clothing(c_title)
            p_tags = extract_position(c_title)
            is_cur = bool(c_tags or p_tags) and len(c_title) >= 10 and len(c_title.split()) >= 3

            # Fast path: already known uncurated video in memory -> skip DB overhead completely
            if eid in self.seen_eids and not is_cur:
                continue
            self.seen_eids.add(eid)

            if self.db.add_video(eid, url=item["url"], title=item["title"],
                                  asset_url=item["asset_url"], is_curated=is_cur):
                new_cnt += 1
                if is_cur:
                    cur_cnt += 1
                self._schedule(eid)

            if is_cur:
                self._submit_download(eid)

        with self.stats_lock:
            self.discovered_total += new_cnt
            self.curated_discovered += cur_cnt

    def _fetch_worker(self) -> None:
        while not self.stop_event.is_set():
            if self.args.max_thumbs and self.assets_done >= self.args.max_thumbs:
                self.stop_event.set()
                break
            try:
                eid = self.page_queue.get(timeout=1.0)
            except queue.Empty:
                if self.pending == 0:
                    break
                continue

            try:
                self._process_page(eid)
            finally:
                with self.sched_lock:
                    self.pending -= 1
                self.page_queue.task_done()

    def _monitor_loop(self):
        last_pages = 0
        last_down = 0
        last_disc = 0
        last_t = time.time()

        while not self.stop_event.is_set():
            time.sleep(5.0)
            now = time.time()
            dt = max(0.001, now - last_t)
            with self.stats_lock:
                pages = self.pages_fetched
                done = self.assets_done
                disc = self.discovered_total
                cur = self.curated_discovered

            q_size = self.page_queue.qsize()
            page_rate = (pages - last_pages) / dt
            down_rate = (done - last_down) / dt
            disc_rate = (disc - last_disc) / dt
            last_pages = pages
            last_down = done
            last_disc = disc
            last_t = now

            log(f"[SPRAY BFS] {page_rate:.1f} pages/s ({disc_rate:,.1f} vids/s) | {down_rate:.1f} curated img/s | "
                f"Frontier Queue: {q_size:,} | Discovered: {disc:,} | Curated Mosaics: {done:,} / {self.args.max_thumbs:,}")

            if self.args.max_thumbs and done >= self.args.max_thumbs:
                log(f"Reached target of {self.args.max_thumbs:,} curated mosaics!")
                self.stop_event.set()
                break

    def run(self):
        os.makedirs(MOSAIC_DIR, exist_ok=True)

        # Start Monitor Thread
        monitor = threading.Thread(target=self._monitor_loop, name="Monitor", daemon=True)
        monitor.start()

        # Start Fetcher Workers
        fetch_threads = []
        log(f"Launching BFS Spray Crawler: {self.args.fetchers} fetcher threads (--rps {self.args.rps}) | {self.args.workers} download threads...")
        for i in range(self.args.fetchers):
            t = threading.Thread(target=self._fetch_worker, name=f"Fetcher-{i}", daemon=True)
            t.start()
            fetch_threads.append(t)

        log("Bootstrapping BFS Frontier from thumbs.db...")
        queued_eids = self.db.pending_frontier(limit=10000)
        log(f"Loaded {len(queued_eids):,} queued videos from database frontier.")
        for eid in queued_eids:
            self._schedule(eid)

        # Also drain any existing pending curated assets
        existing_pending = self.db.pending_assets(limit=5000)
        if existing_pending:
            log(f"Submitting {len(existing_pending):,} existing pending curated assets to download pool...")
            for eid in existing_pending:
                self._submit_download(eid)

        try:
            while not self.stop_event.is_set():
                time.sleep(1.0)
                if self.page_queue.qsize() < 2000:
                    more = self.db.pending_frontier(limit=5000)
                    if more:
                        for eid in more:
                            self._schedule(eid)
        except KeyboardInterrupt:
            log("Interrupted by user. Shutting down gracefully...")
            self.stop_event.set()

        self.stop_event.set()
        for t in fetch_threads:
            t.join(timeout=2.0)
        log("Waiting for active mosaic downloads to complete...")
        self.download_pool.shutdown(wait=True)
        elapsed = time.time() - self.t_start
        log(f"BFS Crawler Finished in {elapsed:.1f}s! Total Curated Mosaics: {self.assets_done:,}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Spray BFS recommendation crawler with live curation filter")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite db path")
    parser.add_argument("--fetchers", type=int, default=16, help="Page fetch worker threads (default: 16)")
    parser.add_argument("--workers", type=int, default=32, help="Contact sheet download worker threads (default: 32)")
    parser.add_argument("--rps", type=float, default=12.0, help="Max page requests per second (default: 12.0)")
    parser.add_argument("--max-thumbs", type=int, default=100000, help="Target curated mosaics to download (default: 100000)")
    parser.add_argument("--retries", type=int, default=3, help="Page fetch retries")
    args = parser.parse_args()

    crawler = CuratedBFSCrawler(args)
    crawler.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
