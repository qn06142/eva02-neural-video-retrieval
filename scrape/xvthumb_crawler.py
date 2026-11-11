#!/usr/bin/env python3
"""xvideos contact-sheet crawler (parallel BFS, deduped by SQLite).

Each video page embeds a `var video_related=[...]` array holding ~40 recommended
videos with their ids, urls and frame urls.  Starting from a few seed pages the
crawler walks that graph breadth-first, records every unique video id in a
SQLite database the first time it is seen, and downloads exactly one CONTACT
SHEET per video.

The contact sheet (`mozaique_listing.jpg`, 1104x624) is a 3x3 grid of 9 frames
of 368x208, served from the same CDN path as the frame it replaces.  One
request therefore yields the whole bag of frames for a video, which is what the
MIL trainer (train_siglip_mil.py) expects: a video is a bag, its title is the
bag-level label, and no single frame is forced to match it.

Re-running resumes: already-known ids are never fetched or downloaded again,
and the BFS frontier is restored from the db.

Parallelism
-----------
  --fetchers  page-fetch worker threads (default 16)
  --workers   contact-sheet download worker threads (default 16)
  --rps       global cap on page requests / second (default 10).  This is the
              real throttle; --fetchers only hides network latency.

Usage
-----
    uv run --no-project --with requests python xvthumb_crawler.py --seeds seeds.txt
    uv run --no-project --with requests python xvthumb_crawler.py --max-pages 0
    uv run --no-project --with requests python xvthumb_crawler.py --report
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import random
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests

sys.path.append(SCRAPE_DIR)
from filter_curated_pairs import clean_title, extract_clothing, extract_position

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOSAIC_DIR = os.path.join(PROJECT_DIR, "mosaics")
SCRAPE_DIR = os.path.join(PROJECT_DIR, "scrape")
DEFAULT_DB = os.path.join(SCRAPE_DIR, "thumbs.db")
DEFAULT_SEEDS = os.path.join(SCRAPE_DIR, "seeds.txt")

BASE_URL = "https://www.xvideos.com"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

EID_RE = re.compile(r"/video\.([A-Za-z0-9]+)")


def mosaic_url(frame_url: str) -> str:
    """xvideos serves a 3x3 contact sheet next to every frame: swap the file name.

    .../<uuid>/<n>/xv_17_t.jpg  ->  .../<uuid>/<n>/mozaique_listing.jpg
    1104x624 = 9 frames of 368x208, i.e. one request yields a whole bag for MIL.
    """
    return re.sub(r"[^/]+\.(?:jpg|jpeg|png|webp|avif)$",
                  "mozaique_listing.jpg", frame_url or "")
JSONLD_RE = re.compile(
    r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', re.S
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    eid         TEXT PRIMARY KEY,
    url         TEXT,
    title       TEXT,
    asset_url   TEXT,
    page_state  TEXT NOT NULL DEFAULT 'queued',   -- queued | fetched | failed
    asset_state TEXT NOT NULL DEFAULT 'pending',  -- pending | done | failed
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


def extract_eid(url: str) -> Optional[str]:
    m = EID_RE.search(url or "")
    return m.group(1) if m else None


def clean_text(s) -> Optional[str]:
    if not isinstance(s, str) or not s.strip():
        return None
    return s.strip()


# --------------------------------------------------------------------------- #
# database
# --------------------------------------------------------------------------- #

class Database:
    """Thread-safe SQLite wrapper (autocommit, one connection, explicit lock)."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(
            path, check_same_thread=False, isolation_level=None, timeout=30
        )
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    def add_video(self, eid: str, url=None, title=None, asset_url=None) -> bool:
        """Insert the video if new; backfill missing fields otherwise.

        Returns True only when the row was newly inserted.
        """
        now = time.time()
        with self.lock:
            row = self.conn.execute(
                "SELECT 1 FROM videos WHERE eid = ?", (eid,)
            ).fetchone()
            if row:
                self.conn.execute(
                    """UPDATE videos SET
                           url       = COALESCE(?, url),
                           title     = COALESCE(?, title),
                           asset_url = COALESCE(?, asset_url)
                       WHERE eid = ?""",
                    (url, clean_text(title), asset_url, eid),
                )
                return False
            self.conn.execute(
                """INSERT INTO videos (eid, url, title, asset_url, discovered)
                   VALUES (?, ?, ?, ?, ?)""",
                (eid, url, clean_text(title), asset_url, now),
            )
            return True

    def set_page_state(self, eid: str, state: str, error: Optional[str] = None):
        with self.lock:
            self.conn.execute(
                "UPDATE videos SET page_state = ?, error = ?, "
                "fetched = CASE WHEN ? = 'fetched' THEN ? ELSE fetched END "
                "WHERE eid = ?",
                (state, error, state, time.time(), eid),
            )

    def set_asset_state(
        self,
        eid: str,
        state: str,
        asset_path: Optional[str] = None,
        error: Optional[str] = None,
    ):
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
            return self.conn.execute(
                "SELECT * FROM videos WHERE eid = ?", (eid,)
            ).fetchone()

    def pending_assets(self, limit: Optional[int] = None) -> list:
        with self.lock:
            sql = ("SELECT eid FROM videos "
                   "WHERE asset_state = 'pending' AND asset_url IS NOT NULL")
            params: tuple = ()
            if limit:
                sql += " LIMIT ?"
                params = (limit,)
            return self.conn.execute(sql, params).fetchall()

    def queued_pages(self, limit: Optional[int] = None) -> list:
        with self.lock:
            sql = "SELECT eid, url FROM videos WHERE page_state = 'queued'"
            params: tuple = ()
            if limit:
                sql += " LIMIT ?"
                params = (limit,)
            return self.conn.execute(sql, params).fetchall()

    def counts(self) -> dict:
        with self.lock:
            row = self.conn.execute(
                """SELECT
                     COUNT(*)                                              AS total,
                     SUM(page_state = 'fetched')                            AS fetched,
                     SUM(page_state = 'failed')                             AS page_failed,
                     SUM(page_state = 'queued')                             AS queued,
                     SUM(asset_state = 'done')                              AS done,
                     SUM(asset_state = 'pending' AND asset_url IS NOT NULL) AS pending,
                     SUM(asset_state = 'failed')                            AS failed
                   FROM videos"""
            ).fetchone()
            return {k: (row[k] or 0) for k in row.keys()}


# --------------------------------------------------------------------------- #
# rate limiter
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Serialises request start times so aggregate rate stays <= rps."""

    def __init__(self, rps: float):
        self.interval = (1.0 / rps) if (rps and rps > 0) else 0.0
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        delay = start - now
        if delay > 0:
            time.sleep(delay)


# --------------------------------------------------------------------------- #
# crawler
# --------------------------------------------------------------------------- #

class Crawler:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.db = Database(args.db)
        self._local = threading.local()
        self.rate_limiter = RateLimiter(args.rps)

        self.stop_event = threading.Event()
        self.interrupted = False
        self.page_queue: "queue.Queue[str]" = queue.Queue()
        self.sched_lock = threading.Lock()
        self.scheduled: set = set()
        self.pending = 0

        self.stats_lock = threading.Lock()
        self.pages_fetched = 0
        self.assets_done = 0
        self.limit_reached = False

        self.download_pool: Optional[ThreadPoolExecutor] = None

    # -- http -------------------------------------------------------------- #

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update(HEADERS)
            self._local.session = s
        return s

    def fetch_page(self, url: str) -> Optional[str]:
        """GET a video page with retries/backoff. None = give up."""
        sess = self._session()
        backoff = 2.0
        for attempt in range(1, self.args.retries + 1):
            self.rate_limiter.wait()
            try:
                r = sess.get(url, timeout=30, allow_redirects=True)
            except requests.RequestException as exc:
                log(f"  ! {url} -> {type(exc).__name__}: {exc} (attempt {attempt})")
                time.sleep(backoff + random.uniform(0, 1))
                backoff = min(backoff * 2, 60)
                continue

            if r.status_code == 200:
                r.encoding = "utf-8"
                return r.text
            if r.status_code in (403, 429, 503):
                wait = backoff + random.uniform(0, 2)
                log(f"  ! HTTP {r.status_code} for {url}; backing off {wait:.1f}s")
                time.sleep(wait)
                backoff = min(backoff * 2, 60)
                continue
            if r.status_code == 404:
                log(f"  - 404 for {url}")
                return None
            log(f"  ! HTTP {r.status_code} for {url} (attempt {attempt})")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
        return None

    # -- parsing ----------------------------------------------------------- #

    def parse_related(self, html: str) -> list:
        """Extract the `video_related` recommendation array from a page."""
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
            out.append(
                {
                    "eid": eid,
                    "url": u if u.startswith("http") else BASE_URL + u,
                    "title": item.get("tf") or item.get("t") or "",
                    "asset_url": thumb,
                }
            )
        return out

    def parse_main(self, html: str, url: str) -> dict:
        """Extract the page's own video (id, title, thumbnail)."""
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
            m = re.search(r"setThumbSlideBig\('([^']+)'", html) or re.search(
                r"setThumbSlide\('([^']+)'", html
            )
            if m:
                thumb = m.group(1)

        return {"eid": eid, "url": url, "title": title, "asset_url": thumb}

    # -- thumb download ---------------------------------------------------- #

    def download_asset(self, eid: str) -> None:
        row = self.db.get_video(eid)
        if row is None or row["asset_state"] == "done":
            return
        asset_url = row["asset_url"]
        if not asset_url:
            self.db.set_asset_state(eid, "failed", error="no thumb url")
            return
        url = mosaic_url(asset_url)

        path = os.path.join(MOSAIC_DIR, f"{eid}.jpg")
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            self.db.set_asset_state(eid, "done", asset_path=path)
            return

        tmp = f"{path}.{threading.get_ident()}.part"
        try:
            sess = self._session()
            r = sess.get(url, timeout=30, headers={"Referer": BASE_URL + "/"})
            body = r.content
            if r.status_code == 200 and len(body) > 1000:
                with open(tmp, "wb") as fh:
                    fh.write(body)
                os.replace(tmp, path)
                self.db.set_asset_state(eid, "done", asset_path=path)
                with self.stats_lock:
                    self.assets_done += 1
            else:
                self.db.set_asset_state(
                    eid, "failed", error=f"http {r.status_code} len {len(body)}"
                )
        except Exception as exc:  # noqa: BLE001 - never let a worker die
            self.db.set_asset_state(eid, "failed", error=str(exc)[:200])
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    # -- seeds ------------------------------------------------------------- #

    def read_seeds(self) -> list:
        urls: list = []
        if self.args.seeds and os.path.exists(self.args.seeds):
            with open(self.args.seeds, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        urls.append(line)
        urls.extend(self.args.seed or [])

        out, seen = [], set()
        for u in urls:
            if not u.startswith("http"):
                u = BASE_URL + u
            eid = extract_eid(u)
            if not eid or eid in seen:
                continue
            seen.add(eid)
            out.append({"eid": eid, "url": u})
        return out

    # -- scheduling -------------------------------------------------------- #

    def _schedule(self, eid: str) -> bool:
        """Enqueue a page fetch exactly once per run; skip already-fetched pages."""
        with self.sched_lock:
            if eid in self.scheduled:
                return False
            row = self.db.get_video(eid)
            if row is not None and row["page_state"] == "fetched":
                return False
            self.scheduled.add(eid)
            self.pending += 1
            self.page_queue.put(eid)
            return True

    def _limit_hit(self) -> bool:
        with self.stats_lock:
            if self.limit_reached:
                return True
            if self.args.max_pages and self.pages_fetched >= self.args.max_pages:
                self.limit_reached = True
            elif self.args.max_thumbs and self.assets_done >= self.args.max_thumbs:
                self.limit_reached = True
            return self.limit_reached

    def _submit_download(self, eid: str) -> None:
        if self.download_pool is None:
            return
        row = self.db.get_video(eid)
        if row is None or row["asset_state"] != "pending":
            return
        self.download_pool.submit(self.download_asset, eid)

    # -- fetch workers ----------------------------------------------------- #

    def _process_page(self, eid: str) -> None:
        row = self.db.get_video(eid)
        url = row["url"] if (row and row["url"]) else f"{BASE_URL}/video.{eid}/"

        html = self.fetch_page(url)
        if html is None:
            self.db.set_page_state(eid, "failed", error="fetch failed")
            return
        if len(html) < 5000 or "video_related" not in html:
            log(f"  ! suspicious page for {eid} ({len(html)} bytes)")
            self.db.set_page_state(eid, "failed", error="suspicious page")
            return

        self.db.set_page_state(eid, "fetched")
        with self.stats_lock:
            self.pages_fetched += 1
            pages = self.pages_fetched

        main = self.parse_main(html, url)
        if main["eid"]:
            self.db.add_video(
                main["eid"], url=url, title=main["title"],
                asset_url=main["asset_url"],
            )
            self._submit_download(main["eid"])

        related = self.parse_related(html)
        new = 0
        for item in related:
            if self.db.add_video(
                item["eid"], url=item["url"], title=item["title"],
                asset_url=item["asset_url"],
            ):
                new += 1
                if not self._limit_hit():
                    self._schedule(item["eid"])
            self._submit_download(item["eid"])

        if pages % 50 == 0:
            with self.sched_lock:
                q = self.page_queue.qsize()
                pending = self.pending
            with self.stats_lock:
                done = self.assets_done
            log(f"pages={pages} queue={q} in_flight={pending} new={new} thumbs={done}")

        if self.args.delay:
            time.sleep(self.args.delay + random.uniform(0, self.args.jitter))

    def _fetch_worker(self) -> None:
        while not self.stop_event.is_set():
            if self._limit_hit():
                self.stop_event.set()
                return
            try:
                eid = self.page_queue.get_nowait()
            except queue.Empty:
                with self.sched_lock:
                    if self.pending == 0:
                        self.stop_event.set()
                        return
                time.sleep(0.05)
                continue
            try:
                self._process_page(eid)
            except Exception as exc:  # noqa: BLE001
                log(f"  ! fetcher error on {eid}: {type(exc).__name__}: {exc}")
            finally:
                with self.sched_lock:
                    self.pending -= 1
                    if self.pending == 0:
                        self.stop_event.set()

    # -- orchestration ----------------------------------------------------- #

    def run(self) -> None:
        os.makedirs(MOSAIC_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.args.db)), exist_ok=True)

        seeds = self.read_seeds()
        if self.args.report:
            self.report(seeds=seeds)
            return

        log(f"DB: {self.args.db}")
        log(f"contact sheets -> {MOSAIC_DIR}")
        log(f"seeds: {[s['eid'] for s in seeds]}")

        self.resume_pending()
        if self.args.max_pages <= 0:
            self.report()
            return

        self.download_pool = ThreadPoolExecutor(
            max_workers=self.args.workers, thread_name_prefix="dl"
        )

        # restore the BFS frontier from the db, then the seeds
        limit = self.args.max_pages * self.args.fetchers
        frontier = self.db.queued_pages(limit=limit)
        for row in frontier:
            self._schedule(row["eid"])
        for s in seeds:
            self.db.add_video(s["eid"], url=s["url"])
            self._schedule(s["eid"])
        log(f"frontier restored: {self.page_queue.qsize()} pages queued")

        fetchers = [
            threading.Thread(target=self._fetch_worker, name=f"fetch-{i}", daemon=True)
            for i in range(self.args.fetchers)
        ]
        for t in fetchers:
            t.start()
        # timed joins: a bare join() can swallow Ctrl-C on Windows until the
        # thread happens to finish, so poll and let the main thread respond
        for t in fetchers:
            while t.is_alive():
                t.join(timeout=0.5)

        if self.stop_event.is_set():
            log("interrupted - cancelling queued downloads")
            self.shutdown()
        else:
            log("crawl finished, waiting for contact sheet downloads...")
            if self.download_pool is not None:
                self.download_pool.shutdown(wait=True)
        self.report()

    def shutdown(self) -> None:
        """Stop accepting work and drop anything still queued."""
        self.stop_event.set()
        if self.download_pool is not None:
            self.download_pool.shutdown(wait=False, cancel_futures=True)
            self.download_pool = None

    def resume_pending(self) -> None:
        rows = self.db.pending_assets(limit=self.args.max_thumbs or None)
        if not rows:
            return
        log(f"resuming {len(rows)} pending thumbnail downloads")
        with ThreadPoolExecutor(
            max_workers=self.args.workers, thread_name_prefix="resume"
        ) as pool:
            for r in rows:
                pool.submit(self.download_asset, r["eid"])

    def report(self, seeds: Optional[list] = None) -> None:
        c = self.db.counts()
        print()
        print("---- crawl report ----")
        print(f"  videos discovered : {c['total']}")
        print(f"  pages fetched     : {c['fetched']}")
        print(f"  pages queued      : {c['queued']}")
        print(f"  pages failed      : {c['page_failed']}")
        print(f"  sheets done       : {c['done']}")
        print(f"  sheets pending    : {c['pending']}")
        print(f"  sheets failed     : {c['failed']}")
        if seeds:
            print(f"  seed ids          : {[s['eid'] for s in seeds]}")
        files = 0
        if os.path.isdir(MOSAIC_DIR):
            files = sum(
                1 for f in os.listdir(MOSAIC_DIR)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            )
        print(f"  sheets on disk    : {files}")
        print("----------------------")


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crawl xvideos recommendations and download thumbnails (deduped)."
    )
    p.add_argument("--seeds", default=DEFAULT_SEEDS,
                   help=f"file with seed video urls (default: {DEFAULT_SEEDS})")
    p.add_argument("--seed", action="append", default=[],
                   help="extra seed url (repeatable)")
    p.add_argument("--db", default=DEFAULT_DB, help=f"sqlite db path (default: {DEFAULT_DB})")
    p.add_argument("--max-pages", type=int, default=250,
                   help="max video pages to fetch (0 = no crawling, default: 250)")
    p.add_argument("--max-thumbs", type=int, default=5000,
                   help="stop crawling after this many new contact sheets "
                        "(0 = unlimited, default: 5000)")
    p.add_argument("--fetchers", type=int, default=16,
                   help="page-fetch worker threads (default: 16)")
    p.add_argument("--workers", type=int, default=16,
                   help="thumbnail-download worker threads (default: 16)")
    p.add_argument("--rps", type=float, default=10.0,
                   help="max page requests per second (default: 10; 0 = unlimited)")
    p.add_argument("--delay", type=float, default=0.0,
                   help="extra sleep after each page fetch (default: 0)")
    p.add_argument("--jitter", type=float, default=0.2,
                   help="random extra delay after each page fetch (default: 0.2)")
    p.add_argument("--retries", type=int, default=4,
                   help="page fetch retries (default: 4)")
    p.add_argument("--report", action="store_true",
                   help="print DB statistics and exit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    crawler = Crawler(args)

    def _sigint(signum, frame):
        crawler.interrupted = True
        if not crawler.stop_event.is_set():
            log("ctrl-c: stopping after the current batch (progress is saved)")
        crawler.stop_event.set()

    try:
        signal.signal(signal.SIGINT, _sigint)
    except (ValueError, OSError):
        pass  # not the main thread, or platform without SIGINT

    try:
        crawler.run()
    except KeyboardInterrupt:
        log("interrupted - progress is saved in the db")
        crawler.shutdown()
        crawler.report()
        return 130
    return 130 if crawler.interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
