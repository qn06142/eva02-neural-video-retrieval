"""Offline tests verifying scrapers process filter-rejected (`skipped_uncurated`) links."""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scrape"))

import crawl_xvideos_curated_bfs as bfs_mod
import harvest_throughput as ht_mod
import xvthumb_crawler as xvt_mod


class TestRejectedLinkScraping(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_rejected_")
        self.db_path = os.path.join(self.tmp_dir, "thumbs.db")
        self.mosaic_dir = os.path.join(self.tmp_dir, "mosaics")
        os.makedirs(self.mosaic_dir, exist_ok=True)

        db = bfs_mod.Database(self.db_path)
        conn = db.conn
        # Seed representative rows across states
        rows = [
            ("eid_pend", "https://example.com/v/eid_pend", "generic clip title", "https://cdn.example/1/xv_1_t.jpg", "fetched", "pending", 0),
            ("eid_rej1", "https://example.com/v/eid_rej1", "random uncurated vlog", "https://cdn.example/2/xv_2_t.jpg", "fetched", "skipped_uncurated", 0),
            ("eid_rej2", "https://example.com/v/eid_rej2", "another uncurated video", "https://cdn.example/3/xv_3_t.jpg", "fetched", "skipped_uncurated", 0),
            ("eid_purg", "https://example.com/v/eid_purg", "purged item", "https://cdn.example/4/xv_4_t.jpg", "fetched", "purged", 0),
            ("eid_done", "https://example.com/v/eid_done", "already done", "https://cdn.example/5/xv_5_t.jpg", "fetched", "done", 1),
            ("eid_dead", "https://example.com/v/eid_dead", "too many attempts", "https://cdn.example/6/xv_6_t.jpg", "fetched", "failed", 3),
            ("eid_nourl", "https://example.com/v/eid_nourl", "missing thumb url", None, "queued", "skipped_uncurated", 0),
        ]
        for eid, url, title, aurl, pstate, astate, att in rows:
            conn.execute(
                """INSERT INTO videos (eid, url, title, asset_url, page_state, asset_state, attempts, discovered)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1.0)""",
                (eid, url, title, aurl, pstate, astate, att),
            )
        conn.close()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_dummy_mosaic(self, eid: str):
        p = os.path.join(self.mosaic_dir, f"{eid}.jpg")
        with open(p, "wb") as f:
            f.write(b"FAKEJPEG" * 200)  # > 1000 bytes
        return p

    def test_bfs_pending_assets_and_download_rejected(self):
        db = bfs_mod.Database(self.db_path)
        with_rejected = db.pending_assets(limit=100, include_rejected=True)
        self.assertEqual(with_rejected[0], "eid_pend")
        self.assertEqual(set(with_rejected), {"eid_pend", "eid_rej1", "eid_rej2"})

        curated_only_ids = db.pending_assets(limit=100, include_rejected=False)
        self.assertEqual(curated_only_ids, ["eid_pend"])

        missing = db.missing_asset_url_frontier(limit=100, include_rejected=True)
        self.assertEqual(missing, ["eid_nourl"])
        db.conn.close()

        # Verify download_asset processes rejected item when curated_only=False
        self._write_dummy_mosaic("eid_rej1")
        args = types.SimpleNamespace(
            db=self.db_path, fetchers=1, workers=1, rps=100.0,
            max_thumbs=0, retries=1, curated_only=False, drain_only=True,
        )
        with patch.object(bfs_mod, "MOSAIC_DIR", self.mosaic_dir):
            crawler = bfs_mod.CuratedBFSCrawler(args)
            crawler.download_asset("eid_rej1")
            row = crawler.db.get_video("eid_rej1")
            self.assertEqual(row["asset_state"], "done")
            div = crawler.db.conn.execute("SELECT title, preview_url FROM videos_diverse WHERE eid='eid_rej1'").fetchone()
            cur = crawler.db.conn.execute("SELECT eid FROM videos_curated WHERE eid='eid_rej1'").fetchone()
            self.assertIsNotNone(div)
            self.assertEqual(div[1], "https://cdn.example/2/preview.mp4")
            self.assertIsNone(cur)  # Uncurated item must not pollute curated table

            # Now test curated_only=True respects skipping
            self._write_dummy_mosaic("eid_rej2")
            args_strict = types.SimpleNamespace(
                db=self.db_path, fetchers=1, workers=1, rps=100.0,
                max_thumbs=0, retries=1, curated_only=True, drain_only=True,
            )
            crawler_strict = bfs_mod.CuratedBFSCrawler(args_strict)
            crawler_strict.download_asset("eid_rej2")
            row2 = crawler_strict.db.get_video("eid_rej2")
            self.assertEqual(row2["asset_state"], "skipped_uncurated")
            crawler.download_pool.shutdown(wait=True)
            crawler_strict.download_pool.shutdown(wait=True)
            crawler.db.conn.close()
            crawler_strict.db.conn.close()

    def test_harvest_throughput_processes_rejected_and_marks_failed(self):
        self._write_dummy_mosaic("eid_rej1")

        # Mock HTTP 404 for eid_rej2 so it marks 'failed' cleanly in DB
        mock_sess = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.content = b""
        mock_sess.get.return_value = mock_resp

        scraper = ht_mod.HighThroughputScraper(
            db_path=self.db_path,
            mosaic_dir=self.mosaic_dir,
            num_workers=2,
            states=("skipped_uncurated",),
            max_attempts=3,
        )
        with patch.object(scraper, "_get_session", return_value=mock_sess):
            scraper.run()

        conn = sqlite3.connect(self.db_path)
        r1 = conn.execute("SELECT asset_state FROM videos WHERE eid='eid_rej1'").fetchone()[0]
        r2 = conn.execute("SELECT asset_state, attempts, error FROM videos WHERE eid='eid_rej2'").fetchone()
        r_purg = conn.execute("SELECT asset_state FROM videos WHERE eid='eid_purg'").fetchone()[0]
        div_eids = {r[0] for r in conn.execute("SELECT eid FROM videos_diverse").fetchall()}
        conn.close()

        self.assertEqual(r1, "done")
        self.assertIn("eid_rej1", div_eids)
        self.assertEqual(r2[0], "failed")
        self.assertEqual(r2[1], 1)
        self.assertIn("404", r2[2])
        self.assertEqual(r_purg, "purged")

    def test_xvthumb_crawler_counts_and_pending_with_rejected(self):
        db = xvt_mod.Database(self.db_path)
        counts = db.counts()
        self.assertEqual(counts["pending"], 1)
        self.assertEqual(counts["rejected"], 2)

        rows_all = [r["eid"] for r in db.pending_assets(include_rejected=True)]
        self.assertEqual(rows_all[0], "eid_pend")
        self.assertEqual(set(rows_all), {"eid_pend", "eid_rej1", "eid_rej2"})

        rows_no_rej = [r["eid"] for r in db.pending_assets(include_rejected=False)]
        self.assertEqual(rows_no_rej, ["eid_pend"])
        db.conn.close()


if __name__ == "__main__":
    unittest.main()
