#!/usr/bin/env python3
"""Online Real-Time Pipeline: High-Throughput Scrape -> Online GPU 9-Tile Encoding.

Pipes scraped mosaics directly to GPU VRAM in real-time batches:
  1. I/O Downloader Pool: 48 worker threads stream mosaics from CDN to disk.
  2. GPU Ingestion Worker: Collects batches of 8 mosaics (72 tiles) as they land on disk.
  3. EVA-02-CLIP Native FP16: Encodes 72 tiles on Tensor Cores (~120ms).
  4. Live Index Append: Appends 9-tile embeddings to keyframes_9tiles and metadata SQLite.
  5. Continuous Throughput: Keeps scraping indefinitely without needing offline encoding stops.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import queue
import re
import signal
import sqlite3
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import torch
import torch.nn.functional as F
from torchvision import transforms
import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

SCRAPE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRAPE_DIR)
DEFAULT_DB = os.path.join(SCRAPE_DIR, "thumbs.db")
DEFAULT_MOSAIC_DIR = os.path.join(PROJECT_DIR, "mosaics")
DEFAULT_EMB_DIR = os.path.join(SCRAPE_DIR, "embeddings_v2")
DEFAULT_WEIGHTS = os.path.join(SCRAPE_DIR, "siglip_runs", "eva02_curated_v2", "best.pt")
DEFAULT_MODEL = "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(336, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


def mosaic_url(frame_url: str) -> str:
    return re.sub(r"[^/]+\.(?:jpg|jpeg|png|webp|avif)$", "mozaique_listing.jpg", frame_url or "")


def preview_mp4_url(asset_url: str) -> Optional[str]:
    if not asset_url:
        return None
    idx = asset_url.find("/xv_")
    if idx != -1:
        return asset_url[:idx] + "/preview.mp4"
    return None


def clean_text_basic(raw_title: Optional[str]) -> str:
    if not raw_title:
        return ""
    text = html.unescape(raw_title)
    return re.sub(r"\s+", " ", text).strip()


def create_session(pool_size: int = 64) -> requests.Session:
    sess = requests.Session()
    sess.headers.update(HEADERS)
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=retries)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def extract_9_tiles(path: str) -> torch.Tensor:
    """Crops 3x3 contact sheet into 9 distinct 336x336 tensors."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            tw, th = w // 3, h // 3
            crops = []
            for r in range(3):
                for c in range(3):
                    box = (c * tw, r * th, (c + 1) * tw, (r + 1) * th)
                    crops.append(IMAGE_TRANSFORM(im.crop(box)))
            return torch.stack(crops, dim=0)  # [9, 3, 336, 336]
    except Exception:
        return torch.zeros(9, 3, 336, 336)


# --------------------------------------------------------------------------- #
# Real-Time GPU Encoder & Live Index Appender
# --------------------------------------------------------------------------- #

class LiveGPUEncoder(threading.Thread):
    def __init__(
        self,
        emb_dir: str,
        db_path: str,
        model_name: str = DEFAULT_MODEL,
        weights: str = DEFAULT_WEIGHTS,
        device: str = "cuda",
        batch_size: int = 8,
    ):
        super().__init__(name="LiveGPUEncoder", daemon=True)
        self.emb_dir = emb_dir
        self.db_path = db_path
        self.model_name = model_name
        self.weights = weights
        self.device = device
        self.batch_size = batch_size
        self.queue: queue.Queue[Tuple[str, str, str, str]] = queue.Queue()
        self.stop_event = threading.Event()
        self.total_encoded = 0

    def run(self):
        print(f"[*] Initializing Live GPU Encoder on {self.device}...")
        model, _, _ = open_clip.create_model_and_transforms(self.model_name)
        if os.path.exists(self.weights):
            print(f"[*] Loading fine-tuned weights from {self.weights}...")
            ckpt = torch.load(self.weights, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
        if self.device == "cuda":
            model = model.half().to(self.device).eval()
        else:
            model = model.to(self.device).eval()

        meta_db = os.path.join(self.emb_dir, "metadata_147k.sqlite3")
        meta_conn = sqlite3.connect(meta_db, timeout=60)
        meta_cur = meta_conn.cursor()
        meta_cur.execute("SELECT MAX(idx), COUNT(*) FROM video_metadata")
        row = meta_cur.fetchone()
        next_idx = (row[0] + 1) if row and row[0] is not None else 0
        total_meta_count = row[1] if row else 0

        # Output raw continuous binary embedding appenders
        raw_9tiles_file = os.path.join(self.emb_dir, "live_keyframes_9tiles.bin")
        raw_1tile_file = os.path.join(self.emb_dir, "live_globals.bin")

        state_file = os.path.join(self.emb_dir, "live_encoding_state.json")
        print(f"[*] Ready for online GPU encoding! (Starting index: {next_idx:,})")

        batch_buffer = []

        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.2)
                batch_buffer.append(item)
            except queue.Empty:
                pass

            if len(batch_buffer) >= self.batch_size or (batch_buffer and self.stop_event.is_set()):
                curr_batch = batch_buffer[:self.batch_size]
                batch_buffer = batch_buffer[self.batch_size:]

                # 1. Parallel crop tiles
                tensors = []
                for b_eid, b_title, b_path, b_preview in curr_batch:
                    tensors.append(extract_9_tiles(b_path))
                batch_tensor = torch.stack(tensors, dim=0).to(self.device)  # [B, 9, 3, 336, 336]
                if self.device == "cuda":
                    batch_tensor = batch_tensor.half()

                B = batch_tensor.size(0)
                flat_tiles = batch_tensor.view(B * 9, 3, 336, 336)

                # 2. Tensor Core FP16 Encode
                with torch.inference_mode():
                    feats = model.encode_image(flat_tiles)  # [B*9, 768]
                    feats = F.normalize(feats.float(), dim=-1).half()  # [B*9, 768] FP16
                    all_9tiles_np = feats.view(B, 9, 768).cpu().numpy()  # [B, 9, 768]
                    center_1tile_np = all_9tiles_np[:, 4, :].copy()      # [B, 768] (Tile 4 = Center)

                # 3. Stream binary appends (zero copy / instant)
                with open(raw_9tiles_file, "ab") as f9:
                    f9.write(all_9tiles_np.tobytes())
                with open(raw_1tile_file, "ab") as f1:
                    f1.write(center_1tile_np.tobytes())

                # 4. Insert into metadata_147k.sqlite3
                meta_rows = []
                for i, (b_eid, b_title, b_path, b_preview) in enumerate(curr_batch):
                    meta_rows.append((next_idx, b_eid, b_title, "[]", "[]", b_path, b_preview))
                    next_idx += 1

                meta_cur.executemany("""
                    INSERT OR REPLACE INTO video_metadata (idx, eid, title, clothing_tags, position_tags, asset_path, preview_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, meta_rows)
                meta_conn.commit()

                self.total_encoded += B
                total_meta_count += B

                # Update live state file
                try:
                    with open(state_file, "w") as sf:
                        json.dump({
                            "total_items": total_meta_count,
                            "completed_items": total_meta_count,
                            "last_idx": next_idx - 1,
                            "last_update": time.time(),
                        }, sf)
                except Exception:
                    pass

                for _ in range(B):
                    self.queue.task_done()

        meta_conn.close()


# --------------------------------------------------------------------------- #
# Orchestrator: Scraper -> GPU Pipeline
# --------------------------------------------------------------------------- #

class OnlineScrapeAndEncodePipeline:
    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        mosaic_dir: str = DEFAULT_MOSAIC_DIR,
        emb_dir: str = DEFAULT_EMB_DIR,
        workers: int = 48,
        limit: Optional[int] = None,
        device: str = "cuda",
    ):
        self.db_path = db_path
        self.mosaic_dir = mosaic_dir
        self.emb_dir = emb_dir
        self.workers = workers
        self.limit = limit
        self.device = device
        os.makedirs(mosaic_dir, exist_ok=True)
        os.makedirs(emb_dir, exist_ok=True)

        self.gpu_encoder = LiveGPUEncoder(emb_dir=self.emb_dir, db_path=self.db_path, device=self.device)
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
                with self.stats_lock:
                    self.cached_count += 1
                self.gpu_encoder.queue.put((eid, clean_title, target_path, preview_url))
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
                    with self.stats_lock:
                        self.downloaded_count += 1
                        self.total_bytes += b_len
                    # Dispatch immediately to live GPU encoding queue
                    self.gpu_encoder.queue.put((eid, clean_title, target_path, preview_url))
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
            enc = self.gpu_encoder.total_encoded
            dl_rate = dl / elapsed
            enc_rate = enc / elapsed

            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"CDN Downloaded: {dl:,} ({dl_rate:.1f}/s) | "
                f"GPU Encoded (9-Tile): {enc:,} ({enc_rate:.1f} vids/s, {enc_rate*9:.1f} frames/s) | "
                f"Cached: {cached:,} | Fail: {fail:,} | GPU Q: {self.gpu_encoder.queue.qsize()}",
                flush=True
            )

    def run(self):
        print(f"[*] Starting Online Scrape + GPU 9-Tile Encoding Pipeline...")
        print(f"[*] Mosaics Directory: {self.mosaic_dir}")
        print(f"[*] Embeddings Directory: {self.emb_dir}")
        print(f"[*] Scraper Threads: {self.workers} | Device: {self.device}")

        # 1. Start GPU Ingestion Engine
        self.gpu_encoder.start()

        # 2. Start Scraper Workers
        workers = []
        for i in range(self.workers):
            t = threading.Thread(target=self.worker_loop, name=f"ScraperWorker-{i}", daemon=True)
            t.start()
            workers.append(t)

        reporter = threading.Thread(target=self.reporter_loop, name="Reporter", daemon=True)
        reporter.start()

        # 3. Stream tasks from database
        conn = sqlite3.connect(self.db_path, timeout=60)
        cur = conn.cursor()

        # Avoid already encoded eids in metadata_147k
        meta_db = os.path.join(self.emb_dir, "metadata_147k.sqlite3")
        cur.execute(f"ATTACH DATABASE '{meta_db}' AS mdb")

        query = """
            SELECT v.eid, v.title, v.asset_url
            FROM videos v
            LEFT JOIN mdb.video_metadata m ON v.eid = m.eid
            WHERE v.asset_url IS NOT NULL 
              AND m.eid IS NULL
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
        print(f"[*] Pushed {pushed:,} candidate download + encode tasks. Streaming...")

        self.task_queue.join()
        self.gpu_encoder.queue.join()
        self.stop_event.set()
        self.gpu_encoder.stop_event.set()
        self.gpu_encoder.join()

        elapsed = time.time() - self.t_start
        print(f"\n[+] Online Scrape + GPU Encode Finished Successfully!")
        print(f"    Downloaded : {self.downloaded_count:,}")
        print(f"    GPU Encoded: {self.gpu_encoder.total_encoded:,} videos ({self.gpu_encoder.total_encoded * 9:,} keyframes)")
        print(f"    Total Time : {elapsed:.1f} s")


def main():
    parser = argparse.ArgumentParser(description="Continuous High-Throughput Scraper + Live GPU 9-Tile Encoder")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to thumbs.db")
    parser.add_argument("--mosaic-dir", default=DEFAULT_MOSAIC_DIR, help="Destination directory for mosaics")
    parser.add_argument("--emb-dir", default=DEFAULT_EMB_DIR, help="Embeddings directory")
    parser.add_argument("--workers", type=int, default=48, help="Scraper threads (default: 48)")
    parser.add_argument("--limit", type=int, default=None, help="Limit items (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    pipeline = OnlineScrapeAndEncodePipeline(
        db_path=args.db,
        mosaic_dir=args.mosaic_dir,
        emb_dir=args.emb_dir,
        workers=args.workers,
        limit=args.limit,
        device=args.device,
    )

    def handle_sigint(sig, frame):
        print("\n[*] Stopping pipeline cleanly...")
        pipeline.stop_event.set()
        pipeline.gpu_encoder.stop_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    pipeline.run()


if __name__ == "__main__":
    main()
