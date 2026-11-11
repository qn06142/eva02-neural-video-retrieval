"""High-performance concurrent downloader for curated Danbooru2023 images.
Uses deepghs/danbooru2023_index byte offsets and HTTP Range requests against nyanko7/danbooru2023.
"""

import os
import sys
import json
import sqlite3
import time
import argparse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download
from PIL import Image
import io

DB_PATH = "scrape/danbooru2023_curated/manifest.sqlite3"
IMG_DIR = "scrape/danbooru2023_curated/images"
HF_BASE_URL = "https://huggingface.co/datasets/nyanko7/danbooru2023/resolve/main"

INDEX_CACHE = {}

def get_bucket_index(bucket: int):
    """Fetch and parse index JSON for a given bucket (cached in memory and disk)."""
    if bucket in INDEX_CACHE:
        return bucket, INDEX_CACHE[bucket]
    try:
        idx_file = f"original/data-{bucket:04d}.json"
        local_path = hf_hub_download(
            "deepghs/danbooru2023_index",
            idx_file,
            repo_type="dataset"
        )
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            files = data.get("files", {})
            INDEX_CACHE[bucket] = files
            return bucket, files
    except Exception as e:
        print(f"[Warn] Failed to load index for bucket {bucket}: {e}", file=sys.stderr, flush=True)
        return bucket, {}

def download_single_image(post_id: int, bucket: int, file_info: dict, max_retries: int = 3):
    """Download single image using byte range request from Hugging Face tar."""
    offset = file_info["offset"]
    size = file_info["size"]
    tar_url = f"{HF_BASE_URL}/original/data-{bucket:04d}.tar"
    dest_path = os.path.join(IMG_DIR, f"{post_id}.jpg")

    headers = {
        "Range": f"bytes={offset}-{offset + size - 1}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DanbooruDownloader/1.0"
    }

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(tar_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()

            if len(data) != size:
                if len(data) < size:
                    raise IOError(f"Incomplete download: {len(data)} / {size} bytes")

            # Validate and convert/save with PIL
            img = Image.open(io.BytesIO(data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dest_path, "JPEG", quality=95)
            
            return post_id, dest_path, os.path.getsize(dest_path), True, None
        except Exception as e:
            if attempt == max_retries - 1:
                return post_id, None, 0, False, str(e)
            time.sleep(1.0 * (attempt + 1))

def run_download(batch_size: int = None, max_workers: int = 16):
    os.makedirs(IMG_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    query = "SELECT post_id, bucket FROM danbooru_curated WHERE is_downloaded = 0 ORDER BY bucket, fav DESC"
    if batch_size:
        query += f" LIMIT {batch_size}"

    cur.execute(query)
    rows = cur.fetchall()
    total_to_download = len(rows)
    print(f"Total pending images to download: {total_to_download:,}", flush=True)

    if total_to_download == 0:
        print("All curated images are already downloaded!", flush=True)
        conn.close()
        return

    # Group by bucket to batch index lookups
    by_bucket = {}
    for pid, b in rows:
        by_bucket.setdefault(b, []).append(pid)

    distinct_buckets = list(by_bucket.keys())
    print(f"Posts span across {len(distinct_buckets)} distinct buckets. Pre-fetching bucket indices...", flush=True)

    t_idx_start = time.time()
    # Fetch needed bucket indices concurrently
    with ThreadPoolExecutor(max_workers=min(max_workers, len(distinct_buckets))) as idx_exec:
        fut_map = {idx_exec.submit(get_bucket_index, b): b for b in distinct_buckets}
        for fut in as_completed(fut_map):
            b, files = fut.result()
            INDEX_CACHE[b] = files

    tasks = []
    skipped_no_index = 0
    for b, pids in by_bucket.items():
        bucket_files = INDEX_CACHE.get(b, {})
        for pid in pids:
            found_info = None
            for ext in [".jpg", ".png", ".jpeg", ".webp"]:
                key = f"./{pid}{ext}"
                if key in bucket_files:
                    found_info = bucket_files[key]
                    break
            if found_info:
                tasks.append((pid, b, found_info))
            else:
                skipped_no_index += 1

    print(f"Loaded all indices and prepared {len(tasks):,} tasks in {time.time()-t_idx_start:.1f}s (Skipped without index: {skipped_no_index})", flush=True)

    # Concurrent image download
    success_count = 0
    fail_count = 0
    total_bytes = 0
    t0 = time.time()

    update_buffer = []

    def flush_updates(c, cur_obj, buf):
        if not buf:
            return
        cur_obj.executemany("""
            UPDATE danbooru_curated 
            SET is_downloaded = ?, image_path = ?, file_bytes = ?
            WHERE post_id = ?
        """, buf)
        c.commit()
        buf.clear()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_image, pid, b, info): pid for pid, b, info in tasks}

        for idx, fut in enumerate(as_completed(futures), start=1):
            pid, dest_path, size_bytes, ok, err = fut.result()
            if ok:
                success_count += 1
                total_bytes += size_bytes
                update_buffer.append((1, dest_path, size_bytes, pid))
            else:
                fail_count += 1
                print(f"[Failed] Post {pid}: {err}", file=sys.stderr, flush=True)

            if len(update_buffer) >= 50:
                flush_updates(conn, cur, update_buffer)

            if idx % 50 == 0 or idx == len(tasks):
                elapsed = time.time() - t0
                speed_mb = (total_bytes / (1024 * 1024)) / max(elapsed, 0.001)
                img_per_sec = idx / max(elapsed, 0.001)
                print(f"Progress: {idx}/{len(tasks)} ({idx/len(tasks)*100:.1f}%) | "
                      f"Success: {success_count}, Fail: {fail_count} | "
                      f"Speed: {speed_mb:.2f} MB/s ({img_per_sec:.1f} img/s) | "
                      f"Total downloaded: {total_bytes/(1024*1024):.1f} MB", flush=True)

    flush_updates(conn, cur, update_buffer)
    conn.close()

    elapsed = time.time() - t0
    print(f"\nDownload finished in {elapsed:.1f}s!", flush=True)
    print(f"Total downloaded: {success_count:,} images ({total_bytes/(1024*1024):.1f} MB)", flush=True)
    if fail_count > 0:
        print(f"Failed downloads: {fail_count}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download curated Danbooru images")
    parser.add_argument("--batch-size", type=int, default=None, help="Limit number of images to download")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker threads")
    args = parser.parse_args()

    run_download(batch_size=args.batch_size, max_workers=args.workers)
