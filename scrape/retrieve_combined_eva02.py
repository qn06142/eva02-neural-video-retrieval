#!/usr/bin/env python3
"""Unified retrieval across curated Danbooru2023 and XVideos using EVA-02-CLIP-L-14-336.

Features:
  - Architecture: EVA-02-CLIP Large (304M vision params, SwiGLU, 2D RoPE, 336x336 resolution)
  - Pretrained / Refined weights: timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k
  - High resolution: 336x336 (576 tokens per image)
  - High capacity: 768-D joint embedding space
  - Multi-threaded parallel disk prefetching & GPU inference
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
from typing import List, Optional, Tuple

import open_clip
import torch
from PIL import Image
from torchvision import transforms

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DANBOORU_DB = os.path.join(PROJECT_DIR, "scrape", "danbooru2023_curated", "manifest.sqlite3")
DEFAULT_XVIDEOS_DB = os.path.join(PROJECT_DIR, "scrape", "thumbs.db")
DEFAULT_EVA_MODEL = "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"
DEFAULT_KB_PATH = os.path.join(PROJECT_DIR, "scrape", "siglip_runs", "combined_kb_eva02.pt")

eva_transform = transforms.Compose([
    transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_danbooru_corpus(danbooru_db: str, limit: Optional[int] = None) -> list[dict]:
    """Load Danbooru records from manifest.sqlite3."""
    if not os.path.exists(danbooru_db):
        log(f"Danbooru database not found at {danbooru_db}")
        return []

    conn = sqlite3.connect(danbooru_db)
    rows = conn.execute("""
        SELECT post_id, character, position_tags, clothing_tags, image_path, fav 
        FROM danbooru_curated WHERE is_downloaded = 1 ORDER BY fav DESC
    """).fetchall()
    conn.close()

    items = []
    for pid, char_name, p_json, c_json, img_path, fav in rows:
        if not img_path:
            continue
        if not os.path.isabs(img_path):
            img_path = os.path.join(PROJECT_DIR, img_path)
        if not os.path.exists(img_path):
            continue

        p_tags = [t.replace("_", " ") for t in json.loads(p_json)]
        c_tags = [t.replace("_", " ") for t in json.loads(c_json)]
        char_clean = char_name.replace("_", " ").title() if char_name else ""

        desc_parts = []
        if char_clean: desc_parts.append(char_clean)
        if p_tags: desc_parts.append(", ".join(p_tags))
        if c_tags: desc_parts.append(", ".join(c_tags))

        title_str = " | ".join(desc_parts) if desc_parts else f"Danbooru #{pid}"

        items.append({
            "id": f"danbooru_{pid}",
            "path": img_path,
            "title": title_str,
            "domain": "danbooru",
            "position_tags": p_tags,
            "clothing_tags": c_tags,
            "character": char_clean,
            "fav": fav
        })
        if limit and len(items) >= limit:
            break

    log(f"Loaded {len(items):,} Danbooru items from manifest")
    return items


def load_xvideos_corpus(db_path: str, limit: Optional[int] = None) -> list[dict]:
    """Load curated XVideos records from thumbs.db:videos_curated."""
    if not os.path.exists(db_path):
        log(f"XVideos database not found at {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT eid, clean_title, clothing_tags, position_tags, asset_path 
        FROM videos_curated ORDER BY eid
    """).fetchall()
    conn.close()

    items = []
    for eid, clean_title, c_json, p_json, asset_path in rows:
        if not asset_path or not os.path.exists(asset_path):
            continue
        c_tags = json.loads(c_json)
        p_tags = json.loads(p_json)

        items.append({
            "id": f"xv_{eid}",
            "path": asset_path,
            "title": clean_title,
            "domain": "xvideos",
            "position_tags": p_tags,
            "clothing_tags": c_tags,
            "character": "",
            "fav": 0
        })
        if limit and len(items) >= limit:
            break

    log(f"Loaded {len(items):,} curated XVideos items from database")
    return items


def _load_and_transform_item(item: dict) -> tuple[Optional[torch.Tensor], dict]:
    path = item["path"]
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_DIR, path)
    domain = item["domain"]
    try:
        with Image.open(path) as raw:
            im = raw.convert("RGB")
            if domain == "xvideos":
                w, h = im.size
                if w >= 3 and h >= 3:
                    im = im.crop((w // 3, h // 3, 2 * w // 3, 2 * h // 3))
            tensor = eva_transform(im)
            return tensor, item
    except Exception:
        return None, item


@torch.no_grad()
def build_kb_embeddings(model, items: list[dict], device: torch.device,
                        batch_size: int = 128, workers: int = 8) -> tuple[torch.Tensor, list[dict]]:
    """Encode all items in batches using multi-threaded prefetching and batched GPU inference."""
    model.eval()
    chunks = []
    valid_items = []
    total = len(items)
    beginning = time.time()
    seen = 0

    prefetch_chunk_size = batch_size * 4
    log(f"Starting parallel EVA-02 embedding: {total:,} images (batch_size={batch_size}, workers={workers})")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, total, prefetch_chunk_size):
            chunk_items = items[i:i + prefetch_chunk_size]
            results = list(pool.map(_load_and_transform_item, chunk_items))

            curr_batch_tensors = []
            curr_batch_items = []
            for tensor, it in results:
                if tensor is not None:
                    curr_batch_tensors.append(tensor)
                    curr_batch_items.append(it)
                if len(curr_batch_tensors) == batch_size:
                    batch = torch.stack(curr_batch_tensors).to(device)
                    with torch.amp.autocast('cuda'):
                        feats = model.encode_image(batch, normalize=True)
                    chunks.append(feats.half().cpu())
                    valid_items.extend(curr_batch_items)
                    seen += len(curr_batch_tensors)
                    curr_batch_tensors = []
                    curr_batch_items = []

            if curr_batch_tensors:
                batch = torch.stack(curr_batch_tensors).to(device)
                with torch.amp.autocast('cuda'):
                    feats = model.encode_image(batch, normalize=True)
                chunks.append(feats.half().cpu())
                valid_items.extend(curr_batch_items)
                seen += len(curr_batch_tensors)

            elapsed = time.time() - beginning
            rate = seen / max(0.1, elapsed)
            remaining = (total - seen) / max(0.1, rate)
            if seen % 2000 < prefetch_chunk_size or i + prefetch_chunk_size >= total:
                log(f"  [EVA-02 encode] {seen:,} / {total:,} images ({seen/total*100:.1f}%) | "
                    f"{rate:.1f} img/s | ~{remaining/60:.1f}m remaining")

    emb = torch.cat(chunks, dim=0)
    return emb, valid_items


def run_query(emb: torch.Tensor, items: list[dict], q: torch.Tensor,
              top: int = 20, device: torch.device = torch.device("cpu"),
              out_dir: Optional[str] = None, query_title: str = "",
              query_image_path: Optional[str] = None,
              copy_images: bool = True,
              domain: str = "all") -> list[dict]:
    """Compute similarities, print ranking, and optionally export matches and copy JPGs."""
    scores = (emb.to(device=device, dtype=torch.float32) @ q.to(device=device, dtype=torch.float32).t()).squeeze(1)

    domain = domain.lower()
    if domain in ("xv", "xvideos"):
        domain = "xvideos"
    elif domain in ("dan", "danbooru"):
        domain = "danbooru"

    if domain in ("xvideos", "danbooru"):
        mask = torch.tensor([it["domain"] == domain for it in items], dtype=torch.bool, device=device)
        scores[~mask] = -float("inf")
        valid_count = int(mask.sum().item())
        k = min(top, valid_count)
    else:
        k = min(top, scores.numel())

    scores = scores.cpu()
    vals, idxs = torch.topk(scores, k)

    print(f"\n{'='*80}")
    print(f"  TOP {k} RESULTS [EVA-02-L-14-336]: {query_title}")
    print(f"{'='*80}")

    export_items = []
    ranking_lines = [
        f"EVA-02-CLIP-L-14-336 RETRIEVAL RANKING: {query_title}",
        f"{'='*80}",
    ]
    for rank, (score_val, idx_val) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        it = items[idx_val]
        dom = it["domain"].upper()
        title = it["title"]
        p = it["path"]
        pos = ", ".join(it.get("position_tags", [])) or "none"
        cloth = ", ".join(it.get("clothing_tags", [])) or "none"

        print(f"[{rank:02d}] {score_val:.4f}  [{dom:<8}]  {title[:55]:<55}")
        print(f"      Pos: {pos:<30} | Attire: {cloth}")
        print(f"      File: {p}")

        ranking_lines.append(
            f"#{rank:02d} | Score: {score_val:.4f} | [{dom}] {title}\n"
            f"      Pos: {pos} | Attire: {cloth}\n"
            f"      Source: {p}\n"
        )

        export_items.append({
            "rank": rank,
            "score": round(score_val, 4),
            "id": it.get("id", f"{it['domain']}_{rank}"),
            "domain": it["domain"],
            "title": title,
            "path": p,
            "position": it.get("position_tags", []),
            "clothing": it.get("clothing_tags", []),
        })

    if out_dir:
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(PROJECT_DIR, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_title)[:40].strip("_") or "query"
        dump_path = os.path.join(out_dir, f"results_{slug}.json")
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(export_items, f, indent=2)

        rank_txt_path = os.path.join(out_dir, "ranking.txt")
        with open(rank_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ranking_lines) + "\n")

        # Copy query image if provided
        if query_image_path:
            if not os.path.isabs(query_image_path):
                query_image_path = os.path.join(PROJECT_DIR, query_image_path)
            if os.path.exists(query_image_path):
                try:
                    with Image.open(query_image_path) as q_img:
                        q_rgb = q_img.convert("RGB")
                        q_out_name = f"00_query_{os.path.basename(query_image_path)}"
                        if not q_out_name.lower().endswith(".jpg"):
                            q_out_name = os.path.splitext(q_out_name)[0] + ".jpg"
                        q_rgb.save(os.path.join(out_dir, q_out_name), "JPEG", quality=95)
                except Exception as e:
                    log(f"Warning: Failed to copy query image: {e}")

        # Copy retrieved images as JPGs
        if copy_images:
            copied_count = 0
            for item in export_items:
                rank = item["rank"]
                score_val = item["score"]
                dom = item["domain"]
                src_path = item["path"]
                item_id = item["id"]
                if not os.path.isabs(src_path):
                    src_path = os.path.join(PROJECT_DIR, src_path)
                if not os.path.exists(src_path):
                    continue

                safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", item_id)[:30]
                base_name = f"{rank:02d}_{score_val:.4f}_{dom}_{safe_id}"

                try:
                    with Image.open(src_path) as img:
                        img_rgb = img.convert("RGB")

                        if dom == "xvideos":
                            # Save full mosaic JPG
                            mosaic_path = os.path.join(out_dir, f"{base_name}_mosaic.jpg")
                            img_rgb.save(mosaic_path, "JPEG", quality=95)

                            # Save center frame JPG (the exact tile EVA-02 matched)
                            w, h = img_rgb.size
                            if w >= 3 and h >= 3:
                                center_tile = img_rgb.crop((w // 3, h // 3, 2 * w // 3, 2 * h // 3))
                                center_path = os.path.join(out_dir, f"{base_name}_center.jpg")
                                center_tile.save(center_path, "JPEG", quality=95)
                                # Primary jpg is the center frame so viewers see the action frame directly
                                prim_path = os.path.join(out_dir, f"{base_name}.jpg")
                                center_tile.save(prim_path, "JPEG", quality=95)
                            else:
                                prim_path = os.path.join(out_dir, f"{base_name}.jpg")
                                img_rgb.save(prim_path, "JPEG", quality=95)
                        else:
                            # Danbooru image
                            out_img_path = os.path.join(out_dir, f"{base_name}.jpg")
                            img_rgb.save(out_img_path, "JPEG", quality=95)

                        copied_count += 1
                except Exception as e:
                    log(f"Warning: failed to save JPG for rank {rank} ({src_path}): {e}")

            log(f"Successfully copied {copied_count} retrieved items as JPGs to: {out_dir}")

    return export_items


def main() -> int:
    parser = argparse.ArgumentParser(description="EVA-02-CLIP unified retrieval across Danbooru and XVideos")
    parser.add_argument("--query", type=str, default="", help="Text query to search for")
    parser.add_argument("--image-query", type=str, default="", help="Path to image query file")
    parser.add_argument("--top", type=int, default=20, help="Number of results to return")
    parser.add_argument("--danbooru-db", type=str, default=DEFAULT_DANBOORU_DB)
    parser.add_argument("--xvideos-db", type=str, default=DEFAULT_XVIDEOS_DB)
    parser.add_argument("--kb-path", type=str, default=DEFAULT_KB_PATH)
    parser.add_argument("--rebuild-kb", action="store_true", help="Force rebuild of knowledge base embeddings")
    parser.add_argument("--weights", type=str, default="", help="Path to fine-tuned checkpoint (best.pt)")
    parser.add_argument("--model-name", type=str, default=DEFAULT_EVA_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="", help="Output directory to copy retrieved JPGs")
    parser.add_argument("--no-copy", action="store_true", help="Do not copy retrieved images into folder")
    parser.add_argument("--domain", type=str, default="all", choices=["all", "danbooru", "xvideos", "dan", "xv"],
                        help="Filter retrieval to a specific dataset: all (default), danbooru, xvideos")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # 1. Load Model & Tokenizer
    log(f"Loading EVA-02 model: {args.model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    if args.weights and os.path.exists(args.weights):
        log(f"Loading fine-tuned checkpoint: {args.weights}")
        ckpt = torch.load(args.weights, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device).eval()

    # 2. Load or Build Knowledge Base
    if os.path.exists(args.kb_path) and not args.rebuild_kb:
        log(f"Loading cached EVA-02 embeddings from {args.kb_path}...")
        data = torch.load(args.kb_path, map_location="cpu")
        emb = data["embeddings"]
        items = data["items"]
        log(f"Loaded {len(items):,} cached embeddings ({emb.shape})")
    else:
        log("Compiling unified corpus from Danbooru and XVideos...")
        dan_items = load_danbooru_corpus(args.danbooru_db)
        xv_items = load_xvideos_corpus(args.xvideos_db)
        all_items = dan_items + xv_items

        if not all_items:
            log("No items found to index! Exiting.")
            return 1

        emb, valid_items = build_kb_embeddings(
            model=model,
            items=all_items,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers
        )

        os.makedirs(os.path.dirname(args.kb_path), exist_ok=True)
        torch.save({"embeddings": emb, "items": valid_items}, args.kb_path)
        log(f"Saved {len(valid_items):,} embeddings to {args.kb_path}")
        items = valid_items

    # 3. Handle Queries
    copy_images = not args.no_copy
    if args.image_query:
        if not os.path.exists(args.image_query):
            print(f"Error: image query not found: {args.image_query}")
            return 1
        log(f"Encoding image query: {args.image_query}...")
        with Image.open(args.image_query) as raw:
            im = raw.convert("RGB")
            tensor = eva_transform(im).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                q_emb = model.encode_image(tensor, normalize=True).cpu()

        img_out = args.out_dir
        if args.out_dir and args.query:
            img_out = os.path.join(args.out_dir, "image_query")

        dom_tag = f" [{args.domain.upper()}]" if args.domain != "all" else ""
        run_query(emb, items, q_emb, top=args.top, device=device,
                  out_dir=img_out, query_title=f"Image: {os.path.basename(args.image_query)}{dom_tag}",
                  query_image_path=args.image_query, copy_images=copy_images, domain=args.domain)

    if args.query:
        log(f"Encoding text query: '{args.query}'...")
        tokens = tokenizer([args.query]).to(device)
        with torch.no_grad():
            q_emb = model.encode_text(tokens, normalize=True).cpu()

        text_out = args.out_dir
        if args.out_dir and args.image_query:
            text_out = os.path.join(args.out_dir, "text_query")

        dom_tag = f" [{args.domain.upper()}]" if args.domain != "all" else ""
        run_query(emb, items, q_emb, top=args.top, device=device,
                  out_dir=text_out, query_title=f"Text: '{args.query}'{dom_tag}",
                  copy_images=copy_images, domain=args.domain)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
