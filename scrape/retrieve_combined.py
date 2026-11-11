#!/usr/bin/env python3
"""Unified retrieval across both Pixiv illustrations and XVideos contact-sheet mosaics.

Builds and queries a unified embedding index across both domains:
  - Pixiv (10k explicit illustrations from manifest.sqlite3)
  - XVideos (118k videos from thumbs.db, represented by center action frame)

Usage
-----
    # 1. Build and save the combined knowledge base embeddings (one-time, ~5 mins)
    .venv/Scripts/python scrape/retrieve_combined.py \\
        --model scrape/siglip_runs/pixiv_refined_v1/best \\
        --save-kb scrape/siglip_runs/combined_kb.pt

    # 2. Query with text (using accurate centroid or vague multi-query TTA)
    .venv/Scripts/python scrape/retrieve_combined.py \\
        --model scrape/siglip_runs/pixiv_refined_v1/best \\
        --load-kb scrape/siglip_runs/combined_kb.pt \\
        --text "lingerie" --text-tta accurate

    # 3. Query with an image (e.g. screenshot)
    .venv/Scripts/python scrape/retrieve_combined.py \\
        --model scrape/siglip_runs/pixiv_refined_v1/best \\
        --load-kb scrape/siglip_runs/combined_kb.pt \\
        --query "path/to/query_image.png"
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

import torch
from PIL import Image
from torchvision import transforms

from frame_model import SiglipBagModel as SiglipModel
from retrieval_tta import TEXT_TTA_MODES, encode_text_query, expand_text_query, TTA_POLICIES
from transformers import SiglipProcessor

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PIXIV_DIR = os.path.join(PROJECT_DIR, "scrape", "pixiv_explicit_10k")
DEFAULT_XVIDEOS_DB = os.path.join(PROJECT_DIR, "scrape", "thumbs.db")
DEFAULT_MOSAICS_DIR = os.path.join(PROJECT_DIR, "mosaics")
DEFAULT_MODEL = os.path.join(PROJECT_DIR, "scrape", "siglip_runs", "pixiv_refined_v1", "best")
DEFAULT_KB_PATH = os.path.join(PROJECT_DIR, "scrape", "siglip_runs", "combined_kb.pt")

img_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pixiv_corpus(pixiv_dir: str, limit: Optional[int] = None) -> list[dict]:
    """Load Pixiv records from manifest.sqlite3."""
    db_path = os.path.join(pixiv_dir, "manifest.sqlite3")
    images_dir = os.path.join(pixiv_dir, "images")
    if not os.path.exists(db_path):
        log(f"Pixiv database not found at {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, sha256, metadata FROM images ORDER BY id").fetchall()
    conn.close()

    items = []
    for img_id, sha256, meta_str in rows:
        try:
            meta = json.loads(meta_str)
        except Exception:
            continue
        p = meta.get("path")
        if not p:
            p = os.path.join(images_dir, f"{sha256}.webp")

        tags_info = meta.get("tags_and_ratings", {})
        chars = tags_info.get("character tags", [])
        gens = tags_info.get("general tags", [])
        chars_clean = [c.replace("_", " ").title() for c in chars if c]
        gens_clean = [g.replace("_", " ") for g in gens[:8] if g]
        title_summary = (", ".join(chars_clean) + " | " + ", ".join(gens_clean)) if chars_clean else ", ".join(gens_clean)

        items.append({
            "id": f"pixiv_{img_id}",
            "path": p,
            "title": title_summary or "Explicit Illustration",
            "domain": "pixiv",
        })
        if limit and len(items) >= limit:
            break

    log(f"loaded {len(items):,} Pixiv items from manifest")
    return items


def load_xvideos_corpus(db_path: str, mosaics_dir: str, limit: Optional[int] = None) -> list[dict]:
    """Load XVideos records from thumbs.db and mosaics/."""
    if not os.path.exists(db_path):
        log(f"XVideos database not found at {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT eid, title, asset_path FROM videos WHERE asset_state='done' ORDER BY eid").fetchall()
    conn.close()

    items = []
    for eid, raw_title, asset_path in rows:
        p = asset_path or os.path.join(mosaics_dir, f"{eid}.jpg")
        title = html.unescape(raw_title or "").replace("\r", " ").replace("\n", " ")
        title = re.sub(r"\s+", " ", title).strip()

        items.append({
            "id": f"xv_{eid}",
            "path": p,
            "title": title or f"Video {eid}",
            "domain": "xvideos",
        })
        if limit and len(items) >= limit:
            break

    log(f"loaded {len(items):,} XVideos items from database")
    return items


def _load_and_transform_item(item: dict) -> tuple[Optional[torch.Tensor], dict]:
    path = item["path"]
    domain = item["domain"]
    try:
        with Image.open(path) as raw:
            im = raw.convert("RGB")
            if domain == "xvideos":
                w, h = im.size
                if w >= 3 and h >= 3:
                    im = im.crop((w // 3, h // 3, 2 * w // 3, 2 * h // 3))
            tensor = img_transform(im)
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

    prefetch_chunk_size = batch_size * 4  # 512 items per thread pool batch
    log(f"starting parallel KB embedding: {total:,} images (batch_size={batch_size}, workers={workers})")

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
                    batch = torch.stack(curr_batch_tensors).to(device, dtype=torch.float16)
                    feats = model.get_image_features(pixel_values=batch)
                    if getattr(model, "frame_mixer", None) is not None:
                        feats = model.mix_frame_features(feats[:, None])[:, 0]
                    feats = torch.nn.functional.normalize(feats.float(), dim=-1)
                    chunks.append(feats.half().cpu())
                    valid_items.extend(curr_batch_items)
                    seen += len(curr_batch_tensors)
                    curr_batch_tensors = []
                    curr_batch_items = []

            if curr_batch_tensors:
                batch = torch.stack(curr_batch_tensors).to(device, dtype=torch.float16)
                feats = model.get_image_features(pixel_values=batch)
                if getattr(model, "frame_mixer", None) is not None:
                    feats = model.mix_frame_features(feats[:, None])[:, 0]
                feats = torch.nn.functional.normalize(feats.float(), dim=-1)
                chunks.append(feats.half().cpu())
                valid_items.extend(curr_batch_items)
                seen += len(curr_batch_tensors)

            elapsed = time.time() - beginning
            rate = seen / max(0.1, elapsed)
            remaining = (total - seen) / max(0.1, rate)
            if seen % 5000 < prefetch_chunk_size or i + prefetch_chunk_size >= total:
                log(f"  [KB encode] {seen:,} / {total:,} images ({seen/total*100:.1f}%) | "
                    f"{rate:.1f} img/s | ~{remaining/60:.1f}m remaining")

    emb = torch.cat(chunks, dim=0)
    return emb, valid_items


def run_query(emb: torch.Tensor, items: list[dict], q: torch.Tensor, tta_mode: str,
              top: int = 20, device: torch.device = torch.device("cpu"),
              out_dir: Optional[str] = None, query_title: str = "") -> None:
    """Compute similarities, print ranking, and optionally export matches."""
    scores = emb.to(device) @ q.to(device).t()
    sims = scores.amax(dim=1).cpu() if tta_mode == "vague" else scores.squeeze(1).cpu()

    k = min(top, sims.numel())
    vals, idxs = torch.topk(sims, k)

    print(f"\n{'='*80}")
    print(f"  TOP {k} RESULTS: {query_title}")
    print(f"{'='*80}")
    lines = []
    for rank, (score, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), 1):
        item = items[idx]
        domain_tag = f"[{item['domain'].upper():7s}]"
        line = f"#{rank:02d} | Score: {score:.4f} | {domain_tag} {item['title'][:60]}"
        print(line)
        lines.append(f"{line} | Path: {item['path']}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for rank, (score, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), 1):
            item = items[idx]
            ext = os.path.splitext(item["path"])[-1]
            out_file = os.path.join(out_dir, f"{rank:02d}_{score:.4f}_{item['domain']}_{item['id']}{ext}")
            try:
                shutil.copy2(item["path"], out_file)
            except Exception:
                pass
        with open(os.path.join(out_dir, "ranking.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nCopied top {k} items to: {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Query unified Pixiv + XVideos Knowledge Base.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--query", help="Path to a query image (e.g. screenshot)")
    g.add_argument("--text", help="Text search query")

    ap.add_argument("--text-tta", choices=TEXT_TTA_MODES, default="none",
                    help="Text TTA mode: 'accurate' (tight centroid) or 'vague' (exploratory multi-query max-pool)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Path to refined SigLIP model checkpoint")
    ap.add_argument("--pixiv-dir", default=DEFAULT_PIXIV_DIR)
    ap.add_argument("--xvideos-db", default=DEFAULT_XVIDEOS_DB)
    ap.add_argument("--mosaics-dir", default=DEFAULT_MOSAICS_DIR)
    ap.add_argument("--save-kb", default=DEFAULT_KB_PATH, metavar="PATH",
                    help=f"Build and save unified KB embeddings to PATH (default: {DEFAULT_KB_PATH})")
    ap.add_argument("--load-kb", default=None, metavar="PATH",
                    help="Load precomputed unified KB embeddings from PATH")
    ap.add_argument("--top", type=int, default=20, help="Number of top results to return")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-xvideos", type=int, default=None, help="Optional limit on XVideos samples")
    ap.add_argument("--max-pixiv", type=int, default=None, help="Optional limit on Pixiv samples")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--out-dir", default=None, help="Folder to copy top ranked images into")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    log(f"device: {device} | model: {args.model}")

    processor = SiglipProcessor.from_pretrained(args.model)
    model = SiglipModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    # ---- 1. KB Loading / Building -------------------------------------- #
    kb_path = args.load_kb
    if not kb_path and os.path.exists(DEFAULT_KB_PATH) and not args.max_xvideos and not args.max_pixiv:
        kb_path = DEFAULT_KB_PATH

    if kb_path and os.path.exists(kb_path):
        log(f"loading precomputed KB embeddings from {kb_path}")
        kb_data = torch.load(kb_path, map_location="cpu", weights_only=False)
        items = kb_data["items"]
        emb = kb_data["emb"].float()
        log(f"loaded {len(items):,} corpus embeddings (shape: {tuple(emb.shape)})")
    else:
        pixiv_items = load_pixiv_corpus(args.pixiv_dir, limit=args.max_pixiv)
        xv_items = load_xvideos_corpus(args.xvideos_db, args.mosaics_dir, limit=args.max_xvideos)
        all_items = pixiv_items + xv_items
        if not all_items:
            log("No items found to build knowledge base")
            return 1
        log(f"total combined corpus: {len(all_items):,} items ({len(pixiv_items):,} Pixiv, {len(xv_items):,} XVideos)")

        emb, items = build_kb_embeddings(model, all_items, device, batch_size=args.batch_size, workers=args.workers)
        emb = emb.float()

        target_save = args.save_kb or DEFAULT_KB_PATH
        if target_save and not args.max_xvideos and not args.max_pixiv:
            os.makedirs(os.path.dirname(os.path.abspath(target_save)), exist_ok=True)
            torch.save({"items": items, "emb": emb.half()}, target_save)
            log(f"saved unified KB embeddings ({len(items):,} items) -> {target_save}")
            if not args.text and not args.query:
                return 0

    # ---- 2. Query Encoding --------------------------------------------- #
    tta_mode = "none"
    if args.text:
        def encode_fn(texts):
            enc = processor(text=texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            pad = processor.tokenizer.pad_token_id
            is_pad = enc["input_ids"] == pad
            first = is_pad.float().argmax(dim=1)
            ar = torch.arange(enc["input_ids"].size(1)).unsqueeze(0)
            mask = (ar <= first.unsqueeze(1)).long()
            mask[~is_pad.any(dim=1)] = 1
            feats = model.get_text_features(input_ids=enc["input_ids"].to(device), attention_mask=mask.to(device))
            return torch.nn.functional.normalize(feats.float(), dim=-1)

        q, tta_mode = encode_text_query(encode_fn, args.text, mode=args.text_tta)
        if tta_mode != "none":
            variations = expand_text_query(args.text, mode=args.text_tta)
            print(f"\n[QUERY] text: {args.text!r} (TTA mode: {args.text_tta}, {len(variations)} variants)", flush=True)
            for vi, v in enumerate(variations, 1):
                print(f"  [{vi:02d}] {v}")
        else:
            print(f"\n[QUERY] text: {args.text!r}", flush=True)
        query_title = f"Text Query: {args.text!r} (TTA: {args.text_tta})"

    elif args.query:
        if not os.path.exists(args.query):
            log(f"Query image not found: {args.query}")
            return 1
        with Image.open(args.query) as q_im:
            q_rgb = q_im.convert("RGB")
            print(f"\n[QUERY] image: {args.query} ({q_rgb.size[0]}x{q_rgb.size[1]})", flush=True)
            t_img = img_transform(q_rgb).unsqueeze(0).to(device, dtype=dtype)
            feats = model.get_image_features(pixel_values=t_img)
            if getattr(model, "frame_mixer", None) is not None:
                feats = model.mix_frame_features(feats[:, None])[:, 0]
            q = torch.nn.functional.normalize(feats.float(), dim=-1)
        query_title = f"Image Query: {os.path.basename(args.query)}"
    else:
        log("No query given. Knowledge base is ready.")
        return 0

    # ---- 3. Similarity Search & Ranking -------------------------------- #
    out_dir = args.out_dir
    if not out_dir:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", args.text or os.path.splitext(os.path.basename(args.query))[0]).strip("_")[:40]
        out_dir = os.path.join(PROJECT_DIR, "scrape", "siglip_runs", "results", f"combined_{slug}")

    run_query(emb, items, q, tta_mode, top=args.top, device=device, out_dir=out_dir, query_title=query_title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
