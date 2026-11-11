#!/usr/bin/env python3
"""Live Instant Search across Curated Video Embeddings (Text & Reverse Image Search).

Features:
  - Max-Match Ranking: Searches all 9 tiles and ranks videos by their highest matching frame.
  - Displays which exact tile (0-8) matched the query and crops the winning frame.
  - Fallback to center-action index if 9-tile index is still building.
  - Text, Reverse Image, and Hybrid (Image + Text) search.
  - Automatically exports ranked results & cropped winning frames into scrape/search_results.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import sqlite3
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn.functional as F
from torchvision import transforms
import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EMB_DIR = os.path.join(PROJECT_DIR, "scrape", "embeddings_v2")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_DIR, "scrape", "search_results")
DEFAULT_WEIGHTS = os.path.join(PROJECT_DIR, "scrape", "siglip_runs", "eva02_curated_v2", "best.pt")
DEFAULT_MODEL = "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])


def export_results_images(results: list[dict], results_dir: str) -> int:
    """Clears out the results directory and copies the top retrieved images and winning tile crops."""
    os.makedirs(results_dir, exist_ok=True)
    for fname in os.listdir(results_dir):
        fpath = os.path.join(results_dir, fname)
        try:
            if os.path.isfile(fpath) or os.path.islink(fpath):
                os.unlink(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath)
        except Exception:
            pass

    copied = 0
    for r in results:
        src = r.get("path")
        if src and os.path.exists(src):
            ext = os.path.splitext(src)[1] or ".jpg"
            tile_str = f"_tile{r['winning_tile']}" if "winning_tile" in r else ""
            dst_name = f"rank_{r['rank']:02d}_sim_{r['score']:.4f}{tile_str}_{r['eid']}{ext}"
            dst = os.path.join(results_dir, dst_name)
            try:
                shutil.copy2(src, dst)
                copied += 1

                # If winning tile is known, also export cropped winning frame
                if "winning_tile" in r and r["winning_tile"] is not None:
                    k = r["winning_tile"]
                    row, col = divmod(k, 3)
                    with Image.open(src) as im:
                        w, h = im.size
                        box = (col * w // 3, row * h // 3, (col + 1) * w // 3, (row + 1) * h // 3)
                        crop_im = im.crop(box)
                        crop_dst = os.path.join(results_dir, f"rank_{r['rank']:02d}_sim_{r['score']:.4f}_MATCH_FRAME_{r['eid']}.jpg")
                        crop_im.save(crop_dst, quality=92)
            except Exception:
                pass
    return copied


def get_live_count(state_file: str, default_total: int = 147412) -> tuple[int, int]:
    completed = 0
    total = default_total
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                st = json.load(f)
                completed = st.get("completed_items", 0)
                total = st.get("total_items", total)
        except Exception:
            pass
    return completed, total


def rerank_candidates_9tiles(
    candidate_indices: list[int],
    query_vec: torch.Tensor,
    model,
    conn: sqlite3.Connection,
    device: str = "cuda",
    chunk_size: int = 6,
) -> tuple[list[int], list[float], list[int]]:
    if not candidate_indices:
        return [], [], []

    cur = conn.cursor()
    placeholders = ",".join("?" * len(candidate_indices))
    cur.execute(f"SELECT idx, asset_path FROM video_metadata WHERE idx IN ({placeholders})", candidate_indices)
    path_map = {r[0]: r[1] for r in cur.fetchall()}

    ordered_paths = [path_map.get(i) for i in candidate_indices]

    def load_and_prep(path: Optional[str]) -> torch.Tensor:
        if not path or not os.path.exists(path):
            return torch.zeros(9, 3, 336, 336)
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                w, h = img.size
                tw, th = w // 3, h // 3
                crops = []
                for r in range(3):
                    for c in range(3):
                        tile = img.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
                        crops.append(IMAGE_TRANSFORM(tile))
                return torch.stack(crops, dim=0)
        except Exception:
            return torch.zeros(9, 3, 336, 336)

    # 1. Parallel loading & cropping
    with ThreadPoolExecutor(max_workers=8) as ex:
        all_tiles = list(ex.map(load_and_prep, ordered_paths))

    # 2. Chunked GPU inference (chunk_size = 6 videos = 54 tiles)
    all_tile_feats = []
    use_half = (device == "cuda")
    for i in range(0, len(all_tiles), chunk_size):
        chunk = all_tiles[i : i + chunk_size]
        chunk_tensor = torch.stack(chunk, dim=0).to(device)
        if use_half:
            chunk_tensor = chunk_tensor.half()
        n_c = chunk_tensor.size(0)
        flat = chunk_tensor.view(n_c * 9, 3, 336, 336)
        with torch.no_grad():
            with torch.amp.autocast(device if device == "cuda" else "cpu"):
                feats = model.encode_image(flat)
                feats = F.normalize(feats.float(), dim=-1)
        all_tile_feats.append(feats.view(n_c, 9, 768))

    tile_feats = torch.cat(all_tile_feats, dim=0)  # [N_cands, 9, 768]

    # 3. MIL Max-Match
    sims = torch.einsum("d,nkd->nk", query_vec.squeeze(0), tile_feats)  # [N_cands, 9]
    max_scores, best_tiles = sims.max(dim=-1)

    # 4. Sort descending
    sorted_scores, sort_order = torch.sort(max_scores, descending=True)
    sorted_indices = [candidate_indices[i] for i in sort_order.tolist()]
    sorted_winning_tiles = [best_tiles[i].item() for i in sort_order.tolist()]
    sorted_scores_list = sorted_scores.tolist()

    return sorted_indices, sorted_scores_list, sorted_winning_tiles


def load_search_engine(
    emb_dir: str = DEFAULT_EMB_DIR,
    weights: str = DEFAULT_WEIGHTS,
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    mode: str = "rerank"
):
    metadata_file = os.path.join(emb_dir, "metadata_147k.sqlite3")
    emb_9tiles_file = os.path.join(emb_dir, "keyframes_9tiles_147k.npy")
    state_9tiles_file = os.path.join(emb_dir, "keyframes_9tiles_state.json")
    emb_1tile_file = os.path.join(emb_dir, "globals_147k.npy")
    state_1tile_file = os.path.join(emb_dir, "encoding_state.json")

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata index not found in {emb_dir}")

    # Load 1-tile global mmap
    completed_1, total = get_live_count(state_1tile_file)
    mmap_1tile = np.memmap(emb_1tile_file, dtype="float16", mode="r", shape=(total, 768))
    globals_gpu = None
    if device == "cuda":
        print(f"[*] Caching {completed_1:,} global vectors in GPU VRAM for instant Stage 1 retrieval...")
        globals_gpu = torch.from_numpy(mmap_1tile[:completed_1].copy()).to(device).half()

    # Load 9-tile offline mmap if available
    mmap_9tiles = None
    completed_9 = 0
    if os.path.exists(emb_9tiles_file):
        completed_9, _ = get_live_count(state_9tiles_file)
        if completed_9 > 0:
            mmap_9tiles = np.memmap(emb_9tiles_file, dtype="float16", mode="r", shape=(total, 9, 768))

    # Load multimodal model
    print(f"[*] Loading EVA-02 model ({model_name})...")
    model, _, _ = open_clip.create_model_and_transforms(model_name)
    if os.path.exists(weights):
        ckpt = torch.load(weights, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    if device == "cuda":
        model = model.to(device).half().eval()
    else:
        model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    conn = sqlite3.connect(metadata_file)
    return model, tokenizer, mmap_1tile, globals_gpu, mmap_9tiles, completed_1, completed_9, total, conn


def encode_query_image(image_path: str, model, device: str = "cuda") -> torch.Tensor:
    """Encodes an image query into the 768-d normalized embedding space."""
    with Image.open(image_path) as im:
        w, h = im.size
        if w > 600 and h > 300 and abs(w / h - 16 / 9) < 0.2:
            crop_box = (w // 3, h // 3, 2 * w // 3, 2 * h // 3)
            tile = im.crop(crop_box).convert("RGB")
        else:
            tile = im.convert("RGB")

        tensor = IMAGE_TRANSFORM(tile).unsqueeze(0).to(device)
        if device == "cuda":
            tensor = tensor.half()

    with torch.no_grad():
        with torch.amp.autocast(device if device == "cuda" else "cpu"):
            img_vec = model.encode_image(tensor)
            img_vec = F.normalize(img_vec.float(), dim=-1)
    return img_vec.float()


def search(
    query_text: Optional[str],
    query_image: Optional[str],
    top_k: int,
    model,
    tokenizer,
    mmap_1tile,
    globals_gpu,
    mmap_9tiles,
    completed_1: int,
    completed_9: int,
    conn: sqlite3.Connection,
    mode: str = "rerank",
    device: str = "cuda"
) -> Tuple[list[dict], float, int, str]:
    t0 = time.time()
    vecs = []

    # 1. Encode image if provided
    if query_image and os.path.exists(query_image):
        img_vec = encode_query_image(query_image, model, device=device)
        vecs.append(img_vec)

    # 2. Encode text if provided
    if query_text:
        text_tokens = tokenizer([query_text]).to(device)
        with torch.no_grad():
            text_vec = model.encode_text(text_tokens)
            text_vec = F.normalize(text_vec.float(), dim=-1)
            vecs.append(text_vec.float())

    if not vecs:
        raise ValueError("Must provide either a text query or an image path.")

    if len(vecs) == 2:
        query_vec = F.normalize((vecs[0] * 0.7 + vecs[1] * 0.3).float(), dim=-1)
    else:
        query_vec = vecs[0].float()

    # 3. Compute Scores
    with torch.no_grad():
        if mode == "center":
            if globals_gpu is not None:
                sims = (query_vec.half() @ globals_gpu.T).squeeze(0).float()
            else:
                live_vectors = torch.from_numpy(mmap_1tile[:completed_1].copy()).to(device).float()
                sims = (query_vec @ live_vectors.T).squeeze(0)
            top_scores, top_indices = torch.topk(sims, k=min(top_k, len(sims)))
            winning_tiles = [4] * len(top_indices)
            top_indices = top_indices.tolist()
            top_scores = top_scores.tolist()
            count = completed_1
            mode_desc = "Fast Global (Center-Tile)"

        elif mode == "offline" and completed_9 > 0 and mmap_9tiles is not None:
            live_vectors = torch.from_numpy(mmap_9tiles[:completed_9].copy()).to(device).float()
            sims = torch.einsum('d,nkd->nk', query_vec.squeeze(0), live_vectors)
            max_scores, best_tiles = sims.max(dim=-1)
            top_scores, top_indices = torch.topk(max_scores, k=min(top_k, len(live_vectors)))
            winning_tiles = [best_tiles[idx].item() for idx in top_indices]
            top_indices = top_indices.tolist()
            top_scores = top_scores.tolist()
            count = completed_9
            mode_desc = "Offline Indexed 9-Tile"

        else:
            # Online 9-Tile Rerank (Default)
            if globals_gpu is not None:
                sims = (query_vec.half() @ globals_gpu.T).squeeze(0).float()
            else:
                live_vectors = torch.from_numpy(mmap_1tile[:completed_1].copy()).to(device).float()
                sims = (query_vec @ live_vectors.T).squeeze(0)

            num_candidates = min(max(top_k, 24), 36, completed_1)
            stage1_scores, stage1_indices = torch.topk(sims, k=num_candidates)
            cand_indices = stage1_indices.tolist()

            reranked_indices, reranked_scores, winning_tiles_list = rerank_candidates_9tiles(
                candidate_indices=cand_indices,
                query_vec=query_vec,
                model=model,
                conn=conn,
                device=device,
                chunk_size=6,
            )

            top_indices = reranked_indices[:top_k]
            top_scores = reranked_scores[:top_k]
            winning_tiles = winning_tiles_list[:top_k]
            count = completed_1
            mode_desc = f"Online 9-Tile Rerank (Top {num_candidates})"

    query_ms = (time.time() - t0) * 1000

    cur = conn.cursor()
    results = []
    for score, idx, w_tile in zip(top_scores, top_indices, winning_tiles):
        cur.execute("SELECT eid, title, clothing_tags, position_tags, asset_path FROM video_metadata WHERE idx = ?", (idx,))
        row = cur.fetchone()
        if row:
            results.append({
                "rank": len(results) + 1,
                "score": score,
                "winning_tile": w_tile,
                "idx": idx,
                "eid": row[0],
                "title": row[1],
                "clothing": json.loads(row[2]) if row[2] else [],
                "posture": json.loads(row[3]) if row[3] else [],
                "path": row[4],
            })

    return results, query_ms, count, mode_desc


def is_image_path(text: str) -> bool:
    clean = text.strip('"\'')
    return os.path.exists(clean) and os.path.isfile(clean) and any(clean.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(description="Live text and reverse image search with 9-tile Max Match")
    parser.add_argument("query", nargs="*", help="Text query string or image file path")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to image for reverse image search")
    parser.add_argument("--top-k", type=int, default=12, help="Number of results to return (default: 12)")
    parser.add_argument("--mode", choices=["rerank", "center", "offline"], default="rerank", help="Search ranking mode")
    parser.add_argument("--emb-dir", default=DEFAULT_EMB_DIR, help="Path to embeddings directory")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Directory to export result images")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    model, tokenizer, mmap_1tile, globals_gpu, mmap_9tiles, completed_1, completed_9, total, conn = load_search_engine(
        emb_dir=args.emb_dir,
        device=args.device,
        mode=args.mode
    )

    raw_query = " ".join(args.query).strip() if args.query else ""
    query_image = args.image
    query_text = None

    if raw_query:
        if is_image_path(raw_query):
            query_image = raw_query.strip('"\'')
        else:
            query_text = raw_query

    if query_text or query_image:
        mode_desc = f"Image: {os.path.basename(query_image)}" if query_image else f"Text: \"{query_text}\""
        if query_image and query_text:
            mode_desc = f"Hybrid (Image: {os.path.basename(query_image)} + Text: \"{query_text}\")"
        print(f"\n[Searching via mode='{args.mode}' | {mode_desc}]")

        results, query_ms, count, active_mode = search(
            query_text=query_text,
            query_image=query_image,
            top_k=args.top_k,
            model=model,
            tokenizer=tokenizer,
            mmap_1tile=mmap_1tile,
            globals_gpu=globals_gpu,
            mmap_9tiles=mmap_9tiles,
            completed_1=completed_1,
            completed_9=completed_9,
            conn=conn,
            mode=args.mode,
            device=args.device
        )
        copied = export_results_images(results, args.results_dir)
        print(f"[*] Searched {count:,} videos ({active_mode}) in {query_ms:.2f} ms:")
        print(f"[*] Copied {copied} images to: {args.results_dir}\n")
        for r in results:
            pos_str = f" [{', '.join(r['posture'])}]" if r['posture'] else ""
            clo_str = f" ({', '.join(r['clothing'])})" if r['clothing'] else ""
            tile_info = f" [Frame #{r['winning_tile']}]" if "winning_tile" in r else ""
            print(f"#{r['rank']:02d} [Score: {r['score']:.4f}]{tile_info}{pos_str}{clo_str} - {r['title'][:65]}")
        conn.close()
        return

    print(f"\n[*] Results directory: {args.results_dir}")
    print("=" * 70)
    print(f"INTERACTIVE SEARCH SHELL (Mode: {args.mode})")
    print("  - Text query: e.g. 'maid uniform doggystyle'")
    print("  - Reverse image: Drag & drop an image file into terminal")
    print("  - Hybrid: type 'path/to/image.jpg + text prompt'")
    print("  - Type 'exit' to quit")
    print("=" * 70 + "\n")

    while True:
        try:
            line = input("Search > ").strip()
            if not line or line.lower() in ("exit", "quit", "q"):
                break

            q_text = None
            q_img = None

            if "+" in line:
                parts = line.split("+", 1)
                p0 = parts[0].strip().strip('"\'')
                p1 = parts[1].strip()
                if is_image_path(p0):
                    q_img = p0
                    q_text = p1
                else:
                    q_text = line
            elif is_image_path(line):
                q_img = line.strip('"\'')
            else:
                q_text = line

            results, query_ms, count, active_mode = search(
                query_text=q_text,
                query_image=q_img,
                top_k=args.top_k,
                model=model,
                tokenizer=tokenizer,
                mmap_1tile=mmap_1tile,
                globals_gpu=globals_gpu,
                mmap_9tiles=mmap_9tiles,
                completed_1=completed_1,
                completed_9=completed_9,
                conn=conn,
                mode=args.mode,
                device=args.device
            )
            copied = export_results_images(results, args.results_dir)
            print(f"[*] Searched {count:,} videos ({active_mode}) in {query_ms:.2f} ms (Copied {copied} images to {args.results_dir}):\n")
            for r in results:
                pos_str = f" [{', '.join(r['posture'])}]" if r['posture'] else ""
                clo_str = f" ({', '.join(r['clothing'])})" if r['clothing'] else ""
                tile_info = f" [Frame #{r['winning_tile']}]" if "winning_tile" in r else ""
                print(f"#{r['rank']:02d} [Score: {r['score']:.4f}]{tile_info}{pos_str}{clo_str} - {r['title'][:65]}")
            print()
        except (KeyboardInterrupt, EOFError):
            break

    conn.close()


if __name__ == "__main__":
    main()
