#!/usr/bin/env python3
"""Interactive Web Viewer for Semantic Video Retrieval (FastAPI + Modern Dark UI).

Features:
  - Text search with instant sub-millisecond retrieval across 147,412 videos.
  - Drag & drop Reverse Image Search (and hybrid text+image search).
  - Mode toggle: 9-Tile Max Match vs Fast Global Index.
  - Interactive results grid: visual preview, similarity scores, posture/clothing pills.
  - Dynamic frame view: switch between full 3x3 storyboard and exact matching frame.
  - "Find Similar" one-click button on any video card.
  - Runs locally on http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import gc
import io
import json
import os
import sqlite3
import time
from typing import Optional

import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn.functional as F
from torchvision import transforms
import open_clip

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_scrape = os.path.join(PROJECT_DIR, "scrape")
if sys_scrape not in sys.path:
    sys.path.append(sys_scrape)
from hover_pacing import fetch_and_analyze_hover_pacing
DEFAULT_EMB_DIR = os.path.join(PROJECT_DIR, "scrape", "embeddings_v2")
DEFAULT_WEIGHTS = os.path.join(PROJECT_DIR, "scrape", "siglip_runs", "eva02_curated_v2", "best.pt")
DEFAULT_MODEL = "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

app = FastAPI(title="Semantic Video Retrieval Engine", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global runtime state
STATE = {
    "model": None,
    "tokenizer": None,
    "mmaps_1tile": [],
    "globals_gpu": None,
    "mmaps_9tiles": [],
    "tiles_gpu_flat": None,
    "tiles_gpu_count": 0,
    "metadata_db": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "total_1tile": 0,
    "total_9tiles": 0,
    "emb_dir": DEFAULT_EMB_DIR,
}


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


def _open_index_pair(
    base_path: str,
    live_path: str,
    state_file: str,
    row_tail_shape: tuple[int, ...],
    max_sql_rows: int,
) -> tuple[list[tuple[np.memmap, int]], int]:
    """Open base .npy memmap plus optional live_*.bin memmap, bounded strictly to SQLite rows."""
    row_bytes = int(np.prod(row_tail_shape)) * 2
    segments: list[tuple[np.memmap, int]] = []
    remaining = max_sql_rows

    if os.path.exists(base_path):
        file_rows = os.path.getsize(base_path) // row_bytes
        c_state, t_state = get_live_count(state_file, default_total=file_rows)
        valid_base = min(file_rows, c_state if c_state > 0 else file_rows, remaining)
        if valid_base > 0:
            shape = (max(file_rows, t_state),) + row_tail_shape
            mm = np.memmap(base_path, dtype="float16", mode="r", shape=shape)
            segments.append((mm, valid_base))
            remaining -= valid_base

    if remaining > 0 and os.path.exists(live_path):
        live_rows = os.path.getsize(live_path) // row_bytes
        valid_live = min(live_rows, remaining)
        if valid_live > 0:
            mm_live = np.memmap(live_path, dtype="float16", mode="r", shape=(live_rows,) + row_tail_shape)
            segments.append((mm_live, valid_live))
            remaining -= valid_live

    total_rows = max_sql_rows - remaining
    return segments, total_rows


def _copy_segments_to_gpu(
    segments: list[tuple[np.memmap, int]],
    total_rows: int,
    row_tail_shape: tuple[int, ...],
    device: str,
    chunk_rows: int = 25000,
) -> torch.Tensor:
    """Stream base + live memmap segments sequentially into a single contiguous FP16 GPU tensor."""
    target_shape = (total_rows,) + row_tail_shape
    gpu_buf = torch.empty(target_shape, dtype=torch.float16, device=device)
    cursor = 0
    for mm, count in segments:
        for start in range(0, count, chunk_rows):
            end = min(start + chunk_rows, count)
            span = end - start
            gpu_buf[cursor:cursor + span].copy_(torch.from_numpy(mm[start:end].copy()), non_blocking=False)
            cursor += span
    return gpu_buf


def init_engine(emb_dir: str = DEFAULT_EMB_DIR, weights: str = DEFAULT_WEIGHTS, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
    print(f"[*] Initializing Retrieval Engine on {device}...")
    STATE["emb_dir"] = emb_dir
    STATE["device"] = device

    meta_db = os.path.join(emb_dir, "metadata_147k.sqlite3")
    if not os.path.exists(meta_db):
        raise FileNotFoundError(f"Metadata index not found at {meta_db}")
    STATE["metadata_db"] = meta_db

    conn = sqlite3.connect(meta_db)
    sql_row = conn.execute("SELECT COUNT(*), COALESCE(MAX(idx), -1) FROM video_metadata").fetchone()
    conn.close()
    max_sql_rows = min(int(sql_row[0]), int(sql_row[1]) + 1) if sql_row else 0

    # Reset cached tensors
    STATE["globals_gpu"] = None
    STATE["tiles_gpu_flat"] = None
    STATE["tiles_gpu_count"] = 0

    # 1. Load model first so free VRAM calculation is exact (cast .half() on CPU before .to(device)!)
    print(f"[*] Loading EVA-02 model ({model_name})...")
    model, _, _ = open_clip.create_model_and_transforms(model_name)
    if os.path.exists(weights):
        ckpt = torch.load(weights, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        del ckpt, state_dict
    model = model.half().to(device).eval() if device == "cuda" else model.to(device).eval()
    STATE["model"] = model
    STATE["tokenizer"] = open_clip.get_tokenizer(model_name)

    # 2. Load 1-tile index (base globals_147k.npy + appended live_globals.bin)
    segs_1, c1 = _open_index_pair(
        os.path.join(emb_dir, "globals_147k.npy"),
        os.path.join(emb_dir, "live_globals.bin"),
        os.path.join(emb_dir, "encoding_state.json"),
        (768,),
        max_sql_rows,
    )
    STATE["mmaps_1tile"] = segs_1
    STATE["total_1tile"] = c1
    if c1 > 0:
        print(f"[*] Center-Tile Index: {c1:,} videos loaded ({len(segs_1)} shard(s)).")
        if device == "cuda":
            gb1 = (c1 * 768 * 2) / (1024 ** 3)
            print(f"[*] Caching {c1:,} center vectors to GPU VRAM ({gb1:.2f} GB)...")
            STATE["globals_gpu"] = _copy_segments_to_gpu(segs_1, c1, (768,), device)

    # 3. Load 9-tile index (base keyframes_9tiles_147k.npy + appended live_keyframes_9tiles.bin)
    segs_9, c9 = _open_index_pair(
        os.path.join(emb_dir, "keyframes_9tiles_147k.npy"),
        os.path.join(emb_dir, "live_keyframes_9tiles.bin"),
        os.path.join(emb_dir, "keyframes_9tiles_state.json"),
        (9, 768),
        max_sql_rows,
    )
    STATE["mmaps_9tiles"] = segs_9
    STATE["total_9tiles"] = c9
    if c9 > 0:
        need_bytes = c9 * 9 * 768 * 2
        need_gb = need_bytes / (1024 ** 3)
        print(f"[*] 9-Tile Max Match Index: {c9:,} videos ({c9 * 9:,} keyframes, {need_gb:.2f} GB) across {len(segs_9)} shard(s).")
        if device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            free_vram = need_bytes + (600 * 1024 * 1024)
            try:
                free_vram, _ = torch.cuda.mem_get_info()
            except Exception:
                pass
            if free_vram > need_bytes + (512 * 1024 * 1024):
                print(f"[*] Caching all {c9 * 9:,} keyframes to GPU VRAM ({need_gb:.2f} GB) for Exhaustive MIL...")
                STATE["tiles_gpu_flat"] = _copy_segments_to_gpu(segs_9, c9, (9, 768), device).view(c9 * 9, 768)
                STATE["tiles_gpu_count"] = c9
            else:
                print(
                    f"[*] VRAM headroom ({free_vram / 1024**3:.2f} GB free) < {need_gb + 0.5:.2f} GB needed; "
                    "using chunked GPU MIL over memory-mapped shards."
                )

    print("[*] Engine ready!")


def extract_dense_tokens_batch(batch_tensor: torch.Tensor, model, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs EVA-02 ViT forward_features + projection head to extract both:
      - cls_vecs: [B, 768] L2-normalized global frame embeddings
      - patch_vecs: [B, 576, 768] L2-normalized 24x24 dense spatial patch embeddings
    """
    with torch.no_grad():
        with torch.amp.autocast(device if device == "cuda" else "cpu"):
            feats = model.visual.trunk.forward_features(batch_tensor)  # [B, 577, 1024]
            projected = model.visual.trunk.head(model.visual.trunk.fc_norm(feats))  # [B, 577, 768]
            projected = F.normalize(projected.float(), dim=-1)
    cls_vecs = projected[:, 0, :]       # [B, 768]
    patch_vecs = projected[:, 1:, :]    # [B, 576, 768]
    return cls_vecs, patch_vecs


def rerank_candidates_dense_patches(
    candidate_indices: list[int],
    winning_tiles: list[int],
    stage1_scores: list[float],
    query_vec: torch.Tensor,
    model,
    metadata_db: str,
    device: str = "cuda",
    query_patches: Optional[torch.Tensor] = None,
    query_patch_weights: Optional[torch.Tensor] = None,
    chunk_size: int = 12,
    top_patch_k: int = 36,
) -> tuple[list[int], list[float], list[int]]:
    """
    Stage-2 ViT Dense Spatial Patch Token Reranker (24x24 = 576 spatial patches per winning keyframe).
    Given Stage-1 candidate videos and their winning tile (0..8), crops the winning frame,
    extracts all 576 spatial patch tokens from EVA-02, and computes fine-grained local patch similarity
    (plus ColBERT-style Patch-to-Patch MaxSim when a query image is provided).
    Returns: (reranked_indices, reranked_scores, reranked_winning_tiles)
    """
    if not candidate_indices:
        return [], [], []

    conn = sqlite3.connect(metadata_db)
    cur = conn.cursor()
    placeholders = ",".join("?" * len(candidate_indices))
    cur.execute(f"SELECT idx, asset_path FROM video_metadata WHERE idx IN ({placeholders})", candidate_indices)
    path_map = {r[0]: r[1] for r in cur.fetchall()}
    conn.close()

    ordered_tasks = [(path_map.get(idx), tile_idx) for idx, tile_idx in zip(candidate_indices, winning_tiles)]

    def load_winning_tile(task: tuple[Optional[str], int]) -> torch.Tensor:
        path, tile_idx = task
        if not path or not os.path.exists(path):
            return torch.zeros(3, 336, 336)
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                w, h = img.size
                r, c = divmod(tile_idx, 3)
                box = (c * w // 3, r * h // 3, (c + 1) * w // 3, (r + 1) * h // 3)
                tile = img.crop(box)
                return IMAGE_TRANSFORM(tile)
        except Exception:
            return torch.zeros(3, 336, 336)

    # 1. Parallel loading of each candidate's winning keyframe crop
    with ThreadPoolExecutor(max_workers=8) as ex:
        tile_tensors = list(ex.map(load_winning_tile, ordered_tasks))

    # 2. Chunked GPU forward pass to extract [N_cands, 576, 768] dense spatial patch tokens
    all_cls = []
    all_patches = []
    use_half = (device == "cuda")
    for i in range(0, len(tile_tensors), chunk_size):
        chunk = tile_tensors[i : i + chunk_size]
        batch = torch.stack(chunk, dim=0).to(device)
        if use_half:
            batch = batch.half()
        cls_v, patch_v = extract_dense_tokens_batch(batch, model, device=device)
        all_cls.append(cls_v)
        all_patches.append(patch_v)

    cand_cls = torch.cat(all_cls, dim=0)          # [M, 768]
    cand_patches = torch.cat(all_patches, dim=0)  # [M, 576, 768]

    # 3. Dense Spatial Patch Scoring
    q_vec = query_vec.squeeze(0).float()  # [768]
    # Prompt/Query-to-Patch similarity across all 576 spatial locations in each frame: [M, 576]
    patch_sims = torch.einsum("d,mpd->mp", q_vec, cand_patches)
    # Average over Top-K most responsive spatial patches (e.g. top 36 patches = top 6.25% local region)
    topk_patch_vals, _ = torch.topk(patch_sims, k=min(top_patch_k, patch_sims.size(1)), dim=-1)
    local_patch_score = topk_patch_vals.mean(dim=-1)  # [M]

    # If the user uploaded/selected a query image, also compute ColBERT-style Patch-to-Patch MaxSim
    if query_patches is not None and query_patch_weights is not None:
        # query_patches: [P_q, 768], cand_patches: [M, 576, 768] -> sim matrix [M, P_q, 576]
        p2p_sim = torch.einsum("qd,mpd->mqp", query_patches.float(), cand_patches)
        max_per_qpatch, _ = p2p_sim.max(dim=-1)  # [M, P_q]
        colbert_maxsim = (max_per_qpatch * query_patch_weights.unsqueeze(0)).sum(dim=-1)  # [M]
        dense_score = 0.50 * local_patch_score + 0.50 * colbert_maxsim
    else:
        dense_score = local_patch_score

    # Combine Stage-1 global keyframe similarity with Stage-2 Dense Spatial Patch similarity
    s1_tensor = torch.tensor(stage1_scores, device=device, dtype=torch.float32)
    final_scores = 0.55 * s1_tensor + 0.45 * dense_score

    sorted_scores, sort_order = torch.sort(final_scores, descending=True)
    order_list = sort_order.tolist()
    sorted_indices = [candidate_indices[i] for i in order_list]
    sorted_winning_tiles = [winning_tiles[i] for i in order_list]
    sorted_scores_list = sorted_scores.tolist()

    return sorted_indices, sorted_scores_list, sorted_winning_tiles


def encode_image_pil(
    im: Image.Image,
    model,
    device: str = "cuda",
    return_patches: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    tile = im.convert("RGB")
    tensor = IMAGE_TRANSFORM(tile).unsqueeze(0).to(device)
    if device == "cuda":
        tensor = tensor.half()
    if return_patches:
        cls_vec, patch_vecs = extract_dense_tokens_batch(tensor, model, device=device)
        # Select top 64 most salient foreground patches (highest alignment with global CLS token)
        salience = torch.einsum("bd,bpd->bp", cls_vec, patch_vecs).squeeze(0)  # [576]
        top_sal, top_idx = torch.topk(salience, k=64)
        salient_patches = patch_vecs.squeeze(0)[top_idx]  # [64, 768]
        salient_weights = F.softmax(top_sal * 8.0, dim=-1)  # [64]
        return cls_vec.float(), salient_patches.float(), salient_weights.float()
    with torch.no_grad():
        with torch.amp.autocast(device if device == "cuda" else "cpu"):
            img_vec = model.encode_image(tensor)
            img_vec = F.normalize(img_vec.float(), dim=-1)
    return img_vec.float(), None, None


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        content=HTML_CONTENT,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )


@app.get("/api/status")
def status():
    c1 = STATE["total_1tile"]
    c9 = STATE["total_9tiles"]
    return {
        "device": STATE["device"],
        "count_1tile": c1,
        "total_1tile": c1,
        "count_9tiles": c9,
        "total_9tiles": c9,
        "has_9tiles": c9 > 0,
    }


def _search_1tile(query_vec: torch.Tensor, top_k: int, device: str) -> tuple[list[int], list[float], int]:
    """Rank all loaded 1-tile shards (from GPU cache or chunked mmap)."""
    c1 = STATE["total_1tile"]
    if STATE.get("globals_gpu") is not None:
        sims = (query_vec.half() @ STATE["globals_gpu"].T).squeeze(0).float()
        top_scores, top_indices = torch.topk(sims, k=min(top_k, len(sims)))
        return top_indices.tolist(), top_scores.tolist(), c1

    q_half = query_vec.squeeze(0).half()
    chunk_size = 50000
    score_chunks = []
    for mm, count in STATE["mmaps_1tile"]:
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            vecs = torch.from_numpy(mm[start:end].copy()).to(device).half()
            score_chunks.append(torch.mv(vecs, q_half).float())
    all_sims = torch.cat(score_chunks)
    top_scores, top_indices = torch.topk(all_sims, k=min(top_k, len(all_sims)))
    return top_indices.tolist(), top_scores.tolist(), c1


def _search_9tile(query_vec: torch.Tensor, top_k: int, device: str) -> tuple[list[int], list[float], list[int], int, str]:
    """Rank all loaded 9-tile bags via MIL max-pooling (GPU VRAM cache or chunked mmap)."""
    if STATE.get("tiles_gpu_flat") is not None:
        c = STATE["tiles_gpu_count"]
        sims = torch.mv(STATE["tiles_gpu_flat"], query_vec.squeeze(0).half()).view(c, 9).float()
        max_scores, best_tiles = sims.max(dim=-1)
        top_scores, top_indices = torch.topk(max_scores, k=min(top_k, c))
        winning_tiles = [best_tiles[idx].item() for idx in top_indices]
        label = f"Exhaustive 9-Tile MIL ({(c * 9) / 1e6:.2f}M Keyframes)"
        return top_indices.tolist(), top_scores.tolist(), winning_tiles, c, label

    c9 = STATE["total_9tiles"]
    if c9 > 0 and STATE.get("mmaps_9tiles"):
        chunk_size = 25000
        q_half = query_vec.squeeze(0).half()
        scores_all = []
        best_tiles_all = []
        for mm, count in STATE["mmaps_9tiles"]:
            for start in range(0, count, chunk_size):
                end = min(start + chunk_size, count)
                chunk_tensor = torch.from_numpy(mm[start:end].copy()).to(device).half()
                sims = torch.einsum("d,nkd->nk", q_half, chunk_tensor).float()
                max_s, b_tiles = sims.max(dim=-1)
                scores_all.append(max_s)
                best_tiles_all.append(b_tiles)
        all_scores = torch.cat(scores_all)
        all_best_tiles = torch.cat(best_tiles_all)
        top_scores, top_indices = torch.topk(all_scores, k=min(top_k, len(all_scores)))
        winning_tiles = [all_best_tiles[idx].item() for idx in top_indices]
        label = f"Chunked 9-Tile MIL ({(c9 * 9) / 1e6:.2f}M Keyframes)"
        return top_indices.tolist(), top_scores.tolist(), winning_tiles, c9, label

    idx_list, score_list, c1 = _search_1tile(query_vec, top_k, device)
    return idx_list, score_list, [4] * len(idx_list), c1, "Center Tile Fallback"


@app.get("/api/thumb/{eid}")
def get_thumbnail(eid: str, tile: Optional[int] = None):
    conn = sqlite3.connect(STATE["metadata_db"])
    cur = conn.cursor()
    cur.execute("SELECT asset_path FROM video_metadata WHERE eid = ?", (eid,))
    row = cur.fetchone()
    conn.close()

    if not row or not os.path.exists(row[0]):
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    path = row[0]
    if tile is not None and 0 <= tile <= 8:
        try:
            with Image.open(path) as im:
                w, h = im.size
                r, c = divmod(tile, 3)
                box = (c * w // 3, r * h // 3, (c + 1) * w // 3, (r + 1) * h // 3)
                crop_im = im.crop(box)
                buf = io.BytesIO()
                crop_im.save(buf, format="JPEG", quality=88)
                return Response(content=buf.getvalue(), media_type="image/jpeg")
        except Exception:
            pass

    with open(path, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/pacing/{eid}")
def get_pacing(eid: str):
    conn = sqlite3.connect(STATE["metadata_db"])
    cur = conn.cursor()
    cur.execute("SELECT preview_url FROM video_metadata WHERE eid = ?", (eid,))
    row = cur.fetchone()
    conn.close()

    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Preview video URL not found")

    res = fetch_and_analyze_hover_pacing(eid, row[0])
    if not res:
        raise HTTPException(status_code=502, detail="Failed to analyze hover preview")
    return res


@app.post("/api/search")
async def search_api(
    query: Optional[str] = Form(None),
    mode: str = Form("auto"),
    top_k: int = Form(24),
    image_b64: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    model = STATE["model"]
    tokenizer = STATE["tokenizer"]
    device = STATE["device"]

    vecs = []
    query_patches = None
    query_patch_weights = None
    need_dense_patches = mode in ("dense_rerank", "online_rerank")

    # 1. Process uploaded image (either base64 string or multipart file)
    if image_b64:
        try:
            header, _, b64_data = image_b64.partition(",")
            raw_bytes = base64.b64decode(b64_data if b64_data else header)
            im = Image.open(io.BytesIO(raw_bytes))
            img_vec, query_patches, query_patch_weights = encode_image_pil(
                im, model, device=device, return_patches=need_dense_patches
            )
            vecs.append(img_vec)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
    elif file and file.filename:
        content = await file.read()
        try:
            im = Image.open(io.BytesIO(content))
            img_vec, query_patches, query_patch_weights = encode_image_pil(
                im, model, device=device, return_patches=need_dense_patches
            )
            vecs.append(img_vec)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # 2. Process query text
    clean_query = (query or "").strip()
    if clean_query:
        tokens = tokenizer([clean_query]).to(device)
        with torch.no_grad():
            text_vec = model.encode_text(tokens)
            text_vec = F.normalize(text_vec.float(), dim=-1)
            vecs.append(text_vec.float())

    if not vecs:
        raise HTTPException(status_code=400, detail="Must provide either a text query or an image file.")

    # 3. Hybrid blend
    if len(vecs) == 2:
        query_vec = F.normalize((vecs[0] * 0.7 + vecs[1] * 0.3).float(), dim=-1)
    else:
        query_vec = vecs[0].float()

    # 4. Determine mode & execute search
    c1 = STATE["total_1tile"]
    c9 = STATE["total_9tiles"]

    t0 = time.time()
    with torch.no_grad():
        if mode == "center":
            top_indices, top_scores, total_searched = _search_1tile(query_vec, top_k, device)
            winning_tiles = [4] * len(top_indices)
            active_mode = "Fast Center Tile (1-Tile Instant)"

        elif mode in ("dense_rerank", "online_rerank"):
            num_candidates = min(max(top_k * 2, 36), 60, c9 if c9 > 0 else c1)
            s1_indices_list, s1_scores_list, s1_tiles, total_searched, _ = _search_9tile(query_vec, num_candidates, device)

            # Stage 2: ViT Dense Spatial Patch Token Rerank (24x24 = 576 spatial patches per candidate winning frame)
            reranked_indices, reranked_scores, winning_tiles_list = rerank_candidates_dense_patches(
                candidate_indices=s1_indices_list,
                winning_tiles=s1_tiles,
                stage1_scores=s1_scores_list,
                query_vec=query_vec,
                model=model,
                metadata_db=STATE["metadata_db"],
                device=device,
                query_patches=query_patches,
                query_patch_weights=query_patch_weights,
                chunk_size=12,
                top_patch_k=36,
            )
            top_indices = reranked_indices[:top_k]
            top_scores = reranked_scores[:top_k]
            winning_tiles = winning_tiles_list[:top_k]
            active_mode = f"ViT Dense Spatial Patch Rerank (Top {len(s1_indices_list)} x 576 Patches)"

        else:
            top_indices, top_scores, winning_tiles, total_searched, active_mode = _search_9tile(query_vec, top_k, device)

    query_ms = (time.time() - t0) * 1000

    conn = sqlite3.connect(STATE["metadata_db"])
    cur = conn.cursor()
    results = []
    scores_list = top_scores if isinstance(top_scores, list) else top_scores.tolist()
    indices_list = top_indices if isinstance(top_indices, list) else top_indices.tolist()
    tiles_list = winning_tiles if isinstance(winning_tiles, list) else winning_tiles.tolist()
    for score, idx, w_tile in zip(scores_list, indices_list, tiles_list):
        cur.execute("SELECT eid, title, clothing_tags, position_tags, preview_url FROM video_metadata WHERE idx = ?", (idx,))
        row = cur.fetchone()
        if row:
            results.append({
                "rank": len(results) + 1,
                "score": round(score, 4),
                "winning_tile": w_tile,
                "idx": idx,
                "eid": row[0],
                "title": row[1],
                "clothing": json.loads(row[2]) if row[2] else [],
                "posture": json.loads(row[3]) if row[3] else [],
                "thumb_url": f"/api/thumb/{row[0]}",
                "tile_url": f"/api/thumb/{row[0]}?tile={w_tile}",
                "video_url": f"https://www.xvideos.com/video.{row[0]}/_",
                "preview_url": row[4] or "",
            })
    conn.close()

    return {
        "mode": active_mode,
        "searched": total_searched,
        "query_ms": round(query_ms, 2),
        "results": results,
    }


# =============================================================================
# Embedded Modern Dark-Mode UI
# =============================================================================

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Semantic Video Retrieval Engine</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0c0e14;
      --card-bg: #161922;
      --card-hover: #1e2230;
      --border: #262b3a;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --pill-pos: #3b82f6;
      --pill-clo: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: rgba(22, 25, 34, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 16px 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .logo {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .logo-badge {
      background: linear-gradient(135deg, #6366f1, #a855f7);
      color: #fff;
      font-size: 0.72rem;
      padding: 3px 8px;
      border-radius: 6px;
      font-weight: 600;
      text-transform: uppercase;
    }
    .status-badge {
      font-size: 0.82rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 10px #10b981;
    }
    main {
      flex: 1;
      max-width: 1600px;
      width: 100%;
      margin: 0 auto;
      padding: 32px 24px;
    }
    .search-panel {
      max-width: 900px;
      margin: 0 auto 36px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .search-row {
      display: flex;
      gap: 12px;
      align-items: center;
    }
    .search-input-wrap {
      flex: 1;
      position: relative;
    }
    .search-input {
      width: 100%;
      background: #0f121a;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 18px;
      color: #fff;
      font-size: 1.02rem;
      outline: none;
      transition: all 0.2s;
    }
    .search-input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }
    .btn {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 10px;
      padding: 14px 24px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
    }
    .btn:hover { background: var(--accent-hover); }
    .drop-btn {
      background: #0f121a;
      border: 1px dashed var(--border);
      color: var(--text-muted);
      border-radius: 10px;
      padding: 13px 18px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.9rem;
      transition: all 0.2s;
    }
    .drop-btn:hover {
      border-color: var(--accent);
      color: var(--text);
    }
    .image-preview-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: #1f2433;
      border: 1px solid var(--accent);
      border-radius: 8px;
      padding: 4px 10px;
      font-size: 0.8rem;
      margin-top: 10px;
    }
    .image-preview-chip img {
      width: 22px;
      height: 22px;
      object-fit: cover;
      border-radius: 4px;
    }
    .image-preview-chip .remove-img {
      cursor: pointer;
      color: #ef4444;
      font-weight: bold;
    }
    .controls-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    .mode-select {
      background: #0f121a;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 6px;
      outline: none;
      font-size: 0.85rem;
    }
    .meta-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      font-size: 0.9rem;
      color: var(--text-muted);
    }
    .results-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 24px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    }
    .card:hover {
      transform: translateY(-4px);
      border-color: var(--accent);
      box-shadow: 0 12px 24px rgba(0, 0, 0, 0.5);
    }
    .thumb-wrap {
      position: relative;
      width: 100%;
      padding-top: 56.25%; /* 16:9 */
      background: #000;
      overflow: hidden;
      display: block;
      cursor: pointer;
    }
    .thumb-img {
      position: absolute;
      top: 0; left: 0; width: 100%; height: 100%;
      object-fit: cover;
      transition: transform 0.25s, opacity 0.2s;
    }
    .thumb-wrap:hover .thumb-img {
      transform: scale(1.03);
    }
    .score-badge {
      position: absolute;
      top: 10px; left: 10px;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(6px);
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.3);
      z-index: 2;
    }
    .tile-badge {
      position: absolute;
      top: 10px; right: 10px;
      background: rgba(12, 14, 20, 0.88);
      border: 1px solid rgba(99, 102, 241, 0.45);
      backdrop-filter: blur(6px);
      padding: 3px 7px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 700;
      color: #c7d2fe;
      z-index: 3;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
    }
    .tile-matrix {
      display: grid;
      grid-template-columns: repeat(3, 8px);
      grid-gap: 2px;
      pointer-events: auto;
    }
    .matrix-cell {
      width: 8px;
      height: 6px;
      background: rgba(255, 255, 255, 0.22);
      border-radius: 1px;
      transition: background 0.12s, transform 0.12s;
      cursor: pointer;
    }
    .matrix-cell:hover {
      background: #38bdf8 !important;
      transform: scale(1.35);
    }
    .matrix-cell.active-win {
      background: #818cf8;
      box-shadow: 0 0 5px rgba(99, 102, 241, 0.9);
    }
    .hover-video {
      position: absolute;
      top: 0; left: 0; width: 100%; height: 100%;
      object-fit: cover;
      display: none;
      z-index: 1;
      pointer-events: none;
    }
    .pacing-container {
      position: absolute;
      bottom: 8px;
      left: 8px;
      right: 8px;
      display: flex;
      flex-direction: column;
      gap: 5px;
      z-index: 2;
      pointer-events: none;
    }
    .pacing-badge {
      align-self: flex-start;
      background: rgba(12, 14, 20, 0.88);
      backdrop-filter: blur(8px);
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 700;
      color: #fbbf24;
      border: 1px solid rgba(251, 191, 36, 0.35);
      display: inline-flex;
      align-items: center;
      gap: 6px;
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
      transition: color 0.15s, border-color 0.15s, box-shadow 0.08s ease-out;
    }
    .pacing-badge .live-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: currentColor;
      box-shadow: 0 0 6px currentColor;
      display: none;
      transform-origin: center center;
      transition: transform 0.04s ease-out, box-shadow 0.04s ease-out, background-color 0.04s ease-out;
    }
    .swatch-timeline {
      display: none;
      width: 100%;
      height: 4px;
      gap: 3px;
      border-radius: 2px;
      overflow: hidden;
      background: rgba(0, 0, 0, 0.6);
      padding: 1px;
    }
    .swatch-segment {
      flex: 1;
      height: 100%;
      border-radius: 1px;
      opacity: 0.55;
      transition: opacity 0.15s, transform 0.15s;
    }
    .swatch-segment.active {
      opacity: 1;
      transform: scaleY(1.6);
      box-shadow: 0 0 8px currentColor;
    }
    .card-body {
      padding: 16px;
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .card-title {
      font-size: 0.95rem;
      font-weight: 600;
      line-height: 1.35;
      margin-bottom: 12px;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      transition: color 0.2s;
    }
    .card-title:hover {
      color: #818cf8;
    }
    .tags-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 14px;
    }
    .pill {
      font-size: 0.72rem;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 4px;
      cursor: pointer;
      transition: filter 0.2s;
    }
    .pill:hover { filter: brightness(1.2); }
    .pill-pos { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .pill-clo { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .card-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 10px;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
      font-size: 0.8rem;
    }
    .toggle-frame-btn {
      background: none;
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: 6px;
      padding: 4px 8px;
      cursor: pointer;
      font-size: 0.75rem;
    }
    .toggle-frame-btn:hover { color: #fff; border-color: var(--accent); }
    .find-similar-btn {
      background: none;
      border: none;
      color: #818cf8;
      cursor: pointer;
      font-weight: 600;
    }
    .find-similar-btn:hover { text-decoration: underline; }
    .watch-btn {
      background: rgba(99, 102, 241, 0.15);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.35);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      text-decoration: none;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 3px;
    }
    .watch-btn:hover {
      background: var(--accent);
      color: #fff;
    }
    .loader {
      display: none;
      margin: 40px auto;
      text-align: center;
      color: var(--text-muted);
    }
    .spinner {
      width: 36px;
      height: 36px;
      border: 3px solid rgba(255, 255, 255, 0.1);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin: 0 auto 12px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

  <header>
    <div class="logo">
      <span>Neural Video Retrieval</span>
      <span class="logo-badge">EVA-02 v2</span>
    </div>
    <div class="status-badge" id="headerStatus">
      <div class="status-dot"></div>
      <span id="catalogCount">Connecting...</span>
    </div>
  </header>

  <main>
    <div class="search-panel">
      <form id="searchForm">
        <div class="search-row">
          <div class="search-input-wrap">
            <input type="text" id="queryInput" class="search-input" placeholder="Type prompt (e.g. 'maid uniform doggystyle', 'stockings heels')...">
          </div>
          <label class="drop-btn" title="Upload image for reverse search">
            <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
            <span>Image</span>
            <input type="file" id="fileInput" accept="image/*" style="display:none;">
          </label>
          <button type="submit" class="btn">Search</button>
        </div>

        <div id="imageChip" class="image-preview-chip" style="display:none;">
          <img id="chipThumb" alt="">
          <span id="chipName">image.jpg</span>
          <span class="remove-img" id="removeImg">&times;</span>
        </div>

        <div class="controls-row">
          <div>
            <span>Ranking Mode:</span>
            <select id="modeSelect" class="mode-select">
              <option value="9tile" selected>🎯 Exhaustive 9-Tile MIL (All 1.33M Keyframes ~8ms)</option>
              <option value="dense_rerank">🔬 Dense Patch Rerank (9-Tile MIL &rarr; 576 ViT Spatial Patches)</option>
              <option value="center">⚡ Fast Center Tile (1-Tile Instant ~2ms)</option>
            </select>
          </div>
          <div>
            <span>Results:</span>
            <select id="topKSelect" class="mode-select">
              <option value="12">12</option>
              <option value="24" selected>24</option>
              <option value="48">48</option>
              <option value="96">96</option>
            </select>
          </div>
        </div>
      </form>
    </div>

    <div class="meta-bar">
      <div id="resultStats">Ready to search.</div>
      <div id="latencyStats"></div>
    </div>

    <div id="loader" class="loader">
      <div class="spinner"></div>
      <div>Running neural similarity search...</div>
    </div>

    <div id="resultsGrid" class="results-grid"></div>
  </main>

  <script>
    if (window.location.href.endsWith('?')) {
      window.history.replaceState(null, '', '/');
    }
    let activeImageB64 = null;
    let pacingAbortCtrl = null;

    async function updateStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const c1 = data.count_1tile.toLocaleString();
        const c9 = data.count_9tiles.toLocaleString();
        document.getElementById('catalogCount').innerText = `${c1} videos ready (${c9} 9-tile indexed)`;
        if (data.count_9tiles > 0) {
          const kfMillions = ((data.count_9tiles * 9) / 1e6).toFixed(2);
          const opt9 = document.querySelector('#modeSelect option[value="9tile"]');
          if (opt9) opt9.text = `🎯 Exhaustive 9-Tile MIL (All ${kfMillions}M Keyframes)`;
        }
      } catch (e) {
        console.error(e);
      }
    }
    updateStatus();
    setInterval(updateStatus, 15000);

    const fileInput = document.getElementById('fileInput');
    const imageChip = document.getElementById('imageChip');
    const chipThumb = document.getElementById('chipThumb');
    const chipName = document.getElementById('chipName');
    const removeImg = document.getElementById('removeImg');

    function blobToNormalizedB64(fileOrBlob) {
      return new Promise((resolve, reject) => {
        const objUrl = URL.createObjectURL(fileOrBlob);
        const img = new Image();
        img.onload = () => {
          try {
            const canvas = document.createElement('canvas');
            const scale = Math.max(336 / Math.max(1, img.width), 336 / Math.max(1, img.height));
            canvas.width = Math.min(1024, Math.max(1, Math.round(img.width * scale)));
            canvas.height = Math.min(1024, Math.max(1, Math.round(img.height * scale)));
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
            URL.revokeObjectURL(objUrl);
            resolve(dataUrl);
          } catch (err) {
            URL.revokeObjectURL(objUrl);
            reject(err);
          }
        };
        img.onerror = (err) => {
          URL.revokeObjectURL(objUrl);
          reject(err);
        };
        img.src = objUrl;
      });
    }

    async function setUploadedFile(fileObj, labelText) {
      try {
        const b64 = await blobToNormalizedB64(fileObj);
        activeImageB64 = b64;
        chipName.innerText = labelText || fileObj.name || 'pasted_image.jpg';
        chipThumb.src = b64;
        imageChip.style.display = 'inline-flex';
        await executeSearch();
      } catch (err) {
        console.error('Failed to decode image:', err);
        document.getElementById('resultStats').innerText = 'Could not read image file.';
      }
    }

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        setUploadedFile(e.target.files[0]);
      }
    });

    removeImg.addEventListener('click', () => {
      activeImageB64 = null;
      fileInput.value = '';
      imageChip.style.display = 'none';
    });

    // Global capture-phase drag & drop so dropping anywhere on the window (even on the text input)
    // intercepts the file before Firefox attempts to open or insert a file:/// URI
    const dropZone = document.querySelector('.search-panel');
    window.addEventListener('dragover', (e) => {
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
      dropZone.style.borderColor = '#6366f1';
    }, true);
    window.addEventListener('dragleave', (e) => {
      if (!e.relatedTarget) {
        dropZone.style.borderColor = '#262b3a';
      }
    }, true);
    window.addEventListener('drop', (e) => {
      dropZone.style.borderColor = '#262b3a';
      e.preventDefault();
      e.stopPropagation();
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        setUploadedFile(e.dataTransfer.files[0]);
      }
    }, true);

    // Support clipboard paste (Ctrl+V) anywhere on page or inside search input
    window.addEventListener('paste', (e) => {
      const cd = e.clipboardData;
      if (!cd) return;

      // 1. Check for image files in clipboardData.files
      if (cd.files && cd.files.length > 0) {
        for (let i = 0; i < cd.files.length; i++) {
          if (cd.files[i].type && cd.files[i].type.startsWith('image/')) {
            e.preventDefault();
            e.stopPropagation();
            setUploadedFile(cd.files[i], 'Pasted Image');
            return;
          }
        }
      }

      // 2. Check for image items in clipboardData.items (standard screenshot paste)
      if (cd.items && cd.items.length > 0) {
        for (let i = 0; i < cd.items.length; i++) {
          const item = cd.items[i];
          if (item.type && item.type.startsWith('image/')) {
            const blob = item.getAsFile();
            if (blob) {
              e.preventDefault();
              e.stopPropagation();
              setUploadedFile(blob, 'Pasted Screenshot');
              return;
            }
          }
        }
      }
    }, true);

    async function executeSearch() {
      const q = document.getElementById('queryInput').value.trim();
      if (!q && !activeImageB64) return;

      // Abort any pending pacing requests so browser HTTP connections are immediately free
      if (pacingAbortCtrl) {
        pacingAbortCtrl.abort();
        pacingAbortCtrl = null;
      }

      const loader = document.getElementById('loader');
      const grid = document.getElementById('resultsGrid');
      const stats = document.getElementById('resultStats');
      const latStats = document.getElementById('latencyStats');

      loader.style.display = 'block';
      grid.innerHTML = '';
      stats.innerText = 'Searching...';

      const params = new URLSearchParams();
      if (q) params.append('query', q);
      if (activeImageB64) params.append('image_b64', activeImageB64);
      params.append('mode', document.getElementById('modeSelect').value);
      params.append('top_k', document.getElementById('topKSelect').value);

      try {
        const res = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8' },
          body: params.toString()
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        loader.style.display = 'none';

        stats.innerText = `Top ${data.results.length} matches from ${data.searched.toLocaleString()} videos (${data.mode})`;
        latStats.innerText = `${data.query_ms} ms`;

        renderResults(data.results);
      } catch (err) {
        loader.style.display = 'none';
        stats.innerText = 'Error performing search.';
        console.error(err);
      }
    }

    document.getElementById('searchForm').addEventListener('submit', (e) => {
      e.preventDefault();
      executeSearch();
    });
    document.getElementById('modeSelect').addEventListener('change', () => executeSearch());
    document.getElementById('topKSelect').addEventListener('change', () => executeSearch());

    const pacingCache = {};
    const TILE_POSITIONS = [
      "Top-Left", "Top-Mid", "Top-Right",
      "Mid-Left", "Center", "Mid-Right",
      "Bottom-Left", "Bottom-Mid", "Bottom-Right"
    ];

    function renderResults(results) {
      const grid = document.getElementById('resultsGrid');
      grid.innerHTML = '';

      results.forEach(r => {
        const card = document.createElement('div');
        card.className = 'card';

        let posturePills = r.posture.map(p => `<span class="pill pill-pos" onclick="tagSearch('${p}')">${p}</span>`).join('');
        let clothingPills = r.clothing.map(c => `<span class="pill pill-clo" onclick="tagSearch('${c}')">${c}</span>`).join('');

        const matrixCells = [0,1,2,3,4,5,6,7,8].map(k => `
          <div class="matrix-cell ${k === r.winning_tile ? 'active-win' : ''}" 
               title="Frame #${k} (${TILE_POSITIONS[k]})${k === r.winning_tile ? ' ★ Best Match' : ''}"
               onclick="showSpecificTile(event, '${r.eid}', ${k})"
               onmouseenter="previewTile(event, '${r.eid}', ${k})"></div>
        `).join('');

        card.innerHTML = `
          <a href="${r.video_url}" target="_blank" rel="noopener noreferrer" class="thumb-wrap" 
             onmouseenter="playHover(this, '${r.preview_url}', '${r.eid}')" 
             onmousemove="scrubHover(event, this, '${r.eid}')"
             onmouseleave="stopHover(this, '${r.eid}')" 
             title="Open video in new tab (Hover to preview & scrub)">
            <img class="thumb-img" id="img_${r.eid}" src="${r.tile_url}" loading="lazy">
            <video class="hover-video" loop muted playsinline></video>
            <div class="score-badge">#${r.rank} &bull; ${(r.score * 100).toFixed(1)}%</div>
            <div class="tile-badge" title="Frame #${r.winning_tile} (${TILE_POSITIONS[r.winning_tile]}) - Click cells to explore other frames">
              <span>Frame #${r.winning_tile}</span>
              <div class="tile-matrix" onclick="event.preventDefault(); event.stopPropagation();">
                ${matrixCells}
              </div>
            </div>
            <div class="pacing-container" id="pacing_box_${r.eid}" style="display:none;">
              <div class="pacing-badge" id="pacing_${r.eid}"></div>
              <div class="swatch-timeline" id="timeline_${r.eid}">
                <div class="swatch-segment" id="seg_${r.eid}_0"></div>
                <div class="swatch-segment" id="seg_${r.eid}_1"></div>
                <div class="swatch-segment" id="seg_${r.eid}_2"></div>
                <div class="swatch-segment" id="seg_${r.eid}_3"></div>
                <div class="swatch-segment" id="seg_${r.eid}_4"></div>
              </div>
            </div>
          </a>
          <div class="card-body">
            <a href="${r.video_url}" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color:inherit;">
              <div class="card-title" title="${r.title}">${r.title}</div>
            </a>
            <div class="tags-row">${posturePills} ${clothingPills}</div>
            <div class="card-actions">
              <button class="toggle-frame-btn" onclick="toggleTile(event, '${r.eid}', '${r.thumb_url}', '${r.tile_url}')">Full Mosaic</button>
              <button class="find-similar-btn" onclick="findSimilar('${r.eid}', ${r.winning_tile})">Similar &rarr;</button>
              <a href="${r.video_url}" target="_blank" rel="noopener noreferrer" class="watch-btn" title="Open in new tab">Watch ↗</a>
            </div>
          </div>
        `;
        grid.appendChild(card);
      });

      // Throttled 2-worker queue for pacing badges so browser HTTP connections are never saturated
      pacingAbortCtrl = new AbortController();
      const signal = pacingAbortCtrl.signal;
      const queue = results.map(r => r.eid);
      async function pacingWorker() {
        while (queue.length > 0 && !signal.aborted) {
          const eid = queue.shift();
          await loadPacingBadge(eid, signal);
        }
      }
      pacingWorker();
      pacingWorker();
    }

    function applyPacingData(eid, data) {
      const box = document.getElementById(`pacing_box_${eid}`);
      const el = document.getElementById(`pacing_${eid}`);
      if (!box || !el) return;

      el.innerHTML = data.badge;
      el.style.color = data.color;
      el.style.borderColor = data.color + '55';

      if (data.swatch_bpms && data.swatch_colors) {
        for (let s = 0; s < 5; s++) {
          const seg = document.getElementById(`seg_${eid}_${s}`);
          if (seg) {
            seg.style.background = data.swatch_colors[s];
            seg.style.color = data.swatch_colors[s];
            seg.title = `Swatch ${s + 1}s: ${data.swatch_bpms[s]} BPM`;
          }
        }
      }
      box.style.display = 'flex';
    }

    async function loadPacingBadge(eid, signal) {
      if (pacingCache[eid]) {
        applyPacingData(eid, pacingCache[eid]);
        return;
      }
      try {
        const res = await fetch(`/api/pacing/${eid}`, { signal });
        if (!res.ok) return;
        const data = await res.json();
        pacingCache[eid] = data;
        applyPacingData(eid, data);
      } catch (e) {}
    }

    function updatePacingSwatch(eid, sec) {
      const data = pacingCache[eid];
      if (!data || !data.swatch_bpms) return;
      sec = Math.min(4, Math.max(0, sec));
      const bpm = data.swatch_bpms[sec];
      const color = (data.swatch_colors && data.swatch_colors[sec]) ? data.swatch_colors[sec] : data.color;

      const el = document.getElementById(`pacing_${eid}`);
      if (el) {
        el.innerHTML = `<span class="live-dot" style="display:inline-block;"></span>⚡ <b>${bpm} BPM</b> <span style="font-size:0.65rem;opacity:0.8;margin-left:3px;">[sec ${sec + 1}/5]</span>`;
        el.style.color = color;
        el.style.borderColor = color;
      }

      for (let s = 0; s < 5; s++) {
        const seg = document.getElementById(`seg_${eid}_${s}`);
        if (seg) {
          if (s === sec) {
            seg.classList.add('active');
          } else {
            seg.classList.remove('active');
          }
        }
      }
    }

    function resetPacingSwatch(eid) {
      const data = pacingCache[eid];
      if (!data) return;
      const el = document.getElementById(`pacing_${eid}`);
      if (el) {
        el.innerHTML = data.badge;
        el.style.color = data.color;
        el.style.borderColor = data.color + '55';
      }
      for (let s = 0; s < 5; s++) {
        const seg = document.getElementById(`seg_${eid}_${s}`);
        if (seg) seg.classList.remove('active');
      }
      const tl = document.getElementById(`timeline_${eid}`);
      if (tl) tl.style.display = 'none';
    }

    const activeBeatLoops = {};

    function startBeatSync(eid, video) {
      if (activeBeatLoops[eid]) {
        cancelAnimationFrame(activeBeatLoops[eid]);
      }
      function syncLoop() {
        if (video.paused || video.ended) return;
        const data = pacingCache[eid];
        if (data && data.beat_times && data.beat_times.length > 0) {
          const t = video.currentTime;
          let minDiff = 999;
          for (let i = 0; i < data.beat_times.length; i++) {
            const diff = Math.abs(t - data.beat_times[i]);
            if (diff < minDiff) minDiff = diff;
          }

          const dot = document.querySelector(`#pacing_${eid} .live-dot`);
          const badge = document.getElementById(`pacing_${eid}`);
          const curSec = Math.min(4, Math.floor(t));
          const color = (data.swatch_colors && data.swatch_colors[curSec]) ? data.swatch_colors[curSec] : data.color;
          const activeSeg = document.getElementById(`seg_${eid}_${curSec}`);

          // Snappy multi-pulse flash window: 0.085 seconds allows up to 5-6 distinct pulses per second / segment
          const decay = 0.085;
          if (dot && minDiff < decay) {
            const factor = Math.sin((1.0 - minDiff / decay) * Math.PI / 2);
            dot.style.transform = `scale(${1.0 + factor * 1.35})`;
            dot.style.opacity = '1';
            dot.style.background = '#ffffff';
            dot.style.boxShadow = `0 0 ${10 + factor * 16}px #ffffff, 0 0 ${18 + factor * 14}px ${color}`;
            if (badge) {
              badge.style.boxShadow = `0 0 ${factor * 16}px ${color}aa, 0 4px 10px rgba(0,0,0,0.5)`;
              badge.style.borderColor = '#ffffff';
            }
            if (activeSeg) {
              activeSeg.style.boxShadow = `0 0 ${4 + factor * 10}px #ffffff`;
              activeSeg.style.filter = `brightness(${1.0 + factor * 0.8})`;
            }
          } else {
            if (dot) {
              dot.style.transform = 'scale(0.85)';
              dot.style.opacity = '0.45';
              dot.style.background = color;
              dot.style.boxShadow = `0 0 3px ${color}`;
            }
            if (badge) {
              badge.style.boxShadow = '0 4px 10px rgba(0,0,0,0.4)';
              badge.style.borderColor = color + '55';
            }
            if (activeSeg) {
              activeSeg.style.boxShadow = '0 0 6px currentColor';
              activeSeg.style.filter = 'none';
            }
          }
        }
        activeBeatLoops[eid] = requestAnimationFrame(syncLoop);
      }
      activeBeatLoops[eid] = requestAnimationFrame(syncLoop);
    }

    function stopBeatSync(eid) {
      if (activeBeatLoops[eid]) {
        cancelAnimationFrame(activeBeatLoops[eid]);
        delete activeBeatLoops[eid];
      }
      const dot = document.querySelector(`#pacing_${eid} .live-dot`);
      const badge = document.getElementById(`pacing_${eid}`);
      if (dot) {
        dot.style.transform = 'scale(1)';
        dot.style.opacity = '1';
        dot.style.background = 'currentColor';
        dot.style.boxShadow = '0 0 6px currentColor';
      }
      if (badge) {
        badge.style.boxShadow = '0 4px 10px rgba(0,0,0,0.4)';
      }
      for (let s = 0; s < 5; s++) {
        const seg = document.getElementById(`seg_${eid}_${s}`);
        if (seg) {
          seg.style.boxShadow = 'none';
          seg.style.filter = 'none';
        }
      }
    }

    function playHover(wrap, previewUrl, eid) {
      if (!previewUrl) return;
      const video = wrap.querySelector('.hover-video');
      const tl = document.getElementById(`timeline_${eid}`);
      if (tl) tl.style.display = 'flex';

      if (video) {
        if (!video.src || video.src !== previewUrl) {
          video.src = previewUrl;
        }
        video.style.display = 'block';
        video.ontimeupdate = () => {
          const sec = Math.min(4, Math.floor(video.currentTime));
          updatePacingSwatch(eid, sec);
        };
        video.play().catch(() => {});
        startBeatSync(eid, video);
      }
    }

    function scrubHover(event, wrap, eid) {
      const video = wrap.querySelector('.hover-video');
      if (!video) return;
      const rect = wrap.getBoundingClientRect();
      const pct = Math.max(0, Math.min(0.99, (event.clientX - rect.left) / rect.width));
      const dur = video.duration || 5.0;
      const targetTime = pct * dur;
      video.currentTime = targetTime;
      const sec = Math.min(4, Math.floor(targetTime));
      updatePacingSwatch(eid, sec);
    }

    function stopHover(wrap, eid) {
      const video = wrap.querySelector('.hover-video');
      if (video) {
        video.pause();
        video.ontimeupdate = null;
        video.style.display = 'none';
      }
      stopBeatSync(eid);
      resetPacingSwatch(eid);
    }

    function showSpecificTile(event, eid, tileIdx) {
      if (event) {
        event.preventDefault();
        event.stopPropagation();
      }
      const img = document.getElementById(`img_${eid}`);
      if (img) img.src = `/api/thumb/${eid}?tile=${tileIdx}`;
    }

    function previewTile(event, eid, tileIdx) {
      if (event) {
        event.preventDefault();
        event.stopPropagation();
      }
      const img = document.getElementById(`img_${eid}`);
      if (img) img.src = `/api/thumb/${eid}?tile=${tileIdx}`;
    }

    function toggleTile(event, eid, fullUrl, tileUrl) {
      if (event) {
        event.preventDefault();
        event.stopPropagation();
      }
      const img = document.getElementById(`img_${eid}`);
      const btn = event.target;
      if (!img) return;
      if (img.src.includes('tile=')) {
        img.src = fullUrl;
        if (btn) btn.innerText = 'Win Frame';
      } else {
        img.src = tileUrl;
        if (btn) btn.innerText = 'Full Mosaic';
      }
    }

    function tagSearch(tag) {
      document.getElementById('queryInput').value = tag;
      executeSearch();
    }

    async function findSimilar(eid, winningTile = 4) {
      try {
        const res = await fetch(`/api/thumb/${eid}?tile=${winningTile}`);
        const blob = await res.blob();
        document.getElementById('queryInput').value = '';
        await setUploadedFile(blob, `Similar to ${eid} (Frame #${winningTile})`);
      } catch (e) {
        console.error(e);
      }
    }
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Web Viewer for Semantic Video Retrieval")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--emb-dir", default=DEFAULT_EMB_DIR, help="Path to embeddings directory")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to trained checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    init_engine(emb_dir=args.emb_dir, weights=args.weights, device=args.device)
    print(f"\n========================================================")
    print(f" WEB VIEWER ACTIVE AT: http://{args.host}:{args.port}")
    print(f"========================================================\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
