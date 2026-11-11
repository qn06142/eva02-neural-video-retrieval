#!/usr/bin/env python3
"""Retrieve the top-N scraped thumbnails for an image or text query.

Backends: siglip (trained checkpoint), clip (zero-shot), dinov2 (image only).

CPU bottleneck
--------------
Re-decoding and pre-processing 119k JPEGs on every query is the real cost, not
the GPU (the vision tower does ~440 samples/s).  So compute the corpus
embeddings ONCE and reuse them:

    # one time, a few minutes
    .venv/Scripts/python scrape/retrieve.py --backend siglip --save-emb corpus_siglip.pt

    # every query after that is < 1s
    .venv/Scripts/python scrape/retrieve.py --backend siglip --text "lingerie" --load-emb corpus_siglip.pt

Usage
-----
    .venv/Scripts/python scrape/retrieve.py --backend siglip --text "lingerie"
    .venv/Scripts/python scrape/retrieve.py --query "<image.png>" --backend dinov2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil

import torch
from PIL import Image

from retrieval_tta import (TTA_POLICIES, TEXT_TTA_MODES, cache_metadata,
                           checkpoint_fingerprint, encode_tta_images,
                           encode_text_query, expand_text_query,
                           validate_cache, load_corpus)
from training_utils import pool_scores
from torch.utils.data import DataLoader, Dataset


def _worker_init(worker_id):
    """Stop each DataLoader worker from spawning a full set of torch threads."""
    torch.set_num_threads(1)


def _default_out_dir(backend, args):
    if args.text:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", args.text).strip("_")[:40]
    elif args.query:
        slug = os.path.splitext(os.path.basename(args.query))[0][:40]
    else:
        slug = "corpus"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "scrape", "siglip_runs", "results",
                        f"{backend}_{slug}")


class ImageDataset(Dataset):
    def __init__(self, paths: list, grid=None, bag_size=9):
        self.paths = paths
        self.grid = grid
        self.bag_size = bag_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            with Image.open(self.paths[i]) as im:
                image = im.convert("RGB")
            if self.grid is None:
                return image
            from train_siglip_mil import MosaicBagDataset
            tiles = MosaicBagDataset([], None, self.bag_size, self.grid)._tiles(image)
            return [tiles[round(j*(len(tiles)-1)/max(self.bag_size-1, 1))] for j in range(self.bag_size)]
        except (OSError, ValueError) as exc:
            raise OSError(f"Cannot decode image: {self.paths[i]}") from exc


class ImageCollator:
    def __call__(self, images):
        bags = [image if isinstance(image, list) else [image] for image in images]
        return [image for bag in bags for image in bag], len(bags), len(bags[0])


def _norm(x):
    return torch.nn.functional.normalize(x.float(), dim=-1)


class SiglipBackend:
    def __init__(self, ckpt, device, dtype):
        from transformers import SiglipProcessor
        from frame_model import SiglipBagModel as SiglipModel
        self.processor = SiglipProcessor.from_pretrained(ckpt)
        self.model = SiglipModel.from_pretrained(
            ckpt, attn_implementation="sdpa", torch_dtype=dtype
        ).to(device).eval()
        self.device = device

    def pixel_values(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    def _mask(self, ids):
        pad = self.processor.tokenizer.pad_token_id
        is_pad = ids == pad
        first = is_pad.float().argmax(dim=1)
        ar = torch.arange(ids.size(1)).unsqueeze(0)
        m = (ar <= first.unsqueeze(1)).long()
        m[~is_pad.any(dim=1)] = 1
        return m

    @torch.no_grad()
    def encode_image(self, pv):
        return _norm(self.model.get_image_features(pixel_values=pv))

    @torch.no_grad()
    def encode_text(self, texts):
        enc = self.processor(text=texts, padding="max_length", truncation=True,
                             max_length=64, return_tensors="pt")
        feats = self.model.get_text_features(
            input_ids=enc["input_ids"].to(self.device),
            attention_mask=self._mask(enc["input_ids"]).to(self.device),
        )
        return _norm(feats)


class CLIPBackend:
    def __init__(self, ckpt, device, dtype):
        from transformers import CLIPModel, CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained(ckpt)
        self.model = CLIPModel.from_pretrained(ckpt, torch_dtype=dtype).to(device).eval()
        self.device = device

    def pixel_values(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    @torch.no_grad()
    def encode_image(self, pv):
        return _norm(self.model.get_image_features(pixel_values=pv))

    @torch.no_grad()
    def encode_text(self, texts):
        enc = self.processor(text=texts, padding=True, truncation=True,
                             return_tensors="pt")
        return _norm(self.model.get_text_features(
            input_ids=enc["input_ids"].to(self.device),
            attention_mask=enc["attention_mask"].to(self.device),
        ))


class Dinov2Backend:
    def __init__(self, ckpt, device, dtype):
        from transformers import AutoImageProcessor, Dinov2Model
        self.processor = AutoImageProcessor.from_pretrained(ckpt)
        self.model = Dinov2Model.from_pretrained(ckpt, torch_dtype=dtype).to(device).eval()
        self.device = device

    def pixel_values(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    @torch.no_grad()
    def encode_image(self, pv):
        return _norm(self.model(pixel_values=pv).last_hidden_state[:, 0])

    def encode_text(self, texts):
        raise NotImplementedError("DINOv2 has no text tower")


BACKENDS = {
    "siglip": (SiglipBackend, r"scrape\siglip_runs\run1\best"),
    "clip": (CLIPBackend, "openai/clip-vit-base-patch16"),
    "dinov2": (Dinov2Backend, "facebook/dinov2-base"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Top-N thumbnail retrieval.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--query", help="path to a query image")
    g.add_argument("--text", help="text query")
    ap.add_argument("--tta", choices=TTA_POLICIES, default="none")
    ap.add_argument("--text-tta", choices=TEXT_TTA_MODES, default="none",
                    help="Text TTA mode: 'accurate' (tight centroid) or 'vague' (exploratory multi-query max-pool)")
    ap.add_argument("--asset-type", choices=("auto", "thumbnail", "mosaic"), default="auto")
    ap.add_argument("--mosaics", default="mosaics")
    ap.add_argument("--grid", default="3x3")
    ap.add_argument("--bag-size", type=int, default=None)
    ap.add_argument("--pooling", choices=("max", "logmeanexp"), default=None,
                    help="default: checkpoint setting, otherwise max")
    ap.add_argument("--pool-temperature", type=float, default=None)
    ap.add_argument("--tta-crop-scale", type=float, default=.9)
    ap.add_argument("--view-batch-size", type=int, default=64)
    ap.add_argument("--backend", default="siglip", choices=list(BACKENDS))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--db", default=r"scrape\thumbs.db")
    ap.add_argument("--thumbs", default="thumbs")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-thread-limit", action="store_true",
                    help="let workers use default torch threads (faster here)")
    ap.add_argument("--limit", type=int, default=None, help="scan only the first N images")
    ap.add_argument("--out-dir", default=None,
                    help="folder to dump the ranked thumbs into (cleared first). "
                         "Default: scrape/siglip_runs/results/<backend>_<query>")
    ap.add_argument("--no-copy", action="store_true",
                    help="do not copy the ranked thumbs into a folder")
    ap.add_argument("--save-emb", default=None, metavar="PATH",
                    help="compute corpus embeddings, save them, and exit")
    ap.add_argument("--load-emb", default=None, metavar="PATH",
                    help="use precomputed corpus embeddings instead of scanning")
    args = ap.parse_args()

    if args.batch_size < 1 or args.view_batch_size < 1 or args.workers < 0 or args.top < 1:
        ap.error("batch sizes and top must be positive; workers must be nonnegative")
    if not 0 < args.tta_crop_scale <= 1 or (args.limit is not None and args.limit < 1):
        ap.error("Need 0 < tta-crop-scale <= 1 and a positive limit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = tuple(int(v) for v in args.grid.lower().split("x"))
    if len(grid) != 2 or min(grid) < 1:
        ap.error("Grid must have two positive dimensions")
    bag_size = args.bag_size if args.bag_size is not None else grid[0]*grid[1]
    if not 1 <= bag_size <= grid[0]*grid[1]:
        ap.error("Invalid bag size or pooling temperature")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.float32
    cls, default_ckpt = BACKENDS[args.backend]
    ckpt = args.ckpt or default_ckpt
    backend = cls(ckpt, device, dtype)
    args.pooling = args.pooling or getattr(backend.model.config, "retrieval_pooling", "max")
    if args.pool_temperature is None:
        args.pool_temperature = getattr(backend.model.config, "retrieval_pool_temperature", .07)
    if not 0 < args.pool_temperature < float("inf"):
        ap.error("Invalid pooling temperature")
    from pathlib import Path
    from transformers.utils.hub import cached_file
    resolved = ckpt if Path(ckpt).is_dir() else str(Path(cached_file(ckpt, "config.json")).parent)
    metadata = cache_metadata(args.backend, checkpoint_fingerprint(resolved), args.tta, args.tta_crop_scale)

    print(f"device={device} backend={args.backend} ckpt={ckpt}", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"vram used: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # ---- corpus metadata ------------------------------------------------- #
    items, asset_type = load_corpus(args.db, args.thumbs, args.mosaics, args.asset_type)
    metadata.update(asset_type=asset_type, grid=list(grid) if asset_type == "mosaic" else None,
                    bag_size=bag_size if asset_type == "mosaic" else 1,
                    dtype=str(dtype), torch=str(torch.__version__))
    if args.limit:
        items = items[: args.limit]
    eids = [it[1] for it in items]
    paths = [it[0] for it in items]
    metadata["corpus_paths_sha256"] = hashlib.sha256(json.dumps(paths).encode("utf-8")).hexdigest()
    if not paths:
        raise ValueError("No corpus images for the requested asset type")
    print(f"corpus: {len(paths):,} images", flush=True)

    # ---- corpus embeddings ----------------------------------------------- #
    if args.load_emb:
        blob = torch.load(args.load_emb, map_location="cpu", weights_only=True)
        validate_cache(blob, metadata, eids)
        emb = _norm(blob["emb"])
        print(f"loaded corpus embeddings {tuple(emb.shape)} from {args.load_emb}",
              flush=True)
    else:
        ds = ImageDataset(paths, grid if asset_type == "mosaic" else None, bag_size)
        dl = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            collate_fn=ImageCollator(),
            worker_init_fn=(None if args.no_thread_limit else _worker_init),
            persistent_workers=(args.workers > 0),
        )
        chunks = []
        with torch.no_grad():
            for i, (images, b, k) in enumerate(dl):
                features = encode_tta_images(images, backend.pixel_values, backend.encode_image,
                    device, args.tta, args.tta_crop_scale, args.view_batch_size)
                bags = features.reshape(b, k, -1)
                if hasattr(backend.model, "mix_frame_features"):
                    bags = backend.model.mix_frame_features(bags)
                chunks.append((bags if asset_type == "mosaic" else bags[:, 0]).cpu())
                if i % 200 == 0:
                    print(f"  encoded {min((i+1)*args.batch_size, len(paths)):,}"
                          f"/{len(paths):,}", flush=True)
        if not chunks:
            raise ValueError("No corpus images available")
        emb = torch.cat(chunks)
        if args.save_emb:
            torch.save({"eids": eids, "emb": emb.half(), "metadata": metadata},
                       args.save_emb)
            print(f"saved corpus embeddings {tuple(emb.shape)} -> {args.save_emb}",
                  flush=True)
            return 0

    # ---- query ----------------------------------------------------------- #
    tta_mode = "none"
    if args.text:
        q, tta_mode = encode_text_query(backend.encode_text, args.text, mode=args.text_tta)
        if tta_mode != "none":
            variations = expand_text_query(args.text, mode=args.text_tta)
            print(f"text query: {args.text!r} [text-tta: {args.text_tta}, {len(variations)} prompts]", flush=True)
            for vi, v in enumerate(variations, 1):
                print(f"  [{vi:02d}] {v}", flush=True)
        else:
            print(f"text query: {args.text!r}", flush=True)
    elif args.query:
        with Image.open(args.query) as im:
            qimg = im.convert("RGB")
        q = encode_tta_images([qimg], backend.pixel_values, backend.encode_image, device,
                              args.tta, args.tta_crop_scale, args.view_batch_size)
        if hasattr(backend.model, "mix_frame_features"):
            q = backend.model.mix_frame_features(q[:, None])[:, 0]
        print(f"image query: {args.query} ({qimg.size[0]}x{qimg.size[1]})", flush=True)
    else:
        print("(no query given -- embeddings only)", flush=True)
        return 0

    # ---- rank ------------------------------------------------------------ #
    if emb.shape[-1] != q.shape[-1]:
        print(f"ERROR: corpus embedding dim {emb.shape[-1]} does not match "
              f"{args.backend} query dim {q.shape[-1]}. The corpus file belongs "
              f"to a different backend; rebuild it with --save-emb.", flush=True)
        return 1
    if emb.ndim == 3:
        frame_scores = torch.einsum("nkd,qd->nqk", emb.float().to(device), q.to(device))
        pooled = pool_scores(frame_scores, args.pooling, args.pool_temperature)
        sims = pooled.amax(dim=1).cpu() if tta_mode == "vague" else pooled.squeeze(1).cpu()
    else:
        scores = emb.float().to(device) @ q.to(device).t()
        sims = scores.amax(dim=1).cpu() if tta_mode == "vague" else scores.squeeze(1).cpu()
    k = min(args.top, sims.numel())
    vals, idxs = torch.topk(sims, k)
    lines = []
    for rank, (v, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), 1):
        path, eid, title = items[idx]
        lines.append(f"{rank:3d}  {v:.4f}  {path}  |  {title}")

    print()
    print(f"---- top {k} [{args.backend}] ----")
    for line in lines:
        print(line)

    if not args.no_copy:
        out_dir = args.out_dir or _default_out_dir(args.backend, args)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        for rank, (v, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), 1):
            path, eid, title = items[idx]
            shutil.copy2(path, os.path.join(out_dir,
                                            f"{rank:02d}_{v:.4f}_{eid}.jpg"))
        with open(os.path.join(out_dir, "_ranking.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\ncopied {k} images -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
