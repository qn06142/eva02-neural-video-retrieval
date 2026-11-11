#!/usr/bin/env python3
"""MIL fine-tuning of SigLIP on xvideos contact sheets.

Each video is a BAG: `mozaique_listing.jpg` is a 3x3 grid of 9 frames.  The
title describes the whole video, so a given frame may or may not show what the
title mentions ("doggystyle" is likely in the video but not necessarily in the
one frame you picked).  Training on one frame per video therefore optimises the
wrong thing -- it forces a single instance to match a bag-level label.

Here every (title, video) pair is scored by the BEST frame in the bag:

    score(i, j) = max_k  cos(title_i, frame_jk)

and the SigLIP pairwise sigmoid loss is applied at bag level.  Frames are
cropped in memory; nothing is pre-sliced to disk (NTFS hates a million tiny
files).

Usage
-----
    .venv/Scripts/python scrape/train_siglip_mil.py --max-samples 2000 --epochs 1
    .venv/Scripts/python scrape/train_siglip_mil.py --epochs 4
    .venv/Scripts/python scrape/train_siglip_mil.py --init-from scrape/siglip_runs/mil1/best
"""

from __future__ import annotations

import argparse
import numpy as np
import html
import os
import random
import re
import sqlite3
import time
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import (
    SiglipProcessor,
)
from frame_model import SiglipBagModel as SiglipModel

from retrieval_tta import encode_dataset
from weak_supervision import (chunked_features, negative_pair_weights, pair_diagnostics,
                              FrozenTitleTeacher)

from training_utils import (
    add_training_options, choose_device, positive_mask,
    retrieval_recall, run_epochs, seed_worker, select_samples, sigmoid_loss,
    split_samples, validate_args,
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(PROJECT_DIR, "scrape", "thumbs.db")
DEFAULT_THUMBS = os.path.join(PROJECT_DIR, "thumbs")
MOSAIC_DIR = os.path.join(PROJECT_DIR, "mosaics")
DEFAULT_MODEL = "google/siglip-base-patch16-224"
DEFAULT_OUT = os.path.join(PROJECT_DIR, "scrape", "siglip_runs", "mil1")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_title(s) -> str:
    s = html.unescape(s or "")
    s = s.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()


def load_bags(db_path: str, mosaic_dir: str, max_samples: Optional[int],
              min_title_len: int, seed: int = 42) -> list:
    """(mosaic_path, title) for every video whose contact sheet downloaded."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT eid, title, asset_path FROM videos
           WHERE title IS NOT NULL AND asset_state = 'done'
           ORDER BY discovered"""
    ).fetchall()
    conn.close()

    bags, skipped = [], 0
    for r in rows:
        path = r["asset_path"] or os.path.join(mosaic_dir, f"{r['eid']}.jpg")
        title = clean_title(r["title"])
        if len(title) < min_title_len:
            skipped += 1
            continue
        bags.append((path, title))
    if skipped:
        log(f"skipped {skipped} rows (empty or short title)")
    return select_samples(bags, max_samples, seed)


class MosaicBagDataset(Dataset):
    """One item = one video, yielding K tiles cropped from its contact sheet."""

    def __init__(self, bags: list, processor: SiglipProcessor, bag_size: int = 4,
                 grid: tuple = (3, 3), train: bool = False, augment: bool = True):
        if len(grid) != 2 or min(grid) < 1 or not 1 <= bag_size <= grid[0] * grid[1]:
            raise ValueError("Require positive grid dimensions and 1 <= bag-size <= tile count")
        self.bags = bags
        self.bag_size = bag_size
        self.grid = grid
        self.train = train
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]) if (train and augment) else None

    def __len__(self) -> int:
        return len(self.bags)

    def _tiles(self, im: Image.Image) -> list:
        cols, rows = self.grid
        w, h = im.size
        if w < cols or h < rows:
            raise ValueError("Contact sheet smaller than its grid")
        return [im.crop((c * w // cols, r * h // rows,
                         (c + 1) * w // cols, (r + 1) * h // rows))
                for r in range(rows) for c in range(cols)]

    def __getitem__(self, idx: int):
        path, title = self.bags[idx]
        try:
            with Image.open(path) as im:
                tiles = self._tiles(im.convert("RGB"))
        except (OSError, ValueError) as exc:
            raise OSError(f"Cannot decode contact sheet: {path}") from exc
        k = min(self.bag_size, len(tiles))
        # random subset each visit -> over epochs the model sees every frame
        chosen = (random.sample(tiles, k) if self.train else
                  [tiles[round(i * (len(tiles)-1) / max(k-1, 1))] for i in range(k)])
        if self.aug is not None:
            chosen = [self.aug(t) for t in chosen]
        return chosen, title


class BagCollator:
    """Picklable collate: B bags x K tiles -> pixel_values (B*K), plus text."""

    def __init__(self, processor: SiglipProcessor):
        self.processor = processor

    def __call__(self, batch):
        imgs = [im for imgs, _ in batch for im in imgs]
        txts = [t for _, t in batch]
        enc = self.processor(
            text=txts, images=imgs, padding="max_length", truncation=True,
            max_length=64, return_tensors="pt",
        )
        ids = enc["input_ids"]
        pad = self.processor.tokenizer.pad_token_id
        is_pad = ids == pad
        first = is_pad.float().argmax(dim=1)
        ar = torch.arange(ids.size(1)).unsqueeze(0)
        mask = (ar <= first.unsqueeze(1)).long()
        mask[~is_pad.any(dim=1)] = 1
        enc["attention_mask"] = mask
        enc["positive_mask"] = positive_mask(txts)
        return enc, len(batch), len(batch[0][0])


def mil_sigmoid_loss(model, img_feats: torch.Tensor, txt_feats: torch.Tensor,
                     positives=None, pooling="max", temperature=.07, pair_weights=None) -> torch.Tensor:
    """Bag-level sigmoid loss with max or smooth pooling, reduced in FP32."""
    return sigmoid_loss(model, img_feats, txt_feats, positives, pooling, temperature, pair_weights)


@torch.no_grad()
def encode_bags(model, processor, bags, device, bag_size, grid, batch_size, workers,
                tta="none", crop_scale=.9, view_batch_size=64):
    ds = MosaicBagDataset(bags, processor, bag_size=bag_size, grid=grid, train=False)
    return encode_dataset(model, processor, ds, device, batch_size, workers,
                          tta, crop_scale, view_batch_size)


def bag_recall_at_k(txt_feats, img_feats, ks=(1, 5, 10), pooling="max", temperature=.07) -> dict:
    return retrieval_recall(txt_feats, img_feats, ks, pooling=pooling, temperature=temperature)


def main() -> int:
    p = argparse.ArgumentParser(description="MIL fine-tune SigLIP on contact sheets.")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--mosaic-dir", default=MOSAIC_DIR)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--resume", default=None)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--frame-mixer", choices=("none", "residual-set"), default=None,
                   help="optional frame interaction head; default inherits checkpoint architecture")
    p.add_argument("--frame-mixer-dim", type=int, default=256)
    p.add_argument("--frame-mixer-layers", type=int, default=2)
    p.add_argument("--frame-mixer-heads", type=int, default=4)
    p.add_argument("--freeze-backbone", action="store_true",
                   help="train only the frame mixer, preserving image/text encoders and logit calibration")
    p.add_argument("--image-chunk-size", type=int, default=None,
                   help="activation-checkpoint image forwards in chunks; loss still uses the full bag batch")
    p.add_argument("--text-chunk-size", type=int, default=None)
    p.add_argument("--negative-title-threshold", type=float, default=None,
                   help="optional frozen-title similarity threshold for downweighting uncertain negatives")
    p.add_argument("--negative-weight-floor", type=float, default=.25)
    p.add_argument("--eval-bag-size", type=int, default=None,
                   help="validation tiles per bag; default: entire grid")
    p.add_argument("--pooling", choices=("max", "logmeanexp"), default="max")
    p.add_argument("--pool-temperature", type=float, default=.07)
    p.add_argument("--bag-size", type=int, default=4,
                   help="frames sampled per video per training step")
    p.add_argument("--grid", default="3x3", help="contact sheet layout, e.g. 3x3")
    p.add_argument("--batch-size", type=int, default=24, help="bags per step")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4,
                   help="dataloader workers (default: 4; more thrashes the pagefile)")
    p.add_argument("--val-frac", type=float, default=0.01)
    p.add_argument("--val-max", type=int, default=1000)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--min-title-len", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--no-augment", action="store_true")
    add_training_options(p)
    args = p.parse_args()
    validate_args(args)
    for size in (args.image_chunk_size, args.text_chunk_size):
        if size is not None and size < 1:
            raise ValueError("Encoder chunk sizes must be positive")
    if args.negative_title_threshold is not None and not 0 <= args.negative_title_threshold < 1:
        raise ValueError("Need 0 <= negative-title-threshold < 1")
    if not 0 <= args.negative_weight_floor <= 1:
        raise ValueError("Need 0 <= negative-weight-floor <= 1")
    if not np.isfinite(args.pool_temperature) or args.pool_temperature <= 0:
        raise ValueError("--pool-temperature must be finite and positive")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args)
    log(f"device: {device}")
    if device.type == "cuda":
        log(f"gpu: {torch.cuda.get_device_name(0)}")
    grid = tuple(int(x) for x in args.grid.lower().split("x"))
    if len(grid) != 2 or min(grid) < 1 or not 1 <= args.bag_size <= grid[0] * grid[1]:
        raise ValueError("Invalid --grid or --bag-size")
    eval_bag_size = args.eval_bag_size if args.eval_bag_size is not None else grid[0] * grid[1]
    if not 1 <= eval_bag_size <= grid[0] * grid[1]:
        raise ValueError("Invalid --eval-bag-size")

    bags = load_bags(args.db, args.mosaic_dir, args.max_samples, args.min_title_len, args.seed)
    if len(bags) < 50:
        log(f"only {len(bags)} bags -- crawl first, aborting")
        return 1
    log(f"bags: {len(bags):,}  grid={grid}  bag_size={args.bag_size}")

    train_bags, val_bags = split_samples(
        bags, args.val_frac, args.val_max, args.seed, args.out)
    log(f"train: {len(train_bags):,}  val: {len(val_bags):,}")

    src = args.resume or args.init_from or args.model
    log(f"loading model from {src}")
    processor = SiglipProcessor.from_pretrained(src)
    model = SiglipModel.from_pretrained(src, attn_implementation="sdpa")
    model.to(device)
    if args.frame_mixer == "residual-set":
        model.enable_frame_mixer(args.frame_mixer_dim, args.frame_mixer_layers, args.frame_mixer_heads)
    elif args.frame_mixer == "none" and model.frame_mixer is not None:
        raise ValueError("Use a backbone-only checkpoint to start a no-mixer ablation")
    if args.freeze_backbone:
        model.freeze_backbone()
    args.frame_mixer_config = getattr(model.config, "frame_mixer", None)
    log(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    model.config.retrieval_pooling = args.pooling
    model.config.retrieval_pool_temperature = args.pool_temperature
    teacher = (FrozenTitleTeacher(model, args.out, resume=bool(args.resume))
               if args.negative_title_threshold is not None else None)
    args.title_teacher_sha256 = teacher.sha256 if teacher is not None else None

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        MosaicBagDataset(train_bags, processor, bag_size=args.bag_size, grid=grid,
                         train=True, augment=not args.no_augment),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=BagCollator(processor), pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker, generator=generator, drop_last=False,
        **({"prefetch_factor": 1} if args.workers else {}),
    )

    def forward_loss(batch):
        enc, B, K = batch
        img = chunked_features(model.get_image_features,
            dict(pixel_values=enc["pixel_values"].to(device)), args.image_chunk_size,
            recompute=not args.freeze_backbone)
        text_inputs = dict(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device))
        txt = chunked_features(model.get_text_features, text_inputs, args.text_chunk_size,
                               recompute=not args.freeze_backbone)
        positives = enc["positive_mask"].to(device)
        weights = (negative_pair_weights(teacher(**text_inputs, chunk_size=args.text_chunk_size or 32),
                    positives, args.negative_title_threshold, args.negative_weight_floor)
                   if teacher is not None else None)
        diagnostics = pair_diagnostics(weights, enc["positive_mask"])
        diagnostics.update(bags=B, frames=B*K)
        img = model.mix_frame_features(img.reshape(B, K, -1))
        return mil_sigmoid_loss(model, img, txt, enc["positive_mask"],
                                args.pooling, args.pool_temperature, weights), diagnostics

    def evaluate():
        images, texts = encode_bags(model, processor, val_bags, device,
                                   eval_bag_size, grid, args.batch_size, args.workers,
                                   args.eval_tta, args.tta_crop_scale, args.view_batch_size)
        return dict(candidates=len(val_bags), eval_bag_size=eval_bag_size,
                    tta=args.eval_tta, pooling=args.pooling,
                    **bag_recall_at_k(texts, images, pooling=args.pooling, temperature=args.pool_temperature))

    run_epochs(args, model, processor, loader, evaluate, forward_loss, device, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
