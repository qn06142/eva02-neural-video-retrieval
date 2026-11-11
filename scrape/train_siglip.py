#!/usr/bin/env python3
"""Fine-tune SigLIP on scraped xvideos thumbnails + titles.

Data comes straight from the crawler's SQLite db: `videos.title` paired with
`thumbs/<eid>.jpg`.  The model is the standard SigLIP dual encoder with the
official sigmoid (pairwise BCE) contrastive loss, trained with mixed precision.

Usage
-----
    .venv/Scripts/python scrape/train_siglip.py --max-samples 2000 --epochs 1
    .venv/Scripts/python scrape/train_siglip.py                       # full run
    .venv/Scripts/python scrape/train_siglip.py --init-from scrape/siglip_runs/run1/best
"""

from __future__ import annotations

import argparse
import numpy as np
import html
import os
import random
import re
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from transformers import (
    SiglipProcessor,
)
from frame_model import SiglipBagModel as SiglipModel

from retrieval_tta import encode_dataset, load_corpus

from training_utils import (
    add_training_options, choose_device, positive_mask,
    retrieval_recall, run_epochs, seed_worker, select_samples, sigmoid_loss,
    split_samples, validate_args,
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(PROJECT_DIR, "scrape", "thumbs.db")
DEFAULT_THUMBS = os.path.join(PROJECT_DIR, "thumbs")
DEFAULT_MODEL = "google/siglip-base-patch16-224"
DEFAULT_OUT = os.path.join(PROJECT_DIR, "scrape", "siglip_runs", "run1")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_title(s: str) -> str:
    s = html.unescape(s or "")
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_pairs(db_path: str, thumbs_dir: str, max_samples: Optional[int],
               min_title_len: int, seed: int = 42) -> list:
    """Read (image_path, title) pairs from the crawler's db."""
    items, _ = load_corpus(db_path, thumbs_dir, "", asset_type="thumbnail")

    pairs, skipped = [], 0
    for path, eid, raw_title in items:
        if not os.path.exists(path):
            skipped += 1
            continue
        title = clean_title(raw_title)
        if len(title) < min_title_len:
            skipped += 1
            continue
        pairs.append((path, title))
    if skipped:
        log(f"skipped {skipped} rows (missing image or empty title)")
    return select_samples(pairs, max_samples, seed)


class ThumbTitleDataset(Dataset):
    def __init__(self, pairs: list, processor: SiglipProcessor, train: bool = False):
        self.pairs = pairs
        self.processor = processor
        # Training-only augmentation: the eval transform stays deterministic.
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(0.7, 1.0), ratio=(0.85, 1.15),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]) if train else None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path, title = self.pairs[idx]
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
        except (OSError, ValueError) as exc:
            raise OSError(f"Cannot decode thumbnail: {path}") from exc
        if self.aug is not None:
            img = self.aug(img)
        return img, title


def _worker_init(worker_id):
    """DataLoader workers must not each spawn a full set of torch CPU threads."""
    seed_worker(worker_id)


class Collator:
    """Picklable collate fn (module-level so Windows spawn can pickle it).

    SiglipProcessor does not return an attention_mask (its tokenizer only emits
    input_ids), so build one: 1 for real tokens up to and including the eos
    (which shares id 1 with the pad token), 0 for the right-padding after it.
    """

    def __init__(self, processor: SiglipProcessor):
        self.processor = processor

    def __call__(self, batch):
        images = [b[0] for b in batch]
        texts = [b[1] for b in batch]
        enc = self.processor(
            text=texts,
            images=images,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        ids = enc["input_ids"]
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None and "attention_mask" not in enc:
            is_pad = ids == pad_id
            first_pad = is_pad.float().argmax(dim=1)
            ar = torch.arange(ids.size(1)).unsqueeze(0)
            mask = (ar <= first_pad.unsqueeze(1)).long()
            mask[~is_pad.any(dim=1)] = 1
            enc["attention_mask"] = mask
        enc["positive_mask"] = positive_mask(texts)
        return enc


@torch.no_grad()
def encode(model, processor, pairs, device, batch_size, workers,
           tta="none", crop_scale=.9, view_batch_size=64) -> tuple:
    images, texts = encode_dataset(model, processor, ThumbTitleDataset(pairs, processor),
                                   device, batch_size, workers, tta, crop_scale, view_batch_size)
    return images[:, 0], texts


def recall_at_k(text_embeds: torch.Tensor, image_embeds: torch.Tensor,
                ks=(1, 5, 10)) -> dict:
    return retrieval_recall(text_embeds, image_embeds, ks)


def main() -> int:
    p = argparse.ArgumentParser(description="Fine-tune SigLIP on thumb+title pairs.")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--thumbs-dir", default=DEFAULT_THUMBS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--resume", default=None,
                   help="epoch checkpoint containing model, processor and training_state.pt")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                   help="attention implementation (default: sdpa)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4,
                   help="dataloader workers (default: 4; 0 = load in main process)")
    p.add_argument("--val-frac", type=float, default=0.01)
    p.add_argument("--val-max", type=int, default=2000)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--min-title-len", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    add_training_options(p)
    args = p.parse_args()
    validate_args(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args)
    log(f"device: {device}")
    if device.type == "cuda":
        log(f"gpu: {torch.cuda.get_device_name(0)}")

    # ---- data ---------------------------------------------------------- #
    pairs = load_pairs(args.db, args.thumbs_dir, args.max_samples,
                       args.min_title_len, args.seed)
    if len(pairs) < 10:
        log("Not enough thumbnail pairs. For contact-sheet-only data use train_siglip_mil.py.")
        return 1
    log(f"pairs: {len(pairs):,}")

    train_pairs, val_pairs = split_samples(
        pairs, args.val_frac, args.val_max, args.seed, args.out)
    log(f"train: {len(train_pairs):,}  val: {len(val_pairs):,}")

    # ---- model --------------------------------------------------------- #
    src = args.resume or args.init_from or args.model
    log(f"loading model from {src} (attn={args.attn})")
    processor = SiglipProcessor.from_pretrained(src)
    model = SiglipModel.from_pretrained(src, attn_implementation=args.attn)
    if model.frame_mixer is not None:
        raise ValueError("Use train_siglip_mil.py for frame-mixer checkpoints")
    model.to(device)

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        ThumbTitleDataset(train_pairs, processor, train=not args.no_augment),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=Collator(processor), pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker, generator=generator, drop_last=False,
    )

    def forward_loss(batch):
        out = model(input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(device))
        return sigmoid_loss(model, out.image_embeds, out.text_embeds, batch["positive_mask"])

    def evaluate():
        images, texts = encode(model, processor, val_pairs, device,
                               args.batch_size, args.workers,
                               args.eval_tta, args.tta_crop_scale, args.view_batch_size)
        return dict(tta=args.eval_tta, candidates=len(val_pairs), **recall_at_k(texts, images))

    run_epochs(args, model, processor, loader, evaluate, forward_loss, device, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
