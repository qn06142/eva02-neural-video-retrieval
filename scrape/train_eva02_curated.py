#!/usr/bin/env python3
"""Joint training of EVA-02-CLIP-L-14-336 on curated XVideos and Danbooru2023 datasets.

Specialized in:
  - Position / Pose retrieval (e.g., kneeling, from behind, doggystyle, all fours, cowgirl)
  - Clothing / Attire retrieval (e.g., lingerie, stockings, dress, bikini, maid uniform)

Features:
  - Joint multi-modal alignment across both live-action frames and anime illustrations
  - Strict train/validation split (held out 10% on both domains)
  - Non-overfitting prompt generation with tag dropout, shuffling, and syntactic archetypes
  - Parameter-efficient fine-tuning on RTX 4060 8GB GPU (< 4.5 GB VRAM allocated)
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

from PIL import Image
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import open_clip

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_XVIDEOS_DB = os.path.join(PROJECT_DIR, "scrape", "thumbs.db")
DEFAULT_DANBOORU_DB = os.path.join(PROJECT_DIR, "scrape", "danbooru2023_curated", "manifest.sqlite3")
DEFAULT_MODEL = "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"
DEFAULT_OUT = os.path.join(PROJECT_DIR, "scrape", "siglip_runs", "eva02_curated_v2")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Diverse Prompt Generators (Anti-Overfitting)
# =============================================================================

POSITION_SYNONYMS = {
    "kneeling": ["kneeling", "on her knees", "kneeling down", "kneeling pose"],
    "from_behind": ["from behind", "rear view", "back view", "from the back"],
    "all_fours": ["on all fours", "hands and knees", "crawling pose", "all fours"],
    "sitting": ["sitting", "seated", "sitting down", "sitting pose"],
    "lying": ["lying down", "prone", "reclining", "lying on bed"],
    "spread_legs": ["spread legs", "legs spread open", "legs apart"],
    "bent_over": ["bent over", "arching forward", "leaning forward"],
    "arched_back": ["arched back", "arching her spine", "deep arched back"],
    "cowgirl_position": ["riding on top", "cowgirl position", "straddling on top"],
    "doggystyle": ["doggy style", "from behind doggystyle", "doggystyle position"],
    "straddling": ["straddling", "sitting astride", "straddle pose"],
    "squatting": ["squatting", "deep squat", "squatting down"],
    "on_back": ["lying on back", "supine position", "flat on back"],
    "on_stomach": ["lying on stomach", "face down", "flat on stomach", "prone position"],
    "prone_bone": ["prone bone", "lying flat on stomach", "prone position", "face down rear entry"],
    "lotus": ["lotus position", "seated lotus sex", "lotus pose"],
    "spooning": ["spooning", "side by side", "side-lying sex"],
    "standing_split": ["standing split", "standing leg up", "standing splits"],
    "mating_press": ["mating press", "folded legs on back", "mating press pose"],
}

CLOTHING_SYNONYMS = {
    "lingerie": ["sexy lingerie", "sheer lace lingerie", "fine lingerie"],
    "stockings": ["sheer stockings", "black stockings", "thigh high stockings"],
    "thighhighs": ["thighhighs", "over-the-knee socks", "thigh high socks"],
    "dress": ["elegant dress", "sheer dress", "short dress", "tight dress"],
    "bikini": ["tiny bikini", "two-piece bikini", "micro bikini"],
    "swimsuit": ["swimsuit", "one-piece swimsuit", "bathing suit"],
    "maid": ["maid outfit", "French maid costume", "maid uniform"],
    "school_uniform": ["school uniform", "sailor uniform", "pleated school skirt"],
    "underwear": ["underwear", "matching bra and panties", "panties"],
    "panties": ["panties", "lace panties", "silk panties"],
    "bra": ["bra", "lace bra", "demi bra"],
    "bunny_suit": ["bunny suit", "playboy bunny outfit", "bunny girl costume"],
    "skirt": ["short skirt", "mini skirt", "micro skirt"],
    "crop_top": ["crop top", "cropped shirt", "short top"],
}


def clean_tag(tag: str) -> str:
    s = html.unescape(tag or "")
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def format_xvideos_caption(clean_title: str, clothing_tags: list[str],
                            position_tags: list[str], rng: random.Random,
                            diversity: bool = True) -> str:
    """Format diverse natural caption for XVideos curated samples."""
    if not diversity:
        parts = [clean_title]
        if position_tags:
            parts.append(f"in {position_tags[0]} position")
        if clothing_tags:
            parts.append(f"wearing {clothing_tags[0]}")
        return " - ".join(parts)

    pattern = rng.choice(["title_augmented", "pose_first", "search_tube", "natural_sentence"])

    # Resolve synonyms with jitter
    p_tags = [rng.choice(POSITION_SYNONYMS.get(t, [clean_tag(t)])) for t in position_tags]
    c_tags = [rng.choice(CLOTHING_SYNONYMS.get(t, [clean_tag(t)])) for t in clothing_tags]

    pos_phrase = " and ".join(p_tags[:2]) if p_tags else ""
    cloth_phrase = " and ".join(c_tags[:2]) if c_tags else ""

    if pattern == "pose_first" and pos_phrase:
        connector = rng.choice([" featuring ", " with babe ", " showing woman "])
        attire_sub = f" in {cloth_phrase}" if cloth_phrase else ""
        return f"{pos_phrase.capitalize()}{connector}{clean_title}{attire_sub}"

    elif pattern == "search_tube":
        tube_tag = rng.choice(["[HD] ", "[POV] ", "[Amateur] ", "[Exclusive] ", ""])
        tokens = [clean_title]
        if pos_phrase: tokens.append(pos_phrase)
        if cloth_phrase: tokens.append(cloth_phrase)
        rng.shuffle(tokens)
        return f"{tube_tag}{' - '.join(tokens)}"

    elif pattern == "natural_sentence":
        subject = rng.choice(["Babe", "Woman", "Model", "Girl", "Amateur"])
        act = f" posing {pos_phrase}" if pos_phrase else ""
        att = f" wearing {cloth_phrase}" if cloth_phrase else ""
        return f"{subject}{act}{att} - {clean_title}"

    else:  # title_augmented
        aug = []
        if pos_phrase: aug.append(f"in {pos_phrase}")
        if cloth_phrase: aug.append(f"wearing {cloth_phrase}")
        return f"{clean_title} ({', '.join(aug)})" if aug else clean_title


def format_danbooru_caption(character: str, position_tags: list[str],
                             clothing_tags: list[str], all_tags: list[str],
                             rng: random.Random, diversity: bool = True) -> str:
    """Format diverse caption for Danbooru anime illustrations."""
    char_clean = character.replace("_", " ").title() if character else ""

    if not diversity:
        tokens = []
        if char_clean: tokens.append(char_clean)
        tokens.extend([clean_tag(t) for t in position_tags])
        tokens.extend([clean_tag(t) for t in clothing_tags])
        return ", ".join(tokens)

    # Resolve synonyms with jitter
    p_tags = [rng.choice(POSITION_SYNONYMS.get(t, [clean_tag(t)])) for t in position_tags]
    c_tags = [rng.choice(CLOTHING_SYNONYMS.get(t, [clean_tag(t)])) for t in clothing_tags]

    pos_phrase = " and ".join(p_tags[:2]) if p_tags else ""
    cloth_phrase = " and ".join(c_tags[:2]) if c_tags else ""

    pattern = rng.choice(["booru_tags", "natural_sentence", "cosplay_tube", "pose_action"])

    if pattern == "booru_tags":
        # Tag dropout & shuffle
        selected = []
        if char_clean and rng.random() > 0.1:
            selected.append(char_clean.lower())
        for t in p_tags + c_tags:
            if rng.random() > 0.15:
                selected.append(t)
        # Add 1-2 random contextual general tags
        for t in all_tags[:5]:
            if rng.random() < 0.2:
                selected.append(clean_tag(t))
        rng.shuffle(selected)
        return ", ".join(selected) if selected else (char_clean or "anime illustration")

    elif pattern == "natural_sentence":
        subject = char_clean if char_clean else rng.choice(["Waifu", "Anime girl", "Cute anime babe"])
        att = f" wearing {cloth_phrase}" if cloth_phrase else ""
        pos = f" in {pos_phrase} posture" if pos_phrase else ""
        return f"{subject}{att}{pos}"

    elif pattern == "cosplay_tube":
        tube_tag = rng.choice(["[Anime] ", "[Illustration] ", "[Art] ", "[Cosplay] ", ""])
        subject = char_clean if char_clean else "Waifu"
        desc = []
        if pos_phrase: desc.append(pos_phrase)
        if cloth_phrase: desc.append(cloth_phrase)
        return f"{tube_tag}{subject} - {' - '.join(desc)}"

    else:  # pose_action
        subject = char_clean if char_clean else "anime babe"
        if pos_phrase and cloth_phrase:
            return f"{pos_phrase.capitalize()} {subject} in {cloth_phrase}"
        elif pos_phrase:
            return f"{pos_phrase.capitalize()} {subject} pose"
        elif cloth_phrase:
            return f"{subject} wearing {cloth_phrase}"
        return f"{subject} illustration"


# =============================================================================
# PyTorch Datasets
# =============================================================================

class JointCuratedDataset(Dataset):
    """Unified dataset interleaving XVideos and Danbooru2023 items with MIL bag support."""

    def __init__(self, items: list[dict], transform=None, bag_size: int = 4,
                 diversity: bool = True, seed: int = 42):
        self.items = items
        self.transform = transform
        self.bag_size = max(1, bag_size)
        self.diversity = diversity
        self.seed = seed

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str, str]:
        item = self.items[idx]
        domain = item["domain"]
        rng = random.Random(self.seed + idx * 10007)

        # 1. Load image and build MIL bag
        img_path = item["image_path"]
        try:
            im = Image.open(img_path)
            if domain == "xvideos" and item.get("needs_crop", False):
                w, h = im.size
                # 3x3 contact sheet: 9 tiles
                tiles = [
                    im.crop((c * w // 3, r * h // 3, (c + 1) * w // 3, (r + 1) * h // 3))
                    for r in range(3) for c in range(3)
                ]
                if self.diversity:
                    # Training: always include center action frame (tile 4) + sample (bag_size - 1) other tiles
                    other_indices = [i for i in range(9) if i != 4]
                    sampled = rng.sample(other_indices, k=min(self.bag_size - 1, len(other_indices)))
                    chosen_tiles = [tiles[4]] + [tiles[i] for i in sampled]
                else:
                    # Validation: deterministic center tile repeated across bag
                    chosen_tiles = [tiles[4]] * self.bag_size
            else:
                # Danbooru illustration: replicate across bag (receives stochastic augmentations in training)
                chosen_tiles = [im] * self.bag_size

            chosen_tiles = [t.convert("RGB") if t.mode != "RGB" else t for t in chosen_tiles]
        except Exception:
            # Fallback black canvas
            chosen_tiles = [Image.new("RGB", (336, 336), (0, 0, 0)) for _ in range(self.bag_size)]

        if self.transform is not None:
            tile_tensors = [self.transform(t) for t in chosen_tiles]
            image_tensor = torch.stack(tile_tensors, dim=0) # shape: (K, 3, 336, 336)
        else:
            image_tensor = torch.zeros(self.bag_size, 3, 336, 336)

        # 2. Synthesize caption
        if domain == "xvideos":
            caption = format_xvideos_caption(
                item["title"],
                item["clothing_tags"],
                item["position_tags"],
                rng=rng,
                diversity=self.diversity
            )
        else:
            caption = format_danbooru_caption(
                item["character"],
                item["position_tags"],
                item["clothing_tags"],
                item["all_tags"],
                rng=rng,
                diversity=self.diversity
            )

        return image_tensor, caption, domain, str(item["id"])


# =============================================================================
# Data Loading & Partitioning
# =============================================================================

def load_curated_data(xvideos_db: str, danbooru_db: str, val_split: float = 0.1,
                      val_max: int = 1200, danbooru_ratio: float = 0.5,
                      seed: int = 42, max_samples: int = 0) -> Tuple[list[dict], list[dict]]:
    random.seed(seed)
    
    # 1. Load XVideos curated
    log(f"Loading curated XVideos from {xvideos_db}...")
    conn_xv = sqlite3.connect(xvideos_db)
    cur_xv = conn_xv.cursor()
    cur_xv.execute("SELECT eid, clean_title, clothing_tags, position_tags, asset_path FROM videos_curated")
    xv_rows = cur_xv.fetchall()
    conn_xv.close()

    # Fast in-memory disk index via single scandir (0.3s vs 40s of exists calls)
    mosaic_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(xvideos_db))), "mosaics")
    existing_eids = set()
    if os.path.exists(mosaic_dir):
        for entry in os.scandir(mosaic_dir):
            if entry.name.endswith(".jpg"):
                existing_eids.add(entry.name[:-4])

    xv_items = []
    for eid, title, c_json, p_json, asset_path in xv_rows:
        if eid not in existing_eids:
            continue
        xv_items.append({
            "id": f"xv_{eid}",
            "domain": "xvideos",
            "title": title,
            "clothing_tags": json.loads(c_json),
            "position_tags": json.loads(p_json),
            "character": "",
            "all_tags": [],
            "image_path": asset_path,
            "needs_crop": True
        })
    log(f"Valid XVideos items found on disk: {len(xv_items):,}")
    if max_samples > 0 and len(xv_items) > max_samples:
        random.shuffle(xv_items)
        xv_items = xv_items[:max_samples]
        log(f"Subsampled XVideos items to max_samples={max_samples:,}")

    # 2. Load Danbooru curated
    log(f"Loading curated Danbooru from {danbooru_db}...")
    conn_dan = sqlite3.connect(danbooru_db)
    cur_dan = conn_dan.cursor()
    cur_dan.execute("""
        SELECT post_id, character, position_tags, clothing_tags, all_tags, image_path 
        FROM danbooru_curated WHERE is_downloaded = 1
    """)
    dan_rows = cur_dan.fetchall()
    conn_dan.close()

    dan_items = []
    for pid, char, p_json, c_json, a_json, img_path in dan_rows:
        if not img_path:
            continue
        if not os.path.isabs(img_path):
            img_path = os.path.join(PROJECT_DIR, img_path)
        if not os.path.exists(img_path):
            continue
        dan_items.append({
            "id": f"dan_{pid}",
            "domain": "danbooru",
            "title": "",
            "clothing_tags": json.loads(c_json),
            "position_tags": json.loads(p_json),
            "character": char or "",
            "all_tags": json.loads(a_json),
            "image_path": img_path,
            "needs_crop": False
        })
    log(f"Valid Danbooru items found on disk: {len(dan_items):,}")

    # Split each domain individually
    random.shuffle(xv_items)
    random.shuffle(dan_items)

    n_val_xv = int(len(xv_items) * val_split)
    n_val_dan = int(len(dan_items) * val_split)

    val_xv = xv_items[:n_val_xv]
    train_xv = xv_items[n_val_xv:]

    val_dan = dan_items[:n_val_dan]
    train_dan = dan_items[n_val_dan:]

    # Domain balance training set according to danbooru_ratio
    # E.g. If danbooru has 3,000 and xvideos has 23,000, we oversample danbooru so both domains participate equally in gradients
    if danbooru_ratio > 0 and len(train_dan) > 0:
        target_dan_count = int(len(train_xv) * (danbooru_ratio / (1.0 - danbooru_ratio)))
        oversampled_dan = []
        while len(oversampled_dan) < target_dan_count:
            oversampled_dan.extend(train_dan)
        train_dan = oversampled_dan[:target_dan_count]
        random.shuffle(train_dan)

    train_items = train_xv + train_dan
    val_items = val_xv + val_dan

    random.shuffle(train_items)
    random.shuffle(val_items)

    if val_max > 0 and len(val_items) > val_max:
        log(f"Subsampling validation items from {len(val_items):,} to val_max={val_max:,} for fast evaluation.")
        val_items = val_items[:val_max]

    log(f"Dataset split complete: Train={len(train_items):,} (XV={len(train_xv):,}, Dan={len(train_dan):,}) | "
        f"Val={len(val_items):,}")

    return train_items, val_items


# =============================================================================
# Validation / Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_dataset(model: nn.Module, val_loader: DataLoader, tokenizer, device: str,
                     pooling: str = "max", pool_temperature: float = 0.07) -> dict:
    """Evaluate contrastive loss and top-k retrieval on validation set with MIL bag pooling."""
    model.eval()
    total_loss = 0.0
    all_img_feats = []
    all_text_feats = []

    for imgs, captions, domains, ids in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        text_tokens = tokenizer(list(captions)).to(device)

        with torch.amp.autocast('cuda'):
            b, k, c, h, w = imgs.shape
            img_feat = model.encode_image(imgs.view(b * k, c, h, w))
            img_feat = F.normalize(img_feat, dim=-1).view(b, k, -1)

            text_feat = model.encode_text(text_tokens)
            text_feat = F.normalize(text_feat, dim=-1)

            sims = torch.einsum("id,jkd->ijk", text_feat, img_feat)
            if pooling == "max":
                scores = sims.amax(dim=-1)
            else:
                scores = pool_temperature * (torch.logsumexp(sims / pool_temperature, dim=-1) - math.log(k))

            logit_scale = model.logit_scale.exp()
            logits_t2v = logit_scale * scores
            logits_v2t = logits_t2v.t()
            labels = torch.arange(b, device=device)

            loss = (F.cross_entropy(logits_t2v, labels) + F.cross_entropy(logits_v2t, labels)) / 2
            total_loss += loss.item() * b

            all_img_feats.append(img_feat.cpu())
            all_text_feats.append(text_feat.cpu())

    n_samples = sum(len(f) for f in all_img_feats)
    if n_samples == 0:
        return {"loss": 0.0, "r1": 0.0, "r5": 0.0}

    mean_loss = total_loss / n_samples

    # Compute Global Top-K Recall across validation subset
    img_mat = torch.cat(all_img_feats, dim=0) # (N, K, D)
    txt_mat = torch.cat(all_text_feats, dim=0) # (N, D)
    n_eval = len(txt_mat)

    chunk_size = 256
    top1_correct = 0
    top5_correct = 0

    for i in range(0, n_eval, chunk_size):
        t_chunk = txt_mat[i:i+chunk_size]
        sim_chunk = torch.einsum("cd,nkd->cnk", t_chunk, img_mat).amax(dim=-1)
        target_indices = torch.arange(i, min(i + chunk_size, n_eval))

        top1_preds = sim_chunk.argmax(dim=-1)
        top1_correct += (top1_preds == target_indices).sum().item()

        top5_preds = sim_chunk.topk(min(5, n_eval), dim=-1).indices
        for row_idx, target_idx in enumerate(target_indices):
            if target_idx in top5_preds[row_idx]:
                top5_correct += 1

    top1 = (top1_correct / n_eval) * 100
    top5 = (top5_correct / n_eval) * 100

    return {"loss": mean_loss, "r1": top1, "r5": top5, "samples": n_samples}


# =============================================================================
# Training Pipeline
# =============================================================================

def train_eva02(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device} (VRAM constraint: <= 7.0 GB)")

    # 1. Load Data
    train_items, val_items = load_curated_data(
        xvideos_db=args.xvideos_db,
        danbooru_db=args.danbooru_db,
        val_split=args.val_split,
        val_max=args.val_max,
        danbooru_ratio=args.danbooru_ratio,
        seed=args.seed,
        max_samples=getattr(args, "max_samples", 0),
    )

    # 2. Setup Model & Preprocessing
    log(f"Loading EVA-02 model: {args.model_name}...")
    model, _, _ = open_clip.create_model_and_transforms(args.model_name)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    # Visual Augmentation for Training:
    # 1. RandomHorizontalFlip (p=0.5): label-invariant across postures & clothing
    # 2. RandomResizedCrop (scale 0.88-1.0): spatial framing invariance
    # 3. ColorJitter: lighting and white-balance invariance
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            336,
            scale=(0.88, 1.0),
            ratio=(0.9, 1.11),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    # Validation Transform: strictly deterministic evaluation
    val_transform = transforms.Compose([
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    if getattr(args, "weights", None) and os.path.exists(args.weights):
        log(f"Loading initial checkpoint weights: {args.weights}")
        ckpt = torch.load(args.weights, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)

    # 3. Vision Freezing for Robustness and Low VRAM Footprint
    # The pre-trained EVA-02-L MIM trunk has optimal general feature representation.
    # We freeze the visual trunk to prevent forgetting and train the text encoder + joint projection.
    for param in model.visual.trunk.parameters():
        param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_cnt = sum(p.numel() for p in trainable_params)
    total_cnt = sum(p.numel() for p in model.parameters())
    log(f"Trainable parameters: {trainable_cnt:,} / {total_cnt:,} ({trainable_cnt/total_cnt*100:.1f}%)")

    # 4. DataLoaders
    train_dataset = JointCuratedDataset(
        train_items,
        transform=train_transform,
        bag_size=args.bag_size,
        diversity=True,
        seed=args.seed
    )
    val_dataset = JointCuratedDataset(
        val_items,
        transform=val_transform,
        bag_size=args.bag_size,
        diversity=False,
        seed=args.seed
    )

    if args.dynamic_weights:
        log("Calculating dynamic inverse-frequency weights for rare posture balancing...")
        pos_counts = Counter()
        for item in train_items:
            for p in item.get("position_tags", []):
                pos_counts[p] += 1

        med_count = float(np.median(list(pos_counts.values()))) if pos_counts else 100.0
        weights = []
        for item in train_items:
            p_tags = item.get("position_tags", [])
            if p_tags:
                min_c = min(pos_counts.get(p, 1) for p in p_tags)
                w = (med_count / max(1, min_c)) ** args.weight_damping
            else:
                w = 1.0
            weights.append(float(w))

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True
        )
        log(f"Dynamic weighting enabled: balanced {len(pos_counts)} posture categories across {len(weights):,} samples.")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.98))
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')

    # Initial Validation Benchmark
    log("Running zero-shot initial validation benchmark...")
    init_val = evaluate_dataset(model, val_loader, tokenizer, device,
                                pooling=args.pooling, pool_temperature=args.pool_temperature)
    log(f"Initial Zero-Shot Val | Loss: {init_val['loss']:.4f} | R@1: {init_val['r1']:.2f}% | R@5: {init_val['r5']:.2f}%")

    os.makedirs(args.out, exist_ok=True)
    best_r1 = init_val['r1']
    history = []

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, (imgs, captions, domains, ids) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            text_tokens = tokenizer(list(captions)).to(device)

            with torch.amp.autocast('cuda'):
                b, k, c, h, w = imgs.shape
                # Pass vision through frozen trunk without caching graph
                with torch.no_grad():
                    img_feat = model.encode_image(imgs.view(b * k, c, h, w))
                    img_feat = F.normalize(img_feat, dim=-1).view(b, k, -1)

                text_feat = model.encode_text(text_tokens)
                text_feat = F.normalize(text_feat, dim=-1)

                # Pairwise similarities (B, B, K)
                sims = torch.einsum("id,jkd->ijk", text_feat, img_feat)
                if args.pooling == "max":
                    scores = sims.amax(dim=-1)
                else:
                    scores = args.pool_temperature * (torch.logsumexp(sims / args.pool_temperature, dim=-1) - math.log(k))

                logit_scale = model.logit_scale.exp()
                logits_t2v = logit_scale * scores
                logits_v2t = logits_t2v.t()
                labels = torch.arange(b, device=device)

                loss = (F.cross_entropy(logits_t2v, labels) + F.cross_entropy(logits_v2t, labels)) / (2 * args.grad_accum)

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            if step % 50 == 0 or step == len(train_loader):
                cur_lr = scheduler.get_last_lr()[0]
                mem_gb = torch.cuda.memory_allocated() / (1024**3)
                log(f"Epoch {epoch}/{args.epochs} [{step}/{len(train_loader)}] | "
                    f"Loss: {epoch_loss/step:.4f} | LR: {cur_lr:.2e} | VRAM: {mem_gb:.2f} GB")

        epoch_time = time.time() - t0
        avg_train_loss = epoch_loss / len(train_loader)
        log(f"Epoch {epoch} finished in {epoch_time:.1f}s | Train Loss: {avg_train_loss:.4f}")

        # Validation Step
        log(f"Evaluating validation split after Epoch {epoch}...")
        val_metrics = evaluate_dataset(model, val_loader, tokenizer, device,
                                       pooling=args.pooling, pool_temperature=args.pool_temperature)
        log(f"Val Epoch {epoch} | Loss: {val_metrics['loss']:.4f} | "
            f"R@1: {val_metrics['r1']:.2f}% | R@5: {val_metrics['r5']:.2f}%")

        hist_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_r1": val_metrics["r1"],
            "val_r5": val_metrics["r5"],
            "time_sec": epoch_time
        }
        history.append(hist_entry)

        # Save latest
        latest_path = os.path.join(args.out, "latest.pt")
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history
        }, latest_path)

        # Save best
        if val_metrics["r1"] >= best_r1:
            best_r1 = val_metrics["r1"]
            best_path = os.path.join(args.out, "best.pt")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "history": history
            }, best_path)
            log(f"*** New best checkpoint saved with R@1: {best_r1:.2f}% -> {best_path} ***")

    with open(os.path.join(args.out, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    log(f"Joint training complete! Best R@1: {best_r1:.2f}%. Output directory: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint training of EVA-02 on XVideos & Danbooru")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--danbooru-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--xvideos-db", type=str, default=DEFAULT_XVIDEOS_DB)
    parser.add_argument("--danbooru-db", type=str, default=DEFAULT_DANBOORU_DB)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--dynamic-weights", action="store_true", default=True, help="Enable dynamic inverse-frequency posture weighting")
    parser.add_argument("--no-dynamic-weights", dest="dynamic_weights", action="store_false")
    parser.add_argument("--weight-damping", type=float, default=0.5, help="Exponent damping for inverse-frequency weights (default 0.5)")
    parser.add_argument("--weights", type=str, default="", help="Path to initial weights (e.g. eva02_curated_v1/best.pt)")
    parser.add_argument("--max-samples", type=int, default=0, help="Max XVideos samples to use (0 for all available)")
    parser.add_argument("--bag-size", type=int, default=4, help="Number of tiles sampled per video in MIL bag (default 4)")
    parser.add_argument("--pooling", choices=["max", "logmeanexp"], default="max", help="MIL score pooling across bag tiles")
    parser.add_argument("--pool-temperature", type=float, default=0.07, help="LogMeanExp temperature if pooling=logmeanexp")
    parser.add_argument("--val-max", type=int, default=1200, help="Maximum held-out validation samples for fast benchmarking (default 1200)")

    args = parser.parse_args()
    train_eva02(args)
