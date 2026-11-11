#!/usr/bin/env python3
"""Refine SigLIP on explicit Pixiv illustrations + tag/title-adapted captions.

Adapts WD-tagger Pixiv tags into natural adult video titles and hybrid captions
to match SigLIP's target distribution (XVideos titles), while optionally mixing in
real XVideos replay samples to prevent catastrophic forgetting.

Usage
-----
    # Refine from the xvideos frame-interaction checkpoint using mixed video-title prompts
    .venv/Scripts/python scrape/train_siglip_pixiv.py \\
        --init-from scrape/siglip_runs/frame_interaction_adaptive_v2/best \\
        --out scrape/siglip_runs/pixiv_refined_v1 \\
        --prompt-style mixed \\
        --epochs 3 --lr 1e-5 --batch-size 32 --grad-accum 4

    # With 20% XVideos replay to anchor natural video titles
    .venv/Scripts/python scrape/train_siglip_pixiv.py \\
        --init-from scrape/siglip_runs/frame_interaction_adaptive_v2/best \\
        --out scrape/siglip_runs/pixiv_refined_replay_v1 \\
        --prompt-style mixed \\
        --xvideos-ratio 0.2 \\
        --epochs 3 --lr 1e-5
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
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from transformers import SiglipProcessor

from frame_model import SiglipBagModel as SiglipModel
from retrieval_tta import encode_dataset, load_corpus
from training_utils import (
    add_training_options, choose_device, positive_mask,
    retrieval_recall, run_epochs, seed_worker, select_samples, sigmoid_loss,
    split_samples, validate_args,
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PIXIV_DIR = os.path.join(PROJECT_DIR, "scrape", "pixiv_explicit_10k")
DEFAULT_XVIDEOS_DB = os.path.join(PROJECT_DIR, "scrape", "thumbs.db")
DEFAULT_XVIDEOS_THUMBS = os.path.join(PROJECT_DIR, "thumbs")
DEFAULT_MODEL = "google/siglip-base-patch16-224"
DEFAULT_OUT = os.path.join(PROJECT_DIR, "scrape", "siglip_runs", "pixiv_refined_v1")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_tag(tag: str) -> str:
    """Normalize a tag: unescape, replace underscores, collapse spaces."""
    s = html.unescape(tag or "")
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def synthesize_video_title(char_tags: list[str], gen_tags: list[str],
                           rng: Optional[random.Random] = None,
                           diversity: bool = False) -> str:
    """Translate WD-tagger tags into a natural video title with diverse syntactic structures and synonym variations."""
    if rng is None:
        rng = random.Random()
    gen_set = set(g.lower() for g in gen_tags)

    # 1. Subject / Character Identification
    chars = [clean_tag(c).title() for c in char_tags if c]
    if chars:
        char_name = chars[0]
        if diversity and rng.random() < 0.4:
            subject = rng.choice([f"Sexy {char_name}", f"Cute {char_name}", f"Waifu {char_name}", char_name])
        else:
            subject = char_name
    elif "2girls" in gen_set:
        subject = rng.choice(["Two anime babes", "Lesbian anime girls", "Two cute waifus", "Anime babes pair"]) if diversity else "Two anime babes"
    elif "multiple_boys" in gen_set or "gangbang" in gen_set:
        subject = rng.choice(["Anime babe gangbang", "Waifu shared by group", "Anime girl bukkake party"]) if diversity else "Anime babe gangbang"
    elif "milf" in gen_set:
        subject = rng.choice(["Hot anime milf", "Mature busty anime woman", "Sexy anime stepmom", "Horny anime housewife"]) if diversity else "Hot anime milf"
    elif "maid" in gen_set:
        subject = rng.choice(["Cute busty maid", "Sexy French maid waifu", "Submissive anime maid", "Horny maid girl"]) if diversity else "Cute busty maid"
    elif "school_uniform" in gen_set:
        subject = rng.choice(["Anime schoolgirl", "Cute student waifu", "Horny school uniform babe"]) if diversity else "Anime schoolgirl"
    elif "bunny_suit" in gen_set:
        subject = rng.choice(["Sexy bunny girl", "Busty bunny suit waifu", "Playful bunny babe"]) if diversity else "Sexy bunny girl"
    elif "1girl" in gen_set:
        subject = rng.choice([
            "Hot anime girl", "Busty anime babe", "Cute waifu", "Amateur anime girl",
            "Beautiful anime babe", "Gorgeous waifu", "Horny anime babe", "Sexy 2D girl"
        ]) if diversity else "Hot anime girl"
    else:
        subject = rng.choice(["Explicit anime", "Hentai animation", "Anime scene", "Waifu fantasy"]) if diversity else "Explicit anime"

    # 2. Key Attributes (Body, Hair, Clothing)
    attrs = []
    # Breasts
    if any(k in gen_set for k in ("huge_breasts", "gigantic_breasts")):
        attrs.append(rng.choice(["huge tits", "massive breasts", "gigantic tits", "heavy bust"]) if diversity else "huge tits")
    elif any(k in gen_set for k in ("large_breasts", "breasts", "big_breasts")):
        attrs.append(rng.choice(["big tits", "busty body", "large breasts", "stacked cleavage"]) if diversity else "big tits")
    elif "small_breasts" in gen_set:
        attrs.append(rng.choice(["petite body", "tiny tits", "slender frame"]) if diversity else "petite body")

    # Hair
    if "blonde_hair" in gen_set:
        attrs.append(rng.choice(["blonde", "blonde hair", "cute blonde"]) if diversity else "blonde")
    elif "red_hair" in gen_set:
        attrs.append(rng.choice(["redhead", "red hair", "fiery redhead"]) if diversity else "redhead")
    elif "pink_hair" in gen_set:
        attrs.append(rng.choice(["pink hair", "pink haired waifu"]) if diversity else "pink hair")
    elif "black_hair" in gen_set:
        attrs.append(rng.choice(["black hair", "dark hair"]) if diversity else "black hair")
    elif "blue_hair" in gen_set:
        attrs.append(rng.choice(["blue hair", "blue haired"]) if diversity else "blue hair")
    elif "white_hair" in gen_set or "silver_hair" in gen_set:
        attrs.append(rng.choice(["silver hair", "white hair"]) if diversity else "silver hair")

    # Legwear & Outfits
    if any(k in gen_set for k in ("fishnets", "fishnet_stockings")):
        attrs.append(rng.choice(["fishnet stockings", "sexy fishnets"]) if diversity else "fishnet stockings")
    elif any(k in gen_set for k in ("thighhighs", "stockings")):
        attrs.append(rng.choice(["black stockings", "thigh high stockings", "sheer stockings"]) if diversity else "black stockings")
    elif "lingerie" in gen_set or "underwear" in gen_set:
        attrs.append(rng.choice(["sexy lingerie", "lace underwear", "sheer lingerie"]) if diversity else "sexy lingerie")
    elif "bikini" in gen_set or "swimsuit" in gen_set:
        attrs.append(rng.choice(["tiny bikini", "sexy swimsuit", "micro bikini"]) if diversity else "tiny bikini")

    connector = rng.choice([" with ", " wearing ", " showing ", " in ", " flaunting "]) if diversity else " with "
    attr_phrase = (connector + " and ".join(attrs[:2])) if attrs else ""

    # 3. Actions / Sexual Acts
    actions = []
    actions_gerund = []
    actions_past = []

    if any(k in gen_set for k in ("deepthroat", "irrumatio")):
        actions.append(rng.choice(["deepthroat blowjob", "deepthroat head", "intense deepthroat"]) if diversity else "deepthroat blowjob")
        actions_gerund.append("deepthroating cock")
        actions_past.append("deepthroated")
    elif any(k in gen_set for k in ("fellatio", "oral")):
        actions.append(rng.choice(["sucking cock", "sloppy blowjob", "giving head", "oral sex"]) if diversity else "sucking cock")
        actions_gerund.append("sucking cock")
        actions_past.append("given oral")

    if "cunnilingus" in gen_set:
        actions.append(rng.choice(["eating wet pussy", "eating her out", "licking pussy"]) if diversity else "eating wet pussy")
        actions_gerund.append("licking her wet pussy")
        actions_past.append("eaten out")

    if "paizuri" in gen_set:
        actions.append(rng.choice(["titfuck", "boob job", "busty titfuck"]) if diversity else "titfuck")
        actions_gerund.append("titfucking")
        actions_past.append("titfucked")

    if "doggystyle" in gen_set or "from_behind" in gen_set:
        actions.append(rng.choice(["fucked doggystyle", "bent over from behind", "doggy style pounding"]) if diversity else "fucked doggystyle")
        actions_gerund.append("pounding from behind")
        actions_past.append("fucked doggystyle")
    elif any(k in gen_set for k in ("cowgirl_position", "riding", "reverse_cowgirl")):
        actions.append(rng.choice(["riding cock", "bouncing on cock", "cowgirl position"]) if diversity else "riding cock")
        actions_gerund.append("riding cock")
        actions_past.append("ridden hard")
    elif any(k in gen_set for k in ("anal", "anal_penetration")):
        actions.append(rng.choice(["hard anal pounding", "tight anal sex", "anal fuck"]) if diversity else "hard anal pounding")
        actions_gerund.append("pounding her tight ass")
        actions_past.append("anal fucked")
    elif any(k in gen_set for k in ("sex", "hetero", "penetration", "vaginal")):
        actions.append(rng.choice(["getting pounded bareback", "intense hardcore sex", "getting fucked hard"]) if diversity else "getting pounded bareback")
        actions_gerund.append("fucking hard bareback")
        actions_past.append("pounded bareback")
    elif any(k in gen_set for k in ("masturbation", "fingering")):
        actions.append(rng.choice(["fingering wet pussy", "solo masturbation", "rubbing clit"]) if diversity else "fingering wet pussy")
        actions_gerund.append("fingering her wet pussy")
        actions_past.append("fingered")

    # Climax / Outcome
    outcomes = []
    if any(k in gen_set for k in ("creampie", "nakadashi")):
        outcomes.append(rng.choice(["huge creampie", "filled with cum", "creampied inside", "warm creampie"]) if diversity else "huge creampie")
    elif any(k in gen_set for k in ("facial", "cum_on_face")):
        outcomes.append(rng.choice(["facial cumshot", "cum on face", "messy facial"]) if diversity else "facial cumshot")
    elif "squirt" in gen_set or "squirting" in gen_set:
        outcomes.append(rng.choice(["squirting orgasm", "gushing squirt", "wild squirting"]) if diversity else "squirting orgasm")

    # Setting / Environment
    setting = ""
    if diversity and rng.random() < 0.35:
        if "outdoors" in gen_set or "nature" in gen_set:
            setting = rng.choice([" outdoors", " in public", " outside"])
        elif "on_bed" in gen_set or "bedroom" in gen_set:
            setting = rng.choice([" in bed", " on the bed", " in bedroom"])
        elif "pool" in gen_set or "beach" in gen_set:
            setting = rng.choice([" by the pool", " at the beach"])
        elif "onsen" in gen_set or "hot_springs" in gen_set:
            setting = rng.choice([" in the hot spring", " in onsen"])

    # POV modifier
    pov_active = "pov" in gen_set
    pov_prefix = (rng.choice(["POV ", "POV - ", "First Person POV "]) if diversity else "POV ") if pov_active else ""

    # Quality / Tag prefix (tube styling)
    tube_tag = ""
    if diversity and rng.random() < 0.25:
        tube_tag = rng.choice(["[Uncensored] ", "[HD] ", "[4K] ", "[Amateur] ", "[Exclusive] ", "[Cosplay] "])

    # Choose Syntactic Pattern
    pattern = rng.choice(["active", "gerund", "passive", "search", "tagged"]) if diversity else "active"

    all_actions = actions[:2] + outcomes[:1]
    action_str = (" and ".join(all_actions)) if all_actions else "showing sexy body"

    if pattern == "gerund" and actions_gerund:
        title = f"{tube_tag}{pov_prefix}{actions_gerund[0]} {subject}{attr_phrase}{setting}"
    elif pattern == "passive" and actions_past:
        outcome_str = f" with {outcomes[0]}" if outcomes else ""
        title = f"{tube_tag}{pov_prefix}{subject}{attr_phrase} gets {actions_past[0]}{setting}{outcome_str}"
    elif pattern == "search":
        # Search query syntax: concise keywords without filler prepositions
        act_kw = " ".join(actions[:2] + outcomes[:1])
        title = f"{subject} {' '.join(attrs[:2])} {act_kw}".strip()
    elif pattern == "tagged":
        title = f"{tube_tag}{subject}{attr_phrase} - {action_str}{setting}"
    else:  # "active" standard pattern
        title = f"{tube_tag}{pov_prefix}{subject}{attr_phrase} {action_str}{setting}"

    return re.sub(r"\s+", " ", title).strip()


def format_tag_list(character_tags: list[str], general_tags: list[str],
                    shuffle: bool = False, dropout: float = 0.0,
                    max_tags: int = 50, rng: Optional[random.Random] = None) -> str:
    """Format character and general tags into a comma-separated list."""
    if rng is None:
        rng = random.Random()
    chars = [clean_tag(t) for t in character_tags if clean_tag(t)]
    gens = [clean_tag(t) for t in general_tags if clean_tag(t)]

    if dropout > 0.0 and gens:
        gens = [t for t in gens if rng.random() > dropout]

    if shuffle and gens:
        gens = list(gens)
        rng.shuffle(gens)

    combined = chars + gens
    if max_tags > 0:
        combined = combined[:max_tags]
    return ", ".join(combined)


def format_pixiv_caption(char_tags: list[str], gen_tags: list[str],
                         style: str = "mixed", shuffle: bool = False,
                         dropout: float = 0.0, max_tags: int = 40,
                         diversity: bool = False,
                         rng: Optional[random.Random] = None) -> str:
    """Construct prompt according to the selected style (mixed, video-title, hybrid, search-query, tag-list)."""
    if rng is None:
        rng = random.Random()

    active_style = style
    if style == "mixed":
        # 30% video title, 30% hybrid, 25% search-query, 15% tag list
        roll = rng.random()
        if roll < 0.30:
            active_style = "video-title"
        elif roll < 0.60:
            active_style = "hybrid"
        elif roll < 0.85:
            active_style = "search-query"
        else:
            active_style = "tag-list"

    if active_style == "video-title":
        return synthesize_video_title(char_tags, gen_tags, rng=rng, diversity=diversity)
    elif active_style == "hybrid":
        title = synthesize_video_title(char_tags, gen_tags, rng=rng, diversity=diversity)
        tags_str = format_tag_list(char_tags, gen_tags, shuffle=shuffle, dropout=dropout, max_tags=max_tags, rng=rng)
        sep = rng.choice([" | ", " - ", " // ", " "]) if diversity else " | "
        return f"{title}{sep}{tags_str}"
    elif active_style == "search-query":
        # Raw space-separated keyword search query without prepositions
        title = synthesize_video_title(char_tags, gen_tags, rng=rng, diversity=True)
        gens_sample = [clean_tag(t) for t in gen_tags if clean_tag(t)]
        if shuffle:
            rng.shuffle(gens_sample)
        if dropout > 0.0:
            gens_sample = [t for t in gens_sample if rng.random() > dropout]
        selected_gens = " ".join(gens_sample[:8])
        return f"{title} {selected_gens}".strip()
    else:  # tag-list
        return format_tag_list(char_tags, gen_tags, shuffle=shuffle, dropout=dropout, max_tags=max_tags, rng=rng)


def load_pixiv_records(pixiv_dir: str, prompt_style: str = "mixed",
                       max_samples: Optional[int] = None,
                       min_tags: int = 3, seed: int = 42) -> list[dict]:
    """Query manifest.sqlite3 in the Pixiv directory and return valid image records."""
    db_path = os.path.join(pixiv_dir, "manifest.sqlite3")
    images_dir = os.path.join(pixiv_dir, "images")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Pixiv manifest database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, sha256, metadata FROM images ORDER BY id").fetchall()
    conn.close()

    records = []
    skipped_missing = 0
    skipped_tags = 0

    for img_id, sha256, meta_str in rows:
        try:
            meta = json.loads(meta_str)
        except Exception:
            continue

        img_path = meta.get("path")
        if not img_path or not os.path.exists(img_path):
            img_path = os.path.join(images_dir, f"{sha256}.webp")
            if not os.path.exists(img_path):
                skipped_missing += 1
                continue

        tags_info = meta.get("tags_and_ratings", {})
        char_tags = tags_info.get("character tags", [])
        gen_tags = tags_info.get("general tags", [])

        if len(char_tags) + len(gen_tags) < min_tags:
            skipped_tags += 1
            continue

        # Canonical prompt for deterministic grouping/splitting: hybrid format with seed
        canonical = format_pixiv_caption(char_tags, gen_tags, style="hybrid",
                                         shuffle=False, dropout=0.0,
                                         rng=random.Random(img_id))
        if not canonical:
            skipped_tags += 1
            continue

        records.append({
            "id": img_id,
            "sha256": sha256,
            "path": img_path,
            "char_tags": char_tags,
            "gen_tags": gen_tags,
            "canonical": canonical,
            "is_xvideos": False,
        })

    if skipped_missing or skipped_tags:
        log(f"Pixiv loader: skipped {skipped_missing} missing files, {skipped_tags} insufficient-tag items")

    if not records:
        return []

    records = sorted(records, key=lambda r: r["path"])
    random.Random(seed).shuffle(records)
    if max_samples is not None:
        records = records[:max_samples]
    return records


def load_xvideos_replay_pairs(db_path: str, thumbs_dir: str, count: int,
                              min_title_len: int = 3, seed: int = 42) -> list[dict]:
    """Load real XVideos (path, title) pairs to replay during training."""
    if not os.path.exists(db_path):
        log(f"XVideos replay db not found at {db_path}; continuing without replay")
        return []

    items, _ = load_corpus(db_path, thumbs_dir, "", asset_type="thumbnail")
    valid = []
    for path, eid, raw_title in items:
        if not os.path.exists(path):
            continue
        title = html.unescape(raw_title or "").replace("\r", " ").replace("\n", " ")
        title = re.sub(r"\s+", " ", title).strip()
        if len(title) >= min_title_len:
            valid.append({
                "id": f"xv_{eid}",
                "sha256": eid,
                "path": path,
                "char_tags": [],
                "gen_tags": [],
                "canonical": title,
                "is_xvideos": True,
            })

    if not valid:
        return []
    valid = sorted(valid, key=lambda r: r["id"])
    random.Random(seed).shuffle(valid)
    return valid[:count]


class PixivAdaptedDataset(Dataset):
    """Dataset yielding (PIL.Image, caption) with training augmentations and target-adapted prompts."""

    def __init__(self, records: list[dict], processor: SiglipProcessor,
                 prompt_style: str = "mixed", train: bool = False,
                 tag_shuffle: bool = True, tag_dropout: float = 0.15,
                 max_tags: int = 40, augment: bool = True):
        self.records = records
        self.processor = processor
        self.prompt_style = prompt_style
        self.train = train
        self.tag_shuffle = tag_shuffle and train
        self.tag_dropout = tag_dropout if train else 0.0
        self.max_tags = max_tags

        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(0.7, 1.0), ratio=(0.85, 1.15),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]) if (train and augment) else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        rec = self.records[idx]
        path = rec["path"]
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
        except (OSError, ValueError) as exc:
            raise OSError(f"Cannot decode image: {path}") from exc

        if self.aug is not None:
            img = self.aug(img)

        # For XVideos replay items, always use the real title
        if rec.get("is_xvideos"):
            return img, rec["canonical"]

        # For Pixiv items, format according to prompt style
        if self.train:
            caption = format_pixiv_caption(
                rec["char_tags"], rec["gen_tags"],
                style=self.prompt_style,
                shuffle=self.tag_shuffle,
                dropout=self.tag_dropout,
                max_tags=self.max_tags,
                diversity=True,
            )
        else:
            caption = rec["canonical"]

        return img, caption


class Collator:
    """Picklable collator building token IDs, attention mask, pixel values, and positive masks."""

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
def encode(model, processor, records, device, batch_size, workers,
           prompt_style="hybrid", tta="none", crop_scale=0.9, view_batch_size=64) -> Tuple[torch.Tensor, torch.Tensor]:
    ds = PixivAdaptedDataset(records, processor, prompt_style=prompt_style, train=False, augment=False)
    images, texts = encode_dataset(model, processor, ds, device, batch_size, workers,
                                   tta, crop_scale, view_batch_size)
    return images[:, 0], texts


def recall_at_k(text_embeds: torch.Tensor, image_embeds: torch.Tensor,
                ks=(1, 5, 10)) -> dict:
    return retrieval_recall(text_embeds, image_embeds, ks)


def configure_parameter_freezing(model, args):
    """Set trainable parameter flags across backbone and frame mixer."""
    has_mixer = getattr(model, "frame_mixer", None) is not None

    if args.freeze_backbone:
        model.vision_model.requires_grad_(False)
        model.text_model.requires_grad_(False)
        log("Backbone (vision and text encoders) frozen")
    else:
        if args.freeze_vision:
            model.vision_model.requires_grad_(False)
            log("Vision encoder frozen")
        else:
            model.vision_model.requires_grad_(True)

        if args.freeze_text:
            model.text_model.requires_grad_(False)
            log("Text encoder frozen")
        else:
            model.text_model.requires_grad_(True)

    if has_mixer:
        if args.train_frame_mixer:
            model.frame_mixer.requires_grad_(True)
            log("Frame mixer trainable")
        else:
            model.frame_mixer.requires_grad_(False)
            log("Frame mixer frozen (preserving trained video interaction weights)")


def main() -> int:
    p = argparse.ArgumentParser(description="Refine SigLIP on explicit Pixiv illustrations with video-title adapted prompts.")
    p.add_argument("--pixiv-dir", default=DEFAULT_PIXIV_DIR,
                   help=f"Directory containing manifest.sqlite3 and images/ (default: {DEFAULT_PIXIV_DIR})")
    p.add_argument("--prompt-style", choices=["mixed", "video-title", "hybrid", "tag-list"], default="mixed",
                   help="Prompt format for training: mixed (default, 45% video-title, 40% hybrid, 15% tags), video-title, hybrid, or tag-list")
    p.add_argument("--xvideos-db", default=DEFAULT_XVIDEOS_DB,
                   help="Optional XVideos SQLite db to mix replay samples into training")
    p.add_argument("--xvideos-thumbs", default=DEFAULT_XVIDEOS_THUMBS,
                   help="Directory of XVideos thumbnails")
    p.add_argument("--xvideos-ratio", type=float, default=0.0,
                   help="Fraction of training set composed of real XVideos replay samples (0.0 to 0.5, default: 0.0)")

    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Default model or Hugging Face Hub ID (default: {DEFAULT_MODEL})")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"Output directory for checkpoints and metrics (default: {DEFAULT_OUT})")
    p.add_argument("--resume", default=None,
                   help="Epoch checkpoint containing model, processor, and training_state.pt")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--no-augment", action="store_true", help="Disable training image augmentations")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                   help="Attention implementation (default: sdpa)")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers (default: 0 for stable Windows memory)")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--val-max", type=int, default=1000)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Optional limit on total samples loaded")
    p.add_argument("--min-tags", type=int, default=3,
                   help="Minimum number of tags required per image")
    p.add_argument("--min-title-len", type=int, default=3,
                   help="Minimum characters in caption prompt (default: 3)")
    p.add_argument("--max-tags", type=int, default=40,
                   help="Maximum number of tags included in caption prompt")
    p.add_argument("--tag-shuffle", action="store_true", default=True,
                   help="Shuffle general tags during training")
    p.add_argument("--no-tag-shuffle", dest="tag_shuffle", action="store_false")
    p.add_argument("--tag-dropout", type=float, default=0.05,
                   help="Probability of dropping individual general tags during training (default: 0.05)")

    p.add_argument("--freeze-backbone", action="store_true",
                   help="Freeze both vision and text encoders")
    p.add_argument("--freeze-vision", action="store_true", help="Freeze vision encoder only")
    p.add_argument("--freeze-text", action="store_true", help="Freeze text encoder only")
    p.add_argument("--train-frame-mixer", action="store_true",
                   help="Train the frame mixer head along with the backbone (default is frozen)")
    p.add_argument("--use-frame-mixer", action="store_true", default=True,
                   help="Pass single-image features through frame mixer if present")
    p.add_argument("--no-use-frame-mixer", dest="use_frame_mixer", action="store_false")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    add_training_options(p)
    args = p.parse_args()
    validate_args(args)

    if not 0.0 <= args.xvideos_ratio <= 0.8:
        raise ValueError("--xvideos-ratio must be between 0.0 and 0.8")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args)
    log(f"device: {device}")
    if device.type == "cuda":
        log(f"gpu: {torch.cuda.get_device_name(0)}")

    # ---- 1. Data loading ------------------------------------------------ #
    log(f"loading Pixiv data from {args.pixiv_dir} (prompt-style={args.prompt_style})")
    records = load_pixiv_records(args.pixiv_dir, args.prompt_style,
                                 args.max_samples, args.min_tags, args.seed)
    if len(records) < 10:
        log(f"Not enough valid Pixiv samples found ({len(records)}). Ensure download is running or completed.")
        return 1
    log(f"valid Pixiv samples: {len(records):,}")

    # Optional XVideos replay samples
    if args.xvideos_ratio > 0.0:
        xv_target = int(len(records) * (args.xvideos_ratio / (1.0 - args.xvideos_ratio)))
        replay_records = load_xvideos_replay_pairs(
            args.xvideos_db, args.xvideos_thumbs, xv_target,
            min_title_len=args.min_title_len, seed=args.seed
        )
        if replay_records:
            log(f"interleaving {len(replay_records):,} XVideos replay pairs ({len(replay_records)/(len(records)+len(replay_records))*100:.1f}%)")
            records.extend(replay_records)
            random.Random(args.seed).shuffle(records)

    samples_for_split = [(r["path"], r["canonical"]) for r in records]
    path_to_record = {r["path"]: r for r in records}

    train_pairs, val_pairs = split_samples(
        samples_for_split, args.val_frac, args.val_max, args.seed, args.out
    )
    train_records = [path_to_record[p] for p, _ in train_pairs]
    val_records = [path_to_record[p] for p, _ in val_pairs]
    log(f"train set: {len(train_records):,}  val set: {len(val_records):,}")

    # ---- 2. Model loading ----------------------------------------------- #
    src = args.resume or args.init_from or args.model
    log(f"loading model from {src} (attn={args.attn})")
    processor = SiglipProcessor.from_pretrained(src)
    model = SiglipModel.from_pretrained(src, attn_implementation=args.attn)
    has_mixer = getattr(model, "frame_mixer", None) is not None
    if has_mixer:
        log("Model has ResidualFrameMixer attached")
    else:
        log("Standard SigLIP dual encoder without frame mixer")

    configure_parameter_freezing(model, args)
    model.to(device)

    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"trainable parameters: {train_params:,} / {total_params:,} ({train_params/total_params*100:.1f}%)")

    # ---- 3. DataLoaders ------------------------------------------------- #
    generator = torch.Generator().manual_seed(args.seed)
    train_ds = PixivAdaptedDataset(
        train_records, processor, prompt_style=args.prompt_style,
        train=not args.no_augment, tag_shuffle=args.tag_shuffle,
        tag_dropout=args.tag_dropout, max_tags=args.max_tags,
    )
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=Collator(processor),
        pin_memory=(device.type == "cuda"), worker_init_fn=seed_worker,
        generator=generator, drop_last=False,
    )

    # ---- 4. Forward loss and evaluation closures ------------------------ #
    def forward_loss(batch):
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device),
        )
        img_embeds = out.image_embeds
        if has_mixer and args.use_frame_mixer:
            img_embeds = model.mix_frame_features(img_embeds[:, None])[:, 0]

        return sigmoid_loss(model, img_embeds, out.text_embeds, batch["positive_mask"])

    def evaluate():
        images, texts = encode(
            model, processor, val_records, device,
            args.batch_size, args.workers,
            prompt_style="hybrid",
            tta=args.eval_tta, crop_scale=args.tta_crop_scale,
            view_batch_size=args.view_batch_size,
        )
        recalls = recall_at_k(texts, images)
        return dict(tta=args.eval_tta, candidates=len(val_records), **recalls)

    # ---- 5. Run training epochs ----------------------------------------- #
    run_epochs(args, model, processor, loader, evaluate, forward_loss, device, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
