"""Deterministic image-view ensembles shared by validation and retrieval."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import sqlite3

from PIL import ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gpu_pacing import paced_gpu_call

TTA_POLICIES = ("none", "flip", "five-crop", "ten-crop")
TTA_VERSION = 1
TEXT_TTA_MODES = ("none", "accurate", "vague")


def expand_text_query(text: str, mode: str = "none") -> list[str]:
    """Expand a query text into targeted or exploratory prompt variations without rigid overfitting.

    Modes:
      - 'none': returns [text]
      - 'accurate': focused domain extensions (video title, hybrid, booru) for high-precision centroid matching
      - 'vague': diverse exploratory variations across acts, settings, and styles for max-similarity retrieval
    """
    if mode not in TEXT_TTA_MODES:
        raise ValueError(f"Invalid text TTA mode: {mode} (choose from {TEXT_TTA_MODES})")

    clean = " ".join(text.strip().split())
    if mode == "none" or not clean:
        return [clean]

    low = clean.lower()

    if mode == "accurate":
        # Targeted variants across multiple domain syntaxes:
        # 1. Exact query
        # 2. Category / Explicit modifier
        # 3. Subject-integrated phrase
        # 4. First-person / POV perspective
        # 5. Hybrid search / tube syntax
        # 6. Danbooru tag syntax
        # 7. Stylized anime / illustration syntax
        return [
            clean,
            f"Explicit {clean}",
            f"Hot girl with {clean}",
            f"POV {clean}",
            f"{clean} | {low}, explicit",
            f"1girl, {low}, solo, explicit",
            f"Anime illustration of {clean}",
            f"[Uncensored] {clean}",
            f"{clean} high quality",
        ]

    elif mode == "vague":
        # Diverse multi-aspect exploratory variations spanning acts, settings, perspectives, and styling
        return [
            clean,
            f"POV hot girl with {clean} posing on bed",
            f"Busty babe with {clean} getting pounded bareback",
            f"Couple having passionate sex with {clean}",
            f"Explicit anime waifu with {clean}",
            f"Hot girl showing {clean} bareback with creampie",
            f"1girl, solo, {low}, stockings, breasts, blush",
            f"1girl, 1boy, {low}, sex, hetero, oral",
            f"Beautiful sexy {clean} in bedroom",
            f"Amateur {clean} bareback",
            f"POV {clean} deepthroat blowjob",
            f"{clean} | high quality, detailed, explicit",
            f"Japanese babe {clean} outdoors",
            f"{clean} doggystyle creampie",
            f"Cute waifu {clean} teasing nude",
        ]


def encode_text_query(encode_fn, text: str, mode: str = "none") -> tuple[torch.Tensor, str]:
    """Encode query with optional text TTA.

    Returns:
      (q_tensor, mode):
        - If mode in ('none', 'accurate'): (1, D) normalized vector
        - If mode == 'vague': (M, D) multi-query tensor for max-pool similarity
    """
    variants = expand_text_query(text, mode=mode)
    features = encode_fn(variants)  # shape (M, D), normalized

    if mode == "accurate":
        centroid = F.normalize(features.mean(dim=0, keepdim=True), dim=-1)
        return centroid, "accurate"
    elif mode == "vague":
        return features, "vague"
    else:
        return features, "none"


def image_views(image, policy="none", crop_scale=.9):
    """Corner/center crops preserve aspect ratio; scale is per side, not area."""
    if policy not in TTA_POLICIES or not math.isfinite(crop_scale) or not 0 < crop_scale <= 1:
        raise ValueError("Invalid TTA policy or crop scale (expected 0 < scale <= 1)")
    if policy == "none":
        return [image]
    if policy == "flip":
        return [image, ImageOps.mirror(image)]
    w, h = image.size
    cw, ch = max(1, round(w * crop_scale)), max(1, round(h * crop_scale))
    offsets = [(0, 0), (w-cw, 0), (0, h-ch), (w-cw, h-ch), ((w-cw)//2, (h-ch)//2)]
    crops = [image.crop((x, y, x+cw, y+ch)) for x, y in offsets]
    return crops if policy == "five-crop" else crops + [ImageOps.mirror(im) for im in crops]


@torch.inference_mode()
def encode_tta_images(images, pixel_values, encode_image, device, policy="none",
                      crop_scale=.9, view_batch_size=64):
    """Normalize each view, average per image, then normalize for cosine search.

    Preprocessing and GPU forward batches contain at most view_batch_size
    views, regardless of the number of input images or the selected policy.
    """
    if not images or view_batch_size < 1:
        raise ValueError("Need images and a positive view batch size")
    pending, owners, summed = [], [], None

    def flush():
        nonlocal summed
        pixels = pixel_values(pending).to(device, non_blocking=True)
        features = F.normalize(paced_gpu_call(encode_image, pixels).float(), dim=-1)
        if features.ndim != 2 or len(features) != len(pending) or not torch.isfinite(features).all():
            raise ValueError("Image encoder returned invalid features")
        if summed is None:
            summed = features.new_zeros((len(images), features.shape[-1]))
        # Owners are contiguous: reduce views per image without GPU atomic adds.
        first = 0
        while first < len(owners):
            last = first + 1
            while last < len(owners) and owners[last] == owners[first]:
                last += 1
            summed[owners[first]] += features[first:last].sum(0)
            first = last
        pending.clear()
        owners.clear()

    for index, im in enumerate(images):
        for view in image_views(im, policy, crop_scale):
            pending.append(view)
            owners.append(index)
            if len(pending) == view_batch_size:
                flush()
    if pending:
        flush()
    # All images receive the same number of views, so sum and mean normalize identically.
    return F.normalize(summed, dim=-1)


def text_inputs(processor, titles, device):
    enc = processor(text=titles, padding="max_length", truncation=True,
                    max_length=64, return_tensors="pt")
    ids = enc["input_ids"]
    if "attention_mask" not in enc:
        pad = processor.tokenizer.pad_token_id
        mask = torch.ones_like(ids)
        if pad is not None:
            is_pad = ids == pad
            first = is_pad.long().argmax(dim=1)
            mask = (torch.arange(ids.shape[1])[None, :] <= first[:, None]).long()
            mask[~is_pad.any(dim=1)] = 1
        enc["attention_mask"] = mask
    return {k: enc[k].to(device) for k in ("input_ids", "attention_mask")}


class EvalCollator:
    """Pass PIL images and titles without allocating all views as pixel tensors."""
    def __call__(self, batch):
        images, titles = zip(*batch)
        bags = [im if isinstance(im, list) else [im] for im in images]
        k = len(bags[0])
        if any(len(bag) != k for bag in bags):
            raise ValueError("Evaluation bags must have equal frame counts")
        return [im for bag in bags for im in bag], list(titles), len(batch), k


@torch.inference_mode()
def encode_dataset(model, processor, dataset, device, batch_size, workers,
                   tta="none", crop_scale=.9, view_batch_size=64):
    from training_utils import seed_worker
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        collate_fn=EvalCollator(), worker_init_fn=seed_worker)
    image_chunks, text_chunks = [], []
    for images, titles, b, k in loader:
        features = encode_tta_images(images,
            lambda ims: processor(images=ims, return_tensors="pt")["pixel_values"],
            lambda px: model.get_image_features(pixel_values=px), device,
            tta, crop_scale, view_batch_size)
        bags = features.reshape(b, k, -1)
        if hasattr(model, "mix_frame_features"):
            bags = model.mix_frame_features(bags)
        image_chunks.append(bags.cpu())
        text_chunks.append(F.normalize(paced_gpu_call(model.get_text_features,
            **text_inputs(processor, titles, device)).float(), dim=-1).cpu())
    if not image_chunks:
        raise ValueError("Cannot evaluate an empty dataset")
    return torch.cat(image_chunks), torch.cat(text_chunks)


def checkpoint_fingerprint(checkpoint):
    """Hash local weights/config; hash once per process, never during workers."""
    root = Path(checkpoint)
    if not root.is_dir():
        raise ValueError("Cache identity requires a local checkpoint or resolved local Hub snapshot")
    files = sorted(p for p in root.iterdir() if p.is_file() and (
        p.suffix in (".json", ".safetensors", ".bin", ".model") and p.name != "training_state.pt"))
    if not any(p.suffix in (".safetensors", ".bin") for p in files):
        raise ValueError("Checkpoint contains no model weights")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as fh:
            while chunk := fh.read(4 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def validate_cache(blob, expected, eids):
    if blob.get("metadata") != expected:
        raise ValueError("Embedding cache model/preprocessing/TTA identity differs; rebuild --save-emb")
    if blob.get("eids") != eids:
        raise ValueError("Embedding cache corpus order differs; rebuild --save-emb")
    emb = blob.get("emb")
    if not isinstance(emb, torch.Tensor) or emb.ndim not in (2, 3) or len(emb) != len(eids) or not torch.isfinite(emb).all():
        raise ValueError("Invalid embedding cache")
    if expected.get("asset_type") == "mosaic" and (emb.ndim != 3 or emb.shape[1] != expected["bag_size"]):
        raise ValueError("Cache frame count differs from requested contact-sheet layout")
    if expected.get("asset_type") == "thumbnail" and emb.ndim != 2:
        raise ValueError("Expected thumbnail embeddings")


def cache_metadata(backend, checkpoint_sha256, policy, crop_scale):
    import transformers
    import PIL
    return dict(backend=backend, checkpoint_sha256=checkpoint_sha256,
                tta=policy, crop_scale=crop_scale, tta_version=TTA_VERSION,
                transformers=transformers.__version__, pillow=PIL.__version__)


def load_corpus(db, thumbs, mosaics, asset_type="auto"):
    """Read either crawler schema without inventing unavailable thumbnail columns."""
    conn = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(videos)")}
        kind = ("thumbnail" if "thumb_state" in columns else "mosaic") if asset_type == "auto" else asset_type
        if kind == "mosaic":
            rows = conn.execute("SELECT eid,title,asset_path FROM videos WHERE asset_state='done' ORDER BY eid").fetchall()
            return [(path or os.path.join(mosaics, f"{eid}.jpg"), eid, title or "") for eid, title, path in rows], kind
        if "thumb_state" in columns:
            rows = conn.execute("SELECT eid,title,thumb_path FROM videos WHERE thumb_state='done' ORDER BY eid").fetchall()
            return [(path or os.path.join(thumbs, f"{eid}.jpg"), eid, title or "") for eid, title, path in rows], kind
        rows = conn.execute("SELECT eid,title FROM videos ORDER BY eid").fetchall()
        items = [(os.path.join(thumbs, f"{eid}.jpg"), eid, title or "") for eid, title in rows]
        return [item for item in items if os.path.isfile(item[0])], kind
    finally:
        conn.close()
