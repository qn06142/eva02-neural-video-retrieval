#!/usr/bin/env python3
"""Compare TTA policies on exactly the samples in a saved validation manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import SiglipProcessor
from frame_model import SiglipBagModel as SiglipModel

from retrieval_tta import TTA_POLICIES, checkpoint_fingerprint
from train_siglip import encode
from train_siglip_mil import encode_bags
from training_utils import retrieval_ranks


def paired_interval(candidate, baseline, repetitions=2000, seed=42, alpha=.05):
    """Paired query bootstrap; descriptive when queries are correlated."""
    delta = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    if delta.ndim != 1 or not len(delta) or repetitions < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid bootstrap input")
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, repetitions, 100):
        indices = rng.integers(len(delta), size=(min(100, repetitions-start), len(delta)))
        means.extend(delta[indices].mean(1))
    low, high = np.quantile(means, [alpha/2, 1-alpha/2])
    return dict(delta=float(delta.mean()), low=float(low), high=float(high))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="local SigLIP checkpoint directory")
    p.add_argument("--split", required=True, help="saved split.json (never generates a new split)")
    p.add_argument("--out", required=True, help="new JSON report path")
    p.add_argument("--mode", choices=("thumbnail", "mil"), default="thumbnail")
    p.add_argument("--policies", nargs="+", choices=TTA_POLICIES, default=list(TTA_POLICIES))
    p.add_argument("--crop-scale", type=float, default=.9)
    p.add_argument("--grid", default="3x3")
    p.add_argument("--bag-size", type=int, default=None, help="MIL evaluation frames; default entire grid")
    p.add_argument("--pooling", choices=("max", "logmeanexp"), default=None)
    p.add_argument("--pool-temperature", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--view-batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    # A video-trained frame mixer must be evaluated on bags, not silently dropped.
    if Path(args.out).exists():
        p.error("Report already exists; choose a new --out")
    if min(args.batch_size, args.view_batch_size, args.bootstrap) < 1 or args.workers < 0:
        p.error("Need positive batch sizes/bootstrap and nonnegative workers")
    if not 0 < args.crop_scale <= 1 or (args.pool_temperature is not None and
            (not np.isfinite(args.pool_temperature) or args.pool_temperature <= 0)):
        p.error("Invalid crop scale or pooling temperature")
    grid = tuple(int(v) for v in args.grid.lower().split("x"))
    if len(grid) != 2 or min(grid) < 1:
        p.error("Invalid grid")
    bag_size = args.bag_size if args.bag_size is not None else grid[0] * grid[1]
    if not 1 <= bag_size <= grid[0] * grid[1]:
        p.error("Invalid bag size")
    manifest_bytes = Path(args.split).read_bytes()
    split = json.loads(manifest_bytes)
    indices = split["val_indices"]
    if (len(indices) < 2 or len(set(indices)) != len(indices) or
            any(not isinstance(i, int) or not 0 <= i < len(split["samples"]) for i in indices) or
            set(indices).intersection(split.get("train_indices", []))):
        p.error("Need at least two unique validation samples")
    pairs = [split["samples"][i] for i in indices]
    policies = ["none"] + [v for v in dict.fromkeys(args.policies) if v != "none"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    dtype = ((torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
             if device.type == "cuda" else torch.float32)
    processor = SiglipProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = SiglipModel.from_pretrained(args.checkpoint, local_files_only=True,
        attn_implementation="sdpa", torch_dtype=dtype).to(device).eval()
    if model.frame_mixer is not None and args.mode != "mil":
        p.error("Frame-mixer checkpoints require --mode mil")
    args.pooling = args.pooling or getattr(model.config, "retrieval_pooling", "max")
    if args.pool_temperature is None:
        args.pool_temperature = getattr(model.config, "retrieval_pool_temperature", .07)
    records, baseline = [], None
    for policy in policies:
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.monotonic()
        common = dict(tta=policy, crop_scale=args.crop_scale, view_batch_size=args.view_batch_size)
        if args.mode == "mil":
            images, texts = encode_bags(model, processor, pairs, device, bag_size, grid,
                                        args.batch_size, args.workers, **common)
        else:
            images, texts = encode(model, processor, pairs, device, args.batch_size, args.workers, **common)
        ranks = retrieval_ranks(texts, images, pooling=args.pooling, temperature=args.pool_temperature)
        if device.type == "cuda":
            torch.cuda.synchronize()
        record = dict(tta=policy, seconds=time.monotonic()-started,
                      **{f"recall@{k}": (ranks < k).float().mean().item() for k in (1, 5, 10)})
        hits = (ranks < 1).numpy()
        if baseline is None:
            baseline = hits
        record["paired_recall1"] = paired_interval(hits, baseline, args.bootstrap, args.seed,
                                                    .05 / max(1, len(policies)-1))
        records.append(record)
        print(json.dumps(record), flush=True)
    qualified = [r for r in records if r["paired_recall1"]["low"] > 0]
    recommended = max(qualified, key=lambda r: r["recall@1"])["tta"] if qualified else "none"
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), args=vars(args),
                  checkpoint_sha256=checkpoint_fingerprint(args.checkpoint),
                  split_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  split_provenance=split.get("provenance", "saved training manifest"),
                  candidates=len(pairs), device=str(device), dtype=str(dtype), torch=torch.__version__,
                  transformers=__import__("transformers").__version__, results=records,
                  recommended_tta=recommended,
                  selection_rule="Recall@1 gain with positive Bonferroni-adjusted paired-query bootstrap lower bound; otherwise none",
                  limitation="Validation-only selection; correlated/duplicate queries and tuning reuse limit interval interpretation. Verify on a separate test set.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Validation recommendation: {recommended}; report: {args.out}", flush=True)


if __name__ == "__main__":
    main()
