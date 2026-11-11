#!/usr/bin/env python3
"""Measure an actual CUDA training step with/without encoder chunk checkpointing."""
import argparse
import gc
import json
from pathlib import Path
import tempfile
import time

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scrape"))

import torch
from transformers import SiglipProcessor
from frame_model import SiglipBagModel as SiglipModel

from train_siglip_mil import BagCollator, MosaicBagDataset, load_bags, mil_sigmoid_loss
from weak_supervision import FrozenTitleTeacher, chunked_features, negative_pair_weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="local SigLIP checkpoint")
    p.add_argument("--db", required=True)
    p.add_argument("--mosaic-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--bag-size", type=int, default=9)
    p.add_argument("--grid", default="3x3")
    p.add_argument("--image-chunk-size", type=int, default=4)
    p.add_argument("--text-chunk-size", type=int, default=2)
    p.add_argument("--with-title-teacher", action="store_true")
    args = p.parse_args()
    if Path(args.out).exists() or min(args.batch_size, args.bag_size, args.image_chunk_size, args.text_chunk_size) < 1:
        p.error("Need a new output file and positive batch/chunk sizes")
    if not torch.cuda.is_available():
        p.error("CUDA is required for this memory benchmark")
    grid = tuple(int(v) for v in args.grid.lower().split("x"))
    if len(grid) != 2 or min(grid) < 1 or args.bag_size > grid[0]*grid[1]:
        p.error("Invalid grid or bag size")
    torch.manual_seed(42)
    processor = SiglipProcessor.from_pretrained(args.model, local_files_only=True)
    model = SiglipModel.from_pretrained(args.model, local_files_only=True, attn_implementation="sdpa").cuda().train()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bags = load_bags(args.db, args.mosaic_dir, args.batch_size, 3)
    if len(bags) != args.batch_size:
        p.error("Not enough valid bags for requested benchmark batch")
    dataset = MosaicBagDataset(bags, processor, args.bag_size, grid, train=False)
    enc, b, k = BagCollator(processor)([dataset[i] for i in range(len(dataset))])
    pixels = dict(pixel_values=enc["pixel_values"].cuda())
    texts = {key: enc[key].cuda() for key in ("input_ids", "attention_mask")}
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    results = []
    with tempfile.TemporaryDirectory() as scratch:
        teacher = FrozenTitleTeacher(model, scratch) if args.with_title_teacher else None
        for name, image_chunk, text_chunk in (("full", None, None),
                ("checkpointed", args.image_chunk_size, args.text_chunk_size)):
            model.load_state_dict(original)
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
            torch.manual_seed(42)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            record = dict(mode=name)
            try:
                with torch.autocast("cuda", dtype=dtype):
                    images = chunked_features(model.get_image_features, pixels, image_chunk)
                    text = chunked_features(model.get_text_features, texts, text_chunk)
                    weights = (negative_pair_weights(teacher(**texts), enc["positive_mask"], .9, .25)
                               if teacher else None)
                    images = model.mix_frame_features(images.reshape(b, k, -1))
                    loss = mil_sigmoid_loss(model, images, text,
                                            enc["positive_mask"], "logmeanexp", .07, weights)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                record.update(status="ok", loss=loss.item(), gradient_norm=float(norm))
            except torch.OutOfMemoryError:
                record["status"] = "out_of_memory"
            record.update(seconds=time.monotonic()-started,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            results.append(record)
            print(json.dumps(record), flush=True)
            del optimizer, scaler
            # Also free graph references after an OOM before attempting the next mode.
            images = text = loss = weights = None
    report = dict(args=vars(args), gpu=torch.cuda.get_device_name(), dtype=str(dtype),
                  torch=str(torch.__version__), transformers=__import__("transformers").__version__,
                  bags=b, frames_per_bag=k, results=results,
                  limitation="One cold AdamW step per mode, including optimizer state allocation; not a steady-state throughput or retrieval-quality benchmark. Model weights are changed only in this temporary process and never saved.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
