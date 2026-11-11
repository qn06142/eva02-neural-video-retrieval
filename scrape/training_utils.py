"""Shared, testable training and evaluation mechanics for the retrieval trainers."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F
from gpu_pacing import configure_gpu_pacing, paced_gpu_call


def title_key(title):
    return " ".join(unicodedata.normalize("NFKC", title).casefold().split())


def seed_worker(worker_id):
    torch.set_num_threads(1)
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def select_samples(samples, maximum, seed=42):
    """A deterministic subset without the crawler's discovery-order bias."""
    samples = sorted(samples)
    random.Random(seed).shuffle(samples)
    return samples if maximum is None else samples[:maximum]


def positive_mask(titles):
    keys = [title_key(t) for t in titles]
    return torch.tensor([[a == b for b in keys] for a in keys], dtype=torch.bool)


def pool_scores(scores, mode="max", temperature=.07):
    if mode == "max":
        return scores.amax(-1)
    if mode != "logmeanexp" or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid pooling mode or temperature")
    return temperature * (torch.logsumexp(scores / temperature, dim=-1) - math.log(scores.shape[-1]))


def sigmoid_loss(model, images, text, positives=None, pooling="max", temperature=.07, pair_weights=None):
    """FP32 pairwise SigLIP loss, with known duplicate titles as positives."""
    with torch.autocast(images.device.type, enabled=False):
        images = F.normalize(images.float(), dim=-1)
        text = F.normalize(text.float(), dim=-1)
        scores = (pool_scores(torch.einsum("id,jkd->ijk", text, images), pooling, temperature)
                  if images.ndim == 3 else text @ images.T)
        logits = scores * model.logit_scale.float().exp() + model.logit_bias.float()
        if positives is None:
            positives = torch.eye(len(text), device=logits.device, dtype=torch.bool)
        targets = positives.to(device=logits.device, dtype=logits.dtype) * 2 - 1
        terms = -F.logsigmoid(targets * logits)
        if pair_weights is not None:
            weights = pair_weights.detach().to(device=logits.device, dtype=logits.dtype)
            if weights.shape != logits.shape or not torch.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
                raise ValueError("Pair weights must match logits and lie in [0,1]")
            # Known positives always keep their original supervision strength.
            weights = torch.where(positives.to(device=logits.device, dtype=torch.bool), 1., weights)
            terms = terms * weights
        return terms.sum(-1).mean()


def split_samples(samples, val_frac, val_max, seed, out_dir):
    """Keep identical normalized titles or asset paths on the same side.

    Persist the exact split and reject a changed corpus/config on reuse. This
    catches crawl growth as well as changed labels instead of moving the holdout.
    This is exact metadata grouping, not perceptual duplicate detection.
    """
    samples = sorted(samples)
    payload = json.dumps(samples, ensure_ascii=False, separators=(",", ":"))
    signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    config = dict(seed=seed, val_frac=val_frac, val_max=val_max)
    path = Path(out_dir) / "split.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["sha256"] != signature or manifest["config"] != config:
            raise ValueError("Corpus or split settings changed; use a new --out directory.")
        train, val = manifest["train_indices"], manifest["val_indices"]
        if sorted(train + val) != list(range(len(samples))) or not train or not val:
            raise ValueError("Invalid split manifest")
    else:
        parent = list(range(len(samples)))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        keys = {}
        for i, (asset, title) in enumerate(samples):
            for key in (("asset", os.path.normcase(os.path.abspath(asset))),
                        ("title", title_key(title))):
                if key in keys:
                    parent[root(i)] = root(keys[key])
                else:
                    keys[key] = i
        groups = {}
        for i in range(len(samples)):
            groups.setdefault(root(i), []).append(i)
        shuffled = list(groups.values())
        random.Random(seed).shuffle(shuffled)
        target = max(1, min(val_max, int(len(samples) * val_frac)))
        val, train = [], []
        for group in shuffled:
            if len(val) < target and len(val) + len(group) <= val_max:
                val.extend(group)
            else:
                train.extend(group)
        if not train or len(val) < 2:
            raise ValueError("Need at least two held-out samples and one training group; "
                             "increase --val-frac/--val-max or supply more distinct data.")
        manifest = dict(version=1, sha256=signature, config=config,
                        train_indices=train, val_indices=val, samples=samples)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return [samples[i] for i in train], [samples[i] for i in val]


def retrieval_ranks(text, images, chunk_size=64, pooling="max", temperature=.07):
    """Exact diagonal recall with bounded (chunk,N,K) temporary storage.

    Images may be (N,D) or bags (N,K,D). Equal scores break ties by corpus
    index, so constant embeddings cannot obtain perfect recall.
    """
    n = len(text)
    if n == 0 or len(images) != n or chunk_size < 1:
        raise ValueError("Expected nonempty aligned embeddings and positive k/chunk size")
    text = F.normalize(text.float(), dim=-1)
    images = F.normalize(images.float(), dim=-1)
    if not torch.isfinite(text).all() or not torch.isfinite(images).all():
        raise ValueError("Non-finite evaluation embeddings")
    ranks = []
    indices = torch.arange(n, device=text.device)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        queries = text[start:stop]
        if images.ndim == 3:
            scores = pool_scores(torch.einsum("id,jkd->ijk", queries, images), pooling, temperature)
        else:
            scores = queries @ images.T
        own = indices[start:stop]
        positive = scores[torch.arange(stop-start, device=text.device), own, None]
        rank = ((scores > positive) | ((scores == positive) & (indices < own[:, None]))).sum(1)
        ranks.append(rank)
    return torch.cat(ranks)


def retrieval_recall(text, images, ks=(1, 5, 10), chunk_size=64, pooling="max", temperature=.07):
    if any(k < 1 for k in ks):
        raise ValueError("Recall k must be positive")
    ranks = retrieval_ranks(text, images, chunk_size, pooling, temperature)
    return {f"recall@{k}": (ranks < min(k, len(ranks))).sum().item() / len(ranks) for k in ks}


def validate_args(args):
    configure_gpu_pacing(args.gpu_duty_cycle, args.gpu_min_pause_ms, args.gpu_target_temp)
    if not 0 < args.tta_crop_scale <= 1 or args.view_batch_size < 1:
        raise ValueError("Need 0 < tta-crop-scale <= 1 and view-batch-size >= 1")
    for key in ("epochs", "batch_size", "grad_accum", "val_max", "log_every", "min_title_len"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.workers < 0 or not 0 < args.val_frac < 1:
        raise ValueError("Need workers >= 0 and 0 < val-frac < 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.clip) or args.clip <= 0:
        raise ValueError("Learning rate and gradient clip must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("Need weight-decay >= 0 and 0 <= warmup-ratio < 1")
    if args.resume and args.init_from:
        raise ValueError("Choose --resume or --init-from")
    if args.resume and not (Path(args.resume) / "training_state.pt").is_file():
        raise ValueError("--resume requires an epoch checkpoint with training_state.pt; "
                         "use --init-from for older model-only checkpoints")
    if args.resume and not all((Path(args.out) / name).is_file() for name in
                               ("split.json", "train_args.json")):
        raise ValueError("Resume requires --out to point to the original run directory")
    if not args.resume and (Path(args.out) / "train_args.json").exists():
        raise ValueError("Output already contains a run; resume it or choose a new --out")


def add_training_options(parser):
    parser.add_argument("--gpu-target-temp", type=float, default=None,
                        help="optional software temperature target in C; adjusts duty below its configured ceiling")
    parser.add_argument("--gpu-duty-cycle", type=float, default=.3,
                        help="CUDA work/rest target per encoder chunk (default: 0.3)")
    parser.add_argument("--gpu-min-pause-ms", type=float, default=50.,
                        help="minimum rest after CUDA calls, including validation (default: 50ms)")
    from retrieval_tta import TTA_POLICIES
    parser.add_argument("--eval-tta", choices=TTA_POLICIES, default="none")
    parser.add_argument("--tta-crop-scale", type=float, default=.9)
    parser.add_argument("--view-batch-size", type=int, default=64)
    parser.add_argument("--init-from", help="initialize weights from a model-only checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-baseline", action="store_true", help="skip starting-model validation")


def choose_device(args):
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def optimizer_step(model, optimizer, scheduler, scaler, clip):
    """Unscale before clipping; don't advance LR when fp16 overflow skips a step."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if not scaler.is_enabled() and not torch.isfinite(norm):
        raise FloatingPointError("Non-finite gradient norm")
    previous = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= previous
    if updated:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return updated


def accumulation_weight(step, sample_count, batch_size, accumulation):
    """Weight microbatch means by queries in the current accumulation window."""
    window_start = (step // accumulation) * accumulation * batch_size
    window_samples = min(accumulation * batch_size, sample_count - window_start)
    batch_samples = min(batch_size, sample_count - step * batch_size)
    return batch_samples / window_samples


def append_metrics(out_dir, record):
    with open(Path(out_dir) / "metrics.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, allow_nan=False) + "\n")


def run_epochs(args, model, processor, loader, evaluate, forward_loss, device, log):
    """Shared epoch-boundary resumable loop; forward_loss returns scalar loss."""
    from transformers import get_cosine_schedule_with_warmup

    if not len(loader):
        raise ValueError("Training loader is empty")
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (no_decay if param.ndim < 2 or any(k in name for k in
             ("bias", "layer_norm", "logit_scale")) else decay).append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    total_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    precision = str(dtype) if device.type == "cuda" else "float32"
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not bf16)
    start, best, updates = 0, -1.0, 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        # Only load trusted local checkpoints: RNG state requires weights_only=False.
        state = torch.load(Path(args.resume) / "training_state.pt", map_location="cpu", weights_only=False)
        if Path(state["args"]["out"]).resolve() != out.resolve():
            raise ValueError("Resume requires the original --out directory")
        if state["precision"] != precision:
            raise ValueError("Resume requires the original precision/device type; use --init-from for a new run")
        completed = [int(p.parent.name.split("-")[-1]) for p in out.glob("epoch-*/training_state.pt")]
        if completed and state["epoch"] != max(completed):
            raise ValueError("Resume the latest completed epoch; use --init-from for a new experiment")
        ignored = {"resume", "init_from", "out", "log_every", "device", "skip_baseline",
                   "gpu_duty_cycle", "gpu_min_pause_ms", "gpu_target_temp"}
        previous_args = {k: v for k, v in state["args"].items() if k not in ignored}
        current_args = {k: v for k, v in vars(args).items() if k not in ignored}
        # Version-2 checkpoints predate optional TTA and smooth pooling.
        for key, default in dict(eval_tta="none", tta_crop_scale=.9, view_batch_size=64,
                                 pooling="max", pool_temperature=.07, image_chunk_size=None,
                                 text_chunk_size=None, negative_title_threshold=None,
                                 negative_weight_floor=.25, title_teacher_sha256=None,
                                 frame_mixer=None, frame_mixer_dim=256, frame_mixer_layers=2,
                                 frame_mixer_heads=4, freeze_backbone=False, frame_mixer_config=None).items():
            if key in current_args:
                previous_args.setdefault(key, default)
        if previous_args != current_args or state["batches"] != len(loader):
            raise ValueError("Resume must use the original training configuration and epoch budget")
        if state["split_sha256"] != hashlib.sha256((out / "split.json").read_bytes()).hexdigest():
            raise ValueError("Resume split does not match the checkpoint")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best, updates = state["epoch"], state["best"], state["updates"]
        metrics_path = out / "metrics.jsonl"
        if metrics_path.exists():
            records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
            metrics_path.write_text("".join(json.dumps(r) + "\n" for r in records if r["epoch"] <= start),
                                    encoding="utf-8")
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        loader.generator.set_state(state["loader_rng"])
    else:
        (out / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        (out / "environment.json").write_text(json.dumps({
            "torch": torch.__version__, "device": str(device),
            "cuda": torch.version.cuda, "precision": precision,
            "transformers": __import__("transformers").__version__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }, indent=2), encoding="utf-8")
        if not args.skip_baseline:
            metrics = evaluate()
            best = metrics["recall@1"]
            model.save_pretrained(out / "best")
            processor.save_pretrained(out / "best")
            append_metrics(out, dict(epoch=0, phase="baseline", **metrics))
            log(f"baseline: {metrics}")
    log(f"optimizer steps planned={total_steps}; negatives per query <= {args.batch_size - 1}; "
        "gradient accumulation does not enlarge the contrastive negative pool")
    import time
    for epoch in range(start, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        beginning = time.monotonic()
        loss_sum = 0.0
        samples_seen = 0
        diagnostics = {}
        for step, batch in enumerate(loader):
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                result = forward_loss(batch)
                if isinstance(result, tuple):
                    raw_loss, batch_diagnostics = result
                    for key, value in batch_diagnostics.items():
                        diagnostics[key] = diagnostics.get(key, 0.) + float(value)
                else:
                    raw_loss = result
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch+1}, batch {step+1}")
            weight = accumulation_weight(step, len(loader.dataset), args.batch_size, args.grad_accum)
            paced_gpu_call(scaler.scale(raw_loss * weight).backward, device=device)
            batch_samples = min(args.batch_size, len(loader.dataset) - step * args.batch_size)
            loss_sum += raw_loss.item() * batch_samples
            samples_seen += batch_samples
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                updates += paced_gpu_call(optimizer_step, model, optimizer, scheduler,
                                           scaler, args.clip, device=device)
            if (step + 1) % args.log_every == 0:
                mixer = getattr(model, "frame_mixer", None)
                suffix = f" mixer_gate={mixer.gate.detach().tanh().item():.5f}" if mixer is not None else ""
                log(f"epoch {epoch+1}/{args.epochs} batch {step+1}/{len(loader)} "
                    f"loss={loss_sum/samples_seen:.4f} lr={scheduler.get_last_lr()[0]:.2e}" + suffix)
        metrics = evaluate()
        r1 = metrics["recall@1"]
        if getattr(model, "frame_mixer", None) is not None:
            metrics["frame_mixer_gate"] = model.frame_mixer.gate.detach().tanh().item()
        improved = r1 > best
        best = max(best, r1)
        negatives = diagnostics.get("negative_pairs", 0.)
        if negatives:
            diagnostics["mean_negative_weight"] = diagnostics["negative_weight_sum"] / negatives
            diagnostics["downweighted_negative_fraction"] = diagnostics["downweighted_negative_pairs"] / negatives
        append_metrics(out, dict(epoch=epoch+1, phase="validation", loss=loss_sum/samples_seen,
                                seconds=time.monotonic()-beginning, updates=updates,
                                training_diagnostics=diagnostics, **metrics))
        epoch_dir = out / f"epoch-{epoch+1}"
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)
        # Serialized after eval to preserve all RNG consumption at epoch boundaries.
        state = dict(epoch=epoch+1, best=best, updates=updates, args=vars(args), batches=len(loader), precision=precision,
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(), python_rng=random.getstate(),
                     numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     loader_rng=loader.generator.get_state(),
                     split_sha256=hashlib.sha256((out / "split.json").read_bytes()).hexdigest())
        if improved:
            model.save_pretrained(out / "best")
            processor.save_pretrained(out / "best")
        torch.save(state, epoch_dir / "training_state.tmp")
        os.replace(epoch_dir / "training_state.tmp", epoch_dir / "training_state.pt")
        log(f"epoch {epoch+1}: {metrics}; best recall@1={best:.4f}")
