"""Memory-efficient joint batches and conservative weak-label pair weighting."""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from gpu_pacing import paced_gpu_call


def chunked_features(encoder, inputs, chunk_size=None, recompute=True):
    """Keep a joint loss graph while rematerializing each encoder chunk.

    This is activation checkpointing across encoder batches, not GradCache or
    independent gradient accumulation. Encoder inputs must share a batch axis.
    """
    n = len(next(iter(inputs.values())))
    if n < 1 or any(len(value) != n for value in inputs.values()):
        raise ValueError("Expected nonempty inputs with aligned batch axes")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("Encoder chunk size must be positive")
    if chunk_size is None:
        return paced_gpu_call(encoder, **inputs)
    def encode_chunk(**batch):
        return paced_gpu_call(encoder, **batch)
    chunks = []
    for start in range(0, n, chunk_size):
        batch = {key: value[start:start+chunk_size] for key, value in inputs.items()}
        if recompute and torch.is_grad_enabled():
            features = checkpoint(encode_chunk, use_reentrant=False, preserve_rng_state=True, **batch)
        else:
            features = encode_chunk(**batch)
        chunks.append(features)
    return torch.cat(chunks)


@torch.no_grad()
def negative_pair_weights(title_features, positives, threshold=.9, minimum=.25):
    """Downweight ambiguous negatives; never create a positive from similarity.

    Use a frozen teacher, not trainable student features, so the student cannot
    reduce its loss by making unrelated titles similar to each other.
    """
    if not math.isfinite(threshold) or not 0 <= threshold < 1 or not 0 <= minimum <= 1:
        raise ValueError("Need 0 <= threshold < 1 and 0 <= minimum <= 1")
    with torch.autocast(title_features.device.type, enabled=False):
        features = F.normalize(title_features.float(), dim=-1)
        if not torch.isfinite(features).all():
            raise ValueError("Non-finite teacher title features")
        similarity = features @ features.T
        if positives.shape != similarity.shape:
            raise ValueError("Positive mask and teacher batch differ")
        ambiguity = ((similarity-threshold)/(1-threshold)).clamp(0, 1)
        weights = 1 - (1-minimum) * ambiguity
        return weights.masked_fill(positives.to(device=weights.device, dtype=torch.bool), 1.)


def pair_diagnostics(weights, positives):
    if weights is None:
        count = positives.bool().sum().item()
        negatives = positives.numel() - count
        return dict(positive_pairs=count, negative_pairs=negatives,
                    negative_weight_sum=negatives, downweighted_negative_pairs=0)
    positives = positives.to(device=weights.device, dtype=torch.bool)
    negatives = ~positives
    return dict(positive_pairs=positives.sum().item(), negative_pairs=negatives.sum().item(),
                negative_weight_sum=weights[negatives].sum().item(),
                downweighted_negative_pairs=((weights < 1) & negatives).sum().item())


class FrozenTitleTeacher:
    """Keep the starting text encoder fixed, and persist it once for resume."""
    def __init__(self, model, out_dir, resume=False):
        self.encoder = copy.deepcopy(model.text_model).eval().requires_grad_(False)
        path = Path(out_dir) / "title_teacher.pt"
        if resume:
            if not path.is_file():
                raise ValueError("Missing frozen title teacher for resume")
            self.encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        else:
            if path.exists():
                raise ValueError("Teacher file already exists; use a fresh output directory")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({name: tensor.detach().cpu() for name, tensor in self.encoder.state_dict().items()}, path)
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            while block := fh.read(4 * 1024 * 1024):
                digest.update(block)
        self.sha256 = digest.hexdigest()

    @torch.inference_mode()
    def __call__(self, input_ids, attention_mask, chunk_size=32):
        self.encoder.eval()
        features = chunked_features(
            lambda **batch: self.encoder(**batch).pooler_output,
            dict(input_ids=input_ids, attention_mask=attention_mask), chunk_size, recompute=False)
        return features.detach()
