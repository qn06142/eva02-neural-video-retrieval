#!/usr/bin/env python3
"""Video Content Pacing & Temporal Dynamics Auto-Detection.

Analyzes the 9 storyboard keyframe embeddings (t=0..8) across the video timeline to
quantify temporal velocity, narrative escalation, volatility, and pacing class.

Metrics:
  - Step Transitions (d_0..d_7): Cosine distance between adjacent frames (1 - cos(v_t, v_{t+1})).
  - Pacing Velocity (v_bar): Mean transition distance across the timeline.
  - Escalation Slope (m): Linear regression trend of distance over time.
      * Positive slope (m > +0.07): Slow burn (gradual build-up towards climax).
      * Negative slope (m < -0.07): Front-loaded (immediate intense action, tapering off).
      * Near zero: Uniform / steady tempo.
  - Volatility (sigma_v): Standard deviation of transition distances (smooth vs cut-heavy).
  - Peak Action Phase (early/mid/late): Where the largest visual change occurs.

Classification:
  - ⏳ Slow Burn: Low early velocity with strong positive escalation slope.
  - 🌊 Steady / Sensual: Low velocity and low volatility (single posture / continuous scene).
  - ⚡ Fast-Paced: High velocity throughout with sustained motion / multi-posture shifts.
  - 💥 Front-Loaded: High initial intensity with negative escalation slope.
  - 🎬 Montage / Cut-Heavy: Erratic jumps, multiple scene/camera cuts.
  - ⚖️ Balanced Scene: Standard episodic scene progression.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

PACING_CLASSES = {
    "slow_burn": ("Slow Burn", "⏳ Slow Burn", "#f59e0b"),
    "steady": ("Steady / Sensual", "🌊 Steady", "#06b6d4"),
    "fast": ("Fast-Paced", "⚡ Fast-Paced", "#ef4444"),
    "front_loaded": ("Front-Loaded", "💥 Front-Loaded", "#ec4899"),
    "montage": ("Montage / Cut-Heavy", "🎬 Montage", "#8b5cf6"),
    "balanced": ("Balanced Scene", "⚖️ Balanced", "#64748b"),
}


def classify_pacing(
    mean_vel: float,
    slope: float,
    std_vel: float,
    max_vel: float,
) -> Tuple[str, str, str]:
    """Classifies temporal dynamics into a human-interpretable pacing category."""
    if max_vel >= 0.38 or std_vel >= 0.12:
        key = "montage"
    elif slope >= 0.07:
        key = "slow_burn"
    elif slope <= -0.07:
        key = "front_loaded"
    elif mean_vel < 0.085:
        key = "steady"
    elif mean_vel >= 0.16:
        key = "fast"
    else:
        key = "balanced"

    name, badge, color = PACING_CLASSES[key]
    return name, badge, color


def compute_pacing_batch(vectors_9tiles: torch.Tensor) -> List[Dict[str, Any]]:
    """Vectorized calculation of pacing metrics for a batch of 9-tile video embeddings.

    Args:
        vectors_9tiles: Tensor of shape (N, 9, 768), float16 or float32.

    Returns:
        List of N dictionaries containing all pacing dynamics and badges.
    """
    if vectors_9tiles.ndim != 3 or vectors_9tiles.size(1) != 9:
        raise ValueError(f"Expected shape (N, 9, D), got {vectors_9tiles.shape}")

    device = vectors_9tiles.device
    vecs = F.normalize(vectors_9tiles.float(), dim=-1)

    # 1. Step-wise cosine distances: d_t = 1 - cos(v_t, v_{t+1}) for t=0..7
    step_sims = (vecs[:, :8] * vecs[:, 1:]).sum(dim=-1)  # (N, 8)
    dists = torch.clamp(1.0 - step_sims, min=0.0, max=2.0)  # (N, 8)

    mean_vel = dists.mean(dim=-1)  # (N,)
    std_vel = dists.std(dim=-1)  # (N,)
    max_vel, max_idx = dists.max(dim=-1)  # (N,)

    # 2. Linear regression slope of distance over time
    t = torch.linspace(-1.0, 1.0, 8, device=device)
    t_c = t - t.mean()
    t_denom = (t_c**2).sum()
    slopes = (dists * t_c.unsqueeze(0)).sum(dim=-1) / t_denom  # (N,)

    dists_cpu = dists.cpu().tolist()
    mean_vel_cpu = mean_vel.cpu().tolist()
    std_vel_cpu = std_vel.cpu().tolist()
    max_vel_cpu = max_vel.cpu().tolist()
    max_idx_cpu = max_idx.cpu().tolist()
    slopes_cpu = slopes.cpu().tolist()

    results = []
    for i in range(len(vectors_9tiles)):
        mv = mean_vel_cpu[i]
        sl = slopes_cpu[i]
        sv = std_vel_cpu[i]
        mx = max_vel_cpu[i]
        peak_step = max_idx_cpu[i]

        phase = "early" if peak_step <= 2 else ("late" if peak_step >= 5 else "mid")
        name, badge, color = classify_pacing(mv, sl, sv, mx)

        results.append({
            "class": name,
            "badge": badge,
            "color": color,
            "velocity": round(mv, 4),
            "slope": round(sl, 4),
            "volatility": round(sv, 4),
            "peak_step": peak_step,
            "peak_phase": phase,
            "step_distances": [round(d, 4) for d in dists_cpu[i]],
        })

    return results


def compute_pacing_single(vectors_9tiles: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
    """Calculates pacing metrics for a single video with 9 tiles."""
    if isinstance(vectors_9tiles, np.ndarray):
        tensor = torch.from_numpy(vectors_9tiles)
    else:
        tensor = vectors_9tiles

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    batch_res = compute_pacing_batch(tensor)
    return batch_res[0]
