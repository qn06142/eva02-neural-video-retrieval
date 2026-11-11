#!/usr/bin/env python3
"""Hover Video Pacing - Region-Aware Spatial Hotspot Cadence (BPM) Extraction.

Analyzes the canonical 5-second 25fps hover video (preview.mp4) hosted on the CDN.
Decodes frames directly in-memory via PyAV, computes dense optical flow via OpenCV,
and extracts the physical action cadence:
  - 4x4 spatial grid decomposition to prevent destructive interference between
    thrusting hips at the frame edges and counter-moving/small motion in the middle
    (crucial for doggystyle, rear-entry, and side-angle camera postures).
  - Isolates top rhythmic action hotspots via Autocorrelation prominence & variance.
  - Directional projection in the hotspot to cleanly separate forward thrusts from rebounds.
  - Measures true full physical cycles (1 forward + 1 backward = 1 full beat).
  - Per-swatch BPM for each of the 5 seconds of the hover preview clip (t = 0..5s).
  - min_bpm, max_bpm, and avg_bpm across all full-motion cycles.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional
import urllib.request

import av
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DB = os.path.join(PROJECT_DIR, "scrape", "embeddings_v2", "pacing_cache.sqlite3")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def get_bpm_color(bpm: int) -> str:
    """Returns color hex corresponding to action cadence intensity (full cycles)."""
    if bpm < 50:
        return "#06b6d4"  # Cyan (Slow / Sensual)
    elif bpm < 75:
        return "#10b981"  # Emerald Green (Steady / Moderate)
    elif bpm < 95:
        return "#f59e0b"  # Amber (Brisk / Energetic)
    else:
        return "#ef4444"  # Crimson Red (Fast / Intense Climax)


def init_cache_db():
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hover_bpm (
            eid         TEXT PRIMARY KEY,
            avg_bpm     INTEGER,
            min_bpm     INTEGER,
            max_bpm     INTEGER,
            badge       TEXT,
            color       TEXT,
            swatch_bpms TEXT,
            beat_times  TEXT
        )
    """)
    cur.execute("PRAGMA table_info(hover_bpm)")
    cols = [r[1] for r in cur.fetchall()]
    if "swatch_bpms" not in cols:
        cur.execute("ALTER TABLE hover_bpm ADD COLUMN swatch_bpms TEXT")
    if "beat_times" not in cols:
        cur.execute("ALTER TABLE hover_bpm ADD COLUMN beat_times TEXT")
    conn.commit()
    conn.close()


init_cache_db()


def clear_pacing_cache():
    """Wipes legacy cache entries so all items recompute with spatial hotspot decomposition."""
    try:
        conn = sqlite3.connect(CACHE_DB)
        cur = conn.cursor()
        cur.execute("DELETE FROM hover_bpm")
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_cached_pacing(eid: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT avg_bpm, min_bpm, max_bpm, badge, color, swatch_bpms, beat_times FROM hover_bpm WHERE eid = ?",
        (eid,)
    )
    row = cur.fetchone()
    conn.close()
    if row and row[5] and row[6]:
        try:
            swatches = json.loads(row[5])
            beats = json.loads(row[6])
            if isinstance(swatches, list) and len(swatches) == 5:
                return {
                    "avg_bpm": row[0],
                    "min_bpm": row[1],
                    "max_bpm": row[2],
                    "badge": row[3],
                    "color": row[4],
                    "swatch_bpms": swatches,
                    "swatch_colors": [get_bpm_color(b) for b in swatches],
                    "beat_times": beats,
                }
        except Exception:
            pass
    return None


def save_cached_pacing(eid: str, data: Dict[str, Any]):
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO hover_bpm 
        (eid, avg_bpm, min_bpm, max_bpm, badge, color, swatch_bpms, beat_times)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        eid,
        data["avg_bpm"],
        data["min_bpm"],
        data["max_bpm"],
        data["badge"],
        data["color"],
        json.dumps(data.get("swatch_bpms", [])),
        json.dumps(data.get("beat_times", [])),
    ))
    conn.commit()
    conn.close()


def analyze_hover_video_bytes(data: bytes, eid: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Analyzes a 5.0s MP4 video buffer using spatial grid hotspot tracking."""
    try:
        buf = io.BytesIO(data)
        container = av.open(buf)
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 25.0)

        frames = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="bgr24")
            frames.append(cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY))
        container.close()

        if len(frames) < 20:
            return None

        T = len(frames) - 1
        # 4x4 spatial grid = 16 localized cells
        grid_h, grid_w = 4, 4
        num_cells = grid_h * grid_w
        cell_flows = np.zeros((num_cells, T))
        cell_u = np.zeros((num_cells, T))
        cell_v = np.zeros((num_cells, T))
        dh, dw = 68 // grid_h, 120 // grid_w

        for t in range(T):
            sg1 = cv2.resize(frames[t], (120, 68))
            sg2 = cv2.resize(frames[t + 1], (120, 68))
            flow = cv2.calcOpticalFlowFarneback(sg1, sg2, None, 0.5, 2, 10, 2, 5, 1.1, 0)
            u_all = flow[..., 0] - np.median(flow[..., 0])
            v_all = flow[..., 1] - np.median(flow[..., 1])
            mag_all = np.sqrt(u_all**2 + v_all**2)

            idx = 0
            for r in range(grid_h):
                for c in range(grid_w):
                    cell_flows[idx, t] = np.mean(mag_all[r * dh:(r + 1) * dh, c * dw:(c + 1) * dw])
                    cell_u[idx, t] = np.mean(u_all[r * dh:(r + 1) * dh, c * dw:(c + 1) * dw])
                    cell_v[idx, t] = np.mean(v_all[r * dh:(r + 1) * dh, c * dw:(c + 1) * dw])
                    idx += 1

        # 3-pole Butterworth low-pass filter at 4.0 Hz
        nyquist = fps / 2.0
        cutoff = min(4.0, fps * 0.4)
        b, a = butter(3, cutoff / nyquist, btype="low")
        cell_smooth = np.zeros_like(cell_flows)
        cell_rhythm_score = np.zeros(num_cells)

        for idx in range(num_cells):
            sig = filtfilt(b, a, cell_flows[idx])
            cell_smooth[idx] = sig
            x = sig - np.mean(sig)
            std_val = np.std(x)
            if std_val > 0.04:
                corr = np.correlate(x, x, mode="full")[len(x) - 1:]
                corr = corr / (corr[0] + 1e-6)
                min_lag = max(2, int(fps * 0.35))
                max_lag = min(int(fps * 2.2), len(corr) - 1)
                p_c, props = find_peaks(corr[min_lag:max_lag], distance=3, prominence=0.03)
                prom = props["prominences"][0] if len(p_c) > 0 else 0.0
                # Rhythm score: energy x autocorrelation peak prominence
                cell_rhythm_score[idx] = std_val * prom
            else:
                cell_rhythm_score[idx] = 0.0

        # Sort cells to find the true action hotspots (hips/pelvis at edges or center)
        top_indices = np.argsort(cell_rhythm_score)[::-1]
        top_3 = top_indices[:3]
        w = cell_rhythm_score[top_3]
        if np.sum(w) > 0:
            w = w / np.sum(w)
            hotspot_mag = np.sum(cell_smooth[top_3] * w[:, None], axis=0)
            hotspot_u = np.sum(cell_u[top_3] * w[:, None], axis=0)
            hotspot_v = np.sum(cell_v[top_3] * w[:, None], axis=0)
        else:
            hotspot_mag = np.mean(cell_smooth, axis=0)
            hotspot_u = np.mean(cell_u, axis=0)
            hotspot_v = np.mean(cell_v, axis=0)

        # Directional projection in the hotspot
        M = np.stack([filtfilt(b, a, hotspot_u), filtfilt(b, a, hotspot_v)], axis=1)
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        proj = M @ Vt[0]

        # Peak detection on smoothed hotspot motion
        peaks, _ = find_peaks(hotspot_mag, distance=max(3, int(fps * 0.16)), prominence=np.std(hotspot_mag) * 0.18)
        peak_times = peaks / fps

        cycle_centers: List[float] = []
        cycle_bpms: List[float] = []

        if len(peak_times) >= 3:
            # Full cycle between alternating peaks: peak i to peak i+2
            for i in range(len(peak_times) - 2):
                dt_full = peak_times[i + 2] - peak_times[i]
                if 0.40 <= dt_full <= 2.40:
                    bpm = 60.0 / dt_full
                    mid = (peak_times[i] + peak_times[i + 2]) / 2.0
                    cycle_centers.append(mid)
                    cycle_bpms.append(bpm)
        elif len(peak_times) == 2:
            dt_full = (peak_times[1] - peak_times[0]) * 2.0
            if 0.40 <= dt_full <= 2.40:
                cycle_centers.append((peak_times[0] + peak_times[1]) / 2.0)
                cycle_bpms.append(60.0 / dt_full)

        # All physical stroke pulses for multi-pulse visual flashing (handles multiple beats/pulses per segment)
        beat_times = [round(float(p / fps), 3) for p in peaks]

        # Fallback if no cycles
        if not cycle_bpms:
            x = hotspot_mag - np.mean(hotspot_mag)
            fft = np.abs(np.fft.rfft(x))
            freqs = np.fft.rfftfreq(len(x), d=1.0 / fps)
            mask = (freqs >= 0.4) & (freqs <= 2.5)
            if np.any(mask):
                f0 = freqs[mask][np.argmax(fft[mask])]
                fallback_bpm = int(round(f0 * 60.0 / 2.0))
            else:
                fallback_bpm = 55
        else:
            fallback_bpm = int(round(np.mean(cycle_bpms)))
        fallback_bpm = max(25, min(140, fallback_bpm))

        # Calculate BPM for each 1-second swatch (5 swatches across 5.0 seconds: 0-1s, 1-2s, 2-3s, 3-4s, 4-5s)
        swatch_bpms = []
        for s in range(5):
            t_center = s + 0.5
            if cycle_bpms:
                weights = [np.exp(-abs(c - t_center) / 0.8) for c in cycle_centers]
                w_sum = sum(weights)
                if w_sum > 0.05:
                    bpm = int(round(sum(b * w for b, w in zip(cycle_bpms, weights)) / w_sum))
                else:
                    bpm = fallback_bpm
            else:
                bpm = fallback_bpm
            bpm = max(25, min(140, bpm))
            swatch_bpms.append(bpm)

        if cycle_bpms:
            avg_bpm = int(round(np.mean(cycle_bpms)))
            min_bpm = int(round(np.min(cycle_bpms)))
            max_bpm = int(round(np.max(cycle_bpms)))
        else:
            avg_bpm = int(round(np.mean(swatch_bpms)))
            min_bpm = int(round(np.min(swatch_bpms)))
            max_bpm = int(round(np.max(swatch_bpms)))

        badge = f"⚡ {avg_bpm} BPM ({min_bpm}–{max_bpm})"
        color = get_bpm_color(avg_bpm)
        swatch_colors = [get_bpm_color(b) for b in swatch_bpms]

        res = {
            "avg_bpm": avg_bpm,
            "min_bpm": min_bpm,
            "max_bpm": max_bpm,
            "badge": badge,
            "color": color,
            "swatch_bpms": swatch_bpms,
            "swatch_colors": swatch_colors,
            "beat_times": beat_times,
        }

        if eid:
            save_cached_pacing(eid, res)
        return res
    except Exception:
        return None


def fetch_and_analyze_hover_pacing(eid: str, preview_url: str) -> Optional[Dict[str, Any]]:
    """Analyzes hover pacing for a video, using SQLite cache if available."""
    cached = get_cached_pacing(eid)
    if cached:
        return cached

    if not preview_url:
        return None

    try:
        req = urllib.request.Request(preview_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = resp.read()
        return analyze_hover_video_bytes(data, eid=eid)
    except Exception:
        return None
