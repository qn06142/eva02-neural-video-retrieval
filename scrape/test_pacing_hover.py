#!/usr/bin/env python3
"""Test video hover preview analysis for pacing auto-detection."""

import os
import re
import sqlite3
import urllib.request
import cv2
import numpy as np

SCRAPE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRAPE_DIR, "thumbs.db")
TMP_VIDEO = os.path.join(SCRAPE_DIR, "temp_test_hover.mp4")
HEADERS = {"User-Agent": "Mozilla/5.0"}


def analyze_hover_video(url: str):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp, open(TMP_VIDEO, "wb") as f:
            f.write(resp.read())
    except Exception as e:
        return None

    cap = cv2.VideoCapture(TMP_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    prev_gray = None
    flows = []
    diffs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            # Fast downsample for real-time optical flow (< 60ms total)
            sg1 = cv2.resize(prev_gray, (120, 68))
            sg2 = cv2.resize(gray, (120, 68))
            flow = cv2.calcOpticalFlowFarneback(sg1, sg2, None, 0.5, 2, 10, 2, 5, 1.1, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flows.append(float(np.mean(mag)))
            diffs.append(float(np.mean(cv2.absdiff(sg2, sg1))))
        prev_gray = gray
    cap.release()

    if os.path.exists(TMP_VIDEO):
        try:
            os.remove(TMP_VIDEO)
        except Exception:
            pass

    if len(flows) < 15:
        return None

    flows = np.array(flows)
    diffs = np.array(diffs)

    mean_flow = float(np.mean(flows))
    max_flow = float(np.max(flows))
    std_flow = float(np.std(flows))

    # Fast Fourier Transform on flow magnitude signal to extract dominant motion frequency
    flow_c = flows - mean_flow
    fft = np.abs(np.fft.rfft(flow_c))
    freqs = np.fft.rfftfreq(len(flows), d=1.0 / fps)

    # Focus on human rhythm band (0.5 Hz to 3.5 Hz == 30 to 210 beats/min)
    mask = (freqs >= 0.5) & (freqs <= 3.5)
    bpm = 0.0
    if np.any(mask):
        peak_f = freqs[mask][np.argmax(fft[mask])]
        bpm = float(peak_f * 60.0)

    # Cut detection: frame diffs > 3x the median
    cuts = int(np.sum(diffs > max(15.0, np.median(diffs) * 3.0)))

    # Pacing classification
    if cuts >= 4 or std_flow > 1.2:
        category = "🎬 Montage / Cuts"
    elif mean_flow < 0.45 or (bpm > 0 and bpm < 55):
        category = "🌊 Slow / Sensual"
    elif mean_flow > 1.3 or bpm >= 115:
        category = "🔥 Fast / Intense"
    else:
        category = "⚖️ Steady Rhythm"

    return {
        "mean_flow": round(mean_flow, 3),
        "max_flow": round(max_flow, 3),
        "std_flow": round(std_flow, 3),
        "bpm": round(bpm, 1),
        "cuts": cuts,
        "category": category,
        "frames": len(flows),
    }


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT v.eid, v.title, v.asset_url FROM videos v "
        "JOIN videos_curated c ON v.eid = c.eid "
        "WHERE v.asset_url IS NOT NULL LIMIT 8"
    )
    samples = cur.fetchall()

    print(f"Testing Hover Video Pacing Detection on {len(samples)} real videos:\n")
    for eid, title, asset_url in samples:
        preview_url = re.sub(r"[^/]+\.(?:jpg|jpeg|png|webp|avif)$", "preview.mp4", asset_url)
        res = analyze_hover_video(preview_url)
        if res:
            print(f"[{res['category']}] Flow: {res['mean_flow']:.2f} px/f | Cadence: {res['bpm']:3.0f} BPM | Cuts: {res['cuts']}")
            print(f"  Title: {title[:70]}")
            print(f"  Preview URL: {preview_url}\n")


if __name__ == "__main__":
    main()
