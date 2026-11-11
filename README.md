# Neural Video Retrieval Engine (EVA-02-CLIP-L-14-336 + 9-Tile MIL + Dense Patch Reranking)

A high-performance, fine-grained semantic video retrieval and motion-pacing analysis engine built on **EVA-02-CLIP-L-14-336** (`336×336`, `576` spatial patches per frame). Designed to index and search **147,412 videos** (**1,326,708 keyframes**) in real time on consumer GPUs (8 GB VRAM).

---

## ✨ Key Features

### 1. Multi-Resolution Semantic Search & Reranking
- **🎯 Exhaustive 9-Tile MIL Search (~8.9 ms in GPU VRAM)**:
  Every video's 3×3 contact sheet is decomposed into 9 distinct chronological keyframes (`147,412 × 9 = 1,326,708` frames) and cached in FP16 GPU VRAM (`2.04 GB`). Queries are scored against all 1.33M keyframes via Tensor Core matrix-vector multiplication (`torch.mv` + `max(dim=-1)`), returning both the matching video and the exact **winning keyframe (`#0..#8`)**.
- **🔬 ViT Dense Spatial Patch Reranking (`24×24 = 576` Patches per Frame)**:
  Stage-2 late-interaction reranker that extracts all **576 spatial patch tokens** (`model.visual.trunk.forward_features` + projection head) from each candidate's winning keyframe:
  - **Text / Hybrid Queries**: Scores the Top-$K$ most responsive spatial patches (`k=36`, top ~6.25% local region) to capture localized clothing, accessories, or anatomical/pose details diluted by global pooling.
  - **Reverse Image Queries**: Extracts the Top-64 salient foreground patches from the query image and computes **ColBERT-style Patch-to-Patch MaxSim** ($\sum_u w_u \max_{v \in \{1..576\}} \langle q_u, d_{i,v} \rangle$) blended with global frame similarity.
- **⚡ Fast Center-Tile Search (~2 ms)**:
  Instant 1-tile search across the center keyframe (`Tile #4`) of all 147,412 videos.

### 2. Cross-Domain Contrastive Fine-Tuning (`EVA-02-CLIP-L-14-336`)
- Jointly fine-tuned across curated real-world video keyframes and Danbooru2023 / Pixiv explicit illustrations using a unified **779-tag posture & attire taxonomy**.
- Multi-label positive-mask InfoNCE / SigLIP loss with weak supervision, hard-negative mining, and Test-Time Augmentation (TTA).

### 3. 2D Vector Optical Flow Rhythm & BPM HUD (`hover_pacing.py`)
- Analyzes 5-second hover preview videos on the fly using **4×3 spatial grid 2D vector optical flow** (`cv2.calcOpticalFlowFarneback`) + **PCA dominant-axis projection** (`vx, vy`).
- Handles edge-dominant and counter-oscillating camera/body motion (e.g., peripheral thrusting vs. stationary center) by weighting active motion cells via 90th-percentile energy and aligning phase polarity across tiles before zero-crossing peak detection.
- Displays a live **5-segment per-second BPM swatch bar** and a **60fps beat-synchronized pulse indicator** (`requestAnimationFrame`) that flashes on every detected motion stroke.

### 4. Interactive Dark-Mode Web UI (`scrape/web_viewer.py`)
- **Text, Reverse Image (Drag-and-Drop / Clipboard Paste), and Hybrid Search** (`70% Image + 30% Text`).
- **Interactive 3×3 Keyframe Matrix**: Hover or click any cell `0..8` on a result card to inspect individual keyframes or toggle the full 3×3 contact sheet mosaic.
- **One-Click "Similar →" Search**: Uses the card's winning keyframe crop as a reverse-image query.

---

## 📂 Repository Structure

```text
├── scrape/
│   ├── web_viewer.py                  # FastAPI + Dark UI server (9-Tile MIL, Dense Patch Rerank, Live BPM HUD)
│   ├── hover_pacing.py                # 4x3 Spatial Grid 2D Optical Flow + PCA Rhythm/BPM analyzer
│   ├── train_eva02_curated.py         # EVA-02-CLIP-L-14-336 cross-domain contrastive fine-tuning
│   ├── preencode_curated_v2.py        # Offline FP16 center-tile embedding indexer (globals_147k.npy)
│   ├── preencode_9tiles.py            # Offline FP16 9-tile MIL embedding indexer (keyframes_9tiles_147k.npy)
│   ├── crawl_xvideos_curated_bfs.py   # High-concurrency graph BFS crawler with live taxonomy curation
│   ├── filter_curated_pairs.py        # Posture & clothing taxonomy extractor (779 Danbooru tags)
│   ├── download_danbooru2023_curated.py
│   ├── download_pixiv_explicit.py
│   ├── retrieval_tta.py               # Test-Time Augmentation (TTA) for text & image retrieval
│   ├── weak_supervision.py            # Pseudo-labeling & temperature-scaled sharpening
│   ├── FRAME_INTERACTION.md           # Documentation on 9-Tile MIL & temporal frame interaction
│   ├── TRAINING.md                    # Documentation on contrastive training & memory optimization
│   ├── TTA.md                         # Documentation on Test-Time Augmentation
│   └── WEAK_SUPERVISION.md            # Documentation on weak supervision pipeline
├── tests/                             # Unit & integration test suite
├── tools/
│   └── bench_training_memory.py       # VRAM & throughput benchmark utility
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Pre-Encode Video Keyframe Indexes (One-Time)
```bash
# Encode 1-Tile (Center Keyframe) Index -> scrape/embeddings_v2/globals_147k.npy
python scrape/preencode_curated_v2.py

# Encode All 9 Keyframe Tiles per Video -> scrape/embeddings_v2/keyframes_9tiles_147k.npy
python scrape/preencode_9tiles.py
```

### 3. Launch the Web Viewer
```bash
python scrape/web_viewer.py --host 0.0.0.0 --port 7860 --device cuda
```
Then open `http://127.0.0.1:7860` in your browser.
