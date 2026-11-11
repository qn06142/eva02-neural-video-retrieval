# Pixiv Explicit SigLIP Refinement Pipeline

This document describes the pipeline to refine the SigLIP dual encoder (trained on xvideos thumbnails and contact-sheet mosaics) on explicit, un-censored, text-free Pixiv illustrations.

---

## 1. Bridging the Distribution Gap (Pixiv Tags vs. XVideos Video Titles)

SigLIP's existing pre-training and XVideos fine-tuning target **natural adult video titles** (e.g. `"Thick & Curvy Glam Girl with Big Tits Gets Pounded in Fishnet Stockings with Cum in Pussy Bareback"`), whereas Pixiv provides **isolated Booru / WD-tagger keywords** (e.g. `['juliana_(pokemon)', '1girl', 'breasts', 'oral', 'deepthroat', 'creampie']`).

To eliminate this distribution gap, [`train_siglip_pixiv.py`](train_siglip_pixiv.py) incorporates:

1. **Tag-to-Video-Title Translation (`synthesize_video_title`)**:
   Maps WD-tagger tags into natural XVideos title grammar, vocabulary, and cadence:
   - **Characters & Subjects**: e.g. `Juliana (Pokemon)`, `Anime schoolgirl`, `Two anime babes`, `Hot anime milf`
   - **Attributes**: e.g. `large_breasts` -> `big tits`, `huge_breasts` -> `huge tits`, `fishnets` -> `fishnet stockings`
   - **Acts & Positions**: e.g. `fellatio`/`deepthroat` -> `deepthroat blowjob`, `doggystyle` -> `fucked doggystyle`, `cowgirl_position` -> `riding cock`, `creampie` -> `huge creampie`
   - **Synthesized Title Example**:
     > `Juliana (Pokemon) with big tits deepthroat blowjob and huge creampie`

2. **Multi-Style Prompt Sampling (`--prompt-style mixed`)**:
   During training, each sample dynamically samples between formats:
   - **45% Video Title**: Natural sentence title (`"POV Busty anime babe with huge tits riding cock"`)
   - **40% Hybrid**: Video title + pipe separator + tags (`"Busty anime babe riding cock | oral, stockings, creampie"`)
   - **15% Clean Tag List**: Comma-separated tags (`"1girl, big tits, stockings, blowjob"`)
   This ensures the model responds accurately to both natural language queries and tag searches.

3. **Optional XVideos Replay (`--xvideos-ratio`)**:
   Pass `--xvideos-ratio 0.2` to interleave 20% real XVideos thumbnail/title pairs from `thumbs.db` during training, preventing catastrophic forgetting of live-action video titles.

---

## 2. Multi-Frame / Mixer Preservation

When refining from the mosaic training run (`frame_interaction_adaptive_v2`):
- **`ResidualFrameMixer` is frozen by default** (`--freeze-frame-mixer`): preserves learned multi-frame video contextualization weights.
- **Vision and Text backbones are unfrozen**: adapts the dual encoder to anime illustration aesthetics, character identities, and sexual acts.
- **Save format**: All weights and `frame_mixer` configs are saved in the checkpoint, remaining 100% compatible with both video retrieval (`retrieve.py`, `train_siglip_mil.py`) and single-image queries.

---

## 3. Running Refinement

Once mosaic training finishes (or from `epoch-1`/`epoch-2`), launch refinement:

```powershell
& "D:\02_Development_And_Projects\xvideos-diffusion\.venv\Scripts\python.exe" `
    "D:\02_Development_And_Projects\xvideos-diffusion\scrape\train_siglip_pixiv.py" `
    --init-from "D:\02_Development_And_Projects\xvideos-diffusion\scrape\siglip_runs\frame_interaction_adaptive_v2\best" `
    --out "D:\02_Development_And_Projects\xvideos-diffusion\scrape\siglip_runs\pixiv_refined_v1" `
    --pixiv-dir "D:\02_Development_And_Projects\xvideos-diffusion\scrape\pixiv_explicit_10k" `
    --prompt-style mixed `
    --xvideos-ratio 0.2 `
    --epochs 3 `
    --batch-size 32 `
    --grad-accum 4 `
    --lr 1e-5 `
    --workers 0 `
    --device cuda `
    --gpu-duty-cycle 0.6 `
    --gpu-target-temp 70 `
    --log-every 20
```

---

## 4. Configuration Options

| Option | Default | Description |
|---|---|---|
| `--prompt-style` | `mixed` | Prompt format: `mixed` (45% title, 40% hybrid, 15% tags), `video-title`, `hybrid`, `tag-list` |
| `--xvideos-ratio` | `0.0` | Fraction of training set composed of real XVideos replay samples (e.g. `0.2`) |
| `--pixiv-dir` | `scrape/pixiv_explicit_10k` | Path to Pixiv download directory |
| `--init-from` | `None` | Path to starting checkpoint |
| `--out` | `scrape/siglip_runs/pixiv_refined_v1` | Output directory |
| `--freeze-frame-mixer` | `True` | Preserve video mixer weights while training backbone |
| `--train-frame-mixer` | `False` | Also train the frame mixer head |
| `--tag-shuffle` | `True` | Shuffle general tags during training |
| `--tag-dropout` | `0.05` | Randomly drop general tags during training |
| `--gpu-duty-cycle` | `0.3` | Paced GPU duty cycle (0.0 to 1.0) |
| `--gpu-target-temp` | `None` | Automatic temperature throttling target (°C) |

---

## 5. Verification & Testing

Run unit tests for the pipeline:

```powershell
& "D:\02_Development_And_Projects\xvideos-diffusion\.venv\Scripts\python.exe" -m unittest tests/test_pixiv_training.py
```
