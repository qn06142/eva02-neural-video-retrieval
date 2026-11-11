# Model scoring and test-time augmentation

The current local dataset contains contact sheets (`asset_path`), not downloaded
single-frame thumbnails. `retrieve.py --asset-type auto` detects the crawler
schema and uses contact sheets when thumbnail columns are absent. Each sheet is
split into frames; its score is pooled over frame similarities. Explicit
`--asset-type thumbnail` remains available for the older thumbnail schema.
The thumbnail trainer also handles either schema, but requires actual thumbnail
files; use `train_siglip_mil.py` for the current contact-sheet-only corpus.

## Deterministic view ensembles

Both validation encoders and corpus/query-image retrieval support:

| Policy | Views per frame | Operation |
| --- | ---: | --- |
| `none` | 1 | Original frame; default |
| `flip` | 2 | Original plus horizontal mirror |
| `five-crop` | 5 | Four corner crops and a center crop |
| `ten-crop` | 10 | Five crops plus their horizontal mirrors |

`--tta-crop-scale 0.9` retains 90% of each image side in the crop policies
(approximately 81% of image area), preserving the frame's aspect ratio before
the model processor resizes it. Every view is encoded and L2-normalized; the
view embeddings are averaged and normalized again. MIL keeps a separate
ensemble embedding per frame, then pools frame **scores**, preserving the
multi-instance objective. Text queries remain unchanged. Image queries use the
same TTA policy as the corpus.

`--view-batch-size` limits the number of preprocessed views and vision-forward
inputs held in a single batch. It controls memory even with many frames and
ten-crop ensembles. `--batch-size` controls source images or bags delivered by
the loader. PIL inputs still occupy host memory, so avoid unnecessarily large
loader batches and worker counts.

TTA does not update model weights. Flips can change text or directional meaning;
crops can remove relevant context. More views need more preprocessing and vision
forwards and do not guarantee better retrieval. Keep a single-view comparison.

From the repository root:

```powershell
# Build an index once; subsequent queries reuse it.
.venv/Scripts/python scrape/retrieve.py --backend siglip --asset-type mosaic --tta flip --workers 2 --view-batch-size 32 --save-emb scrape/corpus_flip.pt
.venv/Scripts/python scrape/retrieve.py --backend siglip --asset-type mosaic --tta flip --text "example query" --load-emb scrape/corpus_flip.pt --no-copy
```

Cache metadata records a content hash of checkpoint weights/config/processor
assets, backend, library versions, inference dtype, TTA policy/crop scale, corpus
paths/order and frame layout. Model objects stay out of loader workers. Wrong
models, policies, layouts or corpus order now raise an error instead of producing
misaligned rankings. Older caches lack this metadata and must be rebuilt.
Changes to image contents at an unchanged path are not hashed: rebuild the index
after replacing/re-downloading assets. The index retains per-frame embeddings,
so changing score pooling can reuse the same index.

## Smooth multi-instance pooling

Hard max pooling only backpropagates through the best-scoring frame (apart from
ties). The optional `--pooling logmeanexp` uses:

`score = temperature * (logsumexp(frame_scores / temperature) - log(frame_count))`

The normalization avoids an artificial score shift when an identical frame set
is repeated. Positive temperature distributes gradients across frames and
smoothly approaches max as temperature decreases. Training and validation share
the same implementation. Pooling mode and temperature are saved in the model
config; retrieval and the TTA evaluator inherit these settings unless explicitly
overridden. `max` remains the default for new runs and legacy checkpoints.

```powershell
.venv/Scripts/python scrape/train_siglip_mil.py --out scrape/siglip_runs/mil_smooth --pooling logmeanexp --pool-temperature 0.07 --eval-tta flip --view-batch-size 32
```

This is an experiment option, not an established quality gain. Compare max and
smooth pooling with the same frozen train/validation split, actual batch size,
frame count and starting weights. New runs in separate output directories
generate identical assignments when the corpus, seed and split settings match.
Changing TTA mid-resume is rejected because it changes checkpoint selection.
Existing version-2 resumable checkpoints accept the new options at their
backward-compatible defaults.

## Measuring TTA before choosing a policy

The evaluator requires a saved split and never constructs a fresh holdout:

```powershell
.venv/Scripts/python scrape/evaluate_tta.py --checkpoint scrape/siglip_runs/mil_smooth/best --split scrape/siglip_runs/mil_smooth/split.json --mode mil --out scrape/tta_comparison.json --workers 2 --view-batch-size 32
```

It evaluates all requested policies against `none` on the identical candidate
set, with the same model, frame count and pooling. The JSON report records
Recall@1/5/10, end-to-end encoding/ranking seconds, checkpoint/split hashes and
runtime versions. Timing includes loader startup and preprocessing; it is an
operational measurement, not an isolated GPU-throughput benchmark. Policies run
in fixed order, so timings may also reflect warm caches and hardware contention.

A paired query bootstrap estimates the Recall@1 change; interval coverage is
Bonferroni-adjusted for the number of non-baseline policies. The report suggests
a non-baseline policy only when its lower bound is positive, otherwise `none`.
These intervals are descriptive: correlated or duplicate queries and repeated
tuning can weaken their interpretation. Select policies on validation and
confirm the result on a separate untouched test corpus.

Use `evaluate_tta.py` and a recorded manifest for retrieval comparisons.
Obsolete thumbnail-only benchmarks and one-off diagnostics have been retired.

## Checks

```powershell
.venv/Scripts/python -m unittest discover -s tests -t . -p "test_*.py" -v
```

Tests verify deterministic views, streamed/dense ensemble equivalence and batch
bounds, no-TTA numerical equivalence, frame grouping, Windows worker spawning,
smooth-pooling gradients/invariance, both crawler schemas, cache mismatches and
checkpoint identity changes, plus the training/resume regressions.

References: [Torchvision FiveCrop](https://docs.pytorch.org/vision/2.0/generated/torchvision.transforms.FiveCrop.html),
[LogAvgExp pooling paper](https://arxiv.org/abs/2111.01742).
