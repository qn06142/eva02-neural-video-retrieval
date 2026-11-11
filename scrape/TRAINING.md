# Retrieval training and evaluation

See [FRAME_INTERACTION.md](FRAME_INTERACTION.md) for the residual frame set
encoder, frozen-backbone training, checkpoint loading and inference commands.

See [WEAK_SUPERVISION.md](WEAK_SUPERVISION.md) for joint large-batch training
with checkpointed encoder chunks, full-frame supervision and optional frozen-title
negative weighting.

See [TTA.md](TTA.md) for deterministic test-time view ensembles, smooth MIL
pooling, contact-sheet retrieval, cache compatibility and policy evaluation.

The active models here are SigLIP dual encoders for thumbnail/title retrieval
and multi-instance retrieval over contact sheets. These scripts do not train a
diffusion model. Training titles are weak video-level labels: they may describe
content outside a particular frame, and matching a title is only a proxy for
human-rated retrieval relevance.

## What the trainers now guarantee

- A seeded sample of the eligible corpus replaces a discovery-order prefix
  when using `--max-samples`.
- Exact normalized titles and repeated asset paths form connected groups that
  stay wholly in training or validation. `split.json` records the samples,
  assignments, configuration and metadata hash. Reusing the output with changed
  metadata or split settings fails. This does **not** detect re-encoded images,
  near-duplicate videos, or changed file contents at an unchanged path.
- Unreadable assets raise an error naming the file. They never silently become
  blank positive examples. Check and repair the offending asset before resuming.
- The normalized pairwise sigmoid loss runs in float32. Identical normalized
  titles within a batch are positives, reducing known false-negative labels.
  Semantically equivalent but differently worded titles can still be negatives.
- CUDA uses bf16 when supported, otherwise fp16 with gradient scaling. Clipping
  follows unscaling; an overflow-skipped update does not advance the scheduler.
- Every sample is used, including the last short batch. Accumulation windows
  average losses by query count, including short final windows. Accumulation
  increases queries per optimizer update; it does **not** create negatives across
  microbatches. A batch of 24 with accumulation 2 has at most 23 negatives per
  query, not 47. Do not compare raw sigmoid losses across different batch sizes.
- Thumbnail augmentation is enabled for training; `--no-augment` disables it.
  Validation is deterministic. MIL samples training tiles randomly and evaluates
  all grid tiles by default. `--eval-bag-size` explicitly limits evaluation to
  evenly spaced tiles, which changes the evaluation protocol.
- Recall uses chunked exact ranking rather than allocating an N-by-N-by-K
  similarity tensor. Ties break by corpus index. Recall is strict aligned-pair
  retrieval: it can penalize a different relevant video, including a duplicate
  title in validation. It is not a comprehensive relevance metric.
- The starting model is evaluated before training, and `best/` retains it if
  no trained epoch beats its recall@1. `metrics.jsonl` records the baseline,
  candidate count, validation recall, sample-weighted training loss, elapsed
  time, and optimizer updates. `environment.json` records runtime versions.

## Running an experiment

From the repository root, start with a small run in a new output directory:

```powershell
.venv/Scripts/python scrape/train_siglip.py --out scrape/siglip_runs/thumb_v2_smoke --max-samples 2000 --epochs 1 --workers 0
.venv/Scripts/python scrape/train_siglip_mil.py --out scrape/siglip_runs/mil_v2_smoke --max-samples 2000 --epochs 1 --workers 0
```

Use `--device cpu` for a CPU correctness check. For larger runs, choose workers
and batch size after checking throughput and available VRAM. MIL processes
`batch-size * bag-size` images during training; evaluation defaults to the full
grid. A new experiment needs a new `--out`. Existing run directories are protected
against an accidental fresh start.

`--init-from scrape/siglip_runs/run1/best` loads older model/processor weights
and starts a new optimizer and schedule. `--resume` now means a true continuation:

```powershell
.venv/Scripts/python scrape/train_siglip_mil.py --out scrape/siglip_runs/mil_v2 --epochs 4 --resume scrape/siglip_runs/mil_v2/epoch-2
```

Repeat the original training arguments, including the original total epoch
budget, seed, batch size, worker count and data/split settings. Resume from the
latest completed epoch in its original output directory. Each epoch stores
optimizer, scheduler, scaler, Python/NumPy/PyTorch/CUDA RNG and loader-generator
state in `training_state.pt`. Partial epochs replay from the preceding completed
checkpoint; minibatch-level resume is not supported. Workers restart each epoch
so their seeds can be reconstructed. Exact GPU reproducibility still depends on
hardware, library versions and deterministic kernels. Load only trusted local
training-state files; restoring Python/NumPy RNG state requires pickle loading.

## Methodology for deciding whether a run improved

1. Freeze a corpus snapshot and retain the split manifest. Before a substantive
   benchmark, audit video identity and image near-duplicates and group them too;
   title/path grouping alone cannot establish a leakage-free benchmark.
2. Compare the starting checkpoint and fine-tuned epochs on the same validation
   candidate set, with identical grid, frame count, preprocessing and ranking.
   Recall@k depends on candidate count. Historical scores from the old random
   split or first-K-frame evaluation are not directly comparable. Comparing
   thumbnail and MIL models requires a common held-out set of video identities;
   their independently generated manifests do not guarantee that alignment.
3. Change one factor at a time: learning rate, thumbnail augmentation, or training
   bag size. Keep the actual contrastive batch size fixed when studying the
   other factors. Repeat promising configurations over several seeds and report
   the spread, not only the best seed. Use new output directories for each run.
4. Select settings on validation only. Reserve a separate immutable test corpus
   and representative human-rated queries for final reporting. The scripts
   currently create train/validation splits; a separate test set and relevance
   annotations must be supplied before making claims about generalization.
5. Keep the baseline unless the measured gain holds on that final evaluation.
   A lower training loss or a passing correctness test does not establish a
   retrieval-quality improvement.

## Verification

```powershell
.venv/Scripts/python -m unittest discover -s tests -t . -p test_training.py -v
```

The offline suite checks grouped/frozen splits, corrupt-image behavior, tile
boundaries, chunked recall and ties, FP32 loss and gradients, short accumulation
windows, actual tiny SigLIP forward/backward passes, Windows worker spawning,
and exact CPU continuation after interruption. A CUDA-only case checks fp16
overflow skipping and scheduler behavior. Tests use synthetic images and tiny
models and do not download model weights.

References: [PyTorch AMP and accumulation](https://docs.pytorch.org/docs/stable/notes/amp_examples.html),
[SigLIP processor requirements](https://huggingface.co/docs/transformers/model_doc/siglip).
