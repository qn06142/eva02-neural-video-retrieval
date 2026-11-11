# Learning more from the existing contact sheets

The most useful next experiment is to expose the model to all nine frames and
more videos in each **joint contrastive batch**, then test a conservative loss
for uncertain negatives. Both are available in `train_siglip_mil.py`; defaults
preserve the existing objective and memory behavior.

## Joint batches with checkpointed encoder chunks

`--image-chunk-size` and `--text-chunk-size` split encoder forwards into small
chunks. PyTorch activation checkpointing recomputes each chunk during backward
instead of retaining every frame's intermediate activations. The embeddings
from all chunks are concatenated **before** the bag-level loss is computed.

For example, 64 videos still form one 64-by-64 pair matrix, with up to 63
negative videos per query. Processing their 576 frames four at a time does not
reduce that negative pool. This differs from gradient accumulation, where each
microbatch has an independent pair matrix. The mechanism is encoder activation
checkpointing, not an implementation of the GradCache algorithm.

All frames retain differentiable paths to the encoder. RNG replay is enabled
for recomputation. Floating-point kernels can differ with chunk size, and
stochastic masks can differ between full-batch and chunked forwards; this does
not promise bitwise equality on GPU. Tests check the FP32 loss and every model
parameter gradient against the full-batch implementation, plus correct dropout
replay for the same chunking and CUDA mixed-precision backward.

Input pixels, embeddings, parameters, gradients and optimizer state still
consume memory. The feature concatenation and pair matrix also grow with batch
size. This reduces activation memory; it does not make memory usage constant.
Recomputation adds encoder work and is not inherently faster.

## Frozen-title uncertainty weighting

The baseline already treats identical normalized titles as positive pairs.
The optional `--negative-title-threshold` also uses a frozen copy of the
**starting text encoder** to identify highly similar, differently worded titles.
It reduces their negative penalty without declaring them positive.

For a teacher title cosine `s`, threshold `t`, and minimum weight `f`:

`negative_weight = 1 - (1-f) * clamp((s-t)/(1-t), 0, 1)`

All known positives retain weight 1. Negatives below the threshold retain
weight 1. Highly similar negatives approach the chosen floor. By default the
feature is disabled; when enabled, the default floor is 0.25. No uncertain pair
is dropped entirely with that floor.

The weights are detached and come from a fixed teacher. The student cannot
make its own negatives disappear by changing title similarities. The original
pairwise sigmoid reduction is preserved; weights multiply individual negative
terms without renormalizing the row. Smooth MIL pooling remains optional and
uses the same training/evaluation score function.

This is a **heuristic for weak labels**, inspired by false-negative mitigation;
it is not a reproduction of a published debiased contrastive objective. Similar
titles can describe genuinely distinct videos, especially when small wording
changes matter. A low threshold can weaken useful discrimination. Calibrate on
validation, compare to the disabled baseline, and inspect ambiguous cases before
adopting it. A lower weighted training loss is expected by construction and is
not evidence of better retrieval.

The frozen text encoder adds parameters and an extra text-only forward. It is
saved once as `title_teacher.pt` in the run directory, and its content hash is
recorded in training arguments/checkpoints. Keep this file when resuming;
resumed runs restore the original teacher rather than copying the updated
student. It is not needed for inference. Only load teacher/checkpoint files
from trusted sources. Older runs resume with these additions at their disabled
defaults; changing the loss or chunk settings during resume is rejected.

Epoch metrics include bags/frames used, known positive pairs, negative pair
count, negative weight sum, mean negative weight and the fraction downweighted.
These make it possible to see if the heuristic is suppressing too much of the
negative signal.

## Suggested controlled experiment

Use a new output directory for each configuration. Keep corpus, split seed,
validation settings and starting checkpoint fixed. The split manifests should
match before comparing outcomes. Use a starting checkpoint whose exposure to
the test set is known; the legacy `run1` has no saved training split.

1. First compare the existing frame subset with full nine-frame training using
   encoder chunks. Keep batch size, pooling and loss identical to isolate frame
   coverage. Then increase the joint video batch separately if memory allows.
2. At the selected frame/batch configuration, compare max and smooth pooling.
3. Add frozen-title weighting as a separate ablation. Keep TTA disabled while
   choosing training settings, then evaluate any selected TTA policy separately.
4. Repeat promising runs over seeds and report retrieval on an untouched test
   set. Do not select based on training loss or a single best validation seed.

Illustrative commands from the repository root (check capacity with a small
pilot first; these settings are not established optima):

```powershell
# All frames with a shared negative pool, using small encoder chunks.
.venv/Scripts/python scrape/train_siglip_mil.py --out scrape/siglip_runs/full_frames --bag-size 9 --batch-size 16 --grad-accum 1 --image-chunk-size 4 --text-chunk-size 4 --workers 2

# Same experiment plus the optional uncertainty-weighted negative loss.
.venv/Scripts/python scrape/train_siglip_mil.py --out scrape/siglip_runs/full_frames_weighted --bag-size 9 --batch-size 16 --grad-accum 1 --image-chunk-size 4 --text-chunk-size 4 --workers 2 --negative-title-threshold 0.9 --negative-weight-floor 0.25
```

If initializing from a local checkpoint, add the same `--init-from` to both
commands. Merely increasing `--grad-accum` does not increase negatives per query.

## Reproducing the memory measurement

```powershell
.venv/Scripts/python tools/bench_training_memory.py --model scrape/siglip_runs/run1/best --db scrape/thumbs.db --mosaic-dir mosaics --out scrape/memory_comparison.json --batch-size 16 --bag-size 9 --image-chunk-size 4 --text-chunk-size 4 --with-title-teacher
.venv/Scripts/python -m unittest discover -s tests -t . -p "test_*.py" -v
```

The benchmark performs one cold AdamW step per mode with identical initial
weights and input data, including the optional teacher. It does not save the
updated weights. CUDA tensor allocation can exceed physical VRAM on Windows
through memory paging; allocated/reserved bytes are not physical-VRAM residency.
One-step timings include initialization/warm-cache effects and should not be
used as steady-state throughput estimates.

References: [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html),
[Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://arxiv.org/abs/2101.06983),
[False Negative Cancellation](https://openaccess.thecvf.com/content/WACV2022/html/Huynh_Boosting_Contrastive_Self-Supervised_Learning_With_False_Negative_Cancellation_WACV_2022_paper.html).
