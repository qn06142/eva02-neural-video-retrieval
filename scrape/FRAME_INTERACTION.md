# Residual frame-interaction model

## Laptop cooling preset

The first launch stopped during baseline evaluation after a reported laptop
crash. That original job stopped without a completed epoch. The logs do not
identify the crash cause. After explicit user authorization, the cooled
`frame_interaction_cool_v1` job was launched at 11:08 on 7 November 2026
(Bangkok). Inspect its logs for current progress; this guide is not a live status.

Both trainers now default to cooperative GPU pacing: `--gpu-duty-cycle 0.3`
and `--gpu-min-pause-ms 50`. CUDA calls are synchronized before timing; encoder
chunks, validation image/text encoding, backward and optimizer steps receive
deliberate idle time. Checkpoint recomputation is also paced, and nested calls
count existing rest rather than sleeping for the same work twice. This is a
work/rest target, not a hardware power or temperature limit.

The authorized cooled restart uses a new run directory and smaller bursts:

```powershell
.venv/Scripts/python -u scrape/train_siglip_mil.py --init-from scrape/siglip_runs/run1/best --out scrape/siglip_runs/frame_interaction_cool_v1 --frame-mixer residual-set --freeze-backbone --bag-size 9 --batch-size 8 --grad-accum 4 --epochs 3 --lr 0.0001 --warmup-ratio 0.01 --image-chunk-size 4 --text-chunk-size 8 --view-batch-size 4 --workers 0 --log-every 10 --device cuda --gpu-duty-cycle 0.3 --gpu-min-pause-ms 50
```

Compared with the original launch, this processes four images per GPU forward,
reduces source video batches from 32 to 8, and removes loader worker processes
to reduce host-memory/CPU pressure. Gradient accumulation retains 32 queries per
optimizer update but gives only up to seven negatives per query in each joint
loss. The original larger batch had up to 31. Cooldown settings may be changed
on epoch-boundary resume without invalidating the optimizer state; training
batch and model settings must remain consistent. No global GPU power/fan or
Windows power settings have been changed.

`SiglipBagModel` extends the existing SigLIP dual encoder with an optional
permutation-equivariant frame set encoder. Independent image/text encoding and
the bag-level sigmoid loss remain available for legacy checkpoints.

The new branch is:

`normalized 768-D frames -> LayerNorm -> 256-D projection -> two Transformer blocks -> LayerNorm -> 768-D projection`

Each block has four attention heads, a 1024-D feed-forward layer, GELU,
pre-normalization and zero dropout. The output is added to the original
normalized features through a learned `tanh(gate)` scalar, then normalized.
The gate starts at zero; the starting cosine-retrieval function therefore
matches the original model up to floating-point rounding. Branch weights have
nonzero random initialization so the gate can learn immediately. The default
module adds **1,975,809 parameters**.

There are no position embeddings. Frame permutations permute the outputs in
the same way, and final MIL pooling remains order-independent. This matches the
current data loader, which randomly orders selected frames and does not provide
reliable timestamps. The module can learn context between frames, but it does
not model motion or elapsed time.

## Training

```powershell
.venv/Scripts/python -u scrape/train_siglip_mil.py --init-from scrape/siglip_runs/run1/best --out scrape/siglip_runs/frame_interaction_cool_v1 --frame-mixer residual-set --freeze-backbone --bag-size 9 --batch-size 8 --grad-accum 4 --epochs 3 --lr 0.0001 --warmup-ratio 0.01 --image-chunk-size 4 --text-chunk-size 8 --view-batch-size 4 --workers 0 --log-every 10 --gpu-duty-cycle 0.3 --gpu-min-pause-ms 50
```

This first stage trains only the new frame mixer. Image/text encoders and logit
scale/bias are frozen, and encoder modules stay in evaluation mode even while
the mixer is training. Encoder forwards are chunked without recomputation when
the backbone is frozen. The existing training augmentations remain enabled;
validation uses all nine original frames, no TTA, and max-score pooling.
The optional negative-title weighting and smooth pooling are disabled to keep
this architecture experiment separate from loss experiments.

The full eligible corpus is used with the existing seed-42 grouped split and
at most 1,000 validation candidates. The exact manifest, training arguments,
environment and trainable parameter count are saved in the run directory.
The legacy starting checkpoint has no saved original training split; the new
head has a recorded holdout, but original-backbone exposure is unverified.

Epoch directories contain the backbone, mixer configuration/weights, processor,
and resumable optimizer/scheduler/RNG state. `best/` is selected by validation
Recall@1 and retains the zero-gate baseline if no trained epoch improves it.
To resume an interrupted run, repeat the original flags, replace `--init-from`
with `--resume <latest-completed-epoch-directory>`, and retain the same output
directory and original total epoch budget. Do not change freeze/head settings
mid-resume. Selective backbone unfreezing should be a separate measured run.

## Evaluation and retrieval

Use the supplied model class when loading these checkpoints:

```python
from frame_model import SiglipBagModel
model = SiglipBagModel.from_pretrained(checkpoint)
```

The training, TTA evaluator, retrieval utility and memory benchmark all load this
class. `get_image_features()` still returns independent frame features;
`mix_frame_features()` contextualizes each video's frame set after TTA view
aggregation and before MIL scoring. This keeps training and inference aligned.
The architecture and branch weights are saved in the standard config/weight
files, so cache checkpoint hashes include the mixer. Plain `SiglipModel` does
not understand the extra head and must not be used to evaluate these checkpoints.

```powershell
.venv/Scripts/python scrape/evaluate_tta.py --checkpoint scrape/siglip_runs/frame_interaction_cool_v1/best --split scrape/siglip_runs/frame_interaction_cool_v1/split.json --mode mil --policies none --out scrape/frame_interaction_validation.json
.venv/Scripts/python scrape/retrieve.py --ckpt scrape/siglip_runs/frame_interaction_cool_v1/best --asset-type mosaic --save-emb scrape/frame_interaction_index.pt
```

Corpus embeddings retain separate contextualized frame vectors. Text queries
use the frozen text encoder. Image queries are processed as one-frame bags;
this is supported, but image-query retrieval needs its own quality evaluation
because training uses video-level text queries and multi-frame bags.

The thumbnail trainer rejects mixer checkpoints; use the MIL route rather than
silently discarding the architecture. Legacy backbone-only checkpoints continue
to load and behave without a frame mixer.

## Verification

```powershell
.venv/Scripts/python -m unittest discover -s tests -t . -p "test_*.py" -v
```

Tests cover zero-gate baseline equivalence, frame-order equivariance, real
cross-frame interaction, first-step gate learning, frozen-backbone isolation,
architecture/weight round trips, legacy loading, evaluation integration and
CUDA bf16 inference/AMP training, in addition to the existing training/TTA tests.
