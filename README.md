# MIQANet

Minimal implementation of the main method: image-conditioned factor prompts
and direct similarity regression. No deep prompts or cross-batch memory.

## Method

- Frozen OpenAI CLIP ViT-L/14@336px; RGB images resized to 518 x 518 with
  bicubic interpolation, center-cropped and CLIP-normalized.
- Patch features from layers 6, 12, 18 and 24 are projected to 768 dimensions,
  L2-normalized at each layer and averaged, then normalized for similarity.
- Nine clean/degraded prompt pairs use four shared learnable context tokens.
  Prototype attention conditions the context on the image with scale 0.1.
  Factor queries use the normalized difference of masked anchor-token means;
  clean/degraded scoring uses the complete frozen CLIP text encoder.
- Average pooling of patch/text cosine similarities gives 18 values:
  `[9 clean, 9 degraded]`. Prediction is `Linear(18,32) -> GELU ->
  Linear(32,1) -> Sigmoid`.
- Trainable parameters: **3,553,409**. The CLIP backbone is frozen.

The only loss is within-batch pairwise fidelity (prediction and target
temperatures both 0.1) plus `0.2 * (1 - Pearson correlation)`. Scores are
normalized with the training-set minimum and maximum. There are no additional
evidence-pooling, factor-weighting, global-score, adapter or ablation branches.

## Install

The recorded environment is Python 3.10, PyTorch 2.5.1 and CUDA 12.1.

```bash
python -m pip install -r requirements.txt
```

CLIP weights are downloaded automatically to `~/.cache/clip`. For offline use,
pass `--clip-checkpoint /path/to/ViT-L-14-336px.pt` to either command below.
The supplied tokenizer vocabulary must remain in `miqanet/`.

## Data

Use the original fixed train/test CSV splits; this code does not re-split data.
Each CSV needs `image_path` (or `image_name`) and `score`:

```csv
image_path,score
images/example_1.png,72.4
images/example_2.png,38.6
```

Relative paths are resolved against `--image-root`. Scores must be finite and
use the convention **higher is better**. For DMOS databases, reverse the score
direction consistently before evaluation. Dataset images and label CSVs are
not included in this code package.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --train-csv /path/to/train.csv \
  --test-csv /path/to/test.csv \
  --image-root /path/to/dataset \
  --output-dir runs/single \
  --epochs 3 --batch-size 32 --seed 42
```

Use **3 epochs** for Single/Double/Triple and **2 epochs** for
Object/Semantic/Depth/Overall. Each task is trained separately with its own
CSV pair. Defaults match the experiments: AdamW at `3e-4`, weight decay
`1e-3`, cosine decay to `1e-6`, gradient clipping at 1.0, no AMP, no data
augmentation and training `drop_last=True`. The cosine schedule spans the
requested total epochs, so a two-epoch run is not the first two epochs of a
three-epoch run.

Each epoch prints raw test PLCC/SRCC without nonlinear fitting. Outputs are
`config.json`, `train_log.jsonl`, `best.json`, `best.pth`, and `last.pth`.
To reproduce the historical reports, `best.pth` maximizes the mean test
PLCC/SRCC; this is test-selected, not validation-selected. `last.pth` stores
the fixed final epoch. Checkpoints contain the quality head and training MOS
range; frozen CLIP weights are loaded separately.

## Predict and evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --checkpoint runs/single/best.pth \
  --csv /path/to/test.csv \
  --image-root /path/to/dataset \
  --output-csv runs/single/predictions.csv \
  --metrics-json runs/single/metrics.json
```

Predictions contain `image_path`, `score`, normalized `prediction`, and
`prediction_mos_scale` mapped with the source training MOS range. For
cross-database evaluation, use the Overall checkpoint and a CSV containing
every target image. For AGIQA-3K, use quality MOS. Target labels are never used
to normalize predictions or fit a calibration function.

The evaluator also loads historical no-deep Direct Similarity MLP head
checkpoints, discarding only their three inactive bookkeeping tensors. It
rejects checkpoints containing deep prompts.

The release was checked against the experiment implementation on four images:
identical seeded initialization, preprocessing, visual/text features, scores
and losses. After one optimizer step, the largest parameter difference was
less than `7e-7` in the recorded environment. This is a code-equivalence check,
not a rerun of all training experiments. Exact benchmark replication also
requires the same images, fixed splits, epoch count and checkpoint selection.

Third-party attribution and licenses are retained in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
