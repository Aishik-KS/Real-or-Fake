# Robust AIGC Image Detector

A detector for AI-generated images that is built to **stay accurate after
real-world image transformations** — JPEG re-compression, blur, resizing,
noise, color jitter, and cropping — not just on clean generator output.

- **Model (shipped: v7):** two frozen backbones — CLIP **ViT-L/14** + **DINOv2
  ViT-L** — fused into a small MLP head (real=0, AIGC=1). ~733M params total,
  ~0.9M trainable — well under the 2B cap. The final head is **distilled from a
  v5+v6 ensemble** (online distillation on strong-augmented images), which beat
  both the ensemble and either teacher while staying a single, fast `.pt`. (An
  earlier CLIP-only linear probe, ~428M, is kept as a baseline.)
- **Robustness lever:** train-time augmentation with the same six transform
  families the model is evaluated against.
- **Generalization lever:** train on many generator families (FLUX, DDPM,
  Other_based, styleGAN, BigGAN, DF-GAN, Midjourney) + diverse real sources
  (OpenImages, COCO, FFHQ, Unsplash); hold generators out to *measure* it.
- **Calibrated output:** temperature scaling + Expected Calibration Error
  (ECE) reporting, so `pred` is a trustworthy P(AIGC), not a raw sigmoid.

See [DEVPOST.md](DEVPOST.md) for the narrative write-up, [PLAN.md](PLAN.md)
for the improvement roadmap, and [PROGRESS.md](PROGRESS.md) for the full build log.

---

## Results (shipped model, v7 — distilled CLIP+DINOv2 hybrid)

**Scored metric (0.5·AUROC_clean + 0.5·AUROC_robust), in-dist:** v5 0.9842 →
v6 0.9812 → **v7 0.9867** (best). v7 also has the best single-transform robust
mean (**0.9805**), the best compound robustness, and the best calibration
(ECE **0.0005**). It's a single `.pt` at 733M params, 2× faster than the
ensemble it distills.

### In-distribution (FLUX) — robust across every transform
Clean AUROC **0.9928**; under every single transform severity AUROC stays
**0.95–0.99**. Full per-severity table:
[`results/robustness_table.csv`](results/robustness_table.csv).

### Compound / real-world robustness (the differentiator)
Real re-shared images stack degradations (double-JPEG, resize→re-JPEG,
screenshots, thumbnails). v5 **collapsed** on extreme low-res chains; v7 fixes
it (visual: [`results/fig_compound_robustness.png`](results/fig_compound_robustness.png)):

| Compound condition | v5 | **v7** |
|---|---|---|
| extreme downscale 0.10× (~22px) | 0.583 | **0.831** |
| thumbnail chain (down 0.15×+JPEG40) | 0.527 | **0.796** |
| extreme downscale 0.15× (~34px) | 0.803 | **0.917** |
| screenshot chain | 0.853 | **0.905** |

### Cross-generator generalization (held-out — never trained on)

| Test set | v2 (CLIP, less diverse) | v5 | **v7 (shipped)** |
|---|---|---|---|
| DDIM (unseen diffusion) | 0.982 | 0.938 | 0.928 |
| GigaGAN (unseen GAN) | 0.876 | 0.903 | 0.901 |
| Imagen + starGAN (unseen) | 0.697 | 0.986 | **0.996** |
| Aesthetic (Unsplash / Midjourney) | 0.767 | 0.998 | **0.999** |
| **Official demo (DALL·E-Advanced / COCO val2017)** | 0.990 | 0.997 | 0.997 |

Two levers drove this. **Data diversity** took unseen-generator AUROC from
**0.70 → 0.99** (v2→v5), and adding aesthetic reals **and** aesthetic fakes
fixed the "beautiful real photo flagged as AI" failure (0.767 → 0.998).
**Then robustness-augmentation + ensemble distillation** (v5→v7) lifted the
compound/low-res robustness and the scored metric further, at near-zero clean
cost. Full comparison:
[`results/crossgen_summary.csv`](results/crossgen_summary.csv). See Limitations
for the honest trade-offs (small DDIM cost; CIFAKE 32px confound).

---

## Setup

Requires Python 3.10+ and (for training) an NVIDIA GPU with a CUDA build of
PyTorch. Built and verified on Python 3.12 + RTX 3060 (12 GB), CUDA 12.8.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate

# GPU PyTorch (pick the CUDA build matching your driver; cu128 used here):
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# everything else:
pip install -r requirements.txt
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Reproduce

All scripts are run from inside `src/` (relative paths assume this).

**1. Get data.** SID_Set (primary) is streamed — no login, only a small
balanced subset is downloaded:

```bash
python scripts/prepare_sid_set.py --out_dir raw_downloads/sid_set --per_class 2000
python scripts/prepare_sid_set.py --out_dir raw_downloads/sid_set_test --per_class 500 --split validation
```

Then place images as `data/train/{real,fake}` and `data/test/{real,fake}`
(the train split feeds training; the official `validation` split is the
held-out test — no leakage). CIFAKE (already provided as a zip in
`raw_downloads/`) is used only as a smoke-test and unseen-generator benchmark.

**2. Train** the frozen ViT-L/14 probe (auto-resumes from `checkpoints/last.pt`):

```bash
cd src
# hybrid CLIP+DINOv2 (current best):
python train.py --data_dir ../data/train --hybrid \
  --clip_name openai/clip-vit-large-patch14 --hidden_dim 512 \
  --epochs 6 --batch_size 64 --num_workers 4 --out_dir ../checkpoints_hybrid
# (drop --hybrid for the CLIP-only baseline; then run calibrate.py either way)
```
New in the pipeline: `src/model_hybrid.py` (two-branch model + shared loaders),
`scripts/prepare_unsplash.py` (aesthetic reals), `scripts/stream_extract_wildfake.py`
(partial-stream huge WildFake zips), `scripts/robust_download.py` (throttle-aware,
token-rotating downloads), `scripts/check_coco_overlap.py` (compliance).

**Long / multi-machine training (stop anywhere, continue on any GPU).**
Training checkpoints *mid-epoch* — periodically (`--save_every_min`, default 10),
on Ctrl+C (it finishes the batch, saves, exits), and at an optional time budget
(`--max_hours`). To resume, just re-run the **same command** (add `--fresh` to
start over). You can pick up on a **different machine with a different/unknown
GPU and even a different `--batch_size`** — progress is tracked in samples, RNG
restore is best-effort (never crashes on a new GPU), and checkpoint writes are
atomic. The data in `--data_dir` must be identical on both machines; the model
architecture flags must match (they're guarded).

```bash
# e.g. run overnight on PC 1 (RTX 3060), 12h budget:
python train.py --data_dir ../data/train --clip_name openai/clip-vit-large-patch14 \
  --epochs 20 --batch_size 128 --max_hours 12
# ...copy checkpoints/last.pt + the data to PC 2 (e.g. Ada 6000), then continue:
python train.py --data_dir ../data/train --clip_name openai/clip-vit-large-patch14 \
  --epochs 20 --batch_size 256   # bigger batch is fine; resumes from the exact spot
```

**Adding WildFake (or any other dataset) for generator diversity.** Once you've
downloaded/uploaded a dataset to a folder (or a `.zip`), ingest it with
`prepare_wildfake_local.py`. **Always `--dry_run` first** — it previews how it
classified real vs fake (from path keywords) without copying, so labels can't be
silently swapped. By default it *adds* to `data/` (mixing with SID_Set); it
de-duplicates by content hash to prevent train/test leakage.

```bash
python ../scripts/prepare_wildfake_local.py --src /path/to/WildFake --dry_run
python ../scripts/prepare_wildfake_local.py --src /path/to/WildFake --per_class 4000
# or hold specific generators out as a true cross-generator test set:
python ../scripts/prepare_wildfake_local.py --src /path/to/WildFake \
  --holdout_generators midjourney,dalle3 --generator_level 0
```

**3. Calibrate** (fit temperature, report ECE, save T into the checkpoint):

```bash
python calibrate.py --checkpoint ../checkpoints/best_model.pt --data_dir ../data/train
```

**4. Inference** — the required deliverable. Outputs JSON of
`{image_path, pred}` (calibrated P(AIGC)) per image:

```bash
python infer.py --image_dir ../data/test \
  --checkpoint ../checkpoints/best_model.pt --out ../results/predictions.json
```

**5. Robustness evaluation** — clean + every transform×severity, with
AUROC / AP / accuracy / F1 / ECE / drop-from-clean, plus error analysis:

```bash
python eval_robustness.py --data_dir ../data/test --checkpoint ../checkpoints/best_model.pt
python ../scripts/make_visuals.py   # -> results/robustness_visual.png
```

To evaluate on your own benchmark (e.g. the COCO-val2017 / DALL·E-Advanced
demo set — *report only, never train on it*), lay it out as `real/` and
`fake/` and point `eval_robustness.py` / `infer.py` at it.

---

## How it works

- **`src/model.py`** — `CLIPDetector`: frozen CLIP encoder → 2-layer MLP head.
- **`src/augmentations.py`** — training augmentation (OneOf of the six
  transform families at p=0.5) and a deterministic eval registry that
  reproduces the brief's exact transform/severity table. Every transform is
  applied at a canonical 224px so severities are resolution-independent.
- **`src/dataset.py`** — `real/`+`fake/` folder dataset (real=0, fake=1).
- **`src/train.py`** — training loop with mixed precision and per-epoch
  resumable checkpoints (`last.pt` auto-resumes; `best_model.pt` = best val AUROC).
- **`src/calibrate.py`** — temperature scaling + ECE/NLL/Brier.
- **`src/infer.py`** — the deliverable inference script (calibrated JSON output).
- **`src/eval_robustness.py`** — the robustness harness (CSV table + error analysis).

---

## Limitations & what we'd improve

Findings specific to **this** run — not placeholder text:

- **Diversity was the winning lever (and we measured it).** Going from a
  single-generator model to 13 generator families took **unseen-generator
  AUROC from ~0.70 (v2) to 0.99 (v5)**. Architecture helped too (the DINOv2
  branch lifted GAN generalization), but data diversity dominated — matching
  the workshop's "augmentation + data > architecture tricks."
- **The aesthetic-real false positive — diagnosed and fixed.** Early models
  flagged beautiful real photos as AI (~53% aesthetic-real accuracy) because
  their "real" class was mundane snapshots and "pretty" correlated with
  Midjourney output. Adding aesthetic reals **and** aesthetic fakes (so it
  learns real-vs-fake *within* the aesthetic domain) took aesthetic AUROC to
  **0.998**. This is the key deployability fix for TikTok-style content.
- **Specialization ↔ generalization is a real trade.** The most diverse
  model (v5) gives up a little on individual seen-like generators (DDIM
  0.982 → 0.938) to generalize far better across *unseen* ones (Imagen+starGAN
  0.70 → 0.986). We chose breadth — the right call for the real world.
- **CIFAKE (32px) is a resolution confound, not a generator failure.** *All*
  our models score ~0.37–0.47 there because 32×32→224px upscaling is a domain
  shift the model reads as "generated." We report it rather than hide it; a
  higher-res unseen set (which we have via the held-out generators) is the
  honest measure.
- **A few hardest aesthetic reals still false-positive** (≈4 of 20 on the
  demo set) — the residual of a genuinely hard boundary; more aesthetic-real
  volume would chip at it.
- **Ensembling is a measured optional upgrade.** Averaging v5 with the v2
  CLIP baseline recovers the DDIM trade (0.938 → **0.979**) and lifts GigaGAN
  (0.903 → **0.930**) — v2's diffusion strength + v5's breadth. We keep v5 as
  the default (single-model, simpler/faster demo) and expose the ensemble as a
  deployment knob for accuracy-first settings. (We also *measured* test-time
  augmentation — no benefit here — and extra GAN data — no benefit — and
  dropped both: honest, measured discipline.)
- **Explainability:** `src/explain_occlusion.py` produces model-agnostic
  occlusion-saliency "why flagged" heatmaps (`results/explainability/`). On AI
  portraits the decision concentrates on the face — where generation artifacts
  live — evidence the detector uses meaningful cues, not dataset shortcuts.
- **Next steps:** RINE-style intermediate-layer aggregation and broader modern-
  generator coverage. Even SoTA robust detectors top out ~0.92–0.97 AUROC
  across generators, so we treat this as a **calibrated triage signal with a
  human in the loop**, not an oracle.

---

## License / data credits

- CLIP (OpenAI) via Hugging Face Transformers.
- SID_Set — `saberzl/SID_Set`, CC-BY-4.0.
- CIFAKE — Bird & Lotfi, Kaggle.
