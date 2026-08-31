# Robust AI-Generated Image Detector — Project Description

## The problem
AI-generated images are now indistinguishable from photographs to most people
(a 2023 benchmark put human misclassification at **38.7%**), fueling
misinformation, impersonation, and fraud. But the images that actually spread
online are almost never pristine generator output — they've been screenshotted,
re-compressed by a messaging app, downscaled by a CDN, filtered, and re-uploaded
many times. **A detector that only works on clean generator output is useless in
the real world.** The hard part is therefore not clean-image accuracy — it's
**staying accurate after real-world transformations**: JPEG re-compression, blur,
resize, noise, color shifts, and cropping.

## How our solution addresses the problem
We built a detector designed around robustness and generalization from the start,
and validated it on a **completely independent dataset it never saw in training**.

**1. A two-branch frozen-backbone detector (CLIP + DINOv2) — deliberately not a
frequency/artifact detector.** Most fake-image detectors key on high-frequency
generation fingerprints (spectral peaks, PRNU residuals, DCT artifacts) — *exactly
what JPEG, blur, resize, and noise destroy.* We reject that for a reason. Instead
we fuse two complementary, robust frozen backbones — **CLIP ViT-L/14** (semantics)
and **DINOv2 ViT-L** (self-supervised structure/texture) — and train only a small
MLP head on the concatenated features. Both degrade gracefully under compression;
DINOv2 catches generator artifacts CLIP misses (our biggest GAN gain). This is
one model, one forward pass — the two "branches" are backbones fused internally.

**2. Train on the exact transforms we're evaluated on — and on compound chains.**
The highest-ROI robustness lever is showing the model degraded images during
training. We go further than single transforms: we train on **chained** real-world
degradations (double-JPEG, resize→re-JPEG, screenshot→downscale) and **extreme
low-resolution** (down to ~11–48px thumbnails), because that's what actually
happens to a re-shared image.

**3. Diversity for generalization.** We train across **13 generator families** and
5 real sources, and **hold generators out** to measure true cross-generator
generalization. Broadening the training set from one generator to thirteen took
unseen-generator AUROC from **0.70 → 0.99**.

**4. Ensemble distillation into a single fast model.** Our best individual models
were complementary (one strong on the scored metric + cross-gen, one on extreme
low-res). We **distilled the ensemble into one hybrid** via online distillation on
strong-augmented images — and the student **beat both teachers** while staying a
single, fast `.pt` (1× inference instead of 2×). This is the core innovation.

**5. Calibrated confidence, not a raw sigmoid.** We fit **temperature scaling** on
a held-out split and report **Expected Calibration Error (ECE 0.0005)**, so `pred`
is a trustworthy probability that drives a 3-band human-in-the-loop triage pipeline.

## Results (shipped model, v7 — distilled CLIP+DINOv2 hybrid)
**Scored metric (0.5·AUROC_clean + 0.5·AUROC_robust):** **0.9867** (best across our
v2→v7 arc), with the best single-transform robustness (**0.9805**) and best
calibration (**ECE 0.0005**).

**Independent validation (the credibility check):** on **AI-vs-Real**
(a third-party Hugging Face dataset, 2,500 real + 2,500 AI, **never used in
training**, deduplicated against our train set), v7 scores **clean AUROC 0.9986**
and **0.99–0.998 under every single transform and compound chain** — proof the
approach generalizes, not just memorizes.

**In-distribution:** clean AUROC **0.9928**; every single transform severity stays
**0.95–0.99** — graceful, not a cliff.

**Cross-generator (held-out — never trained on):**

| Held-out test | v2 (less diverse) | **v7 (shipped)** |
|---|---|---|
| DDIM (unseen diffusion) | 0.982 | 0.928 |
| GigaGAN (unseen GAN) | 0.876 | 0.901 |
| Imagen + starGAN (unseen) | 0.697 | **0.996** |
| Aesthetic (Unsplash / Midjourney) | 0.767 | **0.999** |
| Official demo (DALL·E-Advanced / COCO val2017) | 0.990 | **0.997** |

**Deployment:** 733M params (<2B), single RTX 3060, **55 img/s** batched. As a
calibrated triage signal, only **1.25%** of unseen-generator traffic lands in the
"uncertain" band needing human review; the rest is confidently auto-decided.

## Honest limitations (measured, not hidden)
- **Specialization ↔ generalization trade:** breadth + robustness cost a little on
  seen-like generators (DDIM 0.982→0.928) to generalize far better on *unseen* ones.
- **Extreme low-res tail:** on ~22px thumbnails AUROC dips to ~0.92  — the genuine frontier; we improved it materially over earlier
  versions and quantify exactly where it stands.
- **CIFAKE (32px):** all our models score low — a *resolution* domain shift, not a
  generator failure; we report it rather than hide it.
- We treat the detector as a **calibrated triage signal with a human in the loop**,
  not an oracle.

## Built with
- **Languages / runtime:** Python 3.12, PyTorch 2.9 (CUDA 12.8), mixed precision.
- **Models / APIs:** OpenAI **CLIP ViT-L/14** + Meta **DINOv2 ViT-L** (both frozen,
  public), via Hugging Face Transformers.
- **Libraries / frameworks:** Hugging Face `transformers` + `datasets`,
  `albumentations` + OpenCV (transform pipelines), `scikit-learn` (AUROC/AP/F1/ECE),
  `pandas`, `matplotlib`, `pyarrow`, `gradio` (demo dashboard).
- **Datasets (public/licensed):** **SID_Set** (`saberzl/SID_Set`, CC-BY-4.0),
  **WildFake** (ModelScope — 13 generator families, streamed/partial-extracted so we
  never pulled the ~2 TB), **Unsplash Lite** (aesthetic reals), **CIFAKE** (Kaggle),
  and **AI-vs-Real** (`Parveshiiii/AI-vs-Real`, **test-only**, independent validation).
  Held-out generators (DDIM, GigaGAN, Imagen, starGAN, Midjourney-Typical) measure
  true generalization. We never train on COCO val2017 / DALL·E-Advanced (report-only).
- **Hardware:** single NVIDIA RTX 3060, 12 GB.

## Why it matters
Per-image **calibrated** confidence plus **graceful degradation** under the
transforms images actually undergo when re-shared makes this usable as a platform
**triage signal** — auto-pass the confident reals, flag the confident fakes, route
the ~1% uncertain to a human — rather than a brittle benchmark number. By
quantifying the cross-generator gap and the low-res frontier honestly, we show
exactly where such a detector can and cannot be trusted.
