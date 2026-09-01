# Project Description — Robust AI-Generated Image Detector

> **Demo video (YouTube, public):** ⛳ _paste link here_
> **Public code repository:** ⛳ _paste GitHub link here_

**One line:** a detector that tells real photos from AI-generated ones and *stays
accurate after the real-world transforms images actually undergo when re-shared* —
JPEG, resize, blur, noise, color shifts, crops, screenshots, and thumbnails.

---

## How our solution addresses the problem statement

The challenge is not accuracy on clean generator output — it's **robustness**
(accuracy survives real-world post-processing) and **generalization** (works on
generators never seen in training). We address both directly:

1. **A two-branch frozen-backbone detector, deliberately not a frequency/artifact
   detector.** Most fake-image detectors key on high-frequency generation
   fingerprints (spectral peaks, PRNU residuals, DCT artifacts) — *exactly what JPEG,
   blur, resize, and noise destroy.* We reject that. Instead we fuse two complementary,
   robust frozen backbones — **CLIP ViT-L/14** (semantics) + **DINOv2 ViT-L**
   (self-supervised structure/texture) — and train only a small calibrated MLP head on
   the concatenated features. Both degrade gracefully under compression. It is **one
   model, one forward pass** (the two "branches" are backbones fused internally).

2. **Train on the exact transforms we're evaluated on — plus compound chains and
   extreme low-res.** The highest-ROI robustness lever is showing the model degraded
   images in training. We go further than single transforms: we train on *chained*
   real-world degradations (double-JPEG, resize→re-JPEG, screenshot→downscale) and
   *extreme low-resolution* (down to ~11–48px thumbnails).

3. **Diversity for generalization.** We train across **13 generator families** and
   hold generators out to *measure* cross-generator generalization. Broadening the
   training set from one generator to thirteen took unseen-generator AUROC from
   **0.70 → 0.99**.

4. **Ensemble distillation into one fast model (our key innovation).** Our two best
   models were complementary; we distilled their ensemble into a single hybrid via
   online distillation on augmented images — and the **student beat both teachers**
   while staying a single, fast checkpoint.

5. **Calibrated confidence, not a raw sigmoid.** Temperature scaling + Expected
   Calibration Error (**ECE 0.0005**) make `pred` a trustworthy probability that
   drives a 3-band human-in-the-loop triage pipeline.

**Result (shipped model, v7):** scored metric **0.9867** (0.5·AUROC_clean +
0.5·AUROC_robust); on a completely **independent** third-party test set (AI-vs-Real,
never used in training) it scores **clean AUROC 0.9986** and 0.99–0.998 under every
transform — proof it generalizes, not memorizes. 733M params (< 2B limit), 55 img/s
on one RTX 3060. Full numbers in `3_ROBUSTNESS_SUMMARY.md`.

---

## Development tools used
- **VS Code** — primary editor.
- **Claude Code** — agentic pair-programming / build automation.
- **Git** — version control.
- **NVIDIA RTX 3060 (12 GB)** — all training/eval on a single consumer GPU
  (CUDA 12.8), mixed precision.
- Command-line Python scripts (no notebook dependency); an optional **Gradio**
  dashboard (`app.py`) for the live demo.

## Models / APIs used
- **OpenAI CLIP ViT-L/14** (frozen) — semantic features, via Hugging Face Transformers.
- **Meta DINOv2 ViT-L** (frozen) — self-supervised structure/texture features, via HF.
- Custom trained **MLP head** (~0.9M params) on the fused 1792-d features.
- No external/paid inference APIs — the detector runs fully locally.

## Libraries & frameworks used
- **PyTorch 2.9** (CUDA 12.8) + **torch.amp** (mixed precision) — model & training.
- **Hugging Face `transformers`** — CLIP/DINOv2 backbones; **`datasets`** — streaming.
- **albumentations** + **OpenCV** — the transform/augmentation pipelines.
- **scikit-learn** — AUROC / AP / F1 / accuracy / ECE metrics.
- **pandas**, **numpy**, **pyarrow** — data handling; **matplotlib** — figures.
- **Pillow** — image I/O; **gradio** — the optional demo dashboard.

## Datasets & assets used
- **SID_Set** (`saberzl/SID_Set`, HF, CC-BY-4.0) — FLUX fakes + OpenImages reals.
- **WildFake** (ModelScope) — 13 generator families (DDPM, DDIM, styleGAN, BigGAN,
  DF-GAN, GigaGAN, ADM, VQDM, SD, GALIP, Midjourney, DALL·E-2, …) + COCO/FFHQ reals;
  **streamed / partially extracted** so we never downloaded the ~2 TB.
- **Unsplash Lite** — aesthetic real photography (fixes the "pretty real = fake" bug).
- **CIFAKE** (Kaggle) — smoke-test + low-resolution OOD benchmark.
- **AI-vs-Real** (`Parveshiiii/AI-vs-Real`, HF) — **test-only** independent validation
  (deduplicated against train).
- **Held-out generators** (DDIM, GigaGAN, Imagen, starGAN, Midjourney-Typical) measure
  true cross-generator generalization.
- **Compliance:** all backbones public; all datasets public/licensed; we **never train
  on** COCO val2017 or the DALL·E-Advanced demo set (report-only). Trained head +
  hyperparameters + eval code are open-sourced in the repo.

---

## Team
⛳ _Fill in team members and contributions (e.g. model & training pipeline; data
preparation & robustness harness; evaluation, calibration & write-up)._
