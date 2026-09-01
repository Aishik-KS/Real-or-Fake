# Robust AI-Generated Image Detector 

Detects AI-generated images and **stays accurate after real-world transforms**
(JPEG, resize, blur, noise, color shift, crop, screenshots, thumbnails). Two frozen
backbones (CLIP ViT-L/14 + DINOv2 ViT-L) → a small calibrated head, distilled from an
ensemble into one fast model. real = 0, AIGC = 1.

## Project overview
- **Model:** CLIP ViT-L/14 + DINOv2 ViT-L (both frozen) fused → MLP head. ~733M params
  total (~0.9M trainable), **under the 2B limit**. Single `.pt`, single forward pass.
  The head is **distilled from a v5+v6 ensemble** — the student beat both teachers.
- **Robustness lever:** train on the six evaluated transforms *plus* compound chains
  and extreme low-res.
- **Generalization lever:** train on 13 generator families; hold generators out to
  *measure* it (unseen-gen AUROC 0.70 → 0.99).
- **Calibrated output:** temperature scaling; `pred` is a trustworthy P(AIGC) (ECE 0.0005).
- **Headline:** scored metric (0.5·clean + 0.5·robust) = **0.9867**; independent
  AI-vs-Real test clean AUROC **0.9986**. Full tables: `3_ROBUSTNESS_SUMMARY.md`.


## Project files

```text
Model/
└── best_model.pt

Test Images/
└── Images to evaluate

RunModel.py
README.md
```

## Requirements

- Python 3.10 or newer
- Approximately 5 GB of free disk space
- 16 GB RAM recommended

## Setup in VS Code

1. Open VS Code.
2. Select **File → Open Folder**.
3. Open the folder containing `RunModel.py`.
4. Select **Terminal → New Terminal**.
5. Check that Python is installed:

   ```powershell
   python --version
   ```

6. Install the required packages:

   ```powershell
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   python -m pip install transformers pillow numpy opencv-python-headless
   python -m venv .venv
   pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))

   ```

## Download the model

From the project root, install the Hugging Face CLI and download the model directly into the `Model` folder:

```powershell
python -m pip install -U "huggingface_hub[cli]"
hf download HauntedFrost/TikTok --repo-type model --local-dir "Model"
```

This places the downloaded model files in `Model/`, so the checkpoint is saved where the project expects it:

```text
Model/best_model.pt
```

Once the download finishes, you can run the model immediately.

## Run the model

Place the model checkpoint here:

```text
Model/best_model.pt
```

Place the images you want to check inside `Test Images`, then run:

```powershell
python RunModel.py --image_dir "Test Images" --output predictions.json
```

You can also use the shorter command:

```powershell
python RunModel.py
```

The model may take a few minutes to run on CPU.

## Output

The results are saved in `predictions.json`:

```json
[
  {
    "image_path": "Test Images/example.png",
    "pred": 0.982451
  }
]
```

Score meaning:

- Near `1.0`: more likely AI-generated
- Near `0.0`: more likely real
- Near `0.5`: uncertain

## Choose an image folder

The images do not have to be copied into `Test Images`. That folder is only
provided as the default example.

You can choose either of these methods.

### Option 1: Use a different folder

Keep the images in any folder and provide its path using `--image_dir`:

```powershell
python RunModel.py --image_dir "C:\path\to\images" --output results.json
```

### Option 2: Use `Test Images`

Copy the images into `Test Images` and run:

```powershell
python RunModel.py
```

Using a separate folder is recommended because it keeps the supplied test
images and your own images separate.

Supported image formats are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.

## Limitations & what we'd improve given more time
- **Extreme low-res tail:** on ~22px thumbnails AUROC dips to ~0.92 (well above chance)
  — the genuine frontier. We improved it materially over earlier versions (0.58 → 0.83
  in-dist) via extreme-downscale training; more of it (and a light super-resolution
  front-end) is the next step.
- **Specialization ↔ generalization trade:** breadth (13 generators) costs a little on
  seen-like generators (DDIM 0.982 → 0.928) to generalize far better on unseen ones
  (Imagen+starGAN 0.70 → 0.996) — the right trade for a real feed, but a real one.
- **CIFAKE (32px):** all our models score low — a *resolution* domain shift, not a
  generator failure; reported, not hidden.
- **Residual false positives** on a few hyper-aesthetic real photos (the "too pretty to
  be real" boundary); more aesthetic-real training data would chip at it.
- We treat the detector as a **calibrated triage signal with a human in the loop**, not
  an oracle. Given more time: broader modern-generator coverage and RINE-style
  intermediate-layer aggregation.

  
## License / credits
- CLIP (OpenAI) and DINOv2 (Meta) via Hugging Face Transformers.
- SID_Set (CC-BY-4.0); WildFake (ModelScope); CIFAKE (Kaggle); Unsplash Lite;
  AI-vs-Real (`Parveshiiii/AI-vs-Real`, test-only).
- Repo code + trained weights + hyperparameters open-sourced (add an OSS license, e.g.
  MIT/Apache-2.0, before publishing).
