# Error Analysis Note — Model V7

Representative false positives / false negatives and the trade-offs they reveal.
Figure: `figures/fig_error_analysis.png` (real photos flagged AI on top; AI images
flagged real on bottom). Raw errors with paths + confidences:
[`data/error_analysis.json`](data/error_analysis.json).

## Error rates (threshold 0.5, calibrated)

| Test set | N | Errors | False positives (real→AI) | False negatives (AI→real) |
|---|---|---|---|---|
| In-distribution (FLUX) | 964 | 34 (3.5%) | 25 (5.4% of reals) | 9 (1.8% of fakes) |
| **AI-vs-Real (independent)** | 5000 | 100 (2.0%) | 86 (3.4% of reals) | **14 (0.56% of fakes)** |

## The dominant pattern: the model errs toward *flagging*, not *missing*
On both sets, false positives outnumber false negatives ~3–6×. The detector rarely
lets an AI image through (**FN just 0.56%** on the independent set ≈ 99.4% recall on
fakes); when it errs, it over-flags a hard *real* photo. **For moderation/triage this
is the safer bias** — a flagged real goes to a human reviewer, whereas a missed fake
spreads unchecked — and over-flagged reals mostly land near the boundary, in the
"uncertain" band, not "confident AI."

## Representative false positives — real photos flagged as AI
- **Hyper-aesthetic / heavily-edited reals:** professional portraits and stylized
  shots with smooth skin, shallow depth-of-field, strong color grading — the visual
  signature of Midjourney-style output. Genuinely ambiguous; humans hesitate too.
- **Extreme low-resolution reals:** at ~22–34px a real photo upscaled to 224px is
  mostly interpolation blur, which reads as "synthetic smoothness." This is the
  measured frontier (extreme-downscale AUROC ~0.92) and the main FP source under heavy
  degradation.

## Representative false negatives — AI images flagged as real
- **Photorealistic, low-artifact generations** (modern diffusion of ordinary scenes)
  with no warped hands, garbled text, or impossible lighting — the fakes that are
  simply very good. Few (≤2%), and exactly the cases the whole field still finds hard.

## Trade-offs in the approach (named explicitly)
- **Robustness ↔ clean accuracy:** heavy + compound augmentation costs ~0.005 clean
  AUROC but buys large gains under real-world degradation — worth it (robust is 50% of
  the score) and for deployment.
- **Generalization ↔ specialization:** breadth (13 generators) sacrifices a little on
  seen-like generators (DDIM 0.982→0.928) to generalize far better on unseen ones
  (Imagen+starGAN 0.70→0.996) — the right call for a real feed.
- **Precision ↔ recall (a policy knob, not a limit):** because the output is
  calibrated, moving the flag threshold 0.50→0.65 trades a little recall for higher
  precision on real creators, mapping directly to real risk
  (`figures/fig_operating_point.png`).

## What we'd improve with more time
- Close the extreme-low-res (≤~34px) tail further via more low-res-specialized training
  and a lightweight super-resolution front-end.
- Add more hyper-aesthetic real photography to shave the residual "too-pretty-to-be-real"
  false positives.
