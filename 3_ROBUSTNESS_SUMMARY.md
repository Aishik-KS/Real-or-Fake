# Robustness Evaluation Summary — Model V7

**Final Score = 0.5·AUROC_clean + 0.5·AUROC_robust = `0.9867`**
(clean 0.9928 · mean over all transform severities 0.9805). Metric: **ROC-AUC**
(threshold-free), with accuracy for reference. real = 0, AIGC = 1. Machine-readable
CSVs in `data/`; visuals in `figures/`.

## Clean vs. transformed (the core table)
Held-out FLUX test set; each transform at the brief's exact severities. **AUROC stays
0.96–0.99 under every condition** — graceful degradation, not a cliff.

| Condition | Acc | AUROC |
|---|---|---|
| **Clean** | 0.965 | **0.9928** |
| JPEG q90 / q70 / q50 / q30 | .94 / .93 / .93 / .92 | .985 / .979 / .978 / **.976** |
| Gaussian blur σ0.5 / 1.0 / 2.0 | .95 / .95 / .91 | .989 / .987 / .980 |
| Resize 0.5× / 0.25× (down→up) | .94 / .90 | .986 / **.966** |
| Gaussian noise σ0.02 / 0.05 / 0.10 | .95 / .93 / .90 | .987 / .979 / **.961** |
| Color jitter ±20% | 0.955 | 0.990 |
| Center crop 80% | 0.932 | 0.987 |

Hardest single transforms: noise σ0.10 (.961), resize 0.25× (.966), JPEG q30 (.976).
CSV: [`data/robustness_table.csv`](data/robustness_table.csv). Visual:
`figures/fig_robustness_curve.png`, `figures/robustness_visual.png`.

## Compound / real-world chains (our differentiator)
Re-shared images stack degradations. We built a compound test the single-transform
table doesn't stress; earlier models collapsed on extreme thumbnails, v7 holds up:

| Compound condition | v5 | **v7** |
|---|---|---|
| double-JPEG (q70→q50) | 0.968 | 0.971 |
| screenshot chain (resize+noise+JPEG) | 0.853 | 0.905 |
| extreme downscale 0.10× (~22px) | 0.583 | **0.831** |
| thumbnail chain (down0.15+JPEG40) | 0.527 | **0.796** |

Visual: `figures/fig_compound_robustness.png`. CSV: [`data/compound_robustness_v7.csv`](data/compound_robustness_v7.csv).

## Independent validation — AI-vs-Real (never seen in training)
Third-party HF set (2,500 real + 2,500 AI), deduplicated vs train — the real
generalization proof:

| Condition | AUROC |
|---|---|
| **Clean** | **0.9986** |
| every single transform (JPEG/blur/resize/noise/color/crop) | 0.993 – 0.998 |
| compound chains | 0.99 |

CSV: [`data/ai_vs_real_robustness_table.csv`](data/ai_vs_real_robustness_table.csv).

## Cross-generator generalization (held-out generators)
Diversity is the driver — unseen-gen AUROC 0.70 → 0.99 as we went 1 → 13 generators:

| Held-out generator | v2 | v5 | **v7** |
|---|---|---|---|
| DDIM (diffusion) | 0.982 | 0.938 | 0.928 |
| GigaGAN (GAN) | 0.876 | 0.903 | 0.901 |
| Imagen + starGAN | 0.697 | 0.986 | **0.996** |
| Aesthetic (Unsplash/Midjourney) | 0.767 | 0.998 | **0.999** |
| Official demo (DALL·E-Adv / COCO val2017) | 0.990 | 0.997 | 0.997 |

CSV: [`data/crossgen_summary.csv`](data/crossgen_summary.csv). Visuals:
`figures/fig_ablation_heatmap.png`, `figures/fig_diversity_generalization.png`.

## Calibration
Temperature-scaled: **ECE 0.0005** (in-dist). As a triage signal, only **1.25%** of
unseen-generator traffic lands in the "uncertain" band; precision 0.98 at the 0.65 flag
threshold. `figures/fig_operating_point.png`.
