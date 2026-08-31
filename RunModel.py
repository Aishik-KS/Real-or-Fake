"""Run the trained AIGC detector on every image in a directory.

The submission intentionally consists of only this script plus the ``Model``
and ``Test Images`` directories. Install the runtime packages in the Python
environment used to run the script:

    pip install torch transformers pillow numpy opencv-python-headless

Example:

    python RunModel.py --image_dir "Test Images" --output predictions.json

The output is a JSON array. Each entry has the original ``image_path`` and a
``pred`` value between 0 and 1, where a higher value means the image is more
likely to be AIGC-generated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Thresholds used only for the human-readable verdict printed to the
# console. The JSON output always contains the raw probability regardless
# of these thresholds.
FAKE_THRESHOLD = 0.5
UNCERTAIN_BAND = 0.15

# This checkpoint's architecture is fixed (see make_model_class below), so
# the label shown in the console banner is a constant rather than something
# read out of the checkpoint's training args.
ARCHITECTURE_LABEL = "openai/clip-vit-large-patch14 + facebook/dinov2-large"


def parse_args() -> argparse.Namespace:
    """Parse command-line paths and device selection."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Predict the likelihood that each image is AIGC-generated."
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=script_dir / "Test Images",
        help='Directory to scan recursively (default: "Test Images" beside this script).',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.json"),
        help="Output JSON path (default: predictions.json).",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Optional .pt checkpoint path (default: the only .pt file in Model).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: CUDA when available, otherwise CPU).",
    )
    return parser.parse_args()


def import_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Import heavy optional dependencies and report a useful install command."""
    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn as nn
        from PIL import Image
        from transformers import CLIPConfig, CLIPModel, Dinov2Config, Dinov2Model
    except ImportError as exc:
        raise RuntimeError(
            f"Missing Python dependency: {exc.name!r}. Install dependencies with:\n"
            "    pip install torch transformers pillow numpy opencv-python-headless"
        ) from exc

    transformer_types = (CLIPConfig, CLIPModel, Dinov2Config, Dinov2Model)
    return cv2, np, torch, nn, Image, transformer_types


def find_checkpoint(model_path: Path | None, script_dir: Path) -> Path:
    """Resolve an explicit checkpoint or require exactly one Model/*.pt file."""
    if model_path is not None:
        checkpoint = model_path.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint}")
        return checkpoint

    model_dir = script_dir / "Model"
    checkpoints = sorted(model_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pt model checkpoint found in: {model_dir}")
    if len(checkpoints) > 1:
        names = ", ".join(path.name for path in checkpoints)
        raise RuntimeError(
            f"Multiple checkpoints found in {model_dir}: {names}. "
            "Select one with --model_path."
        )
    return checkpoints[0]


def find_images(image_dir: Path) -> tuple[Path, list[Path]]:
    """Return all supported images below the supplied directory."""
    root = image_dir.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {root}")

    images = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.as_posix().lower(),
    )
    if not images:
        supported = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise RuntimeError(f"No supported images found in {root} ({supported})")
    return root, images


def make_model_class(torch: Any, nn: Any, transformer_types: tuple[Any, Any, Any, Any]) -> Any:
    """Build the exact CLIP-L/14 + DINOv2-Large checkpoint architecture."""
    CLIPConfig, CLIPModel, Dinov2Config, Dinov2Model = transformer_types

    class AIGCDetector(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()

            # Configurations are embedded so loading the complete checkpoint does
            # not download pretrained backbones or require an internet connection.
            clip_config = CLIPConfig(
                projection_dim=768,
                text_config={
                    "vocab_size": 49408,
                    "hidden_size": 768,
                    "intermediate_size": 3072,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 12,
                    "max_position_embeddings": 77,
                    "hidden_act": "quick_gelu",
                    "layer_norm_eps": 1e-5,
                    "attention_dropout": 0.0,
                    "bos_token_id": 49406,
                    "eos_token_id": 49407,
                    "pad_token_id": 1,
                    "projection_dim": 768,
                },
                vision_config={
                    "hidden_size": 1024,
                    "intermediate_size": 4096,
                    "num_hidden_layers": 24,
                    "num_attention_heads": 16,
                    "num_channels": 3,
                    "image_size": 224,
                    "patch_size": 14,
                    "hidden_act": "quick_gelu",
                    "layer_norm_eps": 1e-5,
                    "attention_dropout": 0.0,
                    "projection_dim": 768,
                },
            )
            dino_config = Dinov2Config(
                hidden_size=1024,
                num_hidden_layers=24,
                num_attention_heads=16,
                image_size=518,
                patch_size=14,
                num_channels=3,
                hidden_act="gelu",
                mlp_ratio=4,
                qkv_bias=True,
                layer_norm_eps=1e-6,
                layerscale_value=1.0,
                use_mask_token=True,
                use_swiglu_ffn=False,
                apply_layernorm=True,
            )

            self.clip = CLIPModel(clip_config)
            self.dino = Dinov2Model(dino_config)

            # The buffers reproduce the normalization conversion used at training.
            self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
            self.register_buffer("clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))
            self.register_buffer("in_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("in_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

            self.head = nn.Sequential(
                nn.Linear(768 + 1024, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, pixel_values: Any) -> Any:
            clip_features = self.clip.get_image_features(pixel_values=pixel_values)
            if hasattr(clip_features, "pooler_output"):
                clip_features = clip_features.pooler_output

            rgb = pixel_values * self.clip_std + self.clip_mean
            dino_pixels = (rgb - self.in_mean) / self.in_std
            dino_output = self.dino(pixel_values=dino_pixels)
            dino_features = getattr(dino_output, "pooler_output", None)
            if dino_features is None:
                dino_features = dino_output.last_hidden_state[:, 0]

            features = torch.cat((clip_features, dino_features), dim=-1)
            return self.head(features).squeeze(-1)

    return AIGCDetector


def load_checkpoint(torch: Any, path: Path, device: Any) -> dict[str, Any]:
    """Load the trusted project checkpoint across supported PyTorch versions."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        checkpoint = torch.load(path, map_location=device)

    required = {"model_state_dict", "args"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"Checkpoint is missing required keys: {sorted(missing)}")
    return checkpoint


def preprocess_image(cv2: Any, np: Any, torch: Any, Image: Any, path: Path) -> Any:
    """Load, resize, and CLIP-normalize one RGB image."""
    with Image.open(path) as opened:
        array = np.asarray(opened.convert("RGB"))

    # This matches the training/evaluation pipeline exactly. In particular,
    # OpenCV INTER_LINEAR is retained because changing resize implementations
    # can materially change this model's score for high-frequency images.
    array = cv2.resize(array, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    array = array.astype(np.float32) / 255.0

    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(CLIP_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=tensor.dtype).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def output_image_path(path: Path, original_argument: Path, root: Path) -> str:
    """Keep paths readable while retaining the input directory in each record."""
    try:
        relative = path.relative_to(root)
        base = original_argument if not original_argument.is_absolute() else root
        return (base / relative).as_posix()
    except ValueError:
        return path.as_posix()


def verdict_for(probability: float) -> str:
    """Turn a raw probability into a REAL / AI-GENERATED / UNCERTAIN verdict.

    This is purely for the console table — the JSON output always records
    the raw probability, regardless of these thresholds.
    """
    if abs(probability - 0.5) <= UNCERTAIN_BAND:
        return "UNCERTAIN"
    if probability > FAKE_THRESHOLD:
        return "AI-GENERATED"
    return "REAL"


def print_banner() -> None:
    print()
    print("=" * 72)
    print("                 AIGC IMAGE DETECTOR")
    print("                 REAL vs AI-GENERATED")
    print("=" * 72)
    print()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    try:
        cv2, np, torch, nn, Image, transformer_types = import_dependencies()
        checkpoint_path = find_checkpoint(args.model_path, script_dir)
        image_root, image_paths = find_images(args.image_dir)

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device_name == "auto":
            device_name = "cpu"
        device = torch.device(device_name)

        print_banner()
        print("  Loading model...")

        checkpoint = load_checkpoint(torch, checkpoint_path, device)
        training_args = checkpoint["args"]
        if not isinstance(training_args, dict):
            training_args = vars(training_args)
        if not training_args.get("hybrid", False):
            raise RuntimeError("This script expects the hybrid CLIP + DINOv2 checkpoint.")

        model_class = make_model_class(torch, nn, transformer_types)
        model = model_class(hidden_dim=int(training_args.get("hidden_dim", 512))).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        temperature = float(checkpoint.get("temperature", 1.0))
        if temperature <= 0:
            raise RuntimeError(f"Checkpoint contains invalid temperature: {temperature}")

        print()
        print("  MODEL INFORMATION")
        print("  " + "-" * 68)
        print(f"  Checkpoint   : {checkpoint_path.name}")
        print(f"  Architecture : {ARCHITECTURE_LABEL}")
        print(f"  Device       : {device.type.upper()}")
        print(f"  Images       : {len(image_paths)}")
        print()
        print("=" * 72)
        print()
        print("  RUNNING INFERENCE")
        print()
        print(f"  {'IMAGE':<30}{'PREDICTION':<18}{'AI SCORE':>10}")
        print("  " + "-" * 62)

        predictions: list[dict[str, Any]] = []
        n_real = n_fake = n_uncertain = 0
        with torch.inference_mode():
            for image_path in image_paths:
                try:
                    batch = preprocess_image(cv2, np, torch, Image, image_path).to(device)
                    logit = model(batch)
                    probability = float(torch.sigmoid(logit.float() / temperature).item())
                except Exception as exc:
                    raise RuntimeError(f"Could not process image {image_path}: {exc}") from exc

                verdict = verdict_for(probability)
                if verdict == "REAL":
                    n_real += 1
                elif verdict == "AI-GENERATED":
                    n_fake += 1
                else:
                    n_uncertain += 1

                predictions.append(
                    {
                        "image_path": output_image_path(image_path, args.image_dir, image_root),
                        "pred": round(probability, 6),
                    }
                )
                print(f"  {image_path.name:<30}{verdict:<18}{probability * 100:>8.1f}%")

        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(predictions, output_file, indent=2, ensure_ascii=False)
            output_file.write("\n")

        print()
        print("=" * 72)
        print("                         RESULTS")
        print("=" * 72)
        print()
        print(f"  Total images      : {len(image_paths)}")
        print(f"  Real              : {n_real}")
        print(f"  AI-generated      : {n_fake}")
        print(f"  Uncertain         : {n_uncertain}")
        print()
        print("=" * 72)
        print()
        print("  AI SCORE = calibrated probability that the image is AI-generated.")
        print("  Scores near 50% are treated as uncertain.")
        print()
        print(f"  Wrote {len(predictions)} prediction(s) to: {output_path}")
        print()
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())