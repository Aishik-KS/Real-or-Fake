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
import gc
import json
import sys
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 224
REAL_THRESHOLD = 0.35
AI_THRESHOLD = 0.65
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def prediction_label(probability: float) -> str:
    """Convert an AI confidence score into the label used by the CLI and web app."""
    if probability >= AI_THRESHOLD:
        return "AI-GENERATED"
    if probability <= REAL_THRESHOLD:
        return "REAL"
    return "UNCERTAIN"


def print_information(title: str, fields: list[tuple[str, str]]) -> None:
    """Print a compact, aligned command-line information section."""
    print(f"\n{title}")
    print("-" * len(title))
    label_width = max(len(label) for label, _ in fields)
    for label, value in fields:
        print(f"{label:<{label_width}} : {value}")


def shorten_name(name: str, width: int) -> str:
    """Shorten very long filenames without breaking the table alignment."""
    if len(name) <= width:
        return name
    if width <= 3:
        return name[:width]
    return f"{name[:width - 3]}..."


def print_prediction_header(name_width: int) -> None:
    """Print the heading for the live inference results table."""
    print("\nRUNNING INFERENCE")
    print("-" * (name_width + 32))
    print(f"{'IMAGE':<{name_width}}  {'PREDICTION':<14}  {'AI SCORE':>10}")
    print("-" * (name_width + 32))


def print_summary(predictions: list[dict[str, Any]], output_path: Path) -> None:
    """Print final prediction counts and the saved JSON location."""
    ai_count = sum(item["pred"] >= AI_THRESHOLD for item in predictions)
    real_count = sum(item["pred"] <= REAL_THRESHOLD for item in predictions)
    uncertain_count = len(predictions) - ai_count - real_count
    print_information(
        "INFERENCE COMPLETE",
        [
            ("Images processed", str(len(predictions))),
            ("AI-generated", str(ai_count)),
            ("Real", str(real_count)),
            ("Uncertain", str(uncertain_count)),
            ("Predictions JSON", str(output_path)),
        ],
    )


def load_detector(
    model_path: Path | None = None,
    device_choice: str = "auto",
    script_dir: Path | None = None,
) -> dict[str, Any]:
    """Load and return a reusable detector runtime for CLI or frontend use."""
    base_dir = script_dir or Path(__file__).resolve().parent
    cv2, np, torch, nn, Image, transformer_types = import_dependencies()
    checkpoint_path = find_checkpoint(model_path, base_dir)

    if device_choice not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device choice: {device_choice}")
    if device_choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")

    device_name = "cuda" if device_choice == "auto" and torch.cuda.is_available() else device_choice
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

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

    # Release the checkpoint container after its tensors have been copied into
    # the model. This reduces idle memory use in the long-running web frontend.
    del checkpoint
    gc.collect()

    return {
        "cv2": cv2,
        "np": np,
        "torch": torch,
        "Image": Image,
        "model": model,
        "temperature": temperature,
        "device": device,
        "checkpoint_path": checkpoint_path,
    }


def predict_images(
    runtime: dict[str, Any],
    image_paths: list[Path],
    display_names: list[str] | None = None,
    progress_callback: Any | None = None,
) -> list[dict[str, Any]]:
    """Predict a list of image paths using an already-loaded detector."""
    if not image_paths:
        raise RuntimeError("No images were provided for prediction.")
    if display_names is not None and len(display_names) != len(image_paths):
        raise ValueError("display_names must contain one name for every image path.")

    cv2 = runtime["cv2"]
    np = runtime["np"]
    torch = runtime["torch"]
    Image = runtime["Image"]
    model = runtime["model"]
    device = runtime["device"]
    temperature = runtime["temperature"]

    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, image_path in enumerate(image_paths, start=1):
            try:
                batch = preprocess_image(cv2, np, torch, Image, image_path).to(device)
                logit = model(batch)
                probability = float(torch.sigmoid(logit.float() / temperature).item())
            except Exception as exc:
                raise RuntimeError(f"Could not process image {image_path}: {exc}") from exc

            display_name = display_names[index - 1] if display_names else image_path.as_posix()
            predictions.append(
                {
                    "image_path": display_name,
                    "pred": round(probability, 6),
                }
            )
            if progress_callback is not None:
                progress_callback(index, len(image_paths), display_name, probability)

    return predictions


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    try:
        checkpoint_path = find_checkpoint(args.model_path, script_dir)
        image_root, image_paths = find_images(args.image_dir)
        print("Loading detector, please wait...")
        runtime = load_detector(checkpoint_path, args.device, script_dir)

        display_names = [
            output_image_path(path, args.image_dir, image_root) for path in image_paths
        ]

        print_information(
            "MODEL INFORMATION",
            [
                ("Checkpoint", checkpoint_path.name),
                ("Architecture", "CLIP ViT-L/14 + DINOv2-Large"),
                ("Device", runtime["device"].type.upper()),
                ("Images", str(len(image_paths))),
                ("Image directory", str(image_root)),
            ],
        )

        filename_width = max(len(Path(name).name) for name in display_names)
        name_width = min(max(filename_width, len("IMAGE")), 48)
        print_prediction_header(name_width)

        def print_progress(index: int, total: int, name: str, probability: float) -> None:
            del index, total  # The aligned table provides one row per image.
            filename = shorten_name(Path(name).name, name_width)
            label = prediction_label(probability)
            print(f"{filename:<{name_width}}  {label:<14}  {probability:>9.1%}")

        predictions = predict_images(
            runtime,
            image_paths,
            display_names=display_names,
            progress_callback=print_progress,
        )

        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(predictions, output_file, indent=2, ensure_ascii=False)
            output_file.write("\n")

        print_summary(predictions, output_path)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
