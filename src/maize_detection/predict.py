"""Reusable CPU inference (M7): classify a single corn-leaf image.

``predict(image_path)`` loads the trained checkpoint, applies the pretrained
weights' OWN ``.transforms()`` (CLAUDE.md invariant #4 -- inference preprocessing
can never drift from training), runs a forward pass **on CPU**, and returns:

    {"label": str, "confidence": float, "probabilities": {label: float}}

This is the single entry point a future caller (a script, or the V2 FastAPI
wrapper) builds on. It is deliberately small and dependency-light.

HONESTY CAVEAT (do not strip): this classifier was trained on CONTROLLED
PlantVillage leaves. M6 showed in-domain accuracy 0.984 collapses to ~0.62 on
field photos, with a quarter of diseased field leaves misread as healthy. So a
high `confidence` on a real-world field image is NOT trustworthy. This is a
4-class image classifier, not a diagnosis/treatment tool.
"""

import functools
from pathlib import Path

import torch

from .data import build_transforms
from .evaluate import load_trained_model
from .labels import CANONICAL_LABELS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt"
CPU = torch.device("cpu")  # local inference is CPU-only by contract


@functools.lru_cache(maxsize=2)
def _load_cached_model(checkpoint_path: str):
    """Load + cache the model so repeated predict() calls don't reload the file."""
    model, _ = load_trained_model(Path(checkpoint_path), CPU)
    return model


@functools.lru_cache(maxsize=1)
def _eval_transform():
    """The weights' own eval transforms (resize -> center-crop -> normalize)."""
    return build_transforms(train=False)


def predict(
    image_path: str | Path,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    model: torch.nn.Module | None = None,
) -> dict:
    """Classify one image on CPU. Returns label, confidence, and full probabilities.

    Pass a preloaded `model` to skip the cache (e.g. in a tight batch loop);
    otherwise the checkpoint is loaded once and reused across calls.
    """
    from PIL import Image

    if model is None:
        model = _load_cached_model(str(Path(checkpoint_path)))

    img = Image.open(image_path).convert("RGB")
    x = _eval_transform()(img).unsqueeze(0).to(CPU)  # (1, 3, 224, 224)

    with torch.no_grad():
        probs = model(x).softmax(dim=1)[0]
    conf, idx = probs.max(dim=0)

    return {
        "label": CANONICAL_LABELS[int(idx)],
        "confidence": float(conf),
        "probabilities": {
            CANONICAL_LABELS[i]: float(probs[i]) for i in range(len(CANONICAL_LABELS))
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Classify a corn-leaf image (CPU)")
    parser.add_argument("image", type=Path, help="path to a leaf image")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    result = predict(args.image, args.checkpoint)

    print("=" * 56)
    print(f"image:      {args.image}")
    print(f"prediction: {result['label']}  ({result['confidence']:.1%} confidence)")
    print("-" * 56)
    for label, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {label:<22}{p:>8.1%}")
    print("-" * 56)
    print("NOTE: trained on CONTROLLED images. On real FIELD photos this model is")
    print("unreliable (see M6: ~0.62 field accuracy). Confidence != correctness.")
    print("=" * 56)


if __name__ == "__main__":
    main()
