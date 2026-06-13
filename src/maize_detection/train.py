"""Train the EfficientNet-B0 baseline (M4).

Two-phase transfer learning:
  Phase A - freeze the ImageNet backbone, train only the new 4-class head.
  Phase B - unfreeze everything, fine-tune end-to-end at a lower LR.

Saves the single best-by-validation-loss checkpoint to outputs/checkpoints/best.pt.
Device-agnostic: trains on GPU (Colab/Kaggle) when available, else CPU.

Run a real training run:
    python -m src.maize_detection.train

Quick smoke run (a couple of batches per phase, to prove the loop works):
    python -m src.maize_detection.train --max-batches 2 --epochs-a 1 --epochs-b 1
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from .config import load_config
from .data import build_dataloaders
from .model import build_model, freeze_backbone, unfreeze_all
from .utils import get_device, set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """One pass over `loader`. Trains when `optimizer` is given, else evaluates.

    Returns (average loss, accuracy) over the samples actually seen.
    """
    is_train = optimizer is not None
    model.train(is_train)  # toggles dropout / batchnorm between train & eval modes

    total_loss, correct, seen = 0.0, 0, 0
    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)

            if is_train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)          # de-average to a sum
            correct += (logits.argmax(1) == y).sum().item()
            seen += x.size(0)

    return total_loss / seen, correct / seen


def train_phase(
    model: nn.Module,
    loaders: dict,
    device: torch.device,
    epochs: int,
    lr: float,
    phase: str,
    best: dict,
    ckpt_path: Path,
    max_batches: int | None = None,
) -> dict:
    """Run `epochs` of training for one phase, checkpointing the best val loss.

    `best` is a mutable carry-over dict so the single best checkpoint is tracked
    ACROSS both phases (Phase B keeps improving on Phase A's best).
    Adam optimizes only the params with requires_grad=True, so the same code works
    for the frozen-backbone phase and the full fine-tune phase.
    """
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, device, optimizer, max_batches)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, device, None, max_batches)

        marker = ""
        if va_loss < best["val_loss"]:
            best["val_loss"] = va_loss
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": model.state_dict(), "val_loss": va_loss,
                 "phase": phase, "epoch": epoch},
                ckpt_path,
            )
            marker = "  <- best (saved)"

        print(f"[phase {phase}] epoch {epoch:2d}/{epochs}  "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f}  |  "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}{marker}")

    return best


def run_training(
    cfg: dict,
    *,
    max_batches: int | None = None,
    epochs_a: int | None = None,
    epochs_b: int | None = None,
    device: torch.device | None = None,
) -> dict:
    """Run the full two-phase training. Shared by the CLI and the notebook.

    `epochs_a` / `epochs_b` / `max_batches` override the config when given (used
    for smoke runs). Returns the best val loss, the checkpoint path, and the
    trained model object.
    """
    set_seed(cfg["seed"])
    device = device or get_device()
    print(f"Device: {device}  |  seed: {cfg['seed']}")

    tr = cfg["training"]
    loaders = build_dataloaders(
        batch_size=tr["batch_size"],
        num_workers=tr["num_workers"],
        image_size=cfg["split"]["image_size"],
    )
    print(f"Batches/epoch  train={len(loaders['train'])}  val={len(loaders['val'])}")

    model = build_model().to(device)
    ckpt_path = PROJECT_ROOT / cfg["paths"]["checkpoints"] / "best.pt"
    best = {"val_loss": float("inf")}

    epochs_a = epochs_a if epochs_a is not None else tr["phase_a"]["epochs"]
    epochs_b = epochs_b if epochs_b is not None else tr["phase_b"]["epochs"]

    # Phase A - frozen backbone, head only
    freeze_backbone(model)
    best = train_phase(model, loaders, device, epochs_a, tr["phase_a"]["lr"],
                       "A", best, ckpt_path, max_batches)

    # Phase B - unfreeze all, fine-tune at the lower LR
    unfreeze_all(model)
    best = train_phase(model, loaders, device, epochs_b, tr["phase_b"]["lr"],
                       "B", best, ckpt_path, max_batches)

    print(f"\nDone. Best val loss {best['val_loss']:.4f}  ->  {ckpt_path.relative_to(PROJECT_ROOT)}")
    return {"best_val_loss": best["val_loss"], "checkpoint": ckpt_path, "model": model}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 baseline")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap batches/epoch for a fast smoke run (dev only).")
    parser.add_argument("--epochs-a", type=int, default=None, help="Override Phase A epochs.")
    parser.add_argument("--epochs-b", type=int, default=None, help="Override Phase B epochs.")
    args = parser.parse_args()

    cfg = load_config()
    run_training(cfg, max_batches=args.max_batches,
                 epochs_a=args.epochs_a, epochs_b=args.epochs_b)


if __name__ == "__main__":
    main()
