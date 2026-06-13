"""In-domain evaluation on the held-out controlled (PlantVillage) test split (M5).

Loads the trained checkpoint, runs the **test** split through the model using the
pretrained weights' own eval transforms (CLAUDE.md invariant #4, via
build_dataloaders), and reports accuracy, per-class precision/recall/F1, macro-F1,
the confusion matrix, and the per-class **false-negative rate** — the metric that
matters most for disease detection, since a missed disease is worse than a false
alarm.

INVARIANT #5 — these are CONTROLLED-domain numbers ONLY. They are not the headline
and must never be presented alone: they only mean something next to the M6
cross-domain (field) results. common_rust additionally carries the documented
residual leakage risk (data.COMMON_RUST_LEAKAGE_CAVEAT). Both caveats are printed
by `print_report` so they can't be silently dropped.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .config import load_config
from .data import COMMON_RUST_LEAKAGE_CAVEAT, build_dataloaders
from .labels import CANONICAL_LABELS
from .model import build_model
from .utils import get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_trained_model(checkpoint_path: Path, device: torch.device | None = None):
    """Rebuild the architecture and load trained weights; return (model, ckpt_meta)."""
    device = device or get_device()
    model = build_model().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def collect_predictions(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Run the loader through the model; return (y_true, y_pred) as label indices."""
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y in loader:
        logits = model(x.to(device))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())
    return np.array(y_true), np.array(y_pred)


def compute_metrics(y_true, y_pred, class_names: list[str] = CANONICAL_LABELS) -> dict:
    """Pure metric computation (no torch) — kept separate so it is unit-testable.

    Per-class false-negative rate = FN / (TP + FN) = 1 - recall: of all the leaves
    that truly belong to a class, the fraction the model missed.
    """
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    fn_rate = 1.0 - rec  # per-class false-negative rate

    per_class = {
        name: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "false_negative_rate": float(fn_rate[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }
    return {
        "domain": "controlled (PlantVillage) — IN-DOMAIN ONLY, pair with M6 field results",
        "split": "test",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }


def plot_confusion_matrix(metrics: dict, out_path: Path) -> None:
    """Row-normalized confusion matrix heatmap; cells annotated with raw counts."""
    import matplotlib.pyplot as plt

    cm = np.array(metrics["confusion_matrix"], dtype=float)
    names = metrics["class_names"]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("In-domain confusion matrix (controlled test split)")
    for i in range(len(names)):
        for j in range(len(names)):
            txt = f"{int(cm[i, j])}\n{cm_norm[i, j]:.0%}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalized")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_report(metrics: dict, ckpt: dict | None = None) -> None:
    print("=" * 70)
    print("IN-DOMAIN EVALUATION  --  controlled PlantVillage test split")
    if ckpt:
        print(f"checkpoint: phase {ckpt.get('phase')} epoch {ckpt.get('epoch')} "
              f"val_loss {ckpt.get('val_loss'):.4f}")
    print("=" * 70)
    print(f"accuracy: {metrics['accuracy']:.4f}    macro-F1: {metrics['macro_f1']:.4f}")
    print(f"\n{'class':<22}{'prec':>7}{'recall':>8}{'f1':>7}{'FN-rate':>9}{'n':>7}")
    for name in metrics["class_names"]:
        m = metrics["per_class"][name]
        print(f"{name:<22}{m['precision']:>7.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}"
              f"{m['false_negative_rate']:>9.3f}{m['support']:>7}")
    print("\n" + "-" * 70)
    print("INVARIANT #5: these are CONTROLLED-domain numbers ONLY. Do not present")
    print("them alone -- they are meaningful only beside the M6 field (CD&S) results.")
    print("-" * 70)
    print("common_rust caveat:")
    print(f"  {COMMON_RUST_LEAKAGE_CAVEAT}")
    print("=" * 70)


def main() -> None:
    cfg = load_config()
    device = get_device()
    out_dir = PROJECT_ROOT / cfg["paths"]["outputs"]
    ckpt_path = PROJECT_ROOT / cfg["paths"]["checkpoints"] / "best.pt"

    print(f"Device: {device}  |  checkpoint: {ckpt_path.relative_to(PROJECT_ROOT)}")
    model, ckpt = load_trained_model(ckpt_path, device)
    loaders = build_dataloaders(
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        image_size=cfg["split"]["image_size"],
    )
    y_true, y_pred = collect_predictions(model, loaders["test"], device)
    metrics = compute_metrics(y_true, y_pred)

    metrics_path = out_dir / "metrics_in_domain.json"
    fig_path = out_dir / "figures" / "confusion_matrix_in_domain.png"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({**metrics, "checkpoint": {k: ckpt[k] for k in ("phase", "epoch", "val_loss") if k in ckpt}}, f, indent=2)
    plot_confusion_matrix(metrics, fig_path)

    print_report(metrics, ckpt)
    print(f"\nmetrics -> {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"figure  -> {fig_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
