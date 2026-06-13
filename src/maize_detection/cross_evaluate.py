"""Cross-domain evaluation on CD&S FIELD images (M6) -- the point of the project.

The SAME EfficientNet-B0 trained on controlled PlantVillage leaves is run on
handheld field photos (CD&S, Purdue ACRE) for the only two diseases that overlap
both datasets: northern_leaf_blight and gray_leaf_spot. CD&S has no healthy and
no common_rust class, so cross-domain eval is a 2-class subset -- reported as such.

Preprocessing is identical to in-domain (the pretrained weights' own
``.transforms()`` via data.build_transforms; CLAUDE.md invariant #4), so the only
thing that changed is the *domain*: controlled lab crops -> natural field scenes.
Whatever accuracy we lose here is the honest domain gap, not a preprocessing
artifact.

Two complementary views of the same forward pass (the plan asks for both):

  1. OPEN 4-way -- the model is free to emit any of its 4 logits. Because CD&S only
     contains NLB and GLS, any prediction that lands in healthy/common_rust is a
     LEAKAGE error mode that we count and surface explicitly.

  2. RESTRICTED 2-class -- the decision is forced to argmax over only the
     {NLB, GLS} logits. This is the fair "could it tell the two apart in the
     field?" question, directly comparable to the published 94%->55% GLS benchmark.

INVARIANT #5: this module exists so the in-domain number is never shown alone.
``print_cross_report`` prints the controlled-vs-field gap side by side and re-emits
COMMON_RUST_LEAKAGE_CAVEAT, so the caveats cannot be silently dropped.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .data import (
    COMMON_RUST_LEAKAGE_CAVEAT,
    LABEL_TO_IDX,
    MaizeDataset,
    build_transforms,
)
from .evaluate import compute_metrics, load_trained_model, plot_confusion_matrix
from .labels import CANONICAL_LABELS
from .utils import get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# The only two classes shared by PlantVillage (controlled) and CD&S (field).
OVERLAP_LABELS: list[str] = ["northern_leaf_blight", "gray_leaf_spot"]
OVERLAP_IDX: list[int] = [LABEL_TO_IDX[c] for c in OVERLAP_LABELS]  # -> [2, 3]


# ── Build the CD&S field dataset/loader ───────────────────────────────────────

def load_cds_overlap_rows(overlap_dir: Path) -> list[dict]:
    """Scan data/external/cds_overlap/<canonical>/ and return manifest-style rows.

    No leaf_id and no split column: CD&S is a separate dataset used wholesale for
    evaluation, so there is no within-dataset leakage to guard against here (the
    only rule is that no CD&S image ever touched training -- true by construction).
    """
    suffixes = {".jpg", ".jpeg", ".png"}
    rows: list[dict] = []
    for label in OVERLAP_LABELS:
        cls_dir = overlap_dir / label
        if not cls_dir.exists():
            raise FileNotFoundError(
                f"missing CD&S overlap class dir: {cls_dir}\n"
                "Run: python scripts/download_cds.py --verify"
            )
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in suffixes:
                rows.append({
                    "image_path": img.relative_to(PROJECT_ROOT).as_posix(),
                    "canonical_label": label,
                })
    if not rows:
        raise ValueError(f"no images found under {overlap_dir}")
    return rows


def build_cds_loader(overlap_dir: Path, batch_size: int = 32, num_workers: int = 0):
    """DataLoader over CD&S overlap images using the eval (weights') transforms."""
    from torch.utils.data import DataLoader
    from typing import cast, TYPE_CHECKING

    if TYPE_CHECKING:  # pragma: no cover
        from torch.utils.data import Dataset

    rows = load_cds_overlap_rows(overlap_dir)
    ds = MaizeDataset(rows, build_transforms(train=False))
    loader = DataLoader(
        cast("Dataset", ds), batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return loader, rows


@torch.no_grad()
def collect_logits(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Run the loader once; return (y_true, logits) so both views share one pass."""
    y_true: list[int] = []
    all_logits: list[np.ndarray] = []
    for x, y in loader:
        logits = model(x.to(device))
        all_logits.append(logits.cpu().numpy())
        y_true.extend(y.tolist())
    return np.array(y_true), np.concatenate(all_logits, axis=0)


# ── Metrics (pure: numpy in, dict out -- unit-testable, no torch) ─────────────

def compute_cross_metrics(
    y_true,
    logits,
    class_names: list[str] = CANONICAL_LABELS,
) -> dict:
    """Both views of the field predictions from a single (N, 4) logits array.

    `y_true` holds only the two overlap indices (NLB=2, GLS=3). Returns a dict
    with `open_4way` (free choice + leakage tally) and `restricted_2class`
    (decision forced to the NLB/GLS logits).
    """
    y_true = np.asarray(y_true)
    logits = np.asarray(logits, dtype=float)
    overlap = OVERLAP_IDX

    # --- View 1: OPEN 4-way ----------------------------------------------------
    y_pred_open = logits.argmax(1)
    full = compute_metrics(y_true, y_pred_open, class_names)  # 4x4 confusion
    off_mask = ~np.isin(y_pred_open, overlap)                 # predicted outside NLB/GLS
    leak_by_class = {
        class_names[c]: int((y_pred_open[off_mask] == c).sum())
        for c in range(len(class_names)) if c not in overlap
    }
    open_view = {
        "accuracy_on_overlap": float((y_pred_open == y_true).mean()),
        "per_class": {c: full["per_class"][c] for c in OVERLAP_LABELS},
        "confusion_matrix": full["confusion_matrix"],
        "class_names": list(class_names),
        "leakage": {
            "by_class": leak_by_class,
            "total": int(off_mask.sum()),
            "rate": float(off_mask.mean()),
        },
    }

    # --- View 2: RESTRICTED 2-class -------------------------------------------
    sub = logits[:, overlap]                              # (N, 2): [NLB, GLS] logits
    y_pred_sub = np.asarray(overlap)[sub.argmax(1)]       # back to original idx space
    remap = {orig: i for i, orig in enumerate(overlap)}   # 2->0, 3->1
    yt2 = np.array([remap[int(t)] for t in y_true])
    yp2 = np.array([remap[int(p)] for p in y_pred_sub])
    restricted = compute_metrics(yt2, yp2, OVERLAP_LABELS)

    return {
        "domain": "field (CD&S) -- CROSS-DOMAIN, the project's real test",
        "classes_evaluated": list(OVERLAP_LABELS),
        "n_images": {c: int((y_true == LABEL_TO_IDX[c]).sum()) for c in OVERLAP_LABELS},
        "open_4way": open_view,
        "restricted_2class": restricted,
    }


def compute_domain_gap(in_domain: dict, cross: dict) -> dict:
    """Controlled (M5) vs field (M6) side by side -- the headline of the project."""
    gap: dict = {}
    for c in OVERLAP_LABELS:
        ind = in_domain["per_class"][c]["recall"]
        f_open = cross["open_4way"]["per_class"][c]["recall"]
        f_restr = cross["restricted_2class"]["per_class"][c]["recall"]
        gap[c] = {
            "in_domain_recall": ind,
            "field_recall_open_4way": f_open,
            "field_recall_restricted": f_restr,
            "recall_drop_open": ind - f_open,
            "recall_drop_restricted": ind - f_restr,
        }
    gap["_accuracy"] = {
        "in_domain_accuracy_4class": in_domain["accuracy"],
        "field_accuracy_restricted_2class": cross["restricted_2class"]["accuracy"],
        "field_accuracy_on_overlap_open": cross["open_4way"]["accuracy_on_overlap"],
        "note": (
            "in-domain accuracy is over 4 classes; field restricted accuracy is "
            "over 2 -- NOT like-for-like. Compare per-class recall for the fair read."
        ),
    }
    return gap


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_cross_report(cross: dict, gap: dict | None = None, ckpt: dict | None = None) -> None:
    o, r = cross["open_4way"], cross["restricted_2class"]
    print("=" * 72)
    print("CROSS-DOMAIN EVALUATION  --  CD&S FIELD images (NLB + GLS only)")
    if ckpt:
        print(f"checkpoint: phase {ckpt.get('phase')} epoch {ckpt.get('epoch')} "
              f"val_loss {ckpt.get('val_loss'):.4f}")
    n = cross["n_images"]
    print(f"images: northern_leaf_blight={n['northern_leaf_blight']}  "
          f"gray_leaf_spot={n['gray_leaf_spot']}")
    print("=" * 72)

    print("\n[VIEW 1] OPEN 4-way  (model may predict any of the 4 trained classes)")
    print(f"  accuracy on overlap classes: {o['accuracy_on_overlap']:.4f}")
    print(f"  {'class':<22}{'recall':>8}{'FN-rate':>9}{'n':>7}")
    for c in OVERLAP_LABELS:
        m = o["per_class"][c]
        print(f"  {c:<22}{m['recall']:>8.3f}{m['false_negative_rate']:>9.3f}{m['support']:>7}")
    lk = o["leakage"]
    print(f"  LEAKAGE into non-field classes: {lk['total']} preds "
          f"({lk['rate']:.1%})  -> {lk['by_class']}")

    print("\n[VIEW 2] RESTRICTED 2-class  (decision forced to NLB-vs-GLS logits)")
    print(f"  accuracy: {r['accuracy']:.4f}    macro-F1: {r['macro_f1']:.4f}")
    print(f"  {'class':<22}{'prec':>7}{'recall':>8}{'f1':>7}{'FN-rate':>9}{'n':>7}")
    for c in OVERLAP_LABELS:
        m = r["per_class"][c]
        print(f"  {c:<22}{m['precision']:>7.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}"
              f"{m['false_negative_rate']:>9.3f}{m['support']:>7}")

    if gap:
        print("\n" + "-" * 72)
        print("DOMAIN GAP  --  controlled (M5) vs field (M6), per-class recall")
        print(f"  {'class':<22}{'in-domain':>11}{'field(4way)':>13}{'field(2cls)':>13}{'drop(2cls)':>12}")
        for c in OVERLAP_LABELS:
            g = gap[c]
            print(f"  {c:<22}{g['in_domain_recall']:>11.3f}"
                  f"{g['field_recall_open_4way']:>13.3f}"
                  f"{g['field_recall_restricted']:>13.3f}"
                  f"{g['recall_drop_restricted']:>12.3f}")
        a = gap["_accuracy"]
        print(f"\n  in-domain accuracy (4-class):        {a['in_domain_accuracy_4class']:.4f}")
        print(f"  field accuracy (restricted 2-class): {a['field_accuracy_restricted_2class']:.4f}")

    print("\n" + "-" * 72)
    print("BENCHMARK CONTEXT: the published CD&S study reports ~94% controlled ->")
    print("~55% field accuracy for gray leaf spot. We expect a gap of similar")
    print("character; the numbers above are our honest reproduction, not a target.")
    print("-" * 72)
    print("INVARIANT #5 satisfied: controlled and field results are shown together;")
    print("the field gap -- not the in-domain number -- is the project's finding.")
    print("-" * 72)
    print("common_rust caveat (applies to the in-domain side of this comparison):")
    print(f"  {COMMON_RUST_LEAKAGE_CAVEAT}")
    print("=" * 72)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    from .config import load_config

    cfg = load_config()
    device = get_device()
    out_dir = PROJECT_ROOT / cfg["paths"]["outputs"]
    overlap_dir = PROJECT_ROOT / cfg["paths"]["data_cds_overlap"]
    ckpt_path = PROJECT_ROOT / cfg["paths"]["checkpoints"] / "best.pt"

    print(f"Device: {device}  |  checkpoint: {ckpt_path.relative_to(PROJECT_ROOT)}")
    print(f"CD&S overlap dir: {overlap_dir.relative_to(PROJECT_ROOT)}")

    model, ckpt = load_trained_model(ckpt_path, device)
    loader, _ = build_cds_loader(
        overlap_dir,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )
    y_true, logits = collect_logits(model, loader, device)
    cross = compute_cross_metrics(y_true, logits)

    # Domain gap vs the M5 in-domain metrics, if present.
    in_domain_path = out_dir / "metrics_in_domain.json"
    gap = None
    if in_domain_path.exists():
        with open(in_domain_path, encoding="utf-8") as f:
            gap = compute_domain_gap(json.load(f), cross)
    else:
        print(f"\nNOTE: {in_domain_path.name} not found -- run M5 first for the gap table.")

    # Persist metrics + both confusion figures.
    metrics_path = out_dir / "metrics_cross_domain.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**cross, "domain_gap": gap,
               "checkpoint": {k: ckpt[k] for k in ("phase", "epoch", "val_loss") if k in ckpt}}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fig_dir = out_dir / "figures"
    plot_confusion_matrix(
        cross["restricted_2class"], fig_dir / "confusion_matrix_cross_restricted.png",
        title="Field confusion (CD&S) -- restricted NLB vs GLS",
    )
    plot_confusion_matrix(
        cross["open_4way"], fig_dir / "confusion_matrix_cross_open4way.png",
        title="Field confusion (CD&S) -- open 4-way (shows leakage)",
    )

    print_cross_report(cross, gap, ckpt)
    print(f"\nmetrics -> {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"figures -> {(fig_dir / 'confusion_matrix_cross_restricted.png').relative_to(PROJECT_ROOT)}")
    print(f"           {(fig_dir / 'confusion_matrix_cross_open4way.png').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
