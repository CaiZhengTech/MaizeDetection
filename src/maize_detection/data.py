"""
Data pipeline for MaizeDetection.

M3 will implement here: the leaf_id-aware GroupShuffleSplit (70/15/15) that reads
the PlantVillage manifest, the torchvision Dataset wrappers, and the DataLoaders
(eval uses the pretrained weights' own `.transforms()`; train uses light v2 aug).

Implementation is deliberately deferred to M3 — this module currently exists only
to anchor the leaf_id / common_rust caveat below so it cannot be lost.
"""

# =============================================================================
# CARDINAL CAVEAT — common_rust has NO detectable leaf grouping
# =============================================================================
# Corn has no official `leaf_id` in the source data. We DERIVE leaf_id from the
# PlantVillage filename suffix (see scripts/download_plantvillage.py::derive_leaf_id;
# the value lives in the `leaf_id` column of data/processed/plantvillage_manifest.csv).
#
# This derivation groups multiple shots of one physical leaf for healthy, NLB, and
# GLS. It does NOT work for common_rust: every common_rust image has a unique
# filename key (~1,192 images -> ~1,192 leaves, exactly 1.00 imgs/leaf). If rust
# leaves were in fact photographed multiple times with distinct numbers, our split
# CANNOT detect it, so a same-leaf train/test leak is possible for common_rust ONLY.
#
# Consequences to honor downstream:
#   * The M3 zero-overlap test enforces no shared leaf_id across splits, but for
#     common_rust that guarantee is only as strong as the derived id (singletons).
#   * M5/M6 reporting MUST name this as a residual, unverifiable leakage risk and
#     treat common_rust in-domain recall with appropriate skepticism.
#
# This text is exported as COMMON_RUST_LEAKAGE_CAVEAT so evaluation code can import
# and emit it verbatim — making omission from the report a deliberate act, not an
# accident. See reports/evaluation_report_template.md and the project memory note
# `plantvillage-corn-leaf-id`.
# =============================================================================

COMMON_RUST_LEAKAGE_CAVEAT: str = (
    "common_rust has no detectable leaf grouping: the filename-derived leaf_id "
    "yields exactly one image per leaf (~1,192 images / ~1,192 leaves). Unlike "
    "healthy/NLB/GLS, multi-shot grouping cannot be recovered for rust, so a "
    "same-leaf train/test leak is possible for common_rust only and cannot be "
    "ruled out. Interpret common_rust in-domain recall with this residual, "
    "unverifiable leakage risk in mind."
)

# --- M3 implementation begins below this line ---

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .labels import CANONICAL_LABELS

if TYPE_CHECKING:  # keep torch out of the import path for the pure-split logic
    from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "plantvillage_manifest.csv"
SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "plantvillage_splits.csv"

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
SPLIT_FIELDS = ["image_path", "canonical_label", "leaf_id", "split"]
LABEL_TO_IDX: dict[str, int] = {label: i for i, label in enumerate(CANONICAL_LABELS)}

DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_SEED = 42


# ── Split assignment (leaf-grouped, per-class stratified) ─────────────────────

def assign_splits(
    rows: list[dict],
    ratios: dict[str, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Return a new row list with a 'split' value on every row.

    The split is keyed on ``leaf_id`` so all images of one physical leaf land in
    the SAME split — the cardinal anti-leakage invariant. Splitting is done
    independently *within each canonical class* (leaf_ids are class-pure, see
    derive_leaf_id), which makes the result a stratified group split: every class
    keeps ~the same train/val/test leaf proportions while staying leaf-disjoint.

    Two-stage GroupShuffleSplit per class:
      1. carve off the test leaves (test ratio),
      2. split the remainder into train / val (val ratio rescaled).

    Note: ratios apply to LEAVES, not images; image counts per split drift
    slightly from the ratios when imgs/leaf varies (and common_rust is all
    singletons — see COMMON_RUST_LEAKAGE_CAVEAT).
    """
    _validate_ratios(ratios)
    test_frac = ratios["test"]
    val_frac_of_remainder = ratios["val"] / (ratios["train"] + ratios["val"])

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["canonical_label"]].append(r)

    split_of_leaf: dict[str, str] = {}
    for cls, cls_rows in by_class.items():
        leaves = np.array(sorted({r["leaf_id"] for r in cls_rows}))
        if len(leaves) < 3:
            raise ValueError(
                f"class '{cls}' has only {len(leaves)} leaf group(s); "
                "cannot form non-empty train/val/test. Check the manifest."
            )
        # Stage 1: test vs rest (each "sample" is a leaf; groups == leaves)
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        rest_idx, test_idx = next(gss_test.split(leaves, groups=leaves))
        rest_leaves = leaves[rest_idx]
        for leaf in leaves[test_idx]:
            split_of_leaf[str(leaf)] = "test"
        # Stage 2: val vs train within the remainder
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_frac_of_remainder, random_state=seed
        )
        train_idx, val_idx = next(gss_val.split(rest_leaves, groups=rest_leaves))
        for leaf in rest_leaves[train_idx]:
            split_of_leaf[str(leaf)] = "train"
        for leaf in rest_leaves[val_idx]:
            split_of_leaf[str(leaf)] = "val"

    return [{**r, "split": split_of_leaf[r["leaf_id"]]} for r in rows]


def _validate_ratios(ratios: dict[str, float]) -> None:
    missing = set(SPLIT_NAMES) - set(ratios)
    if missing:
        raise ValueError(f"ratios missing keys: {sorted(missing)}")
    total = sum(ratios[k] for k in SPLIT_NAMES)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")


# ── Manifest I/O ──────────────────────────────────────────────────────────────

def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"manifest not found: {path}\n"
            "Run scripts/download_plantvillage.py first."
        )
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_splits(rows: list[dict], path: Path = SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPLIT_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in SPLIT_FIELDS})


def load_splits(path: Path = SPLITS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"split manifest not found: {path}\n"
            "Run: python -m src.maize_detection.data"
        )
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Verification + summary ────────────────────────────────────────────────────

def leaf_overlap(rows: list[dict]) -> dict[tuple[str, str], set[str]]:
    """Return the intersection of leaf_ids for each pair of splits. Empty == clean."""
    leaves: dict[str, set[str]] = {s: set() for s in SPLIT_NAMES}
    for r in rows:
        leaves[r["split"]].add(r["leaf_id"])
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    return {(a, b): leaves[a] & leaves[b] for a, b in pairs}


def summarize_splits(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"{'split':<7}{'images':>9}{'leaves':>9}   per-class images")
    for split in SPLIT_NAMES:
        srows = [r for r in rows if r["split"] == split]
        n_leaves = len({r["leaf_id"] for r in srows})
        per_cls = Counter(r["canonical_label"] for r in srows)
        cls_str = "  ".join(
            f"{lbl.split('_')[0][:4]}={per_cls.get(lbl, 0)}" for lbl in CANONICAL_LABELS
        )
        lines.append(f"{split:<7}{len(srows):>9}{n_leaves:>9}   {cls_str}")
    total_leaves = len({r["leaf_id"] for r in rows})
    lines.append(f"{'TOTAL':<7}{len(rows):>9}{total_leaves:>9}")

    bad = {k: v for k, v in leaf_overlap(rows).items() if v}
    if bad:
        lines.append("\nLEAF_ID OVERLAP DETECTED (INVALID SPLIT):")
        for (a, b), shared in bad.items():
            lines.append(f"  {a} & {b}: {len(shared)} shared leaves")
    else:
        lines.append("\nleaf_id overlap across splits: 0  (OK - cardinal invariant holds)")
    return "\n".join(lines)


# ── Torch Dataset + transforms + loaders (imported lazily) ────────────────────

def build_transforms(train: bool, image_size: int = 224):
    """Return a torchvision transform.

    Eval/inference uses the pretrained weights' OWN ``.transforms()`` so train-
    and inference-time preprocessing can never drift (CLAUDE.md invariant #4).
    Train adds light geometric augmentation, then the SAME ImageNet normalization
    the weights expect, via ``transforms.v2`` only.
    """
    import torch
    from torchvision.models import EfficientNet_B0_Weights
    from torchvision.transforms import v2

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    if not train:
        return weights.transforms()  # resize -> center-crop -> normalize, exact match

    meta = weights.transforms()
    return v2.Compose([
        v2.PILToTensor(),
        v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(meta.mean), std=list(meta.std)),
    ])


class MaizeDataset:
    """Map-style dataset over one split of the PlantVillage manifest.

    A plain class implementing the ``__len__`` / ``__getitem__`` map-style
    protocol — torch's DataLoader consumes that directly, so this module never
    needs to import torch at module scope (the split logic and its test stay
    torch-free). Each item is ``(transformed_image_tensor, label_idx)``.
    """

    def __init__(self, rows: list[dict], transform, project_root: Path = PROJECT_ROOT):
        self.rows = rows
        self.transform = transform
        self.project_root = project_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        from PIL import Image

        r = self.rows[idx]
        path = self.project_root / r["image_path"]
        img = Image.open(path).convert("RGB")
        return self.transform(img), LABEL_TO_IDX[r["canonical_label"]]


def build_dataloaders(
    splits_path: Path = SPLITS_PATH,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
) -> dict[str, "DataLoader"]:
    """Build train/val/test DataLoaders from the split manifest."""
    from torch.utils.data import DataLoader

    rows = load_splits(splits_path)
    loaders: dict[str, DataLoader] = {}
    for split in SPLIT_NAMES:
        srows = [r for r in rows if r["split"] == split]
        ds = MaizeDataset(srows, build_transforms(train=(split == "train"), image_size=image_size))
        loaders[split] = DataLoader(
            # MaizeDataset is a map-style dataset (implements __len__/__getitem__);
            # torch consumes that protocol directly. cast satisfies the stub, which
            # expects a Dataset subclass.
            cast("Dataset", ds),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
    return loaders


# ── CLI: build and persist the split manifest ─────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build leaf-grouped PlantVillage splits")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out", type=Path, default=SPLITS_PATH)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    split_rows = assign_splits(rows, DEFAULT_RATIOS, seed=args.seed)
    write_splits(split_rows, args.out)

    print("=" * 64)
    print(f"Leaf-grouped splits (seed={args.seed})  ->  {args.out.relative_to(PROJECT_ROOT)}")
    print("=" * 64)
    print(summarize_splits(split_rows))
    print("\nNOTE: common_rust leaves are singletons - see COMMON_RUST_LEAKAGE_CAVEAT.")


if __name__ == "__main__":
    main()
