#!/usr/bin/env python
"""
Download PlantVillage corn images (color) from HuggingFace and write a manifest.

NOTE ON DATASET STRUCTURE (verified 2026-06):
  The `mohanty/PlantVillage` HF dataset no longer serves images via
  load_dataset(config="color") — that path needs the removed `trust_remote_code`,
  and the parquet config stores only file *paths*. All images live inside a single
  ~2.08 GB `data.zip`. We download that zip once (cached by huggingface_hub),
  then extract ONLY the 4 corn color classes (~3,852 images).

LEAF_ID (cardinal invariant):
  The repo's leaf-map.json has NO corn entries, so there is no official corn
  leaf_id. We DERIVE it from the filename suffix (see derive_leaf_id). This is a
  heuristic; common_rust shows no detectable grouping (each image = its own leaf),
  a residual leakage risk documented in the eval report. The M3 split test asserts
  zero leaf_id overlap across splits against this derived id.

Usage (project root, .venv activated):
    python scripts/download_plantvillage.py

No HuggingFace token required. First run downloads ~2.08 GB to the HF cache
(~/.cache/huggingface). Re-runs reuse the cache. To reclaim space afterward:
    huggingface-cli delete-cache      (or delete the datasets--mohanty--PlantVillage cache dir)
"""

import csv
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.maize_detection.labels import PLANTVILLAGE_TO_CANONICAL, validate_mapping

REPO_ID = "mohanty/PlantVillage"
ZIP_FILENAME = "data.zip"

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "plantvillage"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "plantvillage_manifest.csv"
MANIFEST_FIELDS = [
    "image_path", "canonical_label", "original_label",
    "leaf_id", "original_filename", "split",
]

CANONICAL_ORDER = ["healthy", "common_rust", "northern_leaf_blight", "gray_leaf_spot"]
DIVIDER = "=" * 64


def derive_leaf_id(filename: str, canonical_label: str) -> str:
    """Derive a per-leaf group id from a PlantVillage filename.

    Mirrors the grouping signal in the original filenames: the descriptive
    suffix after the last '___', minus the extension and any 'copy N' marker,
    lowercased and prefixed with the canonical label to avoid cross-class
    collisions. Example:
        '...___R.S_HL 5498 copy 2.JPG' (healthy) -> 'healthy:::r.s_hl 5498'
        '...___RS_GLSp 7321.JPG' (gray_leaf_spot) -> 'gray_leaf_spot:::rs_glsp 7321'
    """
    suffix = filename.split("___")[-1]
    suffix = re.sub(r"\.(jpe?g|png)$", "", suffix, flags=re.IGNORECASE)
    suffix = re.sub(r"\s*copy\s*\d*", "", suffix, flags=re.IGNORECASE)
    suffix = suffix.strip().lower()
    return f"{canonical_label}:::{suffix}"


def find_corn_color_members(zf: zipfile.ZipFile) -> list[tuple[str, str, str]]:
    """Return (zip_member, original_label, filename) for every corn color image.

    Robust to whether the zip root is 'raw/color/...' or 'data/raw/color/...':
    we locate the '/color/' segment and read the class folder that follows.
    """
    members: list[tuple[str, str, str]] = []
    for name in zf.namelist():
        if name.endswith("/") or "/color/" not in name:
            continue
        after = name.split("/color/", 1)[1]          # '<class>/<file>'
        segs = after.split("/")
        if len(segs) < 2:
            continue
        original_label = segs[0]
        if original_label not in PLANTVILLAGE_TO_CANONICAL:
            continue
        members.append((name, original_label, segs[-1]))
    return members


def main() -> None:
    validate_mapping(PLANTVILLAGE_TO_CANONICAL)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    for canonical in set(PLANTVILLAGE_TO_CANONICAL.values()):
        (OUT_DIR / canonical).mkdir(parents=True, exist_ok=True)

    print(DIVIDER)
    print("PlantVillage corn download  —  mohanty/PlantVillage (color)")
    print(DIVIDER)
    print(f"Downloading {ZIP_FILENAME} (~2.08 GB) — one-time, cached by HF...")

    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(REPO_ID, ZIP_FILENAME, repo_type="dataset")
    print(f"Zip ready: {zip_path}")

    rows: list[dict] = []
    skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = find_corn_color_members(zf)
        print(f"\nCorn color images found in zip: {len(members):,}")
        if not members:
            print("ERROR: No corn color members found. Zip layout may have changed.")
            print("       Inspect with: python -c \"import zipfile;"
                  f" print(zipfile.ZipFile(r'{zip_path}').namelist()[:20])\"")
            sys.exit(1)

        for i, (member, original_label, filename) in enumerate(
            tqdm(members, desc="Extracting corn-color", unit="img")
        ):
            canonical = PLANTVILLAGE_TO_CANONICAL[original_label]
            leaf_id = derive_leaf_id(filename, canonical)

            suffix = Path(filename).suffix.lower() or ".jpg"
            out_path = OUT_DIR / canonical / f"{i:05d}{suffix}"
            try:
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                if out_path.stat().st_size == 0:
                    raise ValueError("zero-byte image")
            except Exception as exc:
                skipped += 1
                tqdm.write(f"  SKIP corrupt member {member}: {exc}")
                out_path.unlink(missing_ok=True)
                continue

            rows.append({
                "image_path": out_path.relative_to(PROJECT_ROOT).as_posix(),
                "canonical_label": canonical,
                "original_label": original_label,
                "leaf_id": leaf_id,
                "original_filename": filename,
                "split": "",  # assigned in M3
            })

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nSaved {len(rows):,} images  →  {OUT_DIR.relative_to(PROJECT_ROOT)}")
    if skipped:
        print(f"Skipped {skipped} corrupt/empty image(s).")
    print(f"Manifest             →  {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")

    counts = Counter(r["canonical_label"] for r in rows)
    leaves_by_class: dict[str, set[str]] = {}
    for r in rows:
        leaves_by_class.setdefault(r["canonical_label"], set()).add(r["leaf_id"])

    print("\nClass distribution and derived leaf grouping:")
    print(f"  {'class':<22}{'images':>8}{'leaves':>8}{'imgs/leaf':>11}")
    for label in CANONICAL_ORDER:
        n = counts.get(label, 0)
        leaves = len(leaves_by_class.get(label, set()))
        ratio = (n / leaves) if leaves else 0.0
        print(f"  {label:<22}{n:>8}{leaves:>8}{ratio:>11.2f}")

    total_imgs = len(rows)
    total_leaves = len({r["leaf_id"] for r in rows})
    print(f"  {'TOTAL':<22}{total_imgs:>8}{total_leaves:>8}")
    if counts.get("common_rust", 0) and len(leaves_by_class.get("common_rust", set())) == counts.get("common_rust", 0):
        print("\n  NOTE: common_rust has no detectable multi-shot grouping "
              "(each image = its own leaf).")
        print("        Residual leakage risk — must be named in the eval report.")

    print(f"\n{DIVIDER}")
    print("DONE — next: download CD&S, then run 01_data_inspection.ipynb.")
    print(DIVIDER)


if __name__ == "__main__":
    main()
