#!/usr/bin/env python
"""
Download CD&S Dataset_Original from OSF (https://osf.io/s6ru5/).

The script first tries automated download via the public OSF API.
If the project requires a login or the API is unreachable, clear
manual download instructions are printed instead.

Usage:
    python scripts/download_cds.py             # download (auto or manual)
    python scripts/download_cds.py --verify    # verify counts + build overlap dir
    python scripts/download_cds.py --overlap   # rebuild overlap dir from raw data

V1 evaluation: NLB + GLS only (NLS excluded — no canonical match).
Expected counts: NLB=497, GLS=523, NLS=551 (verified against OSF 2026-06).
"""

import argparse
import sys
import shutil
from pathlib import Path
from time import sleep

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.maize_detection.labels import CDS_TO_CANONICAL

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cds"
OVERLAP_DIR = PROJECT_ROOT / "data" / "external" / "cds_overlap"

OSF_PROJECT_ID = "s6ru5"
EXPECTED_COUNTS = {"NLB": 497, "GLS": 523, "NLS": 551}
OVERLAP_CLASSES = {"NLB", "GLS"}
IMG_SUFFIXES = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}
DIVIDER = "=" * 62

MANUAL_STEPS = """
+----------------------------------------------------------+
|  MANUAL DOWNLOAD -- CD&S Dataset (OSF)                   |
+----------------------------------------------------------+
|  1. Open:   https://osf.io/s6ru5/                        |
|  2. Create a free OSF account if prompted                |
|  3. Navigate: Files > Dataset_Original                   |
|  4. Download NLB, GLS, and NLS as separate folders       |
|     (or download the full Dataset_Original ZIP)          |
|  5. Extract / copy to this exact structure:              |
|                                                          |
|     data/raw/cds/NLB/    <- 497 .jpg files               |
|     data/raw/cds/GLS/    <- 523 .jpg files               |
|     data/raw/cds/NLS/    <- 551 .jpg files               |
|                                                          |
|  6. Run: python scripts/download_cds.py --verify         |
+----------------------------------------------------------+
"""


def count_images(cls_dir: Path) -> int:
    if not cls_dir.exists():
        return 0
    return sum(1 for f in cls_dir.iterdir() if f.suffix in IMG_SUFFIXES)


def verify_counts() -> bool:
    print("\nVerifying CD&S image counts...")
    all_ok = True
    for cls, expected in sorted(EXPECTED_COUNTS.items()):
        cls_dir = RAW_DIR / cls
        found = count_images(cls_dir)
        if not cls_dir.exists():
            tag = "MISSING"
            all_ok = False
        elif found == expected:
            tag = "PASS   "
        else:
            tag = "WARN   "
            all_ok = False
        print(f"  {tag}  {cls}/   found={found:>3}, expected={expected}")
    return all_ok


def build_overlap_dir() -> None:
    """Copy NLB + GLS into cds_overlap/ using canonical label folder names."""
    print("\nBuilding overlap directory (NLB + GLS only)...")
    for cls in sorted(OVERLAP_CLASSES):
        canonical = CDS_TO_CANONICAL[cls]
        if canonical is None:  # never true for NLB/GLS — guards type + invariant
            continue
        src = RAW_DIR / cls
        dst = OVERLAP_DIR / canonical
        if not src.exists():
            print(f"  SKIP  {cls}/ not found — download first")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        n = count_images(dst)
        print(f"  {cls}  ->  {dst.relative_to(PROJECT_ROOT)}  ({n} images)")


# ── OSF API helpers ───────────────────────────────────────────────────────────

def _osf_get(url: str, session) -> dict | None:
    try:
        r = session.get(url, timeout=30)
        if r.status_code in (401, 403):
            print(f"  OSF returned {r.status_code} — project requires login.")
            return None
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"  OSF API error: {exc}")
        return None


def _list_all(url: str, session) -> list[dict]:
    """Follow OSF pagination and return all file/folder objects."""
    items: list[dict] = []
    page_url: str | None = url
    while page_url:
        body = _osf_get(page_url, session)
        if body is None:
            break
        items.extend(body.get("data", []))
        page_url = body.get("links", {}).get("next")
    return items


def _download_file(url: str, dest: Path, session) -> bool:
    try:
        r = session.get(url, timeout=180, stream=True)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as exc:
        print(f"    Error downloading {dest.name}: {exc}")
        return False


def try_automated_download() -> bool:
    """Attempt OSF API download. Returns True if all classes retrieved."""
    try:
        import requests
    except ImportError:
        print("requests not installed — skipping automated download.")
        return False

    print("Attempting automated download via OSF public API...")
    session = requests.Session()
    session.headers["User-Agent"] = "MaizeDetection-research/1.0"

    # 1. List storage root
    root_url = f"https://api.osf.io/v2/nodes/{OSF_PROJECT_ID}/files/osfstorage/"
    root_items = _list_all(root_url, session)
    if not root_items:
        return False

    print(f"  Root contents: {[i['attributes']['name'] for i in root_items]}")

    # 2. Find Dataset_Original folder
    ds_folder = next(
        (i for i in root_items if i["attributes"]["name"] == "Dataset_Original"), None
    )
    if ds_folder is None:
        print("  'Dataset_Original' not found in project root.")
        return False

    sub_url = ds_folder["relationships"]["files"]["links"]["related"]["href"]
    class_items = _list_all(sub_url, session)
    print(f"  Dataset_Original contents: {[i['attributes']['name'] for i in class_items]}")

    # 3. Download each class folder
    success = True
    for cls_item in class_items:
        cls_name: str = cls_item["attributes"]["name"]
        if cls_name not in EXPECTED_COUNTS:
            continue

        cls_dir = RAW_DIR / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        already = count_images(cls_dir)
        expected = EXPECTED_COUNTS[cls_name]

        if already == expected:
            print(f"  SKIP  {cls_name}/ already complete ({already} images)")
            continue

        file_url = cls_item["relationships"]["files"]["links"]["related"]["href"]
        file_items = [f for f in _list_all(file_url, session)
                      if f["attributes"]["kind"] == "file"]
        print(f"  Downloading {cls_name}/ ({len(file_items)} files)...")

        try:
            from tqdm import tqdm
            iterator = tqdm(file_items, desc=f"    {cls_name}", unit="img")
        except ImportError:
            iterator = iter(file_items)

        saved = 0
        for file_item in iterator:
            fname: str = file_item["attributes"]["name"]
            dl_url: str = file_item["links"]["download"]
            dest = cls_dir / fname
            if dest.exists():
                saved += 1
                continue
            if _download_file(dl_url, dest, session):
                saved += 1
            sleep(0.04)  # polite rate limiting

        print(f"  {cls_name}: {saved}/{len(file_items)} saved")
        if saved < expected:
            success = False

    return success


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download CD&S dataset from OSF")
    parser.add_argument("--verify", action="store_true",
                        help="Verify counts and build overlap dir (no download)")
    parser.add_argument("--overlap", action="store_true",
                        help="Rebuild overlap dir from existing raw data")
    args = parser.parse_args()

    print(DIVIDER)
    print("CD&S Dataset  —  OSF project s6ru5")
    print("V1 evaluation: NLB (northern_leaf_blight) + GLS (gray_leaf_spot)")
    print("Excluded: NLS — Northern Leaf Spot has no canonical match")
    print(DIVIDER)

    if args.verify:
        ok = verify_counts()
        if ok:
            build_overlap_dir()
            print("\nDone. Run notebooks/01_data_inspection.ipynb to inspect.")
        else:
            print("\nCount mismatch. Re-download or check extraction.")
            print(MANUAL_STEPS)
            sys.exit(1)
        return

    if args.overlap:
        build_overlap_dir()
        return

    # Check if already complete
    if all(count_images(RAW_DIR / cls) == n for cls, n in EXPECTED_COUNTS.items()):
        print("CD&S data already present and counts match.")
        build_overlap_dir()
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAP_DIR.mkdir(parents=True, exist_ok=True)

    success = try_automated_download()
    if not success:
        print(MANUAL_STEPS)
        print("After manual download, run: python scripts/download_cds.py --verify")
        sys.exit(0)

    ok = verify_counts()
    if ok:
        build_overlap_dir()
        print(f"\n{DIVIDER}")
        print("CD&S download complete.")
        print(DIVIDER)
    else:
        print("\nSome counts mismatch after download. Re-run with --verify.")
        print(MANUAL_STEPS)


if __name__ == "__main__":
    main()
