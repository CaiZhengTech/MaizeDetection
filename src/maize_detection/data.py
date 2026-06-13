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
