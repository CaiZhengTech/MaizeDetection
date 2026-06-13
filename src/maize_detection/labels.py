"""
Canonical taxonomy for MaizeDetection V1.

The four canonical label strings are the only valid targets throughout the
codebase. validate_mapping() must be called on any external mapping before use.
"""

from collections.abc import Mapping

CANONICAL_LABELS: list[str] = [
    "healthy",
    "common_rust",
    "northern_leaf_blight",
    "gray_leaf_spot",
]

# Original PlantVillage folder/label names → canonical
PLANTVILLAGE_TO_CANONICAL: dict[str, str] = {
    "Corn_(maize)___healthy":                          "healthy",
    "Corn_(maize)___Common_rust_":                     "common_rust",
    "Corn_(maize)___Northern_Leaf_Blight":             "northern_leaf_blight",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "gray_leaf_spot",
}

# CD&S class names → canonical (None = excluded; no canonical match)
CDS_TO_CANONICAL: dict[str, str | None] = {
    "NLB": "northern_leaf_blight",
    "GLS": "gray_leaf_spot",
    "NLS": None,  # Northern Leaf Spot ≠ Northern Leaf Blight — excluded, not force-fit
}

# Pairs that must never appear in any mapping: (source_key, illegal_target)
_FORBIDDEN: dict[tuple[str, str], str] = {
    ("NLS", "northern_leaf_blight"): (
        "NLS (Northern Leaf Spot, Cochliobolus carbonum) must never map to "
        "'northern_leaf_blight' (Northern Leaf Blight, Exserohilum turcicum) — "
        "they are biologically distinct diseases."
    ),
    ("Southern Rust", "common_rust"): (
        "Southern Rust (Puccinia polysora) must never map to 'common_rust' "
        "(Puccinia sorghi) — they are biologically distinct diseases."
    ),
}


def validate_mapping(mapping: Mapping[str, str | None]) -> None:
    """Raise ValueError if *mapping* contains any illegal or unrecognised target.

    Call this on every external label mapping before it is used anywhere in the
    pipeline. A bad mapping that slips through corrupts evaluation silently;
    this function ensures it fails loudly instead.
    """
    for source, target in mapping.items():
        # Check explicitly forbidden source→target pairs
        if (source, target) in _FORBIDDEN:
            raise ValueError(_FORBIDDEN[(source, target)])

        # Check that non-None targets are canonical
        if target is not None and target not in CANONICAL_LABELS:
            raise ValueError(
                f"Label mapping error: '{source}' → '{target}' is not a canonical label. "
                f"Valid targets: {CANONICAL_LABELS} (or None to exclude)."
            )
