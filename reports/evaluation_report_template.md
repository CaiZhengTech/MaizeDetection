# MaizeDetection — Evaluation Report

> Template. M5 fills the in-domain section; M6 fills the cross-domain section.
> **Lead with the domain gap, not the in-domain headline number.**
> Every section marked **REQUIRED** must be present in the final report.

---

## 1. Summary — the domain gap (REQUIRED, lead with this)

- In-domain (controlled PlantVillage) macro-F1: `<fill M5>`
- Cross-domain (CD&S field, NLB+GLS) macro-F1: `<fill M6>`
- **Gap:** `<controlled − field>`. Narrative comparison to the published 94%→55% GLS benchmark: `<fill>`

## 2. In-domain results — controlled test split (REQUIRED, M5)

- Accuracy, per-class precision / recall / F1, macro-F1: `<fill>`
- Confusion matrix: `<figure>`
- **Per-class false-negative rate** (matters most for disease detection): `<fill>`

## 3. Cross-domain results — CD&S field, NLB + GLS only (REQUIRED, M6)

- 2-class evaluation among the 4 logits; leakage into healthy/common_rust noted as an error mode: `<fill>`
- Per-class FN rate on field images: `<fill>`

## 4. Known confounds & caveats (REQUIRED)

### 4.1 common_rust leaf_id grouping — residual leakage risk (REQUIRED — do NOT omit)

Corn has no official `leaf_id`; it is derived from PlantVillage filenames. The
derivation groups multi-shot images for healthy, NLB, and GLS, **but not for
common_rust**, where every image is its own leaf (~1,192 images / ~1,192 leaves,
1.00 imgs/leaf).

> Canonical wording is exported from code as
> `src.maize_detection.data.COMMON_RUST_LEAKAGE_CAVEAT` — import and emit it
> verbatim so this section can never be silently dropped:
>
> *"common_rust has no detectable leaf grouping: the filename-derived leaf_id
> yields exactly one image per leaf (~1,192 images / ~1,192 leaves). Unlike
> healthy/NLB/GLS, multi-shot grouping cannot be recovered for rust, so a
> same-leaf train/test leak is possible for common_rust only and cannot be ruled
> out. Interpret common_rust in-domain recall with this residual, unverifiable
> leakage risk in mind."*

**Implication:** treat common_rust in-domain recall with skepticism; it may be
optimistically inflated relative to the other three classes.

### 4.2 Other confounds (REQUIRED)

- **Background bias:** PlantVillage backgrounds correlate with class; the model may cheat on background. `<discuss>`
- **GLS class imbalance:** ~513 vs ~1,192; watch GLS recall. `<discuss>`
- **Controlled vs field domain shift:** single detached leaves vs natural field backgrounds. `<discuss>`
