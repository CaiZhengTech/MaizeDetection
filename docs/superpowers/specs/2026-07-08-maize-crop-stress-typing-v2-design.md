# MaizeDetection V2 — Crop Stress Typing (Biotic vs. Abiotic) — Design Spec

**Date:** 2026-07-08
**Branch:** `dev2` (cut from up-to-date `main`)
**Status:** Design — awaiting user review before planning
**Predecessor:** V1 (in-domain 98.4% → field 61.7%; honest domain-gap measurement)

---

## 1. One-paragraph thesis

V1 proved a corn-leaf disease classifier looks great in the lab and struggles in the
field. V2 changes the *question the model answers*: instead of only "which disease?",
it first asks **"is this leaf stressed by a pathogen (biotic) or by a management/soil
problem — a nutrient deficiency (abiotic)?"**, then drills into the specific class.
This matters because the two need opposite responses (spray vs. feed the soil), and it
turns the tool into a **triage aid** that points toward root cause — complementing
soil-remediation work (e.g. Todd) rather than just flagging symptoms. The deliverable is
**not** a "solved" classifier; it is a **field-trustworthy, honestly-measured stress-typing
model whose limits are quantified**.

## 2. What V2 is (and is not)

**In scope:**
- A **two-level** maize stress classifier (see §3).
- **Field-parity data**: field *and* lab imagery on both the biotic and abiotic sides.
- Honest evaluation: per-level metrics, per-class false-negative rate, cross-source /
  leave-one-source-out, a **held-out external test set** never seen in training, and
  explicit **source-confound diagnostics**.
- Retraining on the pooled multi-source data (Colab GPU; local CPU inference preserved).

**Out of scope (deferred):**
- Water/temperature/other abiotic stresses (nutrient deficiency only in V2).
- Frontend / `serve.py` update — a **later milestone within V2**, after model + eval.
- Soil-data fusion, severity estimation, temporal monitoring, agentic advisory — **V3**
  (see §10).
- Multi-label (a leaf being diseased *and* deficient at once) — V3.
- Physical sensor / drone hardware — later.
- Multi-crop expansion — later.

**Framing rule (headline honesty):** V2 is presented as a **"triage aid whose limits are
measured,"** never as a solved biotic/abiotic classifier. The Level-1 gate result is
**suggestive, not proven** (see §7).

## 3. Taxonomy (two-level cascade)

```
Level 1 (gate):   healthy  |  biotic  |  abiotic
Level 2 (biotic):   common_rust | northern_leaf_blight | gray_leaf_spot
Level 2 (abiotic):  nitrogen | phosphorus | potassium | magnesium   (deficiency)
```

- `healthy` is terminal at Level 1.
- Rationale for the cascade (per user): don't score classes that aren't relevant —
  saves compute, and lets the biotic-vs-abiotic decision (the whole thesis) be
  **measured separately** from the fine-grained decision.
- Label integrity carries from V1: `validate_mapping()` hard-fails on biologically wrong
  merges (NLS≠NLB, Southern Rust≠Common Rust). Extended to abiotic: nutrient classes map
  only from datasets that explicitly induced/annotated that nutrient.

## 4. Data sources (the 2×2 grid)

| | **Lab / controlled** | **Field** |
|---|---|---|
| **Biotic** | PlantVillage — rust, NLB, GLS, healthy *(CC BY-SA — verify)* | CD&S (NLB, GLS) *+* **Iowa State *Puccinia sorghi*** (common rust + healthy) |
| **Abiotic** | **MPLD** — phosphorus only, severity levels *(partial)* | **Mendeley Maize Nutrient** — N/P/K/Mg + healthy |

**Verified provenance & credibility:**
- **PlantVillage** — established, widely cited; known background/class-correlation bias
  (documented since V1). Biotic-lab.
- **CD&S** (OSF/Purdue) — peer-published field iPhone photos; already used in V1. Field
  biotic (NLB, GLS only).
- **Iowa State *Puccinia sorghi*** (DOI 10.25380/iastate.23669328) — real **field RGB**
  common-rust + healthy maize leaves, pustule annotations. **License: CC BY-NC 4.0
  (non-commercial).** Closes the V1 common-rust field gap; see §8 commercialization note.
- **Mendeley Maize Nutrient Deficiency** (DOI 10.17632/34gb2gr7p2.1) — real smartphone
  field photos, expert-led surveys, India; N/P/K/Mg + healthy. **CC BY 4.0.** Single
  geography/device; per-class counts to verify at inspection.
- **MPLD** (Zenodo 10279042) — controlled maize, phosphorus induced by omission; 3,934
  images. **CC BY 4.0.** Abiotic-lab, phosphorus only.
- **Held-out external test candidates** (never trained on): Data-in-Brief nitrogen maize
  (Croatia, peer-reviewed, CC BY 4.0) for abiotic-N generalization; a disjoint slice of
  CD&S / a mixed-field set for biotic. Chosen so the final number reflects true transfer.

**Rejected / flagged:**
- Kaggle `smaranjitghose` corn-disease set — its Common Rust (1306) / GLS (574) counts
  match PlantVillage; it is **re-packaged lab data**, not field. Not used as field biotic.
- UPretoria "In-field" set — has field common rust but is **multi-label**; excluded from
  V2 to avoid multi-label complexity (V3 candidate).
- UAV/multispectral rust sets — different modality (not plain RGB); out of scope.

**Every image carries a `source` tag** (dataset + lab/field flag) through the whole
pipeline, so splits and metrics can be computed per-source.

## 5. Model

- **Backbone:** EfficientNet-B0, ImageNet weights, its own `.transforms()` (V1 invariant #4).
- **Cascade:** a Level-1 gate head (`healthy/biotic/abiotic`) → route to a biotic or
  abiotic Level-2 specialist head. Shared backbone to keep it small; heads are cheap.
- **Train it ourselves** (V1 invariant #3): no pre-finetuned plant classifier from a hub.
- Two-phase transfer learning as in V1 (freeze→fine-tune), fixed seed, best-by-val-loss,
  device-agnostic (Colab GPU train, CPU inference).

## 6. Data pipeline & integration methodology

- **leaf_id-grouped splits** (V1 cardinal rule) applied **within each source**; the
  common-rust filename-derived caveat carries forward where applicable.
- **Multi-source integration:** pool sources, split grouped *within* source, track `source`
  everywhere. Report **per-source** and **leave-one-source-out** metrics — never a single
  blended number that hides a source doing all the work.
- **Field-parity for the gate:** the Level-1 gate's headline evaluation runs on
  **field-vs-field** (and cross-source), so it cannot win by detecting lab-vs-field.
- **Background/leaf handling:** a background-masking (or leaf-crop) **ablation** — if gate
  accuracy collapses when backgrounds are removed, it was cheating on background, not
  biology. Report both.
- **External held-out test:** a dataset excluded from all training/validation, evaluated
  once, as the honest predictability number (user's requirement).

## 7. Invariants (carried + new)

1. **leaf_id leakage is the cardinal sin** — grouped splits + zero-overlap test. *(V1)*
2. **Never mis-map labels** — `validate_mapping()` hard-fails; non-overlapping classes
   excluded. *(V1, extended to nutrients)*
3. **Train the model ourselves** — ImageNet backbone + our heads. *(V1)*
4. **Preprocessing must match** — weights' own `.transforms()`. *(V1)*
5. **Report controlled vs. field together; lead with the gap; always show FN-rate.** *(V1)*
6. **NEW — the source confound must be measured and caveated, never hidden.** The gate is
   trained/evaluated to resist the "which dataset?" shortcut, and every gate report prints
   a `SOURCE_CONFOUND_CAVEAT` (in the spirit of V1's `COMMON_RUST_LEAKAGE_CAVEAT`).

## 8. Honest limitations (state these up front, not in a footnote)

1. **Source-confound ceiling.** Even field-vs-field, biotic (US/Purdue/Iowa) and abiotic
   (India) differ by geography/device/photographer. No public dataset provides
   biotic *and* abiotic from the *same* acquisition campaign, so a residual confound
   remains. **The gate is suggestive, not proven.**
2. **Symptom overlap.** Chlorosis, spotting, and wilting are shared by disease and
   deficiency; some leaves are undiagnosable from a single RGB image without soil/context.
   → The tool is **triage**, and its real power arrives with soil-data fusion (V3).
3. **Single-label simplification.** Real leaves can be diseased *and* deficient at once;
   V2 forces one label. Documented; multi-label is V3.
4. **Abiotic domain gap is only partial.** Controlled→field can be measured fully for
   biotic, but abiotic-lab is **phosphorus only** (MPLD). The abiotic gap is a phosphorus-
   scoped, clearly-labeled partial analysis, not a headline.
5. **Commercialization licensing.** The Iowa State common-rust set is **CC BY-NC**. Fine
   for this research/portfolio V2; a paid product needs a commercial license or a swap.

## 9. Evaluation plan (grounded in recent literature)

Recent work (2024–2025 plant-disease few-shot & domain-generalization papers) converges on:
categorize sources as **controlled / field / mixed**; report **per-level** hierarchical
metrics + hierarchical-F1; do **cross-dataset / leave-one-source-out**; add **confound
ablations**. V2 adopts all four, plus V1's FN-rate leadership.

- **Level-1 gate:** accuracy, per-class precision/recall/F1, confusion matrix, **FN-rate**;
  background-masking ablation; field-vs-field and leave-one-source-out.
- **Level-2 specialists:** per-class metrics within biotic and within abiotic.
- **End-to-end cascade:** hierarchical accuracy/F1 across the tree.
- **Domain gap:** biotic controlled→field (full, now incl. common rust); abiotic
  controlled→field (phosphorus-only, flagged).
- **External held-out test:** single honest transfer number.
- Every cross-domain report prints both sides + the source/leakage caveats.

## 10. Milestones (SDLC, one commit each, in order)

- **M1 — Taxonomy + label integrity.** Extend `labels.py` to the two-level tree + nutrient
  maps; extend `validate_mapping()`; tests. `chore:`
- **M2 — Data acquisition + inspection.** Download scripts for Iowa State rust, Mendeley
  nutrient, MPLD; verify counts/licenses; side-by-side lab/field inspection notebook;
  confirm `source` tags. **Checkpoint: show the human.** `data:`
- **M3 — Source-aware leak-safe splits + pipeline.** Grouped-within-source splits, `source`
  tags, background-mask transform; zero-overlap + field-parity tests. **Checkpoint.** `feat:`
- **M4 — Cascade model + training.** Gate + specialist heads; two-phase train. `feat:`
- **M5 — In-domain + per-level evaluation.** Hierarchical metrics, FN-rate, confound
  ablation. `eval:`
- **M6 — Cross-source + domain-gap + external test.** Leave-one-source-out, biotic/abiotic
  gaps, held-out external number, caveats. `eval:`
- **M7 (optional, last) — Frontend + `serve.py` update** to surface the two-level output.
  `feat:`

## 11. V3 vision (recorded now; not built in V2)

V2 is the trustworthy **eyes**; V3 adds **judgment**. Rough build order:

1. **Soil-data fusion (strongest).** Leaf image **+** soil data → resolves the biotic/abiotic
   ambiguity images alone can't. The point where this tool and soil-remediation work
   (Todd) fuse into something neither has alone.
2. **Severity, not just presence.** "How much rust" (Iowa State data supports this) →
   enables measuring change over time.
3. **Verification loop.** Track a field across a season: *did the intervention reduce
   stress?* — the monitoring-product core.
4. **Agentic advisory.** LLM agent turns {prediction + confidence + severity + soil context}
   into **grounded, hedged** advice that defers when unsure.
5. **Safety spine (under all of it):** confidence calibration + abstention — the tool says
   *"not sure, get a soil test"* instead of bluffing.
6. **North star:** multi-crop expansion + edge/drone capture.

**One-sentence north star:** a field-deployed, image-*and*-soil crop-stress monitor that
spots what's wrong, tracks whether fixes work, and gives grounded advice — maize first,
then across crops.

## 12. Open questions for review

- External held-out test set: lock the specific dataset(s) at M2 once counts are verified.
- Mendeley nutrient class balance unknown until download — may affect per-class reporting.
- Confirm PlantVillage color-config license for the commercialization ledger.
