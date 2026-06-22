# MaizeDetection — Project Overview & Interview Explainer

A study-and-speak-from guide to everything that was built and *why*. Organized the
way an interview tends to flow: the pitch, the "why," the technical depth, then the
questions you'll likely get.

---

## 1. The 30-second pitch

> "I built a corn-leaf disease image classifier, but the actual point of the
> project wasn't the accuracy — it was *honesty about generalization*. I trained an
> EfficientNet-B0 on controlled lab images and got 98.4% accuracy, then evaluated
> the exact same model on real field photos and watched it drop to 62%. The project
> is an end-to-end, leak-safe pipeline whose deliverable is a transparently measured
> **domain gap**."

Why this is a *good* portfolio piece: most ML demos report one inflated number.
This one is built around the failure mode that bites real deployments — a model
that looks perfect in the lab and falls apart in the field. Being able to *predict,
measure, and explain* that is a more senior skill than chasing accuracy.

---

## 2. The core thesis: the domain gap

Two image distributions:

- **Controlled (PlantVillage):** single detached leaves, uniform backgrounds, even
  lighting. Easy.
- **Field (CD&S):** handheld iPhone photos, natural clutter, variable lighting,
  whole-plant scenes. Hard.

A known published benchmark on this exact problem saw accuracy fall from ~94%
(controlled) to ~55% (field) for gray leaf spot. **My job was to reproduce a gap of
that character and report it without spin.** I did — same shape of result.

**Why the gap exists (say this crisply):** the controlled images have backgrounds
that correlate with the disease class. The model learns *shortcuts* — background
tone, framing, lighting — that don't exist in field images. So it's not really
learning "what the lesion looks like"; it's partly learning "what a PlantVillage
rust photo looks like." Field images strip those shortcuts away and expose how much
was real signal vs. dataset artifact.

---

## 3. The pipeline, module by module

Built in seven milestones (M1–M7), one commit each, in strict order.

### Labels & taxonomy (`src/maize_detection/labels.py`)

Four canonical classes: `healthy`, `common_rust`, `northern_leaf_blight`,
`gray_leaf_spot`. The non-obvious part is **`validate_mapping()`**, which *hard-fails*
on biologically wrong mappings.

**Talking point:** Northern Leaf **Spot** (NLS) and Northern Leaf **Blight** (NLB)
are different diseases caused by different pathogens, but the names are one word
apart. A naive person merges them and silently corrupts their labels. CD&S has an
NLS class — I **excluded** it rather than force-fitting it to NLB. The validator
encodes forbidden pairs (NLS→NLB, Southern Rust→Common Rust) so the mistake is
impossible to make accidentally. *This is "I take label integrity seriously" in
code form.*

### Data pipeline & the leakage problem (`src/maize_detection/data.py`)

The single most important correctness piece, and the best thing to talk about.

**The problem:** PlantVillage contains *multiple photos of the same physical leaf*.
A naive random split puts the same leaf in both train and test, the model memorizes
that leaf, and your test accuracy is inflated and meaningless. This is **data
leakage**, the cardinal sin of the project.

**The fix:** every split is **grouped by `leaf_id`** so all photos of one leaf stay
in the same split. I used scikit-learn's `GroupShuffleSplit`, run **two-stage,
per-class**:
1. Carve off the test leaves (15%).
2. Split the remainder into train/val (rescaling val to 15/85 of what's left).

Doing it *per class* makes it a **stratified group split** — each class keeps its
~70/15/15 proportion *and* stays leaf-disjoint. Ratios apply to **leaves, not
images**.

**The honest caveat you must mention:** corn has no official `leaf_id` in the source
data. I *derive* it from the filename suffix. That grouping works for healthy/NLB/GLS,
but **common_rust filenames are all unique** — every rust image looks like its own
leaf. So for rust specifically, a same-leaf leak *can't be ruled out*. I exported
this as a constant, `COMMON_RUST_LEAKAGE_CAVEAT`, that gets **printed on every
evaluation run** so the report can never quietly omit it. *This is the detail that
signals real rigor — I documented the weakness in my own anti-leakage measure
instead of hiding it.*

**A test enforces zero leaf_id overlap across all three splits.** No training result
counts as valid until that test passes.

Design note: `MaizeDataset` is a plain map-style class (`__len__`/`__getitem__`)
that *doesn't import torch at module scope*. That keeps the split logic — and its
tests — torch-free and fast. PyTorch's DataLoader accepts any object with that
protocol.

### Model (`src/maize_detection/model.py`)

EfficientNet-B0 with **ImageNet weights** — the *generic* feature extractor, **not**
a pre-finetuned plant-disease model from a model hub. I replace the final layer
(`classifier[1]`, the `Linear`) with my own 4-class head, keeping the dropout.
`freeze_backbone()` and `unfreeze_all()` toggle `requires_grad`.

**Talking point:** the invariant is "train it yourself." Anyone can download a
finished PlantVillage classifier. The point was to demonstrate the *transfer-learning
workflow* — take generic ImageNet features, attach my own head, adapt them with my
own pipeline.

### Training (`src/maize_detection/train.py`)

**Two-phase transfer learning:**
- **Phase A:** freeze the backbone, train only the new head (fast, stable — the head
  learns the 4 classes before we disturb the pretrained features). LR 1e-3, 5 epochs.
- **Phase B:** unfreeze everything, fine-tune end-to-end at a *lower* LR (1e-4, 15
  epochs) so we nudge the features without destroying them.

Adam optimizes only `requires_grad=True` params (same code works frozen or
unfrozen), CrossEntropyLoss, and I checkpoint **best-by-validation-loss** carried
*across both phases*. Fixed seed for reproducibility.

**Practical detail:** a full run is impractical on CPU, so training happens on a
Colab GPU (notebook 02), and the resulting `best.pt` comes back to the local machine
for CPU evaluation. I refactored the loop into a `run_training()` function so the CLI
and the notebook call the exact same code — no logic duplicated in the notebook.

### In-domain evaluation (`src/maize_detection/evaluate.py`, M5)

On the held-out controlled test split: accuracy, per-class precision/recall/F1,
macro-F1, confusion matrix, and the metric I lead with — **per-class false-negative
rate (FN-rate = 1 − recall)**.

**Talking point:** for *disease detection*, a false negative (miss a sick leaf) is
worse than a false positive (false alarm). So recall/FN-rate matters more than raw
accuracy. I separated `compute_metrics()` as a **pure function** (numpy in, dict out,
no torch, no file I/O) specifically so it's unit-testable — and I tested it on
hand-built cases, including verifying FN-rate = 1 − recall by hand.

**Result:** 98.4% accuracy, 0.978 macro-F1. Only real errors are NLB↔GLS confusion
(both are lesion diseases — visually similar).

### Cross-domain evaluation (`src/maize_detection/cross_evaluate.py`, M6) — the centerpiece

The same model, the same preprocessing, run on 1,020 CD&S field images (NLB + GLS
only). I report the **same forward pass two ways** — the cleverest part of the design:

1. **Open 4-way:** the model is free to predict any of its 4 classes. Because CD&S
   only contains NLB and GLS, *any prediction landing in healthy or common_rust is a
   leakage error mode I count explicitly.* This is what deployment actually looks like.
2. **Restricted 2-class:** argmax over *only* the NLB and GLS logits — "forget the
   other two classes exist." This isolates the genuine NLB-vs-GLS signal that survives
   the domain shift.

**Why both?** The open view exposes the dangerous real-world behavior; the restricted
view answers "can it even tell the two field diseases apart?" Reporting only one
would be misleading.

**Results:**
- Restricted 2-class accuracy: **0.617**. Open 4-way accuracy: **0.465**.
- **26.2% of diseased field leaves leaked into "healthy"** (264 of them) — the most
  dangerous failure for a detection tool.
- NLB recall: 0.957 → 0.563. GLS recall: 0.973 → 0.667.

`compute_domain_gap()` produces the controlled-vs-field table directly. Critically:
**invariant #5 — never present the in-domain number alone.** Every cross-domain
report prints both sides plus the caveats.

### Inference (`src/maize_detection/predict.py`, M7)

`predict(image_path) → {label, confidence, probabilities}`, on CPU, with the model
and transform cached (`lru_cache`) so repeated calls don't reload the 16 MB
checkpoint. It uses the **pretrained weights' own `.transforms()`** so inference
preprocessing can never drift from training (invariant #4 — a subtle bug source in a
lot of real systems). The CLI prints the field caveat so nobody trusts a high
confidence on a field photo.

---

## 4. The cross-cutting invariants (your rigor story)

Five rules enforced throughout. Memorize them — they're your "how I think about
correctness" answer:

1. **leaf_id leakage is the cardinal sin** → grouped splits + a zero-overlap test.
2. **Never mis-map labels** → `validate_mapping()` hard-fails; NLS excluded.
3. **Train the model myself** → ImageNet backbone only, my own head.
4. **Preprocessing must match** → eval/inference use the weights' own `.transforms()`.
5. **Report controlled vs field together** → lead with the gap, always show FN-rate.

---

## 5. Engineering practices worth name-dropping

- **40 passing tests**, including the leakage test, label-integrity tests,
  pure-metric tests, and `predict()` output-contract tests that skip cleanly when
  assets (checkpoint/images) are absent — so a fresh clone still passes.
- **Pure functions separated from I/O** (`compute_metrics`, `compute_cross_metrics`)
  so the math is testable without torch or files.
- **One commit per milestone** with conventional prefixes (`feat:`, `eval:`, `data:`).
- **Reproducibility:** fixed seed, config-driven hyperparameters (`baseline.yaml`),
  CPU/GPU device-agnostic code.
- **Secrets hygiene:** no hardcoded tokens; datasets/checkpoints gitignored.
- **Honest scope fence:** explicitly did *not* build FastAPI/Docker/etc. — those are
  V2. Knowing what *not* to build is a signal too.
- **Web frontend** (React 19 + Tailwind CSS v4 + Vite 6): drag-and-drop classifier
  UI with a sample image gallery, domain-gap visualization showing the actual M5/M6
  numbers, dark mode, and interactive class info dialogs. Currently uses a mock
  prediction — ready to connect to a real backend.

---

## 6. Likely interview questions + crisp answers

**"What was the hardest part?"**
The leaf_id leakage. The subtle version: corn has no real leaf_id, so I derived one
from filenames, and that derivation *fails for common_rust*. The mature move wasn't
to hide it — it was to document the residual risk as a first-class, always-printed
caveat and treat that class's metrics with skepticism.

**"Why EfficientNet-B0?"**
Strong accuracy-per-parameter, runs fine on CPU for inference, well-supported in
torchvision with pretrained weights and matching transforms. The spec also said
*not* to run a multi-model bake-off — if the model underperforms, suspect the data
pipeline first. (Good instinct to voice: data > model.)

**"Why is in-domain accuracy not the real result?"**
Controlled images are easy and background-correlated; high accuracy there is partly
the model exploiting dataset artifacts. The field evaluation is the honest test of
whether it learned the disease or the dataset.

**"How would you close the gap?"** (forward-looking)
Domain adaptation / fine-tuning on a small labeled field set; aggressive augmentation
that simulates field conditions (background, lighting, scale); test-time
augmentation; or collecting field training data. But V1's job was to *measure* the
gap rigorously, not paper over it.

**"What's the FN-rate and why do you lead with it?"**
False-negative rate = 1 − recall = the fraction of truly-diseased leaves the model
misses. For disease detection, a miss is costlier than a false alarm, so it maps to
real-world harm. On field images, the open-view NLB FN-rate is 0.64 — the model
misses most field NLB, often calling it healthy.

**"Two evaluation views — why?"**
Open 4-way shows true deployment behavior (and exposes the leak-to-healthy failure);
restricted 2-class isolates the surviving discriminative signal. One number alone
would either overstate or hide the problem.

**"Why did you build a frontend for a research classifier?"**
The domain-gap numbers are the project's centerpiece but they're dry in a table. The
web UI lets someone drag a leaf image and get a prediction while seeing the 98.4% vs
61.7% gap visualized right on the page — it makes the "don't trust this in the field"
message visceral instead of academic. It also shows I can ship a polished end-to-end
product, not just a notebook.

---

## 7. Numbers cheat-sheet (memorize these)

| Metric | Value |
|---|---|
| In-domain accuracy (4-class) | **0.984** |
| In-domain macro-F1 | 0.978 |
| Field accuracy, restricted 2-class | **0.617** |
| Field accuracy, open 4-way (on overlap) | 0.465 |
| Leakage to non-field classes (open) | **26.2%** (264 → healthy, 3 → common_rust) |
| NLB recall: controlled → field (2-class) | 0.957 → 0.563 |
| GLS recall: controlled → field (2-class) | 0.973 → 0.667 |
| Published benchmark (GLS) | ~94% → ~55% |
| Test count | 40 passing |
| Datasets | PlantVillage (train/in-domain), CD&S (field eval, NLB+GLS) |
| Frontend | React 19, Tailwind CSS v4, Vite 6 |

*Full per-class tables and confusion matrices live in `README.md` and the generated
`outputs/metrics_*.json`.*
