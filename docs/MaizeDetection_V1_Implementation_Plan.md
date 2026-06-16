# MaizeDetection — V1 Implementation Plan

**For:** Claude Code (implementation agent)
**Author context:** Solo developer, Windows, building an agricultural-tech CV portfolio project.
**Status:** Research and scoping complete. This document is the build spec for V1.

---

## 0. What this project is (and is not)

**Goal of V1:** Prove that a transfer-learning image classifier can distinguish four corn foliar conditions on *controlled* leaf images, then **honestly measure how badly (or well) that model generalizes to real field images** of the overlapping diseases.

The domain-gap measurement is the point of the project, not an afterthought. A published benchmark on this exact problem saw accuracy drop from **94% on controlled images to 55% on field images** for gray leaf spot. We expect to reproduce a gap of similar character and report it transparently.

**V1 is explicitly NOT:**
- A pesticide/treatment recommender
- A severity or yield-loss estimator
- A drone/edge/object-detection system
- A medical-style "diagnosis" tool
- A thin wrapper around a pre-finetuned HuggingFace classifier (we train the head ourselves)

**Do NOT build in V1** (these are later phases — do not scaffold them now): FastAPI, Docker, any database, cloud deployment, Weights & Biases, ONNX export, object detection, segmentation, or any frontend.

---

## 1. Canonical taxonomy (use these exact strings everywhere)

```
healthy
common_rust
northern_leaf_blight
gray_leaf_spot
```

**Hard rule on label integrity:** never silently merge a similarly-named but biologically different disease.
- Northern Leaf **Spot** (NLS, *Cochliobolus carbonum*) is NOT Northern Leaf **Blight** (NLB, *Exserohilum turcicum*). They must never be mapped together.
- Southern Rust (*Puccinia polysora*) is NOT Common Rust (*Puccinia sorghi*).
- If an external dataset has a class that does not map cleanly to the four canonical labels, it is **excluded from evaluation**, not forced.

---

## 2. Datasets — verified sources, mappings, and roles

### 2.1 PRIMARY TRAINING SET — PlantVillage (controlled / lab images)

- **Authoritative source:** HuggingFace `mohanty/PlantVillage`, config `"color"`. (Backup: GitHub `spMohanty/PlantVillage-Dataset`, `raw/color/`.)
- **Type:** Controlled — single detached leaves, uniform backgrounds.
- **Corn classes and exact original folder/label names → canonical mapping:**

| Original PlantVillage label | Canonical label | Approx count |
|---|---|---|
| `Corn_(maize)___healthy` | `healthy` | ~1,162 |
| `Corn_(maize)___Common_rust_` | `common_rust` | ~1,192 |
| `Corn_(maize)___Northern_Leaf_Blight` | `northern_leaf_blight` | ~985 |
| `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | `gray_leaf_spot` | ~513 |

Total ≈ 3,852 corn images.

- **CRITICAL — data leakage:** PlantVillage contains multiple photos of the *same physical leaf*. The HuggingFace version exposes a `leaf_id` (a.k.a. leaf-group) field. **All train/val/test splitting MUST group by `leaf_id`** so that no leaf appears in more than one split. A naive per-image random split will leak and produce inflated, meaningless metrics. This is the single most important correctness requirement in the project.

- **Known bias to document (not fix in V1):** in PlantVillage corn, backgrounds correlate with class (e.g., rust images tend to have one background tone, healthy another). The model may cheat on background. We note this in the report and treat it as motivation for the field-image evaluation.

### 2.2 PRIMARY EXTERNAL EVALUATION SET (V1) — CD&S (field images)

- **Source:** OSF project `https://osf.io/s6ru5/`, folder **`Dataset_Original`** (raw field JPGs only — ignore the augmented/annotated/severity folders for V1).
- **Type:** Field — handheld iPhone 11 Pro, 3000×3000, natural backgrounds, Purdue ACRE, July 2020.
- **Single disease per image** (clean for classification).
- **Raw counts:** NLB 511, GLS 524, NLS 562.
- **Overlap mapping (USE ONLY THESE TWO for V1 cross-domain eval):**

| CD&S class | Canonical label | Use in V1? |
|---|---|---|
| NLB | `northern_leaf_blight` | YES |
| GLS | `gray_leaf_spot` | YES |
| NLS | (no canonical match) | **EXCLUDE** |

- CD&S has **no healthy and no common_rust** class. That's fine — cross-domain eval is a 2-class subset (NLB, GLS). Report it as such.
- **Download size:** well under 1 GB. Confirm exact size at download time.

### 2.3 PARKED FOR V2 — UPretoria "Diseases of Maize in the Field" (IF)

Do **not** use in V1. Recorded here so V1 interfaces stay compatible with it later.
- **Why parked:** it is **multi-label** — only ~61% of images are single-disease, and Southern Rust has only 39 images. It needs multi-label handling, which is a V2 concern.
- When used in V2 it overlaps PlantVillage on **three** classes (GLS, NCLB→NLB, CR→common_rust), making it a strong harder external test.
- Smallest copy is the Kaggle 224×224 version (`hamishcrazeai/maize-in-field-dataset`); the Figshare original is 10.02 GB — do not download the Figshare version.

---

## 3. Environment (Windows)

- **Python:** 3.12 (3.11 also fine). Avoid 3.13+ for V1 to dodge wheel-availability surprises.
- **Env manager:** `venv` (built in; no Conda needed for CPU-local work).
- **GPU:** training may run on Google Colab or Kaggle (free GPU). **Local inference must run on CPU.**

### 3.1 Dependencies (pin minimally; let pip resolve patch versions)

```
torch
torchvision
scikit-learn
Pillow
matplotlib
numpy
jupyter
datasets          # HuggingFace, for PlantVillage + leaf_id metadata
kagglehub         # only if pulling any Kaggle data later
pyyaml            # config loading
tqdm              # progress bars
```

- **Augmentation:** use `torchvision.transforms.v2` (built in, zero extra deps, no license issues). Do **not** add `albumentations`/`albumentationsx` in V1 (the original is frozen; the X successor is AGPL/commercial).
- Install commands come from the official PyTorch "previous/get-started" selector for the CPU build on Windows. Generate them at build time rather than hard-coding a CUDA variant.

### 3.2 Environment sanity check (must pass before any training)

The agent should produce a small `scripts/check_env.py` that prints and asserts:
- `sys.executable` is inside the project `.venv`
- `torch.__version__`, `torchvision.__version__`, `torch.cuda.is_available()`
- a successful import of every dependency above

On Windows, also document the three-command check in the README: `where python`, `where pip`, `python -c "import sys; print(sys.executable)"` — all must point inside `.venv`.

---

## 4. Model

- **Framework:** PyTorch (dominant in research/industry; best ecosystem fit).
- **Architecture:** **EfficientNet-B0**, pretrained on ImageNet, from `torchvision.models`.
  - Use `EfficientNet_B0_Weights.IMAGENET1K_V1` (or current default) and that weight object's `.transforms()` for preprocessing so eval/inference preprocessing always matches training.
- **Transfer learning approach:** replace the final classifier head with a 4-class head.
  - Phase A: freeze backbone, train head only for a few epochs (sanity / fast convergence).
  - Phase B: unfreeze, fine-tune whole network at a lower LR.
- Do **not** run a multi-model comparison study in V1. If EfficientNet-B0 underperforms, suspect the data pipeline first.

---

## 5. Project structure (target layout)

```
maize-detection/
├── .gitignore
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── baseline.yaml          # paths, hyperparams, class list, seed
├── data/                      # .gitignored
│   ├── raw/
│   │   ├── plantvillage/
│   │   └── cds/               # CD&S Dataset_Original only
│   ├── processed/             # split manifests (CSV) — NOT copies of images if avoidable
│   └── external/
│       └── cds_overlap/       # NLB + GLS only
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_train_baseline.ipynb
│   ├── 03_evaluate_in_domain.ipynb
│   └── 04_evaluate_cross_domain.ipynb
├── src/maize_detection/
│   ├── __init__.py
│   ├── config.py              # load/validate baseline.yaml
│   ├── labels.py              # canonical labels + mapping dicts + validation
│   ├── data.py                # Dataset, transforms, leaf_id-aware splitter, loaders
│   ├── model.py               # build_model(), freeze/unfreeze helpers
│   ├── train.py               # training loop, checkpointing, seed control
│   ├── evaluate.py            # metrics, confusion matrix, per-class FN rate, plots
│   ├── predict.py             # load checkpoint + predict(image_path) on CPU
│   └── utils.py               # seeding, device, IO, logging
├── scripts/
│   ├── check_env.py
│   ├── download_plantvillage.py
│   └── download_cds.py        # documents OSF steps; never hardcodes credentials
├── outputs/                   # .gitignored: checkpoints, figures, metrics json
└── tests/
    ├── test_labels.py         # mapping integrity (NLS != NLB, etc.)
    └── test_data.py           # split has zero leaf_id overlap; shapes; class coverage
```

**`.gitignore` must include:** `data/`, `outputs/`, `.venv/`, `__pycache__/`, `*.pt`, `*.pth`, `.kaggle/`, `*.ckpt`, `.ipynb_checkpoints/`.

**Secrets rule:** never hardcode Kaggle/HF credentials. Read from standard token locations / env vars only. Never print tokens. Keep `kaggle.json` out of the repo.

---

## 6. Build milestones (each ends in a working artifact + a git commit)

> Work strictly in order. Do not skip ahead to later milestones. Show the human the result of each milestone before proceeding past a major checkpoint (after M2 data inspection, and after M3 split verification).

### M1 — Scaffold + environment
- Create the folder structure, `.gitignore`, `requirements.txt`, `pyproject.toml`, `configs/baseline.yaml`, and `scripts/check_env.py`.
- Implement `labels.py` with the canonical list + PlantVillage→canonical and CD&S→canonical maps, plus a `validate_mapping()` that hard-fails if anyone tries to map NLS→NLB.
- **Commit:** `chore: project scaffold, env check, and label taxonomy`

### M2 — Data download + inspection
- `download_plantvillage.py`: pull `mohanty/PlantVillage` (color), filter to the 4 corn classes, retain `leaf_id`.
- `download_cds.py`: document and (where possible) automate fetching CD&S `Dataset_Original` from OSF; verify NLB/GLS/NLS counts.
- `01_data_inspection.ipynb`: class distribution, sample grids per class for BOTH datasets side by side (controlled vs field), image-size stats, corrupt-file check, confirm `leaf_id` present and populated.
- **Checkpoint: pause and show the human the distributions + sample grids.**
- **Commit:** `data: download scripts and inspection notebook`

### M3 — Leak-safe splits + data pipeline
- `data.py`: implement a **GroupShuffleSplit / group-aware stratified split keyed on `leaf_id`**, producing train/val/test ≈ 70/15/15 as **CSV manifests** (image path, canonical label, leaf_id, split).
- Build `Dataset` + `DataLoader` using the pretrained weights' `.transforms()` for eval and a light train-time augmentation (resize/crop/flip/normalize via `transforms.v2`).
- `tests/test_data.py`: assert **zero leaf_id intersection** across splits; assert all 4 classes present in each split; assert tensor shapes/labels.
- **Checkpoint: show the human the split summary + the passing leakage test.**
- **Commit:** `feat: leaf_id-aware splits and CPU/GPU data pipeline`

### M4 — Train baseline
- `model.py` + `train.py`: EfficientNet-B0, 4-class head; Phase A (frozen) then Phase B (fine-tune); fixed seed; checkpoint best-by-val-loss to `outputs/checkpoints/`.
- Keep it small: this is ~50 lines of real logic. Log per-epoch train/val loss + accuracy.
- **Commit:** `feat: EfficientNet-B0 transfer-learning baseline`

### M5 — In-domain evaluation (controlled test set)
- `evaluate.py` + `03_evaluate_in_domain.ipynb`: on the held-out PlantVillage test split report **accuracy, per-class precision/recall/F1, macro-F1, confusion matrix, and per-class false-negative rate** (FN rate matters most for disease detection). Save figures + a `metrics_in_domain.json`.
- **Commit:** `eval: in-domain metrics, confusion matrix, per-class FN rate`

### M6 — Cross-domain evaluation (CD&S field images)
- Build `cds_overlap/` = CD&S NLB + GLS only.
- Run the same trained model, but **restrict the decision/report to the 2 overlapping classes** (document exactly how: e.g., evaluate only on NLB/GLS images and report the 2-class confusion among the 4 logits, noting any leakage into common_rust/healthy as an error mode).
- `04_evaluate_cross_domain.ipynb`: report controlled-vs-field side by side; state the domain gap honestly; compare narratively to the published 94%→55% GLS benchmark.
- **Commit:** `eval: cross-domain field evaluation on CD&S (NLB, GLS)`

### M7 — Reusable CPU inference function
- `predict.py`: `predict(image_path) -> {label, confidence, probabilities}`; loads checkpoint, applies the weights' transforms, runs on CPU; include a tiny demo cell/script.
- **Commit:** `feat: reusable CPU inference function`

**Estimated effort:** ~6–8 focused days for a motivated solo dev.

---

## 7. Evaluation methodology (do this correctly — it's the project's whole value)

1. **In-domain:** stratified, **leaf_id-grouped** train/val/test on PlantVillage. Metrics on the untouched test split.
2. **Cross-domain:** evaluate the *same* model on CD&S field images, **only** for classes that genuinely overlap (NLB, GLS).
3. **Leakage prevention:** group by leaf_id (in-domain); the external set is a different dataset entirely (cross-domain), so no leakage there by construction — but never let any CD&S image touch training.
4. **Honesty in reporting:** lead with the gap, not the headline in-domain number. Expect high in-domain accuracy (controlled images are easy) and a meaningful drop on field images. Report per-class FN rates. Name confounds (background bias, GLS class imbalance ~513 vs ~1,192).

---

## 8. Risks the agent should actively guard against

| Risk | Guard |
|---|---|
| **leaf_id leakage** inflating metrics | Group-aware split + a test that asserts zero overlap. Highest priority. |
| **CD&S download friction** (OSF may need a free account / JS UI) | Document manual steps in `download_cds.py`; verify counts post-download; if blocked, surface to the human rather than silently substituting another dataset. |
| **Inflated in-domain accuracy mistaken for success** | Frame M6 as the real test; always report controlled vs field together. |
| **GLS class imbalance** (~513 vs ~1,192) | Stratified splits; report per-class metrics, watch GLS recall; do NOT add synthetic oversampling in V1. |
| **Forcing non-overlapping labels** (NLS→NLB) | `validate_mapping()` hard-fails; NLS excluded. |
| **Accidental scope creep** (FastAPI/Docker/etc.) | None of those are in V1. Reject if tempted. |
| **Preprocessing mismatch** train vs inference | Always use the pretrained weights' own `.transforms()` for eval/inference. |

---

## 9. V2+ backlog (do NOT build now — listed only to keep V1 interfaces compatible)

1. UPretoria IF as a **multi-label** harder external test (adds 3-class overlap incl. rust).
2. Add CD&S as a *second* external set once UPretoria is in (two continents, two imaging setups).
3. Wrap `predict()` in **FastAPI**; then a minimal dashboard.
4. Docker, cloud deploy, DB storage, experiment tracking.
5. Object detection / lesion localization; edge-camera experiments.

Each V2 item must ship as its own working, demonstrable release.

---

## 10. First actions for Claude Code

1. Confirm Python 3.12 `venv` on Windows; generate the correct **CPU** PyTorch install command from the official selector; write `requirements.txt`.
2. Scaffold the repo per section 5; implement `labels.py` + `validate_mapping()` + `tests/test_labels.py`.
3. Implement `scripts/check_env.py` and confirm it passes.
4. Stop and report back before downloading data, so the human can run the env check on their actual Windows machine.

**Do not** generate a giant finished codebase in one shot. Build milestone by milestone, prefer small focused modules, schemas, and tests, and pause at the M2 and M3 checkpoints.
