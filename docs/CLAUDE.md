# CLAUDE.md — MaizeDetection

Standing instructions for any Claude Code session in this repo. Read this first, every session.

---

## What this project is

A corn-leaf foliar-disease **image classifier** built to (1) train a transfer-learning baseline on *controlled* PlantVillage images across four classes, and (2) **honestly measure the domain gap** when that model is evaluated on *field* images. The domain-gap measurement is the point — not a high accuracy number.

Full build spec lives in `MaizeDetection_V1_Implementation_Plan.md`. This file is the short, always-on rulebook. If the two ever conflict, the plan wins on *what* to build; this file wins on *how* to behave.

---

## Non-negotiable invariants

1. **leaf_id leakage is the cardinal sin.** Every train/val/test split MUST be grouped by `leaf_id` so no physical leaf appears in two splits. A test asserting **zero `leaf_id` intersection across splits** must exist and pass before any training is considered valid. If you can't guarantee it, stop and say so.

2. **Never mis-map labels.** Canonical labels are exactly: `healthy`, `common_rust`, `northern_leaf_blight`, `gray_leaf_spot`. Northern Leaf **Spot** (NLS) ≠ Northern Leaf **Blight** (NLB). Southern Rust ≠ Common Rust. Non-overlapping external classes are **excluded**, never force-fit. `validate_mapping()` must hard-fail on a bad mapping.

3. **Train the model ourselves.** Use a pretrained EfficientNet-B0 *backbone* (ImageNet weights) + our own head + our own pipeline. Do NOT drop in a pre-finetuned PlantVillage classifier from the Hub.

4. **Preprocessing must match.** Use the pretrained weights' own `.transforms()` for eval/inference so train- and inference-time preprocessing never drift.

5. **Report controlled vs field together.** Never present the in-domain number alone. Lead with the gap. Always include per-class false-negative rate.

---

## Scope fence (V1)

**In scope:** data download/inspection, leak-safe splits, EfficientNet-B0 transfer baseline, in-domain eval, CD&S cross-domain eval (NLB + GLS only), CPU inference function.

**Out of scope — do NOT build or scaffold:** FastAPI, Docker, any database, cloud deploy, Weights & Biases, ONNX, object detection, segmentation, any frontend, UPretoria/multi-label work. These are V2+. If a task tempts you toward these, stop and flag it.

This project is **not** a treatment recommender, severity/yield estimator, or medical-style diagnosis tool. Don't add such outputs.

---

## How to work

- **Milestone by milestone, in order** (see plan §6). Do not jump ahead.
- **Pause for the human** after M2 (data inspection) and M3 (split verification). Show results; wait.
- **One commit per milestone**, with the message prefixes given in the plan (`chore:`, `data:`, `feat:`, `eval:`).
- **Small and focused** beats large and complete. Prefer schemas, interfaces, focused modules, and tests over giant dumps of code. Do not generate the whole codebase in one shot.
- **Surface friction, don't paper over it.** If a dataset download is blocked (e.g., CD&S on OSF needs a free account), tell the human — never silently substitute a different dataset.

---

## Environment

- Windows. Python 3.12, `venv`. Avoid 3.13+ for V1.
- Training may use Colab/Kaggle GPU; **local inference must run on CPU**.
- Augmentation via `torchvision.transforms.v2` only. No albumentations in V1.
- Generate the **CPU** PyTorch install command from the official selector; don't hardcode a CUDA build.
- Run `scripts/check_env.py` before training; it must confirm the interpreter is inside `.venv` and all deps import.

---

## Secrets & hygiene

- Never hardcode or print Kaggle/HuggingFace credentials. Read from standard token files / env vars only.
- `.gitignore` must exclude: `data/`, `outputs/`, `.venv/`, `__pycache__/`, `*.pt`, `*.pth`, `*.ckpt`, `.kaggle/`, `.ipynb_checkpoints/`.
- Never commit datasets, checkpoints, or tokens.

---

## Definition of done (per milestone)

A milestone is done when: the artifact runs, its test(s) pass, the result is shown to the human where a checkpoint is required, and exactly one well-formed commit has been made. In-domain results are only "valid" once the leaf_id leakage test passes.
