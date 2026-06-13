"""
check_env.py — MaizeDetection environment sanity check

Run AFTER activating .venv:
    .venv\\Scripts\\activate       (Windows)
    python scripts/check_env.py

Windows quick-check (all three must point inside .venv):
    where python
    where pip
    python -c "import sys; print(sys.executable)"

If any import fails, install dependencies:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
"""

import sys
import importlib

DIVIDER = "=" * 58


def check_venv() -> bool:
    exe = sys.executable.replace("\\", "/")
    if ".venv/" not in exe:
        print(f"FAIL  interpreter is NOT inside .venv")
        print(f"      found: {exe}")
        print(f"      activate with: .venv\\Scripts\\activate")
        return False
    print(f"PASS  interpreter: {exe}")
    return True


def check_import(module_name: str, display_name: str) -> bool:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "(installed)")
        print(f"PASS  {display_name:<22} {version}")
        return True
    except ImportError as exc:
        print(f"FAIL  {display_name:<22} {exc}")
        return False


IMPORTS = [
    ("torch",        "torch"),
    ("torchvision",  "torchvision"),
    ("sklearn",      "scikit-learn"),
    ("PIL",          "Pillow"),
    ("matplotlib",   "matplotlib"),
    ("numpy",        "numpy"),
    ("datasets",     "datasets (HuggingFace)"),
    ("kagglehub",    "kagglehub"),
    ("yaml",         "pyyaml"),
    ("tqdm",         "tqdm"),
    ("nbformat",     "jupyter (nbformat)"),
]


def main() -> None:
    print(DIVIDER)
    print("check_env.py — MaizeDetection")
    print(DIVIDER)

    venv_ok = check_venv()
    print()

    failures = [
        display
        for module, display in IMPORTS
        if not check_import(module, display)
    ]

    print()
    try:
        import torch
        cuda = torch.cuda.is_available()
        status = "OK — CPU-only build" if not cuda else "WARNING — CUDA detected (local inference must still work on CPU)"
        print(f"INFO  torch.cuda.is_available() = {cuda}  [{status}]")
    except ImportError:
        pass

    print()
    print(DIVIDER)
    if not venv_ok or failures:
        if not venv_ok:
            print("FAIL  interpreter is outside .venv — activate first")
        if failures:
            print(f"FAIL  {len(failures)} missing package(s): {failures}")
        sys.exit(1)
    else:
        print("PASS  All checks passed — environment is ready.")
    print(DIVIDER)


if __name__ == "__main__":
    main()
