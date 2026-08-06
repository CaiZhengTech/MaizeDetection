"""Windowless launcher for serve.py, used by the logon-triggered Scheduled Task.

Task Scheduler runs this under ``pythonw.exe`` so no console window appears. In
that context ``sys.stdout``/``sys.stderr`` are ``None``, and serve.py's startup
prints -- plus Werkzeug's per-request log -- would raise on the missing
``.write``. So point both at ``logs/serve.log`` first, then hand off to serve.py
unchanged.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "serve.log"

LOG.parent.mkdir(exist_ok=True)
sys.stdout = sys.stderr = open(LOG, "a", buffering=1, encoding="utf-8")

os.chdir(ROOT)  # serve.py imports src.maize_detection relative to CWD
sys.path.insert(0, str(ROOT))
sys.argv = ["serve.py"]  # keep serve.py's argparse from seeing this launcher's path

runpy.run_path(str(ROOT / "serve.py"), run_name="__main__")
