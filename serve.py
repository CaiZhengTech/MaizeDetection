"""Minimal Flask server for the web frontend (development only).

Serves POST /predict — accepts a multipart image upload, runs the
real EfficientNet-B0 model on CPU, and returns the prediction as JSON.

Usage:
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    python serve.py                  # → http://localhost:5000
    python serve.py --port 8000      # custom port
"""

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
_predict_fn = None


def _get_predict():
    global _predict_fn
    if _predict_fn is None:
        from src.maize_detection.predict import predict
        _predict_fn = predict
    return _predict_fn


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not _allowed(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    tmp_path = Path(tempfile.mkdtemp()) / "upload.jpg"
    try:
        file.save(str(tmp_path))
        result = _get_predict()(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)
        tmp_path.parent.rmdir()

    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaizeDetection dev server")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    print(f"Starting MaizeDetection API on http://localhost:{args.port}")
    print("NOTE: This is a development server. Do not use in production.")
    app.run(host="127.0.0.1", port=args.port, debug=False)
