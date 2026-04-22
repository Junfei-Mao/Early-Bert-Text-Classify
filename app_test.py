"""Minimal Flask API for the text classification model."""

import os

from flask import Flask, jsonify, request

from predict import TextClassifierPredictor

MODEL_DIR = os.getenv("MODEL_DIR", "./output")
DEVICE = os.getenv("DEVICE")

predictor = TextClassifierPredictor(model_dir=MODEL_DIR, device=DEVICE)
app = Flask(__name__)


@app.route("/api/get", methods=["GET"])
@app.route("/api/predict", methods=["GET"])
def predict_endpoint():
    request_id = (request.args.get("id") or "").replace('"', "")
    title = request.args.get("title", "")
    content = request.args.get("content", "")

    if not content:
        return jsonify({"error": "`content` is required"}), 400

    result = predictor.predict(title=title, content=content)
    payload = {
        "id": request_id,
        "label": result["label"],
        "label_id": result["label_id"],
        "confidence": result["confidence"],
    }
    return jsonify(payload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
