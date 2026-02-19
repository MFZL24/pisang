# ===============================
# IMPORT
# ===============================
import io
import os
import json
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F


# ===============================
# CONFIG
# ===============================
MODEL_PATH = "models/ssd_mobilenet_banana.pth"
NUM_CLASSES = 4  # background + 3 kelas
CONFIDENCE_THRESHOLD = 0.5


# ===============================
# INIT FLASK
# ===============================
app = Flask(__name__)


# ===============================
# DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ===============================
# LOAD LABEL MAP FROM COCO
# ===============================
def load_label_map():
    annotation_path = "_annotations.coco.json"

    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as f:
            coco_data = json.load(f)

        mapping = {}
        for cat in coco_data["categories"]:
            mapping[cat["id"]] = cat["name"]

        print("✅ COCO Label Map Loaded:", mapping)
        return mapping

    # fallback jika json tidak ada
    print("⚠️ COCO file tidak ditemukan. Menggunakan default label.")
    return {
    1: "overripe",
    2: "ripe",
    3: "unripe"
}



LABEL_MAP_RAW = load_label_map()


# ===============================
# TRANSLATE LABEL KE INDONESIA
# ===============================
def translate_label(raw_label):
    mapping = {
        "ripe": "Pisang Matang",
        "unripe": "Pisang Mentah",
        "overripe": "Pisang Terlalu Matang",
        "banana": "Pisang"
    }
    return mapping.get(raw_label, raw_label)


# ===============================
# LOAD MODEL
# ===============================
def load_model():
    model = ssdlite320_mobilenet_v3_large(pretrained=True)

    # Sesuaikan jumlah class
    model.head.classification_head.num_classes = NUM_CLASSES

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


model = load_model()
print("✅ Model Loaded Correctly")


# ===============================
# ROUTE HOME
# ===============================
@app.route("/")
def index():
    return render_template("index.html")


# ===============================
# DETECTION ROUTE
# ===============================
@app.route("/predict-detection", methods=["POST"])
def predict_detection():

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    try:
        # ===============================
        # READ IMAGE
        # ===============================
        image_bytes = file.read()

        if len(image_bytes) == 0:
            return jsonify({"success": False, "error": "Empty file"}), 400

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Convert to tensor
        img_tensor = F.to_tensor(img).to(device)

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        # ===============================
        # INFERENCE
        # ===============================
        with torch.no_grad():
            outputs = model(img_tensor)[0]

        detections = []

        for box, score, label in zip(
            outputs["boxes"],
            outputs["scores"],
            outputs["labels"]
        ):
            if score >= CONFIDENCE_THRESHOLD:

                label_id = int(label.cpu().numpy())

                # Skip background
                if label_id == 0:
                    continue

                x1, y1, x2, y2 = box.cpu().numpy().tolist()

                # Get raw label from COCO
                raw_label = LABEL_MAP_RAW.get(label_id, f"class_{label_id}")

                # Translate ke Bahasa Indonesia
                label_name = translate_label(raw_label)

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "score": float(score.cpu().numpy()),
                    "label": label_name,
                    "class_id": label_id
                })

        # ===============================
        # RESPONSE
        # ===============================
        if len(detections) == 0:
            return jsonify({
                "success": False,
                "detections": [],
                "message": "No detections"
            })

        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections)
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
