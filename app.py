# ===============================
# IMPORT
# ===============================
import io
import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F
from skimage.feature import hog
from joblib import load


# ===============================
# CONFIG DETECTION
# ===============================
MODEL_PATH = "models/ssd_mobilenet_banana.pth"
NUM_CLASSES = 4
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
# LOAD LABEL MAP DETECTION
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

    return {
        1: "overripe",
        2: "ripe",
        3: "unripe"
    }


LABEL_MAP_RAW = load_label_map()


def translate_label(raw_label):
    mapping = {
        "ripe": "Pisang Matang",
        "unripe": "Pisang Mentah",
        "overripe": "Pisang Terlalu Matang",
        "banana": "Pisang"
    }
    return mapping.get(raw_label, raw_label)


# ===============================
# LOAD DETECTION MODEL
# ===============================
def load_detection_model():
    model = ssdlite320_mobilenet_v3_large(pretrained=True)
    model.head.classification_head.num_classes = NUM_CLASSES

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


model_detection = load_detection_model()
print("✅ Detection Model Loaded")


# ===============================
# LOAD CLASSIFICATION MODEL
# ===============================
model_svm = load("models/svm_model_pisang.pkl")
scaler = load("models/scaler_pisang.pkl")
pca = load("models/pca_pisang.pkl")
label_map_cls = np.load("models/label_map.npy", allow_pickle=True).item()
idx_to_label = {v: k for k, v in label_map_cls.items()}

IMG_SIZE = (128, 128)

print("✅ Classification Model Loaded")
print("SVM expects:", model_svm.n_features_in_)
print("PCA components:", pca.n_components_)


# ===============================
# FEATURE EXTRACTION (IDENTIK STREAMLIT)
# ===============================
def extract_features_classification(img_bgr):
    img = cv2.resize(img_bgr, IMG_SIZE)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean_rgb = img.mean(axis=(0, 1))
    mean_hsv = hsv.mean(axis=(0, 1))

    hue = hsv[:, :, 0]
    brown_ratio = np.mean((hue >= 10) & (hue <= 25))

    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    feature = np.concatenate([mean_rgb, mean_hsv, [brown_ratio], hog_feat])

    print("Raw Feature:", len(feature))  # HARUS 8107

    feature = scaler.transform([feature])
    feature = pca.transform(feature)

    print("After PCA shape:", feature.shape)

    return feature


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

    try:
        file = request.files["image"]
        image_bytes = file.read()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = F.to_tensor(img).to(device)
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = model_detection(img_tensor)[0]

        detections = []

        for box, score, label in zip(
            outputs["boxes"],
            outputs["scores"],
            outputs["labels"]
        ):
            if score >= CONFIDENCE_THRESHOLD:
                label_id = int(label.cpu().numpy())

                if label_id == 0:
                    continue

                x1, y1, x2, y2 = box.cpu().numpy().tolist()

                raw_label = LABEL_MAP_RAW.get(label_id, f"class_{label_id}")
                label_name = translate_label(raw_label)

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "score": float(score.cpu().numpy()),
                    "label": label_name,
                    "class_id": label_id
                })

        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections)
        })

    except Exception as e:
        print("Detection Error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ===============================
# CLASSIFICATION ROUTE
# ===============================
@app.route("/predict-classification", methods=["POST"])
def predict_classification():

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    try:
        file = request.files["image"]
        image_bytes = file.read()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        # 🔥 PENTING: RGB → BGR
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        feature = extract_features_classification(img_bgr)

        pred = model_svm.predict(feature)[0]
        label = idx_to_label[pred]

        return jsonify({
            "success": True,
            "label": label
        })

    except Exception as e:
        print("Classification Error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(debug=True)