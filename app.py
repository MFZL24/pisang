# app.py
import io
import os
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F

# Import config
from config import (
    MODEL_PATH,
    NUM_CLASSES,
    CONFIDENCE_THRESHOLD,
    UPLOAD_FOLDER,
    RESULT_FOLDER
)

app = Flask(__name__)

# ===============================
# DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# LABEL MAPPING
# ===============================
def load_label_map():
    """Load label map from npy file if exists, otherwise use default mapping"""
    label_map_path = os.path.join(os.path.dirname(__file__), "models", "label_map.npy")
    
    if os.path.exists(label_map_path):
        try:
            labels = np.load(label_map_path, allow_pickle=True).item()
            print(f"Loaded label map: {labels}")
            return labels
        except Exception as e:
            print(f"Error loading label_map.npy: {e}")
    
    # Default label mapping (class 1, 2, 3 = 3 classes of bananas)
    # Based on NUM_CLASSES = 4 (background + 3 classes)
    default_labels = {
        1: "Pisang Mentah",
        2: "Pisang Matang", 
        3: "Pisang Terlalu Matang"
    }
    print(f"Using default label map: {default_labels}")
    return default_labels

LABEL_MAP = load_label_map()

# ===============================
# LOAD MODEL
# ===============================
def load_model():
    # num_classes = 4 (background + 3 kelas pisang)
    num_classes = NUM_CLASSES
    
    # Gunakan large backbone seperti model_loader.py
    model = ssdlite320_mobilenet_v3_large(pretrained=True)
    
    # Sesuaikan classification head
    model.head.classification_head.num_classes = num_classes

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

model = load_model()
print("✅ Model Loaded Correctly")

# ===============================
# HOME ROUTE
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
        # PREPROCESS IMAGE
        # ===============================
        image_bytes = file.read()
        
        # Debug: Print file size
        print(f"📸 Received image: {len(image_bytes)} bytes")
        
        # Validate file is not empty
        if len(image_bytes) == 0:
            print("❌ Error: Empty file content")
            return jsonify({"success": False, "error": "Empty file content"}), 400
        
        # Open and convert image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"✅ Image loaded: {img.size}")
        
        img_tensor = F.to_tensor(img).to(device)

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        # ===============================
        # MODEL INFERENCE
        # ===============================
        with torch.no_grad():
            outputs = model(img_tensor)[0]  # batch_size=1

        detections = []
        threshold = CONFIDENCE_THRESHOLD  # Use config threshold
        
        for box, score, label in zip(
            outputs["boxes"], outputs["scores"], outputs["labels"]
        ):
            if score >= threshold:
                # Skip background class (class 0)
                label_id = int(label.cpu().numpy())
                if label_id == 0:
                    continue
                    
                x1, y1, x2, y2 = box.cpu().numpy().tolist()
                
                # Get label from LABEL_MAP
                label_name = LABEL_MAP.get(label_id, f"Pisang Class {label_id}")
                
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "score": float(score.cpu().numpy()),
                    "label": label_name,
                    "class_id": label_id
                })

        if len(detections) == 0:
            print("⚠️ No detections found above threshold")
            return jsonify({"success": False, "detections": [], "message": "No detections above threshold"})

        print(f"✅ Found {len(detections)} detection(s)")
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
