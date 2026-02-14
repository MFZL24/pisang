import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "ssd_mobilenet_banana.pth")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

NUM_CLASSES = 4  # background + 3 kelas pisang
CONFIDENCE_THRESHOLD = 0.5
