import torch
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/ssd_mobilenet_banana.pth"

def load_model():

    num_classes = 4  # background + 3 kelas

    # SAMA PERSIS seperti Streamlit
    model = ssdlite320_mobilenet_v3_large(pretrained=True)

    # Ganti jumlah class
    model.head.classification_head.num_classes = num_classes

    # Load weight hasil training
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model.to(device)
    model.eval()

    print("✅ Model Loaded Correctly")

    return model


model = load_model()
