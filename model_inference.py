import torch
import numpy as np
from PIL import Image
import io
import os
from timm import create_model
import base64
import gc

IMG_SIZE = 224
CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
DEVICE = torch.device("cpu")
MODEL_PATH = "swin_best_model.pth"

_model = None

def is_likely_xray(img_np):
    h, w = img_np.shape[:2]
    aspect = w / h
    if aspect < 0.6 or aspect > 1.5:
        return False
    
    gray = img_np.mean(axis=2) if img_np.ndim == 3 else img_np
    hist = np.histogram(gray, bins=16, range=(0, 1))[0]
    if np.max(hist) > 0.9 * np.sum(hist):
        return False
    return True

def load_model():
    global _model
    if _model is None:
        print("Loading Swin model (LIGHT VERSION)...")
        
        _model = create_model(
            "swin_tiny_patch4_window7_224",  # 🔥 đổi sang tiny
            pretrained=False,
            num_classes=3
        )

        if os.path.exists(MODEL_PATH):
            _model.load_state_dict(
                torch.load(MODEL_PATH, map_location=DEVICE)
            )

        _model.to(DEVICE).eval()
    return _model

def preprocess(img_pil):
    img = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

def analyze_xray(image_bytes: bytes):
    model = load_model()

    with torch.inference_mode():
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0

        if not is_likely_xray(img_np):
            return {
                "label": "UNKNOWN",
                "probability": 0.0,
                "heatmap": "",
                "error": "Ảnh không phải X-quang phổi."
            }

        img_tensor = preprocess(img_pil).to(DEVICE)

        output = model(img_tensor)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(prob))
        pred_prob = float(prob[pred_idx]) * 100

        del img_tensor, output
        gc.collect()

    return {
        "label": CLASS_NAMES[pred_idx],
        "probability": round(pred_prob, 1),
        "heatmap": ""  # 🔥 bỏ CAM để nhẹ
    }
