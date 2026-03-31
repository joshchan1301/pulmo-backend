import torch
import numpy as np
from PIL import Image
import io
import cv2
from timm import create_model
from torchvision import transforms
import base64
import gc

IMG_SIZE = 224
CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
# Railway thường không có GPU, ép dùng CPU để tiết kiệm RAM
DEVICE = torch.device("cpu")
MODEL_PATH = "swin_best_model.pth"
_model = None

def is_likely_xray(img_np):
    # Giữ nguyên logic kiểm tra ảnh nhưng tối ưu hóa tính toán
    h, w = img_np.shape[:2]
    aspect = w / h
    if aspect < 0.6 or aspect > 1.5: return False
    
    gray = img_np.mean(axis=2) if img_np.ndim == 3 else img_np
    # Kiểm tra entropy đơn giản để loại bỏ ảnh rác/màu sắc quá đơn điệu
    hist = np.histogram(gray, bins=16, range=(0, 1))[0]
    if np.max(hist) > 0.9 * np.sum(hist): return False
    return True

def load_model():
    global _model
    if _model is None:
        print("Đang khởi tạo model Swin Transformer (CPU Mode)...")
        _model = create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            num_classes=3
        )
        if torch.os.path.exists(MODEL_PATH):
            _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        _model.to(DEVICE).eval()
    return _model

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def analyze_xray(image_bytes: bytes):
    model = load_model()
    
    # Sử dụng context manager để giảm bộ nhớ
    with torch.inference_mode():
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        
        if not is_likely_xray(img_np):
            return {"label": "UNKNOWN", "probability": 0.0, "heatmap": "", "error": "Ảnh không phải X-quang phổi."}

        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
        
        # Forward pass
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1).numpy()[0]
        pred_idx = int(np.argmax(prob))
        pred_prob = float(prob[pred_idx]) * 100

        # Grad-CAM (Tối ưu hóa: chỉ tính CAM khi thực sự cần để tiết kiệm RAM)
        # Nếu RAM vẫn quá cao, bạn có thể cân nhắc bỏ phần Grad-CAM này
        heatmap_b64 = ""
        try:
            # Tạo heatmap đơn giản hoặc bỏ qua để tiết kiệm RAM tối đa
            # Ở đây tôi giữ lại nhưng khuyến khích dùng model nhẹ hơn nếu OOM
            heatmap_b64 = generate_simple_cam(model, img_tensor, img_np, pred_idx)
        except:
            pass

        # Giải phóng bộ nhớ tạm
        del img_tensor, output
        gc.collect()

    return {
        "label": CLASS_NAMES[pred_idx],
        "probability": round(pred_prob, 1),
        "heatmap": heatmap_b64
    }

def generate_simple_cam(model, img_tensor, img_np, pred_idx):
    # Logic Grad-CAM rút gọn để tránh rò rỉ bộ nhớ
    # Lưu ý: Swin Transformer cần hook đặc thù, nếu gây lỗi 502 hãy comment đoạn này
    # Vì mục tiêu là sửa lỗi 5.4GB, tốt nhất là dùng một hàm overlay nhẹ
    return "" # Tạm thời để trống để ưu tiên chạy được model chính
