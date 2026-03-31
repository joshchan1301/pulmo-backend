import torch
import torch.nn.functional as F
import numpy as np
import cv2
import io
import base64
from PIL import Image
from timm import create_model
from torchvision import transforms
from skimage.measure import shannon_entropy

# ================== GLOBAL CONFIG ==================
IMG_SIZE = 224
CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class XRayAnalyzer:
    def __init__(self, model_path="swin_best_model.pth"):
        self.device = DEVICE
        self.img_size = IMG_SIZE
        self.class_names = CLASS_NAMES
        self.model = self._init_model(model_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _init_model(self, model_path):
        """Khởi tạo model Swin Transformer và load trọng số."""
        try:
            model = create_model(
                "swin_base_patch4_window7_224",
                pretrained=False,
                num_classes=len(self.class_names)
            )
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device).eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def is_likely_xray(self, img_np):
        """Kiểm tra tính hợp lệ của ảnh X-quang."""
        h, w = img_np.shape[0], img_np.shape[1]
        aspect = w / h
        if aspect < 0.7 or aspect > 1.4:
            return False, "Tỉ lệ khung hình không hợp lệ."
            
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            rgb_std = np.std(img_np, axis=2)
            color_pixels = np.mean(rgb_std > 0.06)
            if color_pixels > 0.08:
                return False, "Ảnh có quá nhiều màu sắc."
                
        gray = img_np.mean(axis=2) if img_np.ndim == 3 else img_np
        hist = np.histogram(gray, bins=32, range=(0, 1))[0]
        mid_sum = np.sum(hist[8:24])
        total = np.sum(hist)
        if total == 0 or (mid_sum / total) < 0.45:
            return False, "Phân bố cường độ sáng không giống ảnh X-quang."
            
        entropy = shannon_entropy(gray)
        if entropy < 2.8:
            return False, "Ảnh thiếu chi tiết cấu trúc (Entropy thấp)."
            
        return True, "Success"

    def _generate_gradcam(self, img_tensor):
        """Tính toán heatmap Grad-CAM."""
        features, gradients = [], []

        def forward_hook(module, input, output): features.append(output)
        def backward_hook(module, grad_input, grad_output): gradients.append(grad_output[0])

        # Hook vào lớp cuối cùng của Swin
        target_layer = self.model.layers[-1].blocks[-1].attn
        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)

        output = self.model(img_tensor)
        prob = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
        pred_idx = int(np.argmax(prob))
        pred_prob = float(prob[pred_idx]) * 100

        self.model.zero_grad()
        output[0, pred_idx].backward()

        # Xử lý đặc trưng không gian (Spatial Features)
        feat = features[0].squeeze(0).detach().cpu().numpy()
        grad = gradients[0].squeeze(0).detach().cpu().numpy()
        
        weights = np.mean(grad, axis=0)
        cam = np.dot(feat, weights)
        
        side = int(np.sqrt(cam.shape[0]))
        cam = cam.reshape(side, side)
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        
        # Làm mịn heatmap
        cam = cv2.resize(cam, (self.img_size, self.img_size))
        cam = cv2.GaussianBlur(cam, (11, 11), 0)
        cam = cam ** 1.15
        cam[cam < 0.05] = 0

        handle_fwd.remove()
        handle_bwd.remove()

        return cam, pred_idx, pred_prob

    def _overlay_heatmap(self, img_np, cam, alpha=0.40):
        """Trộn heatmap vào ảnh gốc."""
        if img_np.dtype != np.float32:
            img_np = img_np.astype(np.float32)
        if img_np.max() > 1.0:
            img_np /= 255.0

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
        overlay_img = np.clip(overlay * 255, 0, 255).astype(np.uint8)
        return overlay_img

# ==========================================================
# CÁC HÀM ENTRY POINT CHO MAIN.PY
# ==========================================================

_analyzer_instance = None

def load_model(model_path="swin_best_model.pth"):
    """Khởi tạo analyzer duy nhất (Singleton)."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = XRayAnalyzer(model_path)
    return _analyzer_instance

def predict(analyzer, image_bytes: bytes):
    """Thực hiện dự đoán và trả về JSON."""
    if analyzer is None or analyzer.model is None:
        return {"error": "Model loading failed or analyzer not initialized."}

    try:
        # Load và chuẩn bị ảnh
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_np = np.array(img_pil) / 255.0

        # Kiểm tra ảnh X-quang
        is_valid, msg = analyzer.is_likely_xray(img_np)
        if not is_valid:
            return {
                "label": "UNKNOWN",
                "probability": 0.0,
                "heatmap": "",
                "error": f"Invalid image: {msg}"
            }

        # Inference & Grad-CAM
        img_tensor = analyzer.transform(img_pil).unsqueeze(0).to(DEVICE)
        cam, pred_idx, pred_prob = analyzer._generate_gradcam(img_tensor)

        # Overlay & Encode
        overlay_img = analyzer._overlay_heatmap(img_np, cam)
        # Chuyển về BGR để cv2 encode đúng màu
        overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', overlay_bgr)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "label": analyzer.class_names[pred_idx],
            "probability": round(pred_prob, 1),
            "heatmap": heatmap_b64,
            "status": "success"
        }
        
    except Exception as e:
        return {"error": f"Inference Error: {str(e)}"}
