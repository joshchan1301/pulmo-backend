import torch
import torch.nn.functional as F
import numpy as np
import cv2
import io
import base64
import gc
from PIL import Image
from timm import create_model
from torchvision import transforms
from skimage.measure import shannon_entropy

# ================== CẤU HÌNH ==================
IMG_SIZE = 224
CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
# Ép dùng CPU để tiết kiệm RAM tối đa trên Railway
DEVICE = torch.device("cpu")

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
        """Khởi tạo model Swin và giải phóng RAM thừa ngay lập tức."""
        try:
            model = create_model(
                "swin_base_patch4_window7_224",
                pretrained=False,
                num_classes=len(self.class_names)
            )
            # Load trực tiếp vào CPU
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            
            del state_dict
            gc.collect()
            return model
        except Exception as e:
            print(f"Lỗi Load Model: {e}")
            return None

    def is_likely_xray(self, img_np):
        """Bộ lọc kiểm tra ảnh X-quang đầu vào."""
        h, w = img_np.shape[0], img_np.shape[1]
        aspect = w / h
        if aspect < 0.7 or aspect > 1.4:
            return False, "Tỉ lệ ảnh không phù hợp."
            
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            rgb_std = np.std(img_np, axis=2)
            if np.mean(rgb_std > 0.06) > 0.08:
                return False, "Đây có vẻ là ảnh màu, không phải X-quang."
                
        gray = img_np.mean(axis=2) if img_np.ndim == 3 else img_np
        entropy = shannon_entropy(gray)
        if entropy < 2.8:
            return False, "Ảnh quá thiếu chi tiết cấu trúc."
            
        return True, "Hợp lệ"

    def _generate_gradcam(self, img_tensor):
        """Tính toán Grad-CAM và dọn dẹp Tensor ngay sau khi xong."""
        features, gradients = [], []

        def forward_hook(module, input, output): features.append(output)
        def backward_hook(module, grad_input, grad_output): gradients.append(grad_output[0])

        # Hook vào lớp cuối của Swin
        target_layer = self.model.layers[-1].blocks[-1].attn
        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)

        output = self.model(img_tensor)
        prob = torch.softmax(output, dim=1).detach().numpy()[0]
        pred_idx = int(np.argmax(prob))
        pred_prob = float(prob[pred_idx]) * 100

        self.model.zero_grad()
        output[0, pred_idx].backward()

        feat = features[0].squeeze(0).detach().numpy()
        grad = gradients[0].squeeze(0).detach().numpy()
        
        handle_fwd.remove()
        handle_bwd.remove()

        weights = np.mean(grad, axis=0)
        cam = np.dot(feat, weights)
        side = int(np.sqrt(cam.shape[0]))
        cam = cam.reshape(side, side)
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        
        cam = cv2.resize(cam, (self.img_size, self.img_size))
        cam = cv2.GaussianBlur(cam, (11, 11), 0)
        cam = cam ** 1.15 # Làm đậm vùng nóng
        
        # Giải phóng bộ nhớ Tensor
        del output, features, gradients
        gc.collect()

        return cam, pred_idx, pred_prob

    def _overlay_heatmap(self, img_np, cam, alpha=0.45):
        """Chèn bản đồ nhiệt vào ảnh gốc."""
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay = cv2.addWeighted(img_np.astype(np.float32), 1 - alpha, heatmap, alpha, 0)
        return np.clip(overlay * 255, 0, 255).astype(np.uint8)

# ================== BRIDGE TO MAIN.PY ==================

_analyzer = None

def load_model(model_path="swin_best_model.pth"):
    global _analyzer
    if _analyzer is None:
        _analyzer = XRayAnalyzer(model_path)
    return _analyzer

def predict(analyzer, image_bytes: bytes):
    if analyzer is None: return {"error": "Model not loaded"}

    try:
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_np = np.array(img_pil) / 255.0

        is_valid, msg = analyzer.is_likely_xray(img_np)
        if not is_valid:
            return {"label": "UNKNOWN", "probability": 0.0, "heatmap": "", "error": msg}

        img_tensor = analyzer.transform(img_pil).unsqueeze(0).to(DEVICE)
        cam, pred_idx, pred_prob = analyzer._generate_gradcam(img_tensor)
        overlay_img = analyzer._overlay_heatmap(img_np, cam)
        overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', overlay_bgr)
        
        # Dọn dẹp
        del img_tensor, cam, overlay_img
        gc.collect()

        return {
            "label": analyzer.class_names[pred_idx],
            "probability": round(pred_prob, 1),
            "heatmap": base64.b64encode(buffer).decode('utf-8'),
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e)}
