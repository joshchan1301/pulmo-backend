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

# ================== CẤU HÌNH HỆ THỐNG ==================
IMG_SIZE = 224
CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
# Ép buộc sử dụng CPU để tiết kiệm RAM khi deploy lên Railway (Gói Free/Pro thấp thường giới hạn RAM)
DEVICE = torch.device("cpu")

class XRayAnalyzer:
    def __init__(self, model_path="swin_best_model.pth"):
        self.device = DEVICE
        self.img_size = IMG_SIZE
        self.class_names = CLASS_NAMES
        self.model = self._init_model(model_path)
        
        # Transform chuẩn cho Swin Transformer
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _init_model(self, model_path):
        """Khởi tạo model và giải phóng RAM ngay sau khi load."""
        try:
            model = create_model(
                "swin_base_patch4_window7_224",
                pretrained=False,
                num_classes=len(self.class_names)
            )
            # Load trọng số trực tiếp vào CPU để tránh overhead
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            
            # Giải phóng biến tạm
            del state_dict
            gc.collect()
            return model
        except Exception as e:
            print(f"Lỗi khởi tạo Model: {e}")
            return None

    def is_likely_xray(self, img_np):
        """Kiểm tra xem ảnh có phải X-quang phổi không để tránh dự đoán sai."""
        h, w = img_np.shape[0], img_np.shape[1]
        aspect = w / h
        if aspect < 0.7 or aspect > 1.4:
            return False, "Tỉ lệ ảnh không hợp lệ (không phải ảnh X-quang chuẩn)."
            
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            rgb_std = np.std(img_np, axis=2)
            color_pixels = np.mean(rgb_std > 0.06)
            if color_pixels > 0.08:
                return False, "Ảnh có màu, không phải ảnh X-quang đen trắng."
                
        gray = img_np.mean(axis=2) if img_np.ndim == 3 else img_np
        hist = np.histogram(gray, bins=32, range=(0, 1))[0]
        mid_sum = np.sum(hist[8:24])
        total = np.sum(hist)
        if total == 0 or (mid_sum / total) < 0.45:
            return False, "Cấu trúc độ sáng không giống ảnh X-quang."
            
        entropy = shannon_entropy(gray)
        if entropy < 2.8:
            return False, "Ảnh quá mờ hoặc thiếu chi tiết (Entropy thấp)."
            
        return True, "Hợp lệ"

    def _generate_gradcam(self, img_tensor):
        """Tối ưu hóa Grad-CAM cho Swin Transformer và dọn dẹp bộ nhớ."""
        features, gradients = [], []

        def forward_hook(module, input, output): features.append(output)
        def backward_hook(module, grad_input, grad_output): gradients.append(grad_output[0])

        # Swin Transformer thường hook vào lớp attention cuối cùng
        target_layer = self.model.layers[-1].blocks[-1].attn
        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)

        # Chạy dự đoán
        output = self.model(img_tensor)
        prob = torch.softmax(output, dim=1).detach().numpy()[0]
        pred_idx = int(np.argmax(prob))
        pred_prob = float(prob[pred_idx]) * 100

        # Lan truyền ngược để lấy Gradient
        self.model.zero_grad()
        output[0, pred_idx].backward()

        # Trích xuất dữ liệu CAM
        feat = features[0].squeeze(0).detach().numpy()
        grad = gradients[0].squeeze(0).detach().numpy()
        
        # Gỡ bỏ hook ngay để tiết kiệm tài nguyên
        handle_fwd.remove()
        handle_bwd.remove()

        weights = np.mean(grad, axis=0)
        cam = np.dot(feat, weights)
        
        # Reshape về kích thước grid (Swin 7x7)
        side = int(np.sqrt(cam.shape[0]))
        cam = cam.reshape(side, side)
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        
        # Làm mịn Heatmap
        cam = cv2.resize(cam, (self.img_size, self.img_size))
        cam = cv2.GaussianBlur(cam, (11, 11), 0)
        cam = cam ** 1.15 # Tăng độ tương phản cho vùng nóng
        cam[cam < 0.05] = 0

        # Dọn dẹp RAM
        del output, features, gradients
        gc.collect()

        return cam, pred_idx, pred_prob

    def _overlay_heatmap(self, img_np, cam, alpha=0.45):
        """Trộn Heatmap vào ảnh gốc với màu sắc trực quan."""
        if img_np.dtype != np.float32:
            img_np = img_np.astype(np.float32)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
        return np.clip(overlay * 255, 0, 255).astype(np.uint8)

# ==========================================================
# CÁC HÀM ENTRY POINT TƯƠNG THÍCH VỚI MAIN.PY
# ==========================================================

_analyzer_instance = None

def load_model(model_path="swin_best_model.pth"):
    """Singleton: Chỉ khởi tạo analyzer một lần để tiết kiệm RAM."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = XRayAnalyzer(model_path)
    return _analyzer_instance

def predict(analyzer, image_bytes: bytes):
    """Hàm xử lý request từ API."""
    if analyzer is None or analyzer.model is None:
        return {"error": "Model chưa được tải thành công."}

    try:
        # Đọc ảnh từ Bytes
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_np = np.array(img_pil) / 255.0

        # Kiểm tra tính hợp lệ của ảnh X-quang
        is_valid, msg = analyzer.is_likely_xray(img_np)
        if not is_valid:
            return {
                "label": "UNKNOWN",
                "probability": 0.0,
                "heatmap": "",
                "error": f"Ảnh không hợp lệ: {msg}"
            }

        # Chuyển đổi Tensor
        img_tensor = analyzer.transform(img_pil).unsqueeze(0).to(DEVICE)
        
        # Chạy Grad-CAM & Dự đoán
        cam, pred_idx, pred_prob = analyzer._generate_gradcam(img_tensor)

        # Tạo ảnh kết quả Overlay
        overlay_img = analyzer._overlay_heatmap(img_np, cam)
        # Chuyển về BGR trước khi encode cho đúng chuẩn OpenCV -> Base64
        overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', overlay_bgr)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')

        # Dọn dẹp bộ nhớ cuối request
        del img_tensor, cam, overlay_img
        gc.collect()

        return {
            "label": analyzer.class_names[pred_idx],
            "probability": round(pred_prob, 1),
            "heatmap": heatmap_b64,
            "status": "success"
        }
        
    except Exception as e:
        return {"error": f"Lỗi hệ thống: {str(e)}"}
