import torch
import numpy as np
from PIL import Image
import io
import cv2
from timm import create_model
from torchvision import transforms
import base64

# =========== CONFIG ===========
IMG_SIZE = 224
CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "swin_best_model.pth"

# =========== LOAD MODEL (singleton) ===========
_model = None


def is_likely_xray(img_np):
    # 1. Check tỷ lệ hình (nếu là ảnh crop dọc/ngang quá mạnh thì loại)
    h, w = img_np.shape[0], img_np.shape[1]
    aspect = w / h
    if aspect < 0.7 or aspect > 1.4:
        return False

    # 2. Grayscale check (R≈G≈B gần bằng nhau với từng pixel)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        # Tính std giữa các kênh màu trên toàn ảnh
        rgb_std = np.std(img_np, axis=2)
        # Nếu quá nhiều pixel lệch màu thì không phải X-ray
        color_pixels = np.mean(rgb_std > 0.06)
        if color_pixels > 0.08:  # Nếu >8% pixel lệch màu quá lớn, không phải X-ray
            return False

    # 3. Histogram phân bố tập trung giữa (không quá tối hoặc quá sáng)
    gray = img_np.mean(axis=2) if img_np.ndim == 3 else img_np
    hist = np.histogram(gray, bins=32, range=(0, 1))[0]
    mid_sum = np.sum(hist[8:24])
    total = np.sum(hist)
    # Ít nhất 45% pixel nằm ở vùng xám (với 32 bins)
    if total == 0 or (mid_sum / total) < 0.45:
        return False

    # 4. Entropy/phức tạp (ảnh trắng/đen quá nhiều sẽ entropy thấp)
    from skimage.measure import shannon_entropy
    entropy = shannon_entropy(gray)
    if entropy < 2.8:  # Nếu entropy thấp (ảnh đơn điệu, scan giấy) thì loại
        return False

    return True


def load_model():
    global _model
    if _model is None:
        _model = create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            num_classes=3
        )
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        _model.to(DEVICE).eval()
    return _model


# =========== TRANSFORM ===========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# =========== MAIN INFERENCE FUNCTION ===========


def analyze_xray(image_bytes: bytes):
    """
    Dự đoán bệnh phổi và trả về: label (str), probability (%), heatmap (base64 PNG)
    """
    model = load_model()
    features, gradients = [], []
    img = Image.open(io.BytesIO(image_bytes)).convert(
        "RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img) / 255.0
    if not is_likely_xray(img_np):
        return {
            "label": "UNKNOWN",
            "probability": 0.0,
            "heatmap": "",
            "error": "The uploaded image does not appear to be a chest X-ray. Please upload a proper chest X-ray image."
        }
    # Định nghĩa hooks cho GradCAM
    def forward_hook(module, input, output): features.append(output)

    def backward_hook(module, grad_input,
                      grad_output): gradients.append(grad_output[0])

    # Đăng ký hook vào layer Attention cuối cùng
    handle_fwd = model.layers[-1].blocks[-1].attn.register_forward_hook(
        forward_hook)
    handle_bwd = model.layers[-1].blocks[-1].attn.register_backward_hook(
        backward_hook)

    # Tiền xử lý ảnh
    img = Image.open(io.BytesIO(image_bytes)).convert(
        "RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img) / 255.0
    if img_np.shape[-1] != 3:
        img_np = np.stack([img_np]*3, axis=-1)
    img_np = img_np.astype(np.float32)  # Đảm bảo cùng dtype với heatmap

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Forward + Predict
    output = model(img_tensor)
    prob = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
    pred_idx = int(np.argmax(prob))
    pred_prob = float(prob[pred_idx]) * 100

    # Backward cho Grad-CAM
    model.zero_grad()
    output[0, pred_idx].backward()

    # Tính GradCAM
    feat = features[0].squeeze(0).detach().cpu().numpy()
    grad = gradients[0].squeeze(0).detach().cpu().numpy()
    weights = np.mean(grad, axis=0)
    cam = np.dot(feat, weights)
    side = int(np.sqrt(cam.shape[0]))
    cam = cam.reshape(side, side)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = cv2.GaussianBlur(cam, (5, 5), 0)
    cam = cam ** 1.4
    cam[cam < 0.05] = 0

    # Overlay heatmap: blend mềm, đúng màu, không chói
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Blend với alpha
    alpha = 0.45  # Độ đậm heatmap (0.3–0.5 là hợp lý)
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    overlay_img = (overlay * 255).astype(np.uint8)

    # Encode overlay thành base64 PNG
    _, buffer = cv2.imencode('.png', overlay_img)
    heatmap_b64 = base64.b64encode(buffer).decode('utf-8')

    # Cleanup hooks
    handle_fwd.remove()
    handle_bwd.remove()

    return {
        "label": CLASS_NAMES[pred_idx],
        "probability": round(pred_prob, 1),
        "heatmap": heatmap_b64
    }
