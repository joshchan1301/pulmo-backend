import torch
import numpy as np
from PIL import Image
import io
import cv2
import base64
from timm import create_model
from torchvision import transforms
from lung_segmentation import get_lung_mask

IMG_SIZE = 224
CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "swin_best_model.pth"
_model = None

def is_likely_xray(img_np):
    h, w = img_np.shape[0], img_np.shape[1]
    aspect = w / h
    if aspect < 0.7 or aspect > 1.4:
        return False
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        rgb_std = np.std(img_np, axis=2)
        color_pixels = np.mean(rgb_std > 0.06)
        if color_pixels > 0.08:
            return False
    gray = img_np.mean(axis=2) if img_np.ndim == 3 else img_np
    hist = np.histogram(gray, bins=32, range=(0, 1))[0]
    mid_sum = np.sum(hist[8:24])
    total = np.sum(hist)
    if total == 0 or (mid_sum / total) < 0.45:
        return False
    from skimage.measure import shannon_entropy
    entropy = shannon_entropy(gray)
    if entropy < 2.8:
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

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def mask_lung_auto(img_np):
    """Trả về mask phổi (giá trị 0/1, cùng size ảnh gốc)."""
    # Chuyển về grayscale đúng input yêu cầu
    if img_np.shape[-1] == 3:
        img_gray = cv2.cvtColor((img_np*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        img_gray = (img_np*255).astype(np.uint8)
    mask = get_lung_mask(img_gray)
    mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.float32)  # Đảm bảo mask chuẩn binary float
    return mask

def analyze_xray(image_bytes: bytes):
    model = load_model()
    features, gradients = [], []
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
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
    def backward_hook(module, grad_input, grad_output): gradients.append(grad_output[0])

    handle_fwd = model.layers[-1].blocks[-1].attn.register_forward_hook(forward_hook)
    handle_bwd = model.layers[-1].blocks[-1].attn.register_backward_hook(backward_hook)

    # Tiền xử lý ảnh
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
    cam = cv2.GaussianBlur(cam, (9, 9), 0)
    cam = cam ** 1.2  # chỉnh nhẹ, tránh quá sắc nét không tự nhiên

    # ==== Mask phổi: chỉ giữ heatmap trong vùng phổi ====
    lung_mask = mask_lung_auto(img_np)
    cam = cam * lung_mask  # Chỉ giữ vùng phổi, ngoài phổi = 0

    # ==== Threshold & chuẩn hóa lần cuối ====
    threshold = max(0.13, np.mean(cam) + np.std(cam)*0.7)
    cam[cam < threshold] = 0
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    # ==== Overlay heatmap ====
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Blend với alpha truyền thống (0.33–0.43)
    alpha = 0.38
    if img_np.shape[-1] == 1:
        img_rgb = np.repeat(img_np, 3, axis=-1)
    else:
        img_rgb = img_np.copy()
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
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
