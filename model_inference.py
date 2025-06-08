import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import cv2
from timm import create_model
from torchvision import transforms
import base64

IMG_SIZE = 224
CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "swin_best_model.pth"
SEG_MODEL_PATH = "unet_lung_seg.pth" # Đặt tên file weight segment tại đây

# === Định nghĩa U-Net (rút gọn) ===
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_channels, 32)
        self.dconv_down2 = DoubleConv(32, 64)
        self.dconv_down3 = DoubleConv(64, 128)
        self.dconv_down4 = DoubleConv(128, 256)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.ConvTranspose2d(256, 256, 2, stride=2)

        self.dconv_up3 = DoubleConv(256+128, 128)
        self.dconv_up2 = DoubleConv(128+64, 64)
        self.dconv_up1 = DoubleConv(64+32, 32)

        self.conv_last = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out

# ===== Segment phổi tự động bằng PyTorch UNet =====
def segment_lung(img_np):
    """input: img_np RGB[0..1] (H,W,3), output: mask phổi (H,W), 0/1"""
    model = UNet()
    model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location="cpu"))
    model.eval()
    img_gray = cv2.cvtColor((img_np*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img_pil = Image.fromarray(img_gray)
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    x = t(img_pil).unsqueeze(0)
    with torch.no_grad():
        mask = model(x)
    mask = torch.sigmoid(mask).squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.float32)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return mask

# =========== Đoạn code inference + heatmap tối ưu ===========

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

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    output = model(img_tensor)
    prob = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
    pred_idx = int(np.argmax(prob))
    pred_prob = float(prob[pred_idx]) * 100

    # Backward cho Grad-CAM
    model.zero_grad()
    output[0, pred_idx].backward()

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
    cam = cam ** 1.2

    # ==== MASK PHỔI: chỉ giữ heatmap trong vùng phổi ====
    lung_mask = segment_lung(img_np)
    cam = cam * lung_mask

    # ==== THRESHOLD thông minh ====
    threshold = max(0.13, np.mean(cam) + np.std(cam)*0.7)
    cam[cam < threshold] = 0
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    # ==== OVERLAY heatmap ====
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    alpha = 0.38
    img_rgb = img_np.copy()
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
    overlay_img = (overlay * 255).astype(np.uint8)

    # Encode overlay thành base64 PNG
    _, buffer = cv2.imencode('.png', overlay_img)
    heatmap_b64 = base64.b64encode(buffer).decode('utf-8')

    handle_fwd.remove()
    handle_bwd.remove()

    return {
        "label": CLASS_NAMES[pred_idx],
        "probability": round(pred_prob, 1),
        "heatmap": heatmap_b64
    }
