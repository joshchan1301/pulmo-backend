# 🫁 Pulmo Vision – Chest X-ray AI Analysis

**Pulmo Vision** là một dự án AI áp dụng **Swin Transformer** để phân tích ảnh X-ray phổi, hỗ trợ phát hiện một số tình trạng:

- **BACTERIAL Pneumonia**
- **VIRAL Pneumonia**
- **NORMAL (healthy)**

Ngoài ra, hệ thống còn sinh ra **Grad-CAM heatmap** để giải thích quyết định của mô hình, giúp người dùng hiểu rõ hơn vùng phổi nào ảnh hưởng đến chẩn đoán.

---

## 🚀 Features

- 📷 **Upload ảnh X-ray** → model phân tích và trả về nhãn, xác suất.
- 🔥 **Grad-CAM visualization** → hiển thị vùng phổi quan trọng mà mô hình tập trung.
- 🧠 **Swin Transformer backbone** → state-of-the-art cho computer vision.
- 🌐 **FastAPI backend** → dễ deploy (Vercel, Render, Railway…).
- 🤖 **Chatbot tích hợp GPT** (API OpenAI) → giải thích kết quả bằng tiếng Việt thân thiện.

---

## 📂 Project Structure

```
pulmo-backend/
│── main.py              # FastAPI server (API routes)
│── model_inference.py   # Model loading, preprocessing, Grad-CAM
│── requirements.txt     # Dependencies
│── .env.example         # API keys & config
│── swin_best_model.pth  # Model checkpoint
```

---

## ⚙️ Installation

1. Clone repo:

```bash
git clone https://github.com/joshchan1301/pulmo-backend.git
cd pulmo-backend
```

2. Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

3. Tạo file `.env`:

```env
OPENAI_API_KEY=sk-xxxxxxx
```

4. Tải model checkpoint (Swin Transformer):
   - Link Google Drive (đang sử dụng):
     ```
     https://drive.google.com/uc?export=download&id=1VD2pXT9aDHmGwn2KiD3aSoXj5nBglVLw
     ```
   - Lưu với tên `swin_best_model.pth` trong thư mục project.

---

## ▶️ Run API server

```bash
uvicorn main:app --reload
```

API docs (Swagger UI):  
👉 http://127.0.0.1:8000/docs

---

## 📡 API Endpoints

### 1. Analyze Chest X-ray

`POST /api/analyze-xray`

- **Input:** `file` (form-data, image `.png/.jpg`)
- **Output JSON:**

```json
{
  "label": "BACTERIAL",
  "probability": 87.5,
  "heatmap": "<base64-encoded PNG>"
}
```

### 2. Pulmo AI Chatbot

`POST /api/chat`

- **Input JSON:**

```json
{ "message": "Ảnh X-ray của tôi có ý nghĩa gì?" }
```

- **Output JSON:**

```json
{
  "reply": "Ảnh X-ray cho thấy dấu hiệu viêm phổi do vi khuẩn, vùng màu đỏ là nơi mô hình tập trung nhiều nhất."
}
```

---

## 📊 Demo

Do your own =)))

---

## 📖 Tech Stack

- **Python, FastAPI, httpx** – API server
- **PyTorch, timm, torchvision** – Deep learning model
- **OpenAI API** – Chatbot
- **Grad-CAM** – Explainable AI

---

## ⚠️ Disclaimer

- Đây **không phải công cụ y tế chính thức**.
- Chỉ dùng cho mục đích **học tập và nghiên cứu AI**.
- Không thay thế chẩn đoán của bác sĩ.

---

✍️ Author: [Trần Hiếu](https://github.com/joshchan1301)
