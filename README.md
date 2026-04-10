# 🫁 Pulmo Vision – Chest X-ray AI Analysis

**Pulmo Vision** is an AI project that applies the **Swin Transformer** to analyze chest X-ray images, supporting the detection of several conditions:

- **BACTERIAL Pneumonia**
- **VIRAL Pneumonia**
- **NORMAL (healthy)**

Additionally, the system generates **Grad-CAM heatmaps** to explain the model’s decision, helping users understand which lung regions contribute to the diagnosis.

---

## 🚀 Features

- 📷 **Upload X-ray images** → the model analyzes and returns predicted label and probability.
- 🔥 **Grad-CAM visualization** → highlights the lung regions the model focuses on.
- 🧠 **Swin Transformer backbone** → state-of-the-art computer vision architecture.
- 🌐 **FastAPI backend** → easy to deploy (Vercel, Render, Railway…).
- 🤖 **Gemini preview-model chatbot** (Google AI Studio) → explains results in a user-friendly way.

---

## 📂 Project Structure

```
pulmo-backend/
│── main.py # FastAPI server (API routes)
│── model_inference.py # Model loading, preprocessing, Grad-CAM
│── requirements.txt # Dependencies
│── .env.example # API keys & config
│── swin_best_model.pth # Model checkpoint
```

---

## ⚙️ Installation

1. Clone repo:

```bash
git clone https://github.com/joshchan1301/pulmo-backend.git
cd pulmo-backend
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` file:

```env
OPENAI_API_KEY=sk-xxxxxxx
```

4. Download pretrained model (Swin Transformer):
   - Google Drive link:
     ```
     https://drive.google.com/uc?export=download&id=1VD2pXT9aDHmGwn2KiD3aSoXj5nBglVLw
     ```
   - Save as swin_best_model.pth in the project folder.

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
{ "message": "What does my X-ray mean?" }
```

- **Output JSON:**

```json
{
  "reply": "The X-ray indicates signs of bacterial pneumonia. The red highlighted areas are where the model focused most."
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

- This is **not a certified medical tool**.
- For **educational and research purposes only**.
- Not a substitute for professional medical diagnosis.

---

✍️ Author: [Tran Hieu](https://github.com/joshchan1301)
