import os
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

from model_inference import analyze_xray  # Hàm predict của bạn

# ================== CONFIG ==================
MODEL_URL = "https://huggingface.co/joshchan1301/x-ray_img_analysis_ai/resolve/main/swin_best_model.pth"
MODEL_PATH = "swin_best_model.pth"

# ================== LOAD ENV ==================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ================== DOWNLOAD MODEL ==================
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Hugging Face...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Model downloaded!")

# ================== INIT APP ==================
app = FastAPI(
    title="Pulmo Vision API",
    description="API for Lung X-ray Analysis and Pulmo AI Chatbot",
    version="1.0.0"
)

# ================== CORS ==================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== LOAD MODEL ON STARTUP ==================
model = None

@app.on_event("startup")
def load_all():
    global model
    download_model()
    print("Loading model into memory...")
    model = analyze_xray.load_model(MODEL_PATH)  # bạn phải có hàm này
    print("Model loaded successfully!")

# ================== X-RAY API ==================
@app.post("/api/analyze-xray", tags=["X-ray Analysis"])
async def analyze_xray_api(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        result = analyze_xray.predict(model, img_bytes)  # sửa lại theo kiểu này
        return JSONResponse(content=result)
    except Exception as e:
        print("X-ray Analysis Error:", str(e))
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error: " + str(e)}
        )

# ================== CHATBOT ==================
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat", tags=["Chatbot"])
async def chat_with_openai(req: ChatRequest):
    if not OPENAI_API_KEY:
        return {"reply": "Lỗi: Thiếu OPENAI_API_KEY trong môi trường server."}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Lưu ý không trả lời quá 300 tokens. "
                                "Bạn là Pulmo AI Assistant, hãy trả lời các câu hỏi về X-ray phổi, "
                                "giải thích kết quả chẩn đoán AI một cách dễ hiểu cho người Việt."
                            )
                        },
                        {
                            "role": "user",
                            "content": req.message
                        }
                    ],
                    "max_tokens": 300,
                    "temperature": 0.7
                }
            )

        if response.status_code == 200:
            data = response.json()
            return {"reply": data["choices"][0]["message"]["content"]}
        else:
            print("OpenAI API Error:", response.text)
            return {"reply": "Sorry, the AI is currently unavailable."}

    except Exception as e:
        print("OpenAI Chat Exception:", str(e))
        return {"reply": "Lỗi kết nối tới AI. Vui lòng thử lại sau."}
