Hugging Face's logo

joshchan1301
/
x-ray_img_analysis_ai 

like
0
Model card
Files
xet
Community
Settings
x-ray_img_analysis_ai
/
main.py

joshchan1301's picture
joshchan1301
Upload 5 files
844c857
verified
raw

Copy download link
history
blame
edit
delete
4.12 kB
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import gdown

from model_inference import analyze_xray  # Import hàm phân tích ảnh

# Load biến môi trường
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def download_model():
    model_url = "https://drive.google.com/uc?export=download&id=1VD2pXT9aDHmGwn2KiD3aSoXj5nBglVLw"
    model_path = "swin_best_model.pth"
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive with gdown...")
        gdown.download(model_url, model_path, quiet=False)
        print("Model downloaded!")


download_model()

# Khởi tạo FastAPI
app = FastAPI(
    title="Pulmo Vision API",
    description="API for Lung X-ray Analysis and Pulmo AI Chatbot",
    version="1.0.0"
)

# CORS: Cho phép FE gọi API từ domain khác
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== API PHÂN TÍCH X-RAY ==========


@app.post("/api/analyze-xray", tags=["X-ray Analysis"])
async def analyze_xray_api(file: UploadFile = File(...)):
    """
    Nhận file ảnh X-ray, trả về nhãn dự đoán, xác suất và ảnh Grad-CAM (base64).
    """
    try:
        img_bytes = await file.read()
        result = analyze_xray(img_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        print("X-ray Analysis Error:", str(e))
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error: " + str(e)}
        )

# ========== API CHATBOT ==========


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat", tags=["Chatbot"])
async def chat_with_openai(req: ChatRequest):
    """
    Nhận tin nhắn từ user, gửi lên OpenAI ChatGPT và trả về phản hồi.
    """
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
                                "Lưu ý không trả lời quá 300 tokens."

                                "Bạn là Pulmo AI Assistant, hãy trả lời các câu hỏi về X-ray phổi, sức khỏe phổi, các bệnh lý liên quan, "
                                "và giải thích kết quả chẩn đoán AI một cách thân thiện, dễ hiểu cho người dùng Việt Nam. "
                                "Nếu có ai đó hỏi bạn bằng tiếng Anh thì hãy trả lời họ bằng tiếng Anh."

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

