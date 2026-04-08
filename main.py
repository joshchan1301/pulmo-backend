import os
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from huggingface_hub import hf_hub_download

from model_inference import analyze_xray

# Load biến môi trường
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Tối ưu hóa việc tải model từ Hugging Face
def download_model():
    # Thay thế repo_id và filename bằng thông tin thực tế của bạn trên HF
    # Nếu chưa có, bạn nên tạo 1 repo public trên HF và upload file .pth lên
    REPO_ID = "joshchan1301/x-ray_img_analysis_ai" 
    FILENAME = "swin_best_model.pth"
    MODEL_PATH = "swin_best_model.pth"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Đang tải model từ Hugging Face: {REPO_ID}...")
        try:
            # Nếu dùng Google Drive link cũ, bạn vẫn có thể dùng gdown nhưng HF ổn định hơn
            # Ở đây tôi hướng dẫn dùng hf_hub_download
            path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=".")
            print(f"Model đã được tải về tại: {path}")
        except Exception as e:
            print(f"Lỗi khi tải model từ HF: {e}")

# Gọi hàm tải model (tùy chọn nếu bạn đã cấu hình HF)
download_model()

app = FastAPI(
    title="Pulmo Vision API",
    description="API phân tích X-quang phổi và Chatbot Pulmo AI",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze-xray", tags=["X-ray Analysis"])
async def analyze_xray_api(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        # Giải phóng dung lượng file sau khi đọc
        result = analyze_xray(img_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Lỗi máy chủ: {str(e)}"}
        )

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat", tags=["Chatbot"])
async def chat_with_openai(req: ChatRequest):
    if not OPENAI_API_KEY:
        return {"reply": "Lỗi: Thiếu API Key cho Chatbot."}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Bạn là Pulmo AI - Chuyên gia phản hồi thông tin sức khỏe phổi. khi nhận câu hỏi, hãy thực hiện theo cấu trúc: Bước 1: Phân tích nhanh từ khóa y khoa trong câu hỏi. Bước 2: Đưa ra lời giải thích dựa trên các nguồn uy tín (WHO, CDC, ATS). Bước 3: Đề xuất hành động tiếp theo (Khám định kỳ, tập thở, hoặc theo dõi triệu chứng). Yêu cầu: Không nói dài dòng, câu trả lời không quá 150 từ. Luôn kết thúc bằng một lời chúc hoặc lời khuyên tích cực."
                        },
                        {"role": "user", "content": req.message}
                    ],
                    "max_tokens": 250,
                    "temperature": 0.7
                }
            )
        if response.status_code == 200:
            return {"reply": response.json()["choices"][0]["message"]["content"]}
        return {"reply": "AI đang bận, vui lòng thử lại sau."}
    except Exception:
        return {"reply": "Lỗi kết nối AI."}
