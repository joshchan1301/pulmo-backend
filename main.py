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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
async def chat_with_ai(req: ChatRequest):
    if not GEMINI_API_KEY:
        return {"reply": "Lỗi: Server chưa cấu hình API Key."}
        
    # SỬA TẠI ĐÂY: Dùng v1beta thay vì v1 cho gemini-1.5-flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    # Payload nên bọc trong try-except riêng nếu cần
    payload = {
        "contents": [{
            "role": "user", # Thêm role để chuẩn cấu trúc
            "parts": [{"text": f"{system_instruction}\n\nNgười dùng hỏi: {req.message}"}]
        }],
        "generationConfig": {
            "maxOutputTokens": 300,
            "temperature": 0.4,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                # Kiểm tra xem có candidate nào không trước khi truy cập
                if data.get('candidates'):
                    ai_reply = data['candidates'][0]['content']['parts'][0]['text']
                    return {"reply": ai_reply}
                return {"reply": "AI không thể đưa ra câu trả lời phù hợp."}
            else:
                # Log chi tiết để bạn dễ debug
                print(f"Gemini Error {response.status_code}: {response.text}")
                return {"reply": f"Lỗi kết nối AI (Mã: {response.status_code})."}
    except Exception as e:
        return {"reply": "Lỗi hệ thống khi kết nối với AI."}
