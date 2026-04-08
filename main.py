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
    # 1. Kiểm tra API Key có tồn tại trên server không
    if not OPENAI_API_KEY:
        print("Lỗi: OPENAI_API_KEY chưa được cấu hình trong Variables!")
        return {"reply": "Lỗi hệ thống: Server chưa có API Key."}
        
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
                            "content": (
                                "Bạn là Pulmo AI - Chuyên gia sức khỏe phổi. "
                                "Quy trình: 1. Phân tích từ khóa. 2. Giải thích theo nguồn uy tín (WHO, CDC). "
                                "3. Đề xuất hành động. Trả lời dưới 150 từ, thân thiện. "
                                "Luôn nhắc người dùng tham khảo ý kiến bác sĩ chuyên khoa."
                            )
                        },
                        {"role": "user", "content": req.message}
                    ],
                    "max_tokens": 300, # Tăng nhẹ để tránh bị cắt chữ giữa chừng
                    "temperature": 0.5 # Giảm xuống để câu trả lời y tế chính xác hơn
                }
            )
        
        # 2. Kiểm tra mã phản hồi từ OpenAI
        if response.status_code == 200:
            data = response.json()
            # Lấy nội dung tin nhắn an toàn hơn
            return {"reply": data["choices"][0]["message"]["content"]}
        
        # 3. Nếu OpenAI báo lỗi (401, 429, 500...), in log để debug
        error_info = response.json()
        print(f"OpenAI API Error: {response.status_code} - {error_info}")
        
        # Trả về thông báo lỗi cụ thể để bạn biết đường sửa
        return {"reply": f"AI đang bận (Lỗi {response.status_code}). Vui lòng thử lại sau."}

    except httpx.ReadTimeout:
        return {"reply": "Yêu cầu xử lý quá lâu, vui lòng thử lại."}
    except Exception as e:
        # In lỗi hệ thống ra console của Railway
        print(f"Exception tại chat_with_openai: {str(e)}")
        return {"reply": "Lỗi kết nối máy chủ AI. Vui lòng kiểm tra lại mạng."}
