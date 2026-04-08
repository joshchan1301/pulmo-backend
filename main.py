import os
import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Import hàm inference từ file của bạn
try:
    from model_inference import analyze_xray
except ImportError:
    # Để tránh crash nếu bạn chưa có file này lúc test
    def analyze_xray(bytes): return {"result": "Model inference not found"}

# 1. Load biến môi trường
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Cấu hình Prompt hệ thống (Định nghĩa Global để tránh NameError)
SYSTEM_INSTRUCTION = (
    "Bạn là Pulmo AI - Chuyên gia sức khỏe phổi. "
    "Quy trình trả lời: 1. Phân tích từ khóa y khoa. 2. Giải thích theo nguồn uy tín (WHO, CDC). "
    "3. Đề xuất hành động tiếp theo. Yêu cầu: Trả lời thân thiện, ngắn gọn dưới 150 từ. "
    "Luôn có câu nhắc người dùng đi khám bác sĩ chuyên khoa để có kết luận chính xác."
)

# 3. Tải model từ Hugging Face
def download_model():
    REPO_ID = "joshchan1301/x-ray_img_analysis_ai" 
    FILENAME = "swin_best_model.pth"
    MODEL_PATH = "swin_best_model.pth"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Đang tải model từ Hugging Face: {REPO_ID}...")
        try:
            path = hf_hub_download(
                repo_id=REPO_ID, 
                filename=FILENAME, 
                local_dir=".",
                token=HF_TOKEN # Thêm token để tránh rate limit
            )
            print(f"Model đã được tải về tại: {path}")
        except Exception as e:
            print(f"Lỗi khi tải model từ HF: {e}")

download_model()

# 4. Khởi tạo FastAPI
app = FastAPI(
    title="Pulmo Vision API",
    description="API phân tích X-quang phổi và Chatbot Pulmo AI",
    version="1.1.0"
)

# 5. Cấu hình CORS (Chỉnh sửa để hoạt động tốt trên Vercel/Railway)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pulmo-vision.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/analyze-xray", tags=["X-ray Analysis"])
async def analyze_xray_api(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        result = analyze_xray(img_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Lỗi máy chủ: {str(e)}"}
        )

@app.post("/api/chat", tags=["Chatbot"])
async def chat_with_ai(req: ChatRequest):
    if not GEMINI_API_KEY:
        return {"reply": "Lỗi: API Key chưa được cấu hình trên Server."}
        
    # SỬA: Endpoint v1beta hỗ trợ tốt nhất cho gemini-1.5-flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": f"{SYSTEM_INSTRUCTION}\n\nNgười dùng hỏi: {req.message}"}]
        }],
        "generationConfig": {
            "maxOutputTokens": 400,
            "temperature": 0.4,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if 'candidates' in data and len(data['candidates']) > 0:
                    ai_reply = data['candidates'][0]['content']['parts'][0]['text']
                    return {"reply": ai_reply}
                return {"reply": "AI không tìm được câu trả lời phù hợp."}
            else:
                # Log lỗi chi tiết để debug trên Railway logs
                print(f"Gemini API Error: Status {response.status_code}, Body: {response.text}")
                return {"reply": f"Hiện tại tôi không thể trả lời (Lỗi {response.status_code})."}
    except Exception as e:
        print(f"Exception: {str(e)}")
        return {"reply": "Lỗi hệ thống khi kết nối với trí tuệ nhân tạo."}
