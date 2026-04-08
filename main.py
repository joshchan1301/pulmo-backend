import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from huggingface_hub import hf_hub_download

# Import hàm inference từ file của bạn
try:
    from model_inference import analyze_xray
except ImportError:
    def analyze_xray(bytes): return {"result": "Model inference module not found"}

# 1. Load biến môi trường
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Cấu hình Prompt hệ thống
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
                token=HF_TOKEN
            )
            print(f"Model đã được tải về tại: {path}")
        except Exception as e:
            print(f"Lỗi khi tải model từ HF: {e}")

download_model()

# 4. Khởi tạo Client Gemini (Sử dụng SDK mới nhất)
# SDK sẽ tự động xử lý endpoint và quản lý kết nối
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Pulmo Vision API",
    description="API phân tích X-quang phổi và Chatbot Pulmo AI",
    version="1.2.0"
)

# 5. Cấu hình CORS
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
        return {"reply": "Lỗi: API Key chưa được cấu hình."}

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=req.message,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low"),
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3
            )
        )
        
        if response.text:
            return {"reply": response.text}
        return {"reply": "AI không thể đưa ra phản hồi lúc này."}

    except Exception as e:
        print(f"Gemini SDK Error: {str(e)}")
        # Trả về thông báo thân thiện cho người dùng
        return {"reply": "Hiện tại Pulmo AI đang bận xử lý dữ liệu khác. Bạn vui lòng thử lại sau nhé!"}
