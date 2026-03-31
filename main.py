import os
import requests
import torch
import gc
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

# Import các hàm từ file model_inference.py đã tối ưu
import model_inference

# ================== CẤU HÌNH ==================
MODEL_URL = "https://huggingface.co/joshchan1301/x-ray_img_analysis_ai/resolve/main/swin_best_model.pth"
MODEL_PATH = "swin_best_model.pth"

# ================== TẢI BIẾN MÔI TRƯỜNG ==================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ================== TẢI MODEL TỪ CLOUD ==================
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Đang tải model từ Hugging Face (File này nặng, vui lòng đợi)...")
        try:
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Đã tải xong model!")
        except Exception as e:
            print(f"Lỗi khi tải file model: {e}")

# ================== KHỞI TẠO APP ==================
app = FastAPI(
    title="Pulmo Vision API",
    description="Hệ thống phân loại X-quang phổi và Chatbot y tế AI",
    version="1.1.0"
)

# Cấu hình CORS để Frontend (React/Vue/HTML) có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Biến toàn cục để giữ instance của Analyzer
analyzer = None

@app.on_event("startup")
async def startup_event():
    global analyzer
    download_model()
    print("Đang nạp Model vào RAM...")
    # Gọi hàm load_model từ model_inference.py
    analyzer = model_inference.load_model(MODEL_PATH)
    print("Model đã sẵn sàng hoạt động!")

# ================== API PHÂN TÍCH X-QUANG ==================
@app.post("/api/analyze-xray", tags=["Chẩn đoán hình ảnh"])
async def analyze_xray_api(file: UploadFile = File(...)):
    global analyzer
    try:
        # Đọc dữ liệu ảnh gửi lên
        img_bytes = await file.read()
        
        # Gọi hàm predict từ model_inference.py
        # Lưu ý: analyzer ở đây là đối tượng XRayAnalyzer đã được khởi tạo
        result = model_inference.predict(analyzer, img_bytes)
        
        # Ép buộc giải phóng bộ nhớ tạm sau request
        gc.collect()
        
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Lỗi xử lý X-ray: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Lỗi máy chủ nội bộ: {str(e)}"}
        )

# ================== CHATBOT PULMO AI ==================
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat", tags=["Trợ lý ảo"])
async def chat_with_openai(req: ChatRequest):
    if not OPENAI_API_KEY:
        return {"reply": "Lỗi: Server chưa cấu hình khóa API OpenAI."}

    try:
        # Tăng timeout lên 60s vì đôi khi OpenAI phản hồi chậm
        async with httpx.AsyncClient(timeout=60.0) as client:
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
                                "Bạn là Pulmo AI Assistant, một chuyên gia hỗ trợ giải thích kết quả X-quang phổi. "
                                "Hãy trả lời bằng tiếng Việt một cách chuyên nghiệp, dễ hiểu và nhẹ nhàng. "
                                "Giới hạn câu trả lời trong khoảng 300 tokens."
                            )
                        },
                        {"role": "user", "content": req.message}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.7
                }
            )

        if response.status_code == 200:
            data = response.json()
            return {"reply": data["choices"][0]["message"]["content"]}
        else:
            return {"reply": f"AI đang bận một chút (Mã lỗi: {response.status_code}). Vui lòng thử lại."}

    except Exception as e:
        print(f"Lỗi kết nối OpenAI: {str(e)}")
        return {"reply": "Hiện tại không thể kết nối tới trợ lý ảo. Vui lòng kiểm tra internet."}


if __name__ == "__main__":
    import uvicorn
    import os
    # Lấy port từ biến môi trường của Railway
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
