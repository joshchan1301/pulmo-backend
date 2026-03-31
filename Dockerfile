# 1. Base image (nhẹ)
FROM python:3.10-slim

# 2. Cài đặt các thư viện hệ thống cần thiết cho OpenCV (Headless)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements trước
COPY requirements.txt .

# 5. Install dependencies (Dùng bản CPU-only đã dặn ở trên)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy toàn bộ code
COPY . .

# 7. KHÔNG dùng EXPOSE 8000 cố định (Railway tự quản lý)

# 8. Chạy app với biến môi trường PORT của Railway
# Lưu ý: Dùng dấu ngoặc kép và truyền biến đúng cách
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
