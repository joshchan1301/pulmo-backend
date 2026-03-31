# 1. Base image (nhẹ)
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements trước (tối ưu cache)
COPY requirements.txt .

# 4. Install dependencies (KHÔNG cache → giảm GB)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code
COPY . .

# 6. Expose port (Railway dùng $PORT)
EXPOSE 8000

# 7. Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
