FROM python:3.10

# Thư mục làm việc trong container
WORKDIR /app

# Copy toàn bộ code vào container
COPY . /app

# Cập nhật pip và cài các package cần thiết
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# EXPOSE port 7860 (bắt buộc với Hugging Face Spaces Docker)
EXPOSE 7860

# Lệnh khởi động app (đúng như local)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
