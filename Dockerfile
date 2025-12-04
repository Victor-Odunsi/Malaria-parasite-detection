FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency file
COPY app/backend/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend source code
COPY app/backend /app/app/backend

# Expose FastAPI port
EXPOSE 7860

# Start the backend with uvicorn directly
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
