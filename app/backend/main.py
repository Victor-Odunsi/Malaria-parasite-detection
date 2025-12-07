"""
FastAPI backend for Malaria Parasite Detection.
Includes Grafana Cloud monitoring and CORS support.
"""

import io
import logging
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from PIL import Image

from app.backend.pusher import start_pusher
from app.backend.config import settings
from app.backend.download_from_hf import download_model
from app.backend.model import load_model, get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Custom business metrics
predictions_total = Counter(
    'malaria_predictions_total', 
    'Total predictions', 
    ['result', 'status']
)
prediction_latency = Histogram(
    'malaria_prediction_latency_seconds', 
    'Prediction processing time', 
    ['result']
)
image_processing_time = Histogram(
    'malaria_image_processing_seconds', 
    'Image load/decode time'
)
model_inference_time = Histogram(
    'malaria_model_inference_seconds', 
    'Model inference time'
)
infection_rate = Gauge(
    'malaria_infection_rate', 
    'Current infection detection rate'
)
image_size_bytes = Histogram(
    'malaria_image_size_bytes', 
    'Uploaded image sizes'
)
errors_total = Counter(
    'malaria_errors_total', 
    'Total errors', 
    ['error_type']
)

# Track infection rate
infections_count = 0
total_predictions = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """ 
    try:
        logger.info("🚀 Starting Malaria Detection API...")
        
        # Ensure directories exist
        settings.make_model_dir()

        start_pusher()
        
        # Download ONNX model from HuggingFace (or use cached)
        logger.info("Downloading/checking ONNX model from HuggingFace...")
        onnx_model_path = download_model(settings)
        
        # Load ONNX model into memory
        logger.info("Loading ONNX model into memory...")
        load_model(onnx_model_path, settings)
        logger.info("✓ ONNX model loaded successfully")
        logger.info("✓ API is ready to accept requests")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise
    
    yield  # Application runs here
    
    # Shutdown (cleanup if needed)
    logger.info("🛑 Shutting down Malaria Detection API...")


# Initialize FastAPI with lifespan
app = FastAPI(
    title="Malaria Parasite Detection API",
    description="Upload blood cell images to detect malaria parasites",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Prediction-Message", "X-Infected", "X-Processing-Time"]
)

# Setup Prometheus metrics for Grafana Cloud
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)

instrumentator.instrument(app).expose(app, endpoint="/metrics")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Malaria Parasite Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict (ONNX)": "/predict (POST)",
            "health": "/health (GET)",
            "metrics": "/metrics (GET)"
        },
        "model": {
            "repo": settings.hf_repo_name,
            "confidence_threshold": settings.confidence_threshold
        }
    }   

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns API status and model readiness.
    """
    try:
        # Check if model is loaded
        model = get_model()
        model_loaded = model is not None
        
        return {
            "status": "healthy",
            "model_loaded": model_loaded,
            "model_path": str(settings.get_model_path()),
        }
    except RuntimeError:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "error": "Model not loaded"
            }
        )


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Predict malaria parasites in uploaded blood cell image.
    
    Args:
        file: Image file (JPG, PNG)
        
    Returns:
        Annotated image if infection detected, original if not
        
    Response Headers:
        X-Prediction-Message: Detection message
        X-Infected: "true" or "false"
        X-Processing-Time: Processing time in seconds
    """
    global infections_count, total_predictions
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            errors_total.labels(error_type='invalid_file_type').inc()
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Please upload an image."
            )
        
        # Read image and measure size
        logger.info(f"Processing image: {file.filename}")
        contents = await file.read()
        image_size_bytes.observe(len(contents))
        
        # Open as PIL Image and measure decode time
        decode_start = time.time()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            errors_total.labels(error_type='image_decode_failed').inc()
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. Could not decode image."
            )
        image_processing_time.observe(time.time() - decode_start)
        
        # Get model and run prediction (measure inference time)
        inference_start = time.time()
        model = get_model()
        result = model.predict(image)
        model_inference_time.observe(time.time() - inference_start)
        
        # Extract results
        result_image = result["image"]
        message = result["message"]
        has_infection = "Malaria" in message
        
        # Update metrics
        total_predictions += 1
        if has_infection:
            infections_count += 1
        
        # Calculate and set infection rate
        infection_rate.set(
            infections_count / total_predictions if total_predictions > 0 else 0
        )
        
        result_label = "infected" if has_infection else "not_infected"
        predictions_total.labels(result=result_label, status='success').inc()
        
        # Log prediction
        latency = time.time() - start_time
        prediction_latency.labels(result=result_label).observe(latency)
        
        logger.info(
            f"Prediction complete: {message} "
            f"(latency: {latency:.2f}s, file: {file.filename})"
        )
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        # Return image with custom headers
        return StreamingResponse(
            img_byte_arr,
            media_type="image/jpeg",
            headers={
                "X-Prediction-Message": message,
                "X-Infected": str(has_infection).lower(),
                "X-Processing-Time": f"{latency:.2f}s"
            }
        )
        
    except HTTPException:
        predictions_total.labels(result='error', status='http_error').inc()
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        errors_total.labels(error_type='unknown').inc()
        predictions_total.labels(result='error', status='system_error').inc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )