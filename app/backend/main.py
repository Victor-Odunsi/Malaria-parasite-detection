import io
import logging
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from PIL import Image

from app.backend.config import settings
from app.backend.download_from_hf import download_model
from app.backend.model import load_model, get_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    try:
        logger.info("🚀 Starting Malaria Detection API...")
        
        # Ensure directories exist
        settings.make_model_dir()
        
        # Download model from HuggingFace (or use cached)
        logger.info("Downloading/checking model from HuggingFace...")
        model_path = download_model(settings)
        
        # Load model into memory
        logger.info("Loading model into memory...")
        load_model(model_path, settings)
        
        logger.info("✓ Model loaded successfully")
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
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "metrics": "/metrics (GET)"
        },
        "model": {
            "repo": settings.hf_repo_name,
            "confidence_threshold": settings.confidence_threshold
        }
    }


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
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Please upload an image."
            )
        
        # Read image
        logger.info(f"Processing image: {file.filename}")
        contents = await file.read()
        
        # Open as PIL Image
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. Could not decode image."
            )
        
        # Get model and run prediction
        model = get_model()
        result = model.predict(image)
        
        # Extract results
        result_image = result["image"]
        message = result["message"]
        has_infection = "Malaria" in message
        
        # Log prediction
        latency = time.time() - start_time
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
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )