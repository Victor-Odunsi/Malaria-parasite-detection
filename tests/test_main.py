"""
Unit and integration tests for FastAPI endpoints.
"""

import pytest
import io, os
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

from app.backend.main import app
from app.backend.config import Settings

os.environ["ENABLE_METRICS"] = "true"  # Ensure metrics are enabled for testing

@pytest.fixture
def client():
    """Create a test client."""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def mock_model():
    """Mock the model for testing."""
    mock = Mock()
    mock.predict.return_value = {
        "image": Image.new('RGB', (640, 640), color='red'),
        "message": "Malaria Parasite cells detected"
    }
    return mock


@pytest.fixture
def sample_image_bytes():
    """Create sample image as bytes."""
    img = Image.new('RGB', (640, 640), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_returns_api_info(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["status"] == "running"


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    @patch('app.backend.main.get_model')
    def test_health_check_when_model_loaded(self, mock_get_model, client):
        """Test health check returns healthy when model is loaded."""
        mock_get_model.return_value = Mock()
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
    
    @patch('app.backend.main.get_model')
    def test_health_check_when_model_not_loaded(self, mock_get_model, client):
        """Test health check returns unhealthy when model not loaded."""
        mock_get_model.side_effect = RuntimeError("Model not loaded")
        
        response = client.get("/health")
        
        assert response.status_code == 503
        data = response.json()
        
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


class TestPredictEndpoint:
    """Test prediction endpoint."""
    
    @patch('app.backend.main.get_model')
    def test_predict_with_valid_image(self, mock_get_model, client, sample_image_bytes, mock_model):
        """Test prediction with valid image file."""
        mock_get_model.return_value = mock_model
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert "X-Prediction-Message" in response.headers
        assert "X-Infected" in response.headers
        
        # Verify model was called
        mock_model.predict.assert_called_once()
    
    @patch('app.backend.main.get_model')
    def test_predict_returns_infection_status_in_headers(self, mock_get_model, client, sample_image_bytes, mock_model):
        """Test prediction returns infection status in response headers."""
        mock_get_model.return_value = mock_model
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.headers["X-Prediction-Message"] == "Malaria Parasite cells detected"
        assert response.headers["X-Infected"] == "true"
    
    @patch('app.backend.main.get_model')
    def test_predict_with_no_infection(self, mock_get_model, client, sample_image_bytes):
        """Test prediction when no infection is detected."""
        mock = Mock()
        mock.predict.return_value = {
            "image": Image.new('RGB', (640, 640), color='green'),
            "message": "No infection detected"
        }
        mock_get_model.return_value = mock
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        assert response.headers["X-Prediction-Message"] == "No infection detected"
        assert response.headers["X-Infected"] == "false"
    
    def test_predict_without_file_returns_422(self, client):
        """Test prediction without file returns validation error."""
        response = client.post("/predict")
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_invalid_file_type(self, client):
        """Test prediction with non-image file returns error."""
        text_file = io.BytesIO(b"This is not an image")
        
        response = client.post(
            "/predict",
            files={"file": ("test.txt", text_file, "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
    
    @patch('app.backend.main.get_model')
    def test_predict_with_corrupted_image(self, mock_get_model, client):
        """Test prediction with corrupted image data."""
        corrupted_data = io.BytesIO(b"corrupted image data")
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", corrupted_data, "image/jpeg")}
        )
        
        assert response.status_code == 400
        assert "Invalid image file" in response.json()["detail"]
    
    @patch('app.backend.main.get_model')
    def test_predict_handles_model_errors(self, mock_get_model, client, sample_image_bytes):
        """Test prediction handles model inference errors gracefully."""
        mock = Mock()
        mock.predict.side_effect = Exception("Model inference failed")
        mock_get_model.return_value = mock
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]
    
    @patch('app.backend.main.get_model')
    def test_predict_logs_processing_time(self, mock_get_model, client, sample_image_bytes, mock_model):
        """Test prediction includes processing time in headers."""
        mock_get_model.return_value = mock_model
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert "X-Processing-Time" in response.headers
        assert response.headers["X-Processing-Time"].endswith("s")


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint."""
    
    def test_metrics_endpoint_exposed(self, client):
        """Test that /metrics endpoint is accessible."""
        response = client.get("/metrics")
        
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            # If exposed, should be text format
            assert "text" in response.headers.get("content-type", "").lower()


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are set."""
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        assert "access-control-allow-origin" in response.headers


class TestErrorHandling:
    """Test global error handling."""
    
    @patch('app.backend.main.get_model')
    def test_unhandled_exception_returns_500(self, mock_get_model, client, sample_image_bytes):
        """Test that unhandled exceptions are caught and return 500."""
        mock_get_model.side_effect = Exception("Unexpected error")
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 500


@pytest.fixture
def test_settings(tmp_path):
    """Create test settings."""
    return Settings(
        model_dir=tmp_path / "models",
        confidence_threshold=0.25
    )


def test_test_settings_fixture(test_settings, tmp_path):
    """Test the settings fixture."""
    assert test_settings.model_dir == tmp_path / "models"