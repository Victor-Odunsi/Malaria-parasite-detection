import os
from pathlib import Path
from app.backend.config import Settings
import pytest

class TestSettings:

    def test_hf_repo_settings(self):
        """Test HuggingFace repository settings."""
        settings = Settings()
        
        assert settings.hf_repo_name == "victor-odunsi/malaria-detector-yolov11n"
        assert settings.hf_model_filename == "best.onnx"
    
    def test_cors_origins_default(self):
        """Test default CORS origins."""
        settings = Settings()
        assert settings.cors_allowed_origins == ["http://localhost:5173",
        "https://malaria-parasite-detection-ecru.vercel.app"]

    def test_model_path_name(self):
        settings = Settings()
        model_path = settings.get_model_path()
        assert model_path.name == 'best.onnx'
        assert model_path == Path('models') / 'best.onnx'

    def test_custom_model_path(self):
        settings = Settings(model_dir = Path('custom/path'))
        assert settings.get_model_path() == Path('custom/path') / 'best.onnx'

    def test_env_variable_override(self, monkeypatch):
        """Test that environment variables override default settings."""
        monkeypatch.setenv('HF_REPO_NAME', 'custom/repo-name')
        monkeypatch.setenv('HF_MODEL_FILENAME', 'custom_model.onnx')
        monkeypatch.setenv('CONFIDENCE_THRESHOLD', '0.75')
        
        settings = Settings()
        
        assert settings.hf_repo_name == 'custom/repo-name'
        assert settings.hf_model_filename == 'custom_model.onnx'
        assert settings.confidence_threshold == 0.75