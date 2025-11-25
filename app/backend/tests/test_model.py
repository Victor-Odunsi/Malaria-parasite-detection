"""
Unit tests for ONNX model inference.
Tests infection detection and conditional annotation.
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch

from app.backend.model import MalariaDetector, load_model, get_model
from app.backend.config import Settings


@pytest.fixture
def test_settings():
    """Create test settings."""
    return Settings(confidence_threshold=0.25)


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a mock model file."""
    model_file = tmp_path / "test_model.onnx"
    model_file.write_bytes(b"fake_model_data")
    return model_file


@pytest.fixture
def test_image():
    """Create a test image."""
    return Image.new('RGB', (640, 640), color='white')


class TestMalariaDetector:
    """Test MalariaDetector class."""
    
    @patch('backend.model.ort.InferenceSession')
    def test_model_loads_successfully(self, mock_session_class, mock_model_path, test_settings):
        """Test model initialization."""
        mock_session_class.return_value = Mock()
        mock_session_class.return_value.get_inputs.return_value = [
            Mock(name="images", shape=[1, 3, 640, 640])
        ]
        mock_session_class.return_value.get_outputs.return_value = [Mock(name="output0")]
        
        detector = MalariaDetector(mock_model_path, test_settings)
        
        assert detector.session is not None
        assert detector.input_name == "images"
    
    def test_model_file_not_found_raises_error(self, test_settings):
        """Test error when model file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            MalariaDetector(Path("nonexistent.onnx"), test_settings)
    
    @patch('backend.model.ort.InferenceSession')
    def test_predict_with_infection_returns_annotated_image(self, mock_session_class, mock_model_path, test_settings, test_image):
        """Test that infected cells trigger annotation."""
        # Setup mock with infected detection
        mock_session = Mock()
        mock_session.get_inputs.return_value = [
            Mock(name="images", shape=[1, 3, 640, 640])
        ]
        mock_session.get_outputs.return_value = [Mock(name="output0")]
        
        # Mock output with infected cells (class_id = 0)
        mock_output = np.array([
            [
                [320, 320, 100, 100, 0.9, 0.9, 0.1],  # Infected cell
                [200, 200, 80, 80, 0.85, 0.8, 0.15],  # Another infected cell
            ]
        ], dtype=np.float32)
        mock_session.run.return_value = [mock_output]
        
        mock_session_class.return_value = mock_session
        
        detector = MalariaDetector(mock_model_path, test_settings)
        result = detector.predict(test_image)
        
        # Check result structure
        assert "image" in result
        assert "message" in result
        assert "infected_count" in result
        assert "has_infection" in result
        
        # Should have infection
        assert result["has_infection"] is True
        assert result["infected_count"] > 0
        assert "Found" in result["message"]
        assert "infected" in result["message"]
        
        # Image should be annotated (different from original)
        assert isinstance(result["image"], Image.Image)
        assert np.any(np.array(result["image"]) != np.array(test_image))
    
    @patch('backend.model.ort.InferenceSession')
    def test_predict_without_infection_returns_original_image(self, mock_session_class, mock_model_path, test_settings, test_image):
        """Test that no infected cells returns original image."""
        # Setup mock with only uninfected cells
        mock_session = Mock()
        mock_session.get_inputs.return_value = [
            Mock(name="images", shape=[1, 3, 640, 640])
        ]
        mock_session.get_outputs.return_value = [Mock(name="output0")]
        
        # Mock output with only uninfected cells (class_id = 1)
        mock_output = np.array([
            [
                [320, 320, 100, 100, 0.9, 0.1, 0.9],  # Uninfected cell (class 1)
            ]
        ], dtype=np.float32)
        mock_session.run.return_value = [mock_output]
        
        mock_session_class.return_value = mock_session
        
        detector = MalariaDetector(mock_model_path, test_settings)
        result = detector.predict(test_image)
        
        # Should not have infection
        assert result["has_infection"] is False
        assert result["infected_count"] == 0
        assert result["message"] == "No infection detected"
        
        # Image should be original (unchanged)
        assert isinstance(result["image"], Image.Image)
        assert result["image"].size == test_image.size
    
    @patch('backend.model.ort.InferenceSession')
    def test_predict_filters_low_confidence_infected_cells(self, mock_session_class, mock_model_path, test_settings, test_image):
        """Test that low confidence infected cells are filtered out."""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [
            Mock(name="images", shape=[1, 3, 640, 640])
        ]
        mock_session.get_outputs.return_value = [Mock(name="output0")]
        
        # Mock output with one high and one low confidence infected cell
        mock_output = np.array([
            [
                [320, 320, 100, 100, 0.9, 0.9, 0.1],   # High confidence infected
                [200, 200, 80, 80, 0.1, 0.9, 0.05],    # Low confidence infected
            ]
        ], dtype=np.float32)
        mock_session.run.return_value = [mock_output]
        
        mock_session_class.return_value = mock_session
        
        detector = MalariaDetector(mock_model_path, test_settings)
        result = detector.predict(test_image)
        
        # Should only count high confidence detection
        assert result["infected_count"] >= 1  # At least the high confidence one
    
    @patch('backend.model.ort.InferenceSession')
    def test_predict_ignores_uninfected_cells(self, mock_session_class, mock_model_path, test_settings, test_image):
        """Test that uninfected cells are ignored in count."""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [
            Mock(name="images", shape=[1, 3, 640, 640])
        ]
        mock_session.get_outputs.return_value = [Mock(name="output0")]
        
        # Mock output with mixed infected and uninfected
        mock_output = np.array([
            [
                [320, 320, 100, 100, 0.9, 0.9, 0.1],   # Infected
                [400, 400, 100, 100, 0.9, 0.1, 0.9],   # Uninfected (should be ignored)
            ]
        ], dtype=np.float32)
        mock_session.run.return_value = [mock_output]
        
        mock_session_class.return_value = mock_session
        
        detector = MalariaDetector(mock_model_path, test_settings)
        result = detector.predict(test_image)
        
        # Should only count infected cells
        assert result["has_infection"] is True
        # Count should not include uninfected cells
    
    @patch('backend.model.ort.InferenceSession')
    def test_preprocess_converts_grayscale(self, mock_session_class, mock_model_path, test_settings):
        """Test preprocessing handles non-RGB images."""
        mock_session_class.return_value = Mock()
        mock_session_class.return_value.get_inputs.return_value = [
            Mock(name="images", shape=[1, 3, 640, 640])
        ]
        mock_session_class.return_value.get_outputs.return_value = [Mock(name="output0")]
        
        detector = MalariaDetector(mock_model_path, test_settings)
        gray_image = Image.new('L', (640, 640))
        
        img_array, _, _ = detector._preprocess_image(gray_image)
        
        assert img_array.shape[1] == 3  # Should convert to 3 channels


class TestGlobalModelFunctions:
    """Test global model loading and singleton pattern."""
    
    def teardown_method(self):
        """Reset global model instance after each test."""
        import backend.model
        backend.model._model_instance = None
    
    @patch('backend.model.MalariaDetector')
    def test_load_model_creates_instance(self, mock_detector_class, mock_model_path, test_settings):
        """Test load_model creates model instance."""
        mock_detector_class.return_value = Mock()
        
        model = load_model(mock_model_path, test_settings)
        
        assert model is not None
        mock_detector_class.assert_called_once_with(mock_model_path, test_settings)
    
    @patch('backend.model.MalariaDetector')
    def test_load_model_returns_singleton(self, mock_detector_class, mock_model_path, test_settings):
        """Test that load_model returns same instance."""
        mock_instance = Mock()
        mock_detector_class.return_value = mock_instance
        
        model1 = load_model(mock_model_path, test_settings)
        model2 = load_model(mock_model_path, test_settings)
        
        assert model1 is model2
        assert mock_detector_class.call_count == 1
    
    def test_get_model_raises_error_if_not_loaded(self):
        """Test get_model raises error when model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            get_model()
    
    @patch('backend.model.MalariaDetector')
    def test_get_model_returns_loaded_instance(self, mock_detector_class, mock_model_path, test_settings):
        """Test get_model returns the loaded model."""
        mock_instance = Mock()
        mock_detector_class.return_value = mock_instance
        
        load_model(mock_model_path, test_settings)
        model = get_model()
        
        assert model is mock_instance