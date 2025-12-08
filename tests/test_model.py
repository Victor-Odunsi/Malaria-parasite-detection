"""
Unit tests for ONNX model inference.
Updated for YOLOv8/v11 format with NMS.
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch

from app.backend.model import MalariaDetector, load_model, get_model
from app.backend.config import Settings
from app.backend.model import _model_instance

@pytest.fixture(autouse=True)
def reset_model_instance():
    global _model_instance
    _model_instance = None

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
    
    @patch('app.backend.model.ort.InferenceSession')
    def test_model_loads_successfully(self, mock_session_class, mock_model_path, test_settings):
        """Test model initialization."""

        # Mock input
        mock_input = Mock()
        mock_input.name = "images"
        mock_input.shape = [1, 3, 640, 640]

        # Mock output
        mock_output = Mock()
        mock_output.name = "output0"

        # Mock session
        mock_session_instance = Mock()
        mock_session_instance.get_inputs.return_value = [mock_input]
        mock_session_instance.get_outputs.return_value = [mock_output]
        mock_session_class.return_value = mock_session_instance

        detector = MalariaDetector(mock_model_path, test_settings)

        assert detector.session is not None
        assert detector.input_name == "images"


    def test_model_file_not_found_raises_error(self, test_settings):
        """Test error when model file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            MalariaDetector(Path("nonexistent.onnx"), test_settings)

    @patch('app.backend.model.ort.InferenceSession')
    def test_preprocess_image(self, mock_session_class, mock_model_path, test_settings, test_image):
        """Test image preprocessing produces correct output."""
        mock_session_class.return_value = Mock()
        mock_session_class.return_value.get_inputs.return_value = [
            Mock(name="images", shape=[1, 3, 640, 640])
        ]
        mock_session_class.return_value.get_outputs.return_value = [Mock(name="output0")]
        
        detector = MalariaDetector(mock_model_path, test_settings)
        img_array, scale, padding = detector._preprocess_image(test_image)
        
        # Check output format
        assert img_array.shape == (1, 3, 640, 640)
        assert img_array.dtype == np.float32
        assert 0 <= img_array.min() <= 1
        assert 0 <= img_array.max() <= 1
        assert isinstance(scale, float)
        assert isinstance(padding, tuple)
        assert len(padding) == 2
    
    @patch('app.backend.model.ort.InferenceSession')
    def test_preprocess_converts_grayscale_to_rgb(self, mock_session_class, mock_model_path, test_settings):
        """Test preprocessing converts non-RGB images to RGB."""
        mock_session_class.return_value = Mock()
        mock_session_class.return_value.get_inputs.return_value = [
            Mock(name="images", shape=[1, 3, 640, 640])
        ]
        mock_session_class.return_value.get_outputs.return_value = [Mock(name="output0")]
        
        detector = MalariaDetector(mock_model_path, test_settings)
        gray_image = Image.new('L', (640, 640))
        
        img_array, _, _ = detector._preprocess_image(gray_image)
        
        assert img_array.shape[1] == 3  # Should have 3 channels
    
    @patch('app.backend.model.ort.InferenceSession')
    def test_extract_infected_cells_with_yolov8_format(self, mock_session_class, mock_model_path, test_settings):
        """Test extraction with YOLOv8/v11 output format (1, 6, 8400)."""
        mock_session_class.return_value = Mock()
        mock_session_class.return_value.get_inputs.return_value = [
            Mock(name="images", shape=[1, 3, 640, 640])
        ]
        mock_session_class.return_value.get_outputs.return_value = [Mock(name="output0")]
        
        detector = MalariaDetector(mock_model_path, test_settings)
        
        # Create YOLOv8/v11 format output: (1, 6, 8400)
        # Format: [x, y, w, h, class0_conf, class1_conf]
        mock_output = np.zeros((1, 6, 8400), dtype=np.float32)
        
        # Add one strong infected detection (class 0)
        # Prediction at index 0: x=320, y=320, w=100, h=100, conf0=0.9, conf1=0.1
        mock_output[0, 0, 0] = 320 
        mock_output[0, 1, 0] = 320 
        mock_output[0, 2, 0] = 100 
        mock_output[0, 3, 0] = 100 
        mock_output[0, 4, 0] = 0.9 
        mock_output[0, 5, 0] = 0.1 
        
        boxes = detector._extract_infected_cells(
            [mock_output],
            scale=1.0,
            padding=(0, 0),
            original_size=(640, 640)
        )
        
        assert len(boxes) == 1
        assert len(boxes[0]) == 4  # (x1, y1, x2, y2)
    
    @patch('app.backend.model.ort.InferenceSession')
    def test_nms_removes_duplicate_boxes(self, mock_session_class, mock_model_path, test_settings):
        """Test that NMS removes overlapping duplicate detections."""
        mock_session_class.return_value = Mock()
        mock_session_class.return_value.get_inputs.return_value = [
            Mock(name="images", shape=[1, 3, 640, 640])
        ]
        mock_session_class.return_value.get_outputs.return_value = [Mock(name="output0")]
        
        detector = MalariaDetector(mock_model_path, test_settings)
        
        # Create overlapping boxes (same object detected 3 times)
        boxes = [
            [100, 100, 200, 200],
            [105, 105, 205, 205],
            [110, 110, 210, 210],
        ]
        scores = [0.9, 0.8, 0.7]
        
        nms_boxes = detector._apply_nms(boxes, scores, iou_threshold=0.5)
        
        # Should keep only the highest confidence box
        assert len(nms_boxes) == 1
        assert nms_boxes[0] == [100, 100, 200, 200]
    
    @patch('app.backend.model.ort.InferenceSession')
    def test_nms_keeps_non_overlapping_boxes(self, mock_session_class, mock_model_path, test_settings):
        """Test that NMS keeps non-overlapping detections."""
        mock_session_class.return_value = Mock()
        mock_session_class.return_value.get_inputs.return_value = [
            Mock(name="images", shape=[1, 3, 640, 640])
        ]
        mock_session_class.return_value.get_outputs.return_value = [Mock(name="output0")]
        
        detector = MalariaDetector(mock_model_path, test_settings)
        
        # Create non-overlapping boxes (different objects)
        boxes = [
            [100, 100, 200, 200],
            [300, 300, 400, 400],
            [500, 500, 600, 600],
        ]
        scores = [0.9, 0.85, 0.8]
        
        nms_boxes = detector._apply_nms(boxes, scores, iou_threshold=0.5)
        
        # Should keep all boxes since they don't overlap
        assert len(nms_boxes) == 3
    
    @patch('app.backend.model.ort.InferenceSession')
    def test_predict_with_infection_returns_annotated_image(self, mock_session_class, mock_model_path, test_settings, test_image):
        """Test that infected cells trigger annotation."""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [
            Mock(name="images", shape=[1, 3, 640, 640])
        ]
        mock_session.get_outputs.return_value = [Mock(name="output0")]
        
        # Create YOLOv8 format output with infected detection
        mock_output = np.zeros((1, 6, 8400), dtype=np.float32)
        mock_output[0, 0, 0] = 320 
        mock_output[0, 1, 0] = 320 
        mock_output[0, 2, 0] = 100 
        mock_output[0, 3, 0] = 100 
        mock_output[0, 4, 0] = 0.9 
        mock_output[0, 5, 0] = 0.1
        
        mock_session.run.return_value = [mock_output]
        mock_session_class.return_value = mock_session
        
        detector = MalariaDetector(mock_model_path, test_settings)
        result = detector.predict(test_image)
        
        assert "image" in result
        assert "message" in result
        assert result["message"] == "Malaria Parasite cells detected"
        assert isinstance(result["image"], Image.Image)
    
    @patch('app.backend.model.ort.InferenceSession')
    def test_predict_without_infection_returns_original_image(self, mock_session_class, mock_model_path, test_settings, test_image):
        """Test that no infected cells returns original image."""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [
            Mock(name="images", shape=[1, 3, 640, 640])
        ]
        mock_session.get_outputs.return_value = [Mock(name="output0")]
        
        # Create output with only uninfected cells (class 1)
        mock_output = np.zeros((1, 6, 8400), dtype=np.float32)
        mock_output[0, 0, 0] = 320
        mock_output[0, 1, 0] = 320
        mock_output[0, 2, 0] = 100
        mock_output[0, 3, 0] = 100
        mock_output[0, 4, 0] = 0.1 
        mock_output[0, 5, 0] = 0.9 
        
        mock_session.run.return_value = [mock_output]
        mock_session_class.return_value = mock_session
        
        detector = MalariaDetector(mock_model_path, test_settings)
        result = detector.predict(test_image)
        
        assert result["message"] == "No infected cell detected"
        assert isinstance(result["image"], Image.Image)


class TestGlobalModelFunctions:
    """Test global model loading and singleton pattern."""
    
    def teardown_method(self):
        """Reset global model instance after each test."""
        import app.backend.model
        app.backend.model._model_instance = None
    
    def test_get_model_raises_error_if_not_loaded(self):
        """Test get_model raises error when model not loaded."""
        global _model_instance
        _model_instance = None
        with pytest.raises(RuntimeError, match="Model not loaded"):
            get_model()
    
    @patch('app.backend.model.MalariaDetector')
    def test_get_model_returns_loaded_instance(self, mock_detector_class, mock_model_path, test_settings):
        """Test get_model returns the loaded model."""
        mock_instance = Mock()
        mock_detector_class.return_value = mock_instance
        
        load_model(mock_model_path, test_settings)
        model = get_model()
        
        assert model is mock_instance