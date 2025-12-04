"""
ONNX model loading and inference for malaria parasite detection.
Returns annotated image only if infected cells are detected.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort

from app.backend.config import Settings

logger = logging.getLogger(__name__)


class MalariaDetector:
    """
    ONNX model wrapper for malaria parasite detection.
    Returns annotated images only when infection is detected.
    """
    
    def __init__(self, model_path: Path, settings: Settings):
        """
        Initialize detector and load ONNX model.
        
        Args:
            model_path: Path to ONNX model file
            settings: Application settings
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.settings = settings
        
        # Load model
        logger.info(f"Loading ONNX model from: {model_path}")
        
        # Resolve to absolute path (helps with Windows permission issues)
        absolute_path = model_path.resolve()
        logger.info(f"Absolute path: {absolute_path}")
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            str(absolute_path),
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"✓ Model loaded - Input: {self.input_name}, Shape: {self.input_shape}")
    
    def predict(self, image: Image.Image) -> Dict:
        """
        Run inference on an image.
        
        Args:
            image: PIL Image in RGB format
            
        Returns:
            Dict with keys:
            - image: PIL Image (annotated if infected, original if not)
            - message: str (e.g., "Found 3 infected cells" or "No infection detected")
            - infected_count: int
            - has_infection: bool
        """
        original_size = image.size
        
        # Preprocess
        input_array, scale, padding = self._preprocess_image(image)
        
        # Inference
        logger.debug(f"Running inference on image of size {original_size}")
        outputs = self.session.run(self.output_names, {self.input_name: input_array})
        
        # Post-process to get infected cells only
        infected_bboxes = self._extract_infected_cells(outputs, scale, padding, original_size)
        
        infected_count = len(infected_bboxes)
        logger.info(f"Found {infected_count} infected cells")
        
        # Return based on detection
        if infected_count > 0:
            # Draw boxes only on infected cells
            annotated_image = self._draw_boxes(image, infected_bboxes)
            message = "Malaria Parasite cells detected"
            
            return {
                "image": annotated_image,
                "message": message
            }
        else:
            # Return original image
            return {
                "image": image,
                "message": "No infected cell detected"
            }
    
    def _preprocess_image(self, image: Image.Image) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess PIL image for YOLO model."""
        # Get target size from model
        target_h, target_w = self.input_shape[2], self.input_shape[3]
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Calculate scaling to maintain aspect ratio
        img_w, img_h = image.size
        scale = min(target_w / img_w, target_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        # Resize and pad (letterbox)
        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        padded = Image.new('RGB', (target_w, target_h), (114, 114, 114))
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        padded.paste(resized, (pad_w, pad_h))
        
        # Convert to array and normalize
        img_array = np.array(padded).astype(np.float32)
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
        img_array /= 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array, scale, (pad_w, pad_h)
    
    def _extract_infected_cells(
        self,
        outputs: list,
        scale: float,
        padding: Tuple[int, int],
        original_size: Tuple[int, int]
    ) -> list:
        """
        Extract bounding boxes for infected cells only (class_id = 0).
        Applies NMS to remove duplicate detections.
        
        Returns:
            List of bounding boxes as (x1, y1, x2, y2) tuples
        """
        infected_boxes = []
        infected_scores = []
        
        output = outputs[0]
        
        # YOLOv8/v11 output format is (1, 6, 8400) -> [batch, features, predictions]
        # Transpose to get (predictions, features)
        if len(output.shape) == 3:
            output = output[0]  # Remove batch: (6, 8400)
            output = output.T   # Transpose: (8400, 6)
        
        pad_w, pad_h = padding
        orig_w, orig_h = original_size
        
        # First pass: collect all infected cells above threshold
        for pred in output:
            if len(pred) < 6:
                continue
            
            # YOLOv8/v11 format: [x, y, w, h, class0_conf, class1_conf]
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]
            
            # Get best class
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            
            # Only keep infected cells (class_id = 0) above threshold
            if class_id != 0 or confidence < self.settings.confidence_threshold:
                continue
            
            # Convert to corner format and scale back to original
            x1 = (x_center - width / 2 - pad_w) / scale
            y1 = (y_center - height / 2 - pad_h) / scale
            x2 = (x_center + width / 2 - pad_w) / scale
            y2 = (y_center + height / 2 - pad_h) / scale
            
            # Clip to image boundaries
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            infected_boxes.append([x1, y1, x2, y2])
            infected_scores.append(confidence)
        
        # Apply NMS to remove duplicates
        if len(infected_boxes) > 0:
            infected_boxes = self._apply_nms(infected_boxes, infected_scores)
        
        return [tuple(box) for box in infected_boxes]
    
    def _apply_nms(self, boxes: list, scores: list, iou_threshold: float = 0.45) -> list:
        """
        Apply Non-Maximum Suppression to remove duplicate boxes.
        
        Args:
            boxes: List of boxes [[x1, y1, x2, y2], ...]
            scores: List of confidence scores
            iou_threshold: IoU threshold for considering boxes as duplicates
            
        Returns:
            List of boxes after NMS
        """
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by confidence (descending)
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            # Keep the box with highest confidence
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union
            
            # Keep boxes with IoU less than threshold
            indices = np.where(iou <= iou_threshold)[0]
            order = order[indices + 1]
        
        return boxes[keep].tolist()
    
    def _draw_boxes(self, image: Image.Image, bboxes: list) -> Image.Image:
        """Draw red bounding boxes on infected cells."""
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", size=16)
        except:
            font = ImageFont.load_default()
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            
            # Draw red box for infected cells
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            
            # Draw label
            label = "infected"
            bbox_coords = draw.textbbox((x1, y1), label, font=font)
            text_width = bbox_coords[2] - bbox_coords[0]
            text_height = bbox_coords[3] - bbox_coords[1]
            
            # Label background
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill=(255, 0, 0)
            )
            
            # Label text
            draw.text((x1 + 2, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
        
        return annotated


# Global singleton instance
_model_instance: Optional[MalariaDetector] = None


def load_model(model_path: Path, settings: Settings) -> MalariaDetector:
    """
    Load the model (singleton pattern).
    
    Args:
        model_path: Path to ONNX model
        settings: Application settings
        
    Returns:
        MalariaDetector instance
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = MalariaDetector(model_path, settings)
        logger.info("Model loaded and ready for inference")
    
    return _model_instance


def get_model() -> MalariaDetector:
    """
    Get the loaded model instance.
    
    Returns:
        MalariaDetector instance
        
    Raises:
        RuntimeError: If model hasn't been loaded yet
    """
    if _model_instance is None:
        raise RuntimeError(
            "Model not loaded. Call load_model() first, "
            "typically during application startup."
        )
    
    return _model_instance