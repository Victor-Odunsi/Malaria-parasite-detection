import os
from pathlib import Path
from unittest.mock import patch, Mock
import pytest
import requests
from app.backend.config import Settings

from app.backend.download_from_hf import (
    download_with_huggingface_hub,
    download_with_requests,
    get_huggingface_url,
    ensure_model_exists,
    download_model,
    ModelDownloadError
)

class TestDownloadFromHf:

    def test_default_model_version(self):
        url = get_huggingface_url('user_model', 'model.onnx')
        assert url == "https://huggingface.co/user_model/resolve/main/model.onnx"

    def test_custom_model_version(self):
        url = get_huggingface_url('user_model', 'model.onnx', revision='v1.0')
        assert url == "https://huggingface.co/user_model/resolve/v1.0/model.onnx"


class TestDownloadWithRequests:

    @patch('app.backend.download_from_hf.requests.get')
    def test_successful_download(self, mock_get, tmp_path):
        mock_response = Mock()
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content = Mock(return_value = [b"chunk1", b"chunk2"])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        local_path = tmp_path / 'model.onnx'
        url = "http://fakeurl.com/model.onnx"

        result = download_with_requests(url, local_path)

        assert result is True
        assert local_path.exists()
        assert local_path.read_bytes() == b"chunk1chunk2"

        @patch('app.backend.download_from_hf.requests.get')
        def test_timeout_error(self, mock_get, tmp_path):
            """Test download handles timeout."""
            mock_get.side_effect = requests.exceptions.Timeout("Connection timeout")
            
            target_path = tmp_path / "model.onnx"
            url = "https://example.com/model.onnx"
            
            result = download_with_requests(url, target_path, timeout=10)
            
            assert result is False
            assert not target_path.exists()
        
        @patch('app.backend.download_from_hf.requests.get')
        def test_http_error(self, mock_get, tmp_path):
            """Test download handles HTTP errors."""
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
            mock_get.return_value = mock_response
            
            target_path = tmp_path / "model.onnx"
            url = "https://example.com/model.onnx"
            
            result = download_with_requests(url, target_path)
            
            assert result is False

class TestDownloadWithHuggingFaceHub:

    @patch('app.backend.download_from_hf.hf_hub_download')
    @patch('app.backend.download_from_hf.shutil.copy2')
    def test_successful_download(self, mock_copy, mock_hf_download, tmp_path):
        cache_path = Path('tmp/hf_cache/model.onnx')
        mock_hf_download.return_value = cache_path

        local_path = tmp_path / 'model.onnx'

        result = download_with_huggingface_hub(
            repo_name='user_model',
            filename='model.onnx',
            local_path=local_path
        )

        assert result is True
        mock_hf_download.assert_called_once()
        mock_copy.assert_called_once_with(cache_path, local_path)

    @patch('app.backend.download_from_hf.hf_hub_download')
    def test_hf_hub_not_installed(self, mock_hf_download, tmp_path):
        """Test graceful handling when HF hub not installed."""
        # Simulate ImportError
        import sys
        with patch.dict(sys.modules, {'huggingface_hub': None}):
            result = download_with_huggingface_hub(
                repo_name='user_model',
                filename='model.onnx',
                local_path=tmp_path / 'model.onnx'
            )
            
            assert result is False

    @patch('app.backend.download_from_hf.hf_hub_download')
    def test_hf_hub_download_fails(self, mock_hf_download, tmp_path):
        """Test handling when hf_hub_download raises an exception."""
        mock_hf_download.side_effect = Exception("Network error")

        result = download_with_huggingface_hub(
            repo_name='user_model',
            filename='model.onnx',
            local_path=tmp_path / 'model.onnx'
        )

        assert result is False


class TestdownloadModel:

    @patch('app.backend.download_from_hf.download_with_huggingface_hub')
    def test_successful_download_via_hf(self,mock_download_hf, tmp_path):
        """Test model download via hf_hub_download."""
        settings = Settings(model_dir=tmp_path / 'models')
        model_path = settings.get_model_path()

        mock_download_hf.return_value = True

        settings.make_model_dir()
        model_path.write_bytes(b"model_data")

        result = download_model(settings)   

        assert result == model_path
        mock_download_hf.assert_called_once()


    @patch('app.backend.download_from_hf.download_with_huggingface_hub')
    @patch('app.backend.download_from_hf.download_with_requests')
    def test_fallback_to_requests(self, mock_requests, mock_hf, tmp_path):
        """Test fallback to requests when HF hub fails."""
        settings = Settings(model_dir=tmp_path / "models")
        model_path = settings.get_model_path()

        mock_hf.return_value = False
        mock_requests.return_value = True
        
        settings.make_model_dir()
        model_path.write_bytes(b"model_data")
        
        result = download_model(settings)
        
        assert result == model_path
        mock_hf.assert_called_once()
        mock_requests.assert_called_once()


    @patch('app.backend.download_from_hf.download_with_huggingface_hub')
    @patch('app.backend.download_from_hf.download_with_requests')
    def test_all_methods_fail(self, mock_requests, mock_hf, tmp_path):
        
        settings = Settings(model_dir=tmp_path / "models")

        mock_hf.return_value = False
        mock_requests.return_value = False

        with pytest.raises(ModelDownloadError) as exc_info:
            download_model(settings)

        assert "Failed to download model from Hugging Face" in str(exc_info.value)
 
class TestCheckModelExists:
    """Test model existence checking."""
    
    def test_model_exists_and_valid(self, tmp_path):
        """Test returns True for valid model."""
        settings = Settings(model_dir=tmp_path / "models")
        model_path = settings.get_model_path()
        
        settings.make_model_dir()
        model_path.write_bytes(b"0" * (2 * 1024 * 1024))
        
        result = ensure_model_exists(settings)
        assert result is True
    
    def test_model_does_not_exist(self, tmp_path):
        """Test returns False when model doesn't exist."""
        settings = Settings(model_dir=tmp_path / "models")
        
        result = ensure_model_exists(settings)
        assert result is False