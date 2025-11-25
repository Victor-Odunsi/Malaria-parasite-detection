import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from app.backend.config import get_settings, Settings
from typing import Optional
import shutil
import logging
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelDownloadError(Exception):
    pass

def download_with_huggingface_hub(repo_name:str, filename:str, local_path:Path, token:Optional[str] = None) -> bool:
    try:
        logger.info('Attempting to download with hf_hub_download')
        file_path = hf_hub_download(
            repo_id=repo_name,
            filename=filename,
            cache_dir=local_path,
            token=token
        )

        shutil.copy2(file_path, local_path)

        logger.info(f"Model downloaded successfully to {local_path}")
        return True
    
    except ImportError:
        logger.warning('huggingface_hub not installed, will try manual download')
        return False    
    except Exception as e:
        logger.warning(f"Failed to download model: {e}, will try manual download")
        return False
    
def download_with_requests(url: str, local_path: Path, chunk_size: int = 8192, timeout: int = 300) -> bool:
    try:
        logger.info(f"Attempting manual download from: {url}")
        
        # Stream download with timeout
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Get total file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(local_path, 'wb') as f, tqdm(
            desc=f"Downloading {local_path.name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        logger.info("✓ Download successful via requests")
        return True
        
    except requests.exceptions.Timeout:
        logger.error(f"Download timed out after {timeout} seconds")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        return False
    
def get_huggingface_url(repo_id: str, filename: str, revision: str = "main") -> str:

    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"

def download_model(settings: Settings):
    model_path = settings.get_model_path()
    settings.make_model_dir()

    success = download_with_huggingface_hub(
        repo_name=settings.hf_repo_name,
        filename=settings.hf_model_filename,
        local_path=model_path,
    )

    if not success:
        logger.info("Falling back to manual download method...")
        url = get_huggingface_url(
            repo_id=settings.hf_repo_name,
            filename=settings.hf_model_filename
        )
        success = download_with_requests(
            url=url,
            local_path=model_path
        )
    if success:
        logger.info(f"Model ready {model_path}")
        return model_path
    
    if model_path.exists():
        model_path.unlink()

    raise ModelDownloadError(
        f"Failed to download model from Hugging Face: {settings.hf_repo_name}/{settings.hf_model_filename}"
    )

def ensure_model_exists(settings: Settings) -> bool:
    model_path = settings.get_model_path()
    if model_path.exists():
        logger.info(f"Model already exists at {model_path}")
        return True
    else:
        return False



