import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    
    hf_repo_name: str = 'victor-odunsi/malaria-detector-yolov11n'
    hf_model_filename: str = 'best.onnx'

    api_port: int = 8000
    api_host: str = '0.0.0.0'

    confidence_threshold: float = 0.40

    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173/")
    cors_allowed_origins: list = [FRONTEND_URL]

    model_dir: Path = Path('models')

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8'
    )

    def get_model_path(self) -> Path:
        return self.model_dir / self.hf_model_filename
    
    def make_model_dir(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()

def get_settings() -> Settings:
    return settings
