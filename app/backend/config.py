import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    hf_repo_name: str = 'victor-odunsi/malaria-detector-yolov11n'
    hf_model_filename: str = 'best.onnx'

    api_port: int = 8000
    api_host: str = '0.0.0.0'

    model_dir: Path = Path('models')

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8'
    )

    def get_model_path(self) -> Path:
        return self.model_dir / self.hf_model_filename

    def get_hf_model_url(self) -> str:
        return f"https://huggingface.co/{self.hf_repo_name}/resolve/main/{self.hf_model_filename}"
    
    def make_model_dir(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()

