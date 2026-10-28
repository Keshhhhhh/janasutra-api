from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "JanaSutra API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    SECRET_KEY: str = "supersecretkey_janasutra_2026_dev_env_only"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    DATABASE_URL: str = "sqlite:///./janasutra.db"
    POTHOLE_MODEL_PATH: str = "backend/ml_models/pothole.onnx"
    POTHOLE_CONF_THRESHOLD: float = 0.2
    AUTHENTICITY_MODEL_PATH: str = "backend/ml_models/authenticity.onnx"
    AUTHENTICITY_THRESHOLD: float = 0.761  # from training/runs/authenticity/eval_report.json (<=3% false flags on real val photos)
    # Comma-separated list of allowed browser origins, or "*"
    CORS_ORIGINS: str = "*"
    # Seeded account passwords — override via env in any public deployment
    ADMIN_PASSWORD: str = "admin123"
    DEMO_PASSWORD: str = "demo123"

    class Config:
        env_file = ".env"

settings = Settings()
