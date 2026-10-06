import os


def _split_csv(val: str):
    return [v.strip() for v in val.split(",") if v.strip()]


class Settings:
    # Supabase JWT verification
    supabase_jwt_secret: str = os.getenv("SUPABASE_JWT_SECRET", "")
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    jwt_audience: str = os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated")

    # CORS
    cors_origins = _split_csv(os.getenv("CORS_ORIGINS", "http://localhost:8080"))
    # Additive regex so common local-dev ports work without editing CORS_ORIGINS
    # (a CORS-blocked request throws in fetch() and is indistinguishable, from the
    # frontend, from "AI backend unavailable" -- silent fallback to the heuristic).
    # Set to empty string to disable and rely on CORS_ORIGINS only (recommended for prod).
    cors_origin_regex: str = os.getenv("CORS_ORIGIN_REGEX", r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$")

    # Model
    model_path: str = os.getenv("LAMA_MODEL_PATH", "/app/models/inpainting_lama.onnx")
    model_input_size: int = int(os.getenv("LAMA_INPUT_SIZE", "512"))
    onnx_providers = _split_csv(os.getenv("ONNX_PROVIDERS", "CPUExecutionProvider"))
    onnx_intra_threads: int = int(os.getenv("ONNX_INTRA_THREADS", "0"))  # 0 = auto-detect os.cpu_count()
    onnx_inter_threads: int = int(os.getenv("ONNX_INTER_THREADS", "1"))  # single-graph inference; 1 is sufficient

    # Request limits
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))  # 15MB per file
    max_image_dimension: int = int(os.getenv("MAX_IMAGE_DIMENSION", "2048"))

    # Rate limiting
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
    rate_limit_window_seconds: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


settings = Settings()
