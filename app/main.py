"""
YoFilter AI Eraser backend — FastAPI + ONNX Runtime LaMa inpainting.

Endpoints:
  GET  /health          liveness/readiness + model status
  POST /api/erase        real LaMa inpainting (auth required)

Security:
  - Supabase JWT verification (HS256 shared secret)
  - Per-user sliding-window rate limiting
  - Strict content-type / size / dimension validation
  - CORS allow-list from env
  - No image is ever written to persistent storage; temp files (if any)
    are cleaned up immediately after each request via try/finally.
"""
import io
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from PIL import Image

from .auth import get_current_user
from .config import settings
from .lama import get_engine
from .rate_limit import RateLimiter
from .validation import validate_image_upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("yofilter.eraser")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load (and warm) the LaMa ONNX session once, here, before Uvicorn starts
    # accepting connections -- Starlette does not signal "ready" / open the
    # listening socket to traffic until this startup block returns. This is
    # what fixes the 21s-on-first-request bug: get_engine() used to be called
    # lazily, from inside the first /api/erase (and /health) handler, so that
    # request paid for both model load AND inference. Calling it here means
    # every request -- including the very first one -- only ever pays for
    # inference.
    logger.info("Startup: loading LaMa model...")
    t0 = time.time()
    engine = get_engine()
    elapsed_ms = int((time.time() - t0) * 1000)
    if engine.is_loaded:
        logger.info(
            "Startup complete: LaMa model loaded providers=%s elapsed_ms=%s",
            engine.providers, elapsed_ms,
        )
    else:
        logger.error("Startup: LaMa model FAILED to load (elapsed_ms=%s) -- /api/erase will 503", elapsed_ms)
    yield
    # Nothing to release explicitly on shutdown: the ORT session and its
    # arena are freed when the process exits.


app = FastAPI(title="YoFilter AI Eraser API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex or None,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

limiter = RateLimiter(max_requests=settings.rate_limit_requests, window_seconds=settings.rate_limit_window_seconds)


@app.get("/health")
def health():
    # get_engine() here just returns the already-constructed singleton from
    # startup (see lifespan above) -- this call itself does not load or
    # block; it's the same object every request (this endpoint included)
    # shares.
    engine = get_engine()
    return {
        "status": "ok",
        "model_loaded": engine.is_loaded,
        "model_path": engine.model_path,
        "providers": engine.providers,
        "device": engine.device,
        "load_time_ms": engine.load_time_ms,
    }


@app.post("/api/erase")
async def erase(
    request: Request,
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    user=Depends(get_current_user),
):
    logger.info("AI request started user=%s", user["sub"])
    logger.info("Backend reached")

    # ---- Rate limit (per authenticated user) ----
    allowed, retry_after = limiter.check(user["sub"])
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Retry in {retry_after}s.")

    # ---- Validate + decode uploads (size/type/dimension caps) ----
    image_bytes = await validate_image_upload(image, settings.max_upload_bytes)
    mask_bytes = await validate_image_upload(mask, settings.max_upload_bytes)

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    if pil_image.width > settings.max_image_dimension or pil_image.height > settings.max_image_dimension:
        raise HTTPException(status_code=400, detail=f"Image exceeds max dimension of {settings.max_image_dimension}px.")

    if pil_image.size != pil_mask.size:
        # Frontend always sends matching dimensions; guard anyway.
        pil_mask = pil_mask.resize(pil_image.size, Image.NEAREST)

    engine = get_engine()
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Inpainting model is not loaded.")

    logger.info("LaMa inference started engine=lama")
    t0 = time.time()
    try:
        result_img = engine.inpaint(pil_image, pil_mask)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallback triggered — inference failed")
        raise HTTPException(status_code=500, detail="Inference failed.") from exc
    finally:
        # No files were ever written to disk for this request — nothing to clean up.
        # (in-memory only, per the "no permanent image storage" requirement)
        pass

    elapsed_ms = int((time.time() - t0) * 1000)
    logger.info("LaMa inference completed elapsed_ms=%s", elapsed_ms)
    logger.info("erase ok user=%s size=%sx%s elapsed_ms=%s", user["sub"], pil_image.width, pil_image.height, elapsed_ms)

    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    logger.info("Result returned engine=lama")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"X-Inference-Ms": str(elapsed_ms), "X-Engine": "lama"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
