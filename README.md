# YoFilter AI Eraser Backend

FastAPI + ONNX Runtime service that runs real LaMa inpainting for the
YoFilter Magic Eraser. CPU by default, GPU-ready.

## Endpoints

- `GET /health` — model load status, ONNX providers, device.
- `POST /api/erase` — `multipart/form-data` with fields `image` and
  `mask` (PNG/JPEG/WebP). Requires `Authorization: Bearer <supabase_jwt>`.
  Returns `image/png`.

## Run locally

```bash
cp .env.example .env        # fill in SUPABASE_JWT_SECRET
docker compose up --build
curl http://localhost:8000/health
```

Model file goes in `models/inpainting_lama.onnx` (already included in
this package, copied from the file you provided).

## GPU

```bash
docker build --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
             --build-arg ONNXRUNTIME_PKG=onnxruntime-gpu \
             -t yofilter-eraser:gpu .
docker run --gpus all -p 8000:8000 --env-file .env \
  -e ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
  yofilter-eraser:gpu
```

## Model I/O (verified against the shipped ONNX file)

- `image`: float32 NCHW, `[0,1]`-normalized RGB, fixed 512x512
- `mask`: float32 NCHW, `[0,1]` binary (1 = area to remove), fixed 512x512
- `output`: float32 NCHW, RGB pixel values in `[0,255]`, 512x512

Arbitrary request resolutions are handled by edge-padding to a square,
resizing to 512x512 for the model, then resizing/cropping the result
back — see `app/lama.py`.

## Security

- Supabase JWT verified per-request (HS256 shared secret, or JWKS if
  the project uses asymmetric signing — see `.env.example`).
- Per-user in-memory sliding-window rate limit (`RATE_LIMIT_REQUESTS`
  / `RATE_LIMIT_WINDOW_SECONDS`). Swap for Redis if you scale to
  multiple processes/pods.
- Uploads are validated for content-type, size (`MAX_UPLOAD_BYTES`)
  and pixel dimensions (`MAX_IMAGE_DIMENSION`) before decoding.
- Images are processed **entirely in memory** — nothing is written to
  disk, so there is no persistent storage and nothing to clean up.
- CORS is an explicit allow-list (`CORS_ORIGINS`).

## Startup-loading fix — what was actually tested in this environment

This sandbox has no outbound network access, so `fastapi`/`uvicorn`
could not be `pip install`-ed here (they were not already present).
`onnxruntime`, `PyJWT`, `Pillow`, and `numpy` *were* already present,
so rather than skip verification, the real `app/main.py`,
`app/lama.py`, and `app/auth.py` — completely unmodified — were
imported and called directly against a small compatibility shim that
only supplies `fastapi`'s class/decorator surface (no HTTP, no
multipart parsing, no dependency injection). Every number below is a
real measurement from that run, on this sandbox's CPU (1 core):

```
STEP 1  lifespan startup  -> model load elapsed_ms=6544, providers=['CPUExecutionProvider']
STEP 2  GET /health        -> {'model_loaded': True, 'load_time_ms': 6544, ...}
STEP 3  1st POST /api/erase -> elapsed_ms=7193 (X-Inference-Ms=7170, X-Engine=lama)
STEP 4  2nd POST /api/erase -> elapsed_ms=6484 (X-Inference-Ms=6462, X-Engine=lama)
        engine.session identity unchanged across both requests: True
        engine.load_time_ms set exactly once (during startup, never again)
```

This confirms, with real code and a real model run (not a mock):
- ✅ The model loads during `lifespan` startup, before any request.
- ✅ Neither `/api/erase` call re-triggers `_load()` — same
  `ort.InferenceSession` object serves both requests.
- ✅ `/health` reports `model_loaded: true` only after startup
  finished loading it.
- ✅ Response carries `X-Engine: lama` and `X-Inference-Ms`.
- ✅ JWT auth: valid HS256 token accepted; missing/malformed tokens
  correctly rejected with 401 (tested both paths directly against the
  real `app/auth.py`).
- ✅ All modules import and byte-compile cleanly.

Note: 6.5s model load / ~7s inference above is *this sandbox's*
single-core CPU, not representative of Windows timing — the reporter's
own `/health` + inference logs (47.5s inference on their machine) are
the reference for real-world CPU timing. What this run demonstrates is
the **shape** of the fix (load-once-at-startup, reuse-forever), which
is CPU-speed-independent.

- ⚠️ **Not verified in this environment** (no outbound network access
  to install `fastapi`/`uvicorn`, or to run `docker build`): the
  actual HTTP transport layer — request routing, CORS enforcement,
  multipart form parsing, and the Docker image build. The code is
  standard FastAPI/uvicorn with no exotic dependencies. To close this
  last gap on your machine (where `fastapi`/`uvicorn` are already
  installed and working per your `/health` check):

  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 8000
  # in another terminal, immediately (no warm-up request needed):
  curl -w "\n%{time_total}s\n" http://localhost:8000/health
  # then two real erases (need a real Supabase access_token):
  curl -s -o out1.png -D - \
    -H "Authorization: Bearer $TOKEN" \
    -F "image=@test.png" -F "mask=@mask.png" \
    http://localhost:8000/api/erase
  curl -s -o out2.png -D - \
    -H "Authorization: Bearer $TOKEN" \
    -F "image=@test.png" -F "mask=@mask.png" \
    http://localhost:8000/api/erase
  ```

  Both should return `200`, carry `X-Engine: lama`, and — since the
  model is now loaded at startup — the first curl should take roughly
  the same time as the second (no ~21s first-request penalty).
