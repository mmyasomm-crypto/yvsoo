"""
LaMa ONNX inpainting engine.

Model I/O (verified against the shipped inpainting_lama_2025jan.onnx):
  input  "image": float32 NCHW, [0,1]-normalized RGB, fixed 512x512
  input  "mask":  float32 NCHW, [0,1] binary (1 = hole to fill), fixed 512x512
  output "output": float32 NCHW, RGB pixel values in [0,255], 512x512

Since the graph has a fixed 512x512 spatial size, arbitrary-resolution
requests are handled by padding to a square (edge-replicate, so the
model sees plausible context instead of a hard black border) and
resizing to 512x512; the result is resized back and cropped to the
original request size.
"""
import logging
import os
import threading
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from .config import settings

logger = logging.getLogger("yofilter.eraser.lama")

_MODEL_SIZE = settings.model_input_size  # 512


def _cgroup_cpu_quota() -> int | None:
    """Return the number of CPUs actually granted to this process by a
    Docker/Kubernetes CPU *quota* (e.g. `--cpus=2`), or None if no quota
    cgroup file is present / parseable.

    This matters because `os.cpu_count()` reports the number of CPUs the
    kernel knows about on the host, which is NOT the same thing as the
    fractional CPU-time quota a container may be throttled to. A container
    with `--cpus=2` on a 32-logical-CPU host still has `os.cpu_count()==32`,
    so blindly using that as `intra_op_num_threads` makes ONNX Runtime spin
    up far more worker threads than the container can ever actually run
    concurrently. Those extra threads don't add parallelism (there's no
    spare CPU time for them) -- they only add scheduling/context-switch
    contention, which is measured (see bench.py) to scale inference time
    up sharply as thread count exceeds real available cores.
    """
    try:
        # cgroup v2: single file "cpu.max" = "<quota> <period>" or "max <period>"
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota_str, period_str = f.read().split()
        if quota_str == "max":
            return None
        quota, period = int(quota_str), int(period_str)
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (FileNotFoundError, ValueError, OSError):
        pass

    try:
        # cgroup v1: two files, quota is -1 when unset/unlimited
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read().strip())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (FileNotFoundError, ValueError, OSError):
        pass

    return None


def _detect_safe_intra_threads() -> int:
    """Pick an intra-op thread count that matches CPUs this process can
    actually use concurrently, taking the most conservative (smallest) of:
      - the CPU affinity mask (respects cpuset-style container limits)
      - any Docker/Kubernetes CPU *quota* limit (respects `--cpus=N` limits)
      - os.cpu_count() (host logical CPU count, used only as a fallback)
    Always returns at least 1.
    """
    candidates = []

    try:
        candidates.append(len(os.sched_getaffinity(0)))  # POSIX only
    except AttributeError:
        pass

    quota_cpus = _cgroup_cpu_quota()
    if quota_cpus is not None:
        candidates.append(quota_cpus)

    if not candidates:
        candidates.append(os.cpu_count() or 4)

    chosen = max(1, min(candidates))
    logger.info(
        "CPU auto-detect: affinity/cgroup candidates=%s os.cpu_count()=%s -> intra_op_num_threads=%s",
        candidates, os.cpu_count(), chosen,
    )
    return chosen


class LamaEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session: ort.InferenceSession | None = None
        self.providers = []
        self.device = "cpu"
        self.load_time_ms: int | None = None
        self._lock = threading.Lock()  # onnxruntime session.run is not guaranteed thread-safe across sessions config
        self._load()

    def _load(self):
        t0 = time.time()
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Single sequential graph, one request in flight per call (we already
            # serialize with self._lock) -- ORT_SEQUENTIAL avoids the overhead of
            # ORT's parallel executor scheduling ops across an inter-op thread pool
            # that this graph shape doesn't benefit from.
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # Intra-op (within-op, e.g. conv) parallelism is what actually speeds
            # up a single inference on CPU -- but only up to the number of CPUs
            # this process can actually use concurrently. Measured (bench.py):
            # threads beyond the real usable core count make inference SLOWER,
            # not faster (thread contention, no added parallelism), so "auto"
            # is affinity/cgroup-quota aware rather than blindly os.cpu_count().
            # Operators can still force an exact value via ONNX_INTRA_THREADS.
            intra_threads = settings.onnx_intra_threads or _detect_safe_intra_threads()
            so.intra_op_num_threads = intra_threads
            so.inter_op_num_threads = settings.onnx_inter_threads
            self.session = ort.InferenceSession(
                self.model_path, sess_options=so, providers=settings.onnx_providers
            )
            self.providers = self.session.get_providers()
            self.device = "cuda" if any("CUDA" in p for p in self.providers) else "cpu"
            self.load_time_ms = int((time.time() - t0) * 1000)
            logger.info(
                "LaMa model loaded from %s providers=%s intra_threads=%s inter_threads=%s elapsed_ms=%s",
                self.model_path, self.providers, intra_threads, settings.onnx_inter_threads, self.load_time_ms,
            )
        except Exception:
            logger.exception("Failed to load LaMa model from %s", self.model_path)
            self.session = None
            self.load_time_ms = int((time.time() - t0) * 1000)

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        if self.session is None:
            raise RuntimeError("Model not loaded")

        orig_w, orig_h = image.size
        side = max(orig_w, orig_h)

        # Pad to a square using edge-replication so the model has real
        # context right up to the border instead of an artificial edge.
        padded_img = Image.new("RGB", (side, side))
        padded_img.paste(image, (0, 0))
        if side > orig_w or side > orig_h:
            padded_img = _edge_pad(image, side)

        padded_mask = Image.new("L", (side, side), color=0)
        padded_mask.paste(mask, (0, 0))

        model_img = padded_img.resize((_MODEL_SIZE, _MODEL_SIZE), Image.BILINEAR)
        model_mask = padded_mask.resize((_MODEL_SIZE, _MODEL_SIZE), Image.NEAREST)

        img_np = np.asarray(model_img, dtype=np.float32) / 255.0  # HWC, [0,1]
        img_np = img_np.transpose(2, 0, 1)[None, ...]  # 1,3,512,512

        mask_np = np.asarray(model_mask, dtype=np.float32) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)[None, None, ...]  # 1,1,512,512

        with self._lock:
            out = self.session.run(None, {"image": img_np, "mask": mask_np})[0]

        out = np.clip(out[0], 0, 255).astype(np.uint8).transpose(1, 2, 0)  # HWC uint8
        out_img = Image.fromarray(out, mode="RGB")

        out_img = out_img.resize((side, side), Image.BILINEAR)
        out_img = out_img.crop((0, 0, orig_w, orig_h))
        return out_img


def _edge_pad(image: Image.Image, side: int) -> Image.Image:
    """Pad `image` to a `side`x`side` square by replicating edge pixels."""
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    pad_h = side - h
    pad_w = side - w
    padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return Image.fromarray(padded)


_engine: LamaEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> LamaEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = LamaEngine(settings.model_path)
    return _engine
