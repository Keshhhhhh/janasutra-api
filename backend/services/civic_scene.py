"""
Civic scene classifier: CLIP ViT-B/32 image encoder (int8 ONNX) compared with
precomputed text-prompt embeddings, for scene-level problems box detectors miss
(fallen trees, garbage, flooding, debris, open manholes). Built and measured by
training/export_clip_scene.py; thresholds and prompts ship in civic_scene.json.
"""

import io
import json
import os
import threading

import numpy as np
from PIL import Image

from backend.config import settings

_state = {}
_lock = threading.Lock()


def available() -> bool:
    return settings.CIVIC_SCENE_ENABLED and os.path.exists(settings.CIVIC_SCENE_MODEL_PATH) \
        and os.path.exists(settings.CIVIC_SCENE_CONFIG_PATH)


def _load():
    if not _state:
        with _lock:
            if not _state:
                import onnxruntime as ort
                cfg = json.load(open(settings.CIVIC_SCENE_CONFIG_PATH))
                _state.update(cfg=cfg, text=np.asarray(cfg["textEmbeddings"], dtype=np.float32),
                              sess=ort.InferenceSession(settings.CIVIC_SCENE_MODEL_PATH,
                                                        providers=["CPUExecutionProvider"]))
    return _state


def _preprocess(image: Image.Image, cfg: dict) -> np.ndarray:
    n = cfg["inputSize"]
    s = n / min(image.size)
    image = image.resize((max(n, round(image.width * s)), max(n, round(image.height * s))), Image.BICUBIC)
    left, top = (image.width - n) // 2, (image.height - n) // 2
    a = (np.asarray(image.crop((left, top, left + n, top + n)), dtype=np.float32) / 255 - cfg["mean"]) / cfg["std"]
    return a.transpose(2, 0, 1)[None].astype(np.float32)


def classify_scene(image_bytes: bytes) -> dict | None:
    """None when disabled/not installed, else {"detected", "issues": [{name, label_en, subcategory_id,
    probability}] above their thresholds, "probabilities": {name: p}}."""
    if not available():
        return None
    st = _load()
    cfg = st["cfg"]
    v = st["sess"].run(None, {"pixels": _preprocess(Image.open(io.BytesIO(image_bytes)).convert("RGB"), cfg)})[0][0]
    z = cfg["logitScale"] * st["text"] @ v
    p = np.exp(z - z.max())
    p /= p.sum()
    probs = {}
    for prompt, x in zip(cfg["prompts"], p):
        probs[prompt["key"]] = probs.get(prompt["key"], 0.0) + float(x)
    issues = [{"name": c["name"], "label_en": c["label_en"], "subcategory_id": c["subcategory_id"],
               "probability": round(probs[c["name"]], 3)}
              for c in cfg["classes"] if probs[c["name"]] >= c["threshold"]]
    issues.sort(key=lambda i: -i["probability"])
    return {"detected": bool(issues), "issues": issues,
            "probabilities": {k: round(v, 3) for k, v in probs.items()}}
