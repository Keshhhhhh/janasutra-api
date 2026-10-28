"""
Road-damage detector for dashcam reports: potholes, alligator cracks, linear
cracks and faded road markings in one YOLO11n model (trained on RDD2022 India +
Czech, see training/prepare_road_damage.py), run with onnxruntime.

The class list, per-class thresholds and the taxonomy subcategory each class
maps to ship next to the model in road_damage.json (written by
training/export_road_damage.py), so a retrained model can change them.
Missing model -> detect_road_damage returns None and the pipeline carries on.
"""

import io
import json
import os
import threading

import numpy as np
from PIL import Image

from backend.config import settings
from backend.services.pothole_detector import NMS_IOU_THRESHOLD, _letterbox, _nms

class Detector:
    """A multi-class YOLO ONNX model plus its JSON config (classes, thresholds, taxonomy map)."""

    def __init__(self, model_path: str, config_path: str):
        self.model_path, self.config_path = model_path, config_path
        self._session = self._config = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return os.path.exists(self.model_path) and os.path.exists(self.config_path)

    def load(self):
        if self._session is None:
            with self._lock:
                if self._session is None:
                    import onnxruntime as ort
                    self._config = json.load(open(self.config_path))
                    self._session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        return self._session, self._config

    def detect(self, image_bytes: bytes) -> dict | None:
        return _detect(self, image_bytes) if self.available() else None


road_damage = Detector(settings.ROAD_DAMAGE_MODEL_PATH, settings.ROAD_DAMAGE_CONFIG_PATH)


def detect_road_damage(image_bytes: bytes) -> dict | None:
    return road_damage.detect(image_bytes)


def _detect(detector: Detector, image_bytes: bytes) -> dict | None:
    """
    Returns None when the model is not installed, else
        {
            "detected": bool, "count": int,
            "detections": [{"class_name", "label_en", "subcategory_id", "confidence",
                            "x1", "y1", "x2", "y2", "area"}],   # 0-1 coords, area = box fraction of the image
            "model": str,
        }
    """
    session, cfg = detector.load()
    classes = cfg["classes"]
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    tensor, scale, pad_x, pad_y = _letterbox(image)
    # output0: (1, 4 box + one score per class, anchors)
    preds = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    scores = preds[4:4 + len(classes)]
    best = scores.argmax(axis=0)
    conf = scores.max(axis=0)
    # Disabled classes (see road_damage.json "note") never pass the threshold
    thresholds = np.array([c["threshold"] if c.get("enabled", True) else 2.0 for c in classes])
    keep = conf >= thresholds[best]
    cx, cy, w, h = preds[:4, keep]
    conf, best = conf[keep], best[keep]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    detections = []
    for k in range(len(classes)):  # NMS per class, like Ultralytics
        idx = np.where(best == k)[0]
        for j in (idx[i] for i in _nms(xyxy[idx], conf[idx], NMS_IOU_THRESHOLD)) if idx.size else []:
            x1, y1, x2, y2 = xyxy[j]
            x1 = min(max((x1 - pad_x) / scale, 0), width) / width
            x2 = min(max((x2 - pad_x) / scale, 0), width) / width
            y1 = min(max((y1 - pad_y) / scale, 0), height) / height
            y2 = min(max((y2 - pad_y) / scale, 0), height) / height
            c = classes[k]
            detections.append({
                "class_name": c["name"], "label_en": c["label_en"], "subcategory_id": c["subcategory_id"],
                "confidence": round(float(conf[j]), 3),
                "x1": round(float(x1), 4), "y1": round(float(y1), 4),
                "x2": round(float(x2), 4), "y2": round(float(y2), 4),
                "area": round(float((x2 - x1) * (y2 - y1)), 4),
            })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return {"detected": bool(detections), "count": len(detections), "detections": detections,
            "model": os.path.basename(detector.model_path)}
