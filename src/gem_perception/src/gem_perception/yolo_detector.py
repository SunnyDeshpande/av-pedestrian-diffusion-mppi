"""YOLO-World text-promptable detector (Ultralytics)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection2D:
    score: float
    bbox_xyxy: np.ndarray            # (4,) int, [x1,y1,x2,y2]
    mask: np.ndarray                 # (H,W) bool, rectangular mask from bbox
    prompt: str


class YoloWorldDetector:
    """Thin wrapper around ultralytics YOLO-World. Picks top-1 by confidence."""

    def __init__(self, weight_path: str, device: str = "cuda", conf: float = 0.05):
        from ultralytics import YOLO
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"YOLO-World weight not found at {weight_path}. "
                f"Run scripts/download_models.py first."
            )
        self.model = YOLO(weight_path)
        self.device = device
        self.conf = conf
        self._prompt: Optional[str] = None

    def set_prompt(self, prompt: str) -> None:
        prompt = (prompt or "").strip()
        if not prompt:
            return
        if prompt == self._prompt:
            return
        # YOLO-World expects a list of class names
        classes = [p.strip() for p in prompt.split(",") if p.strip()]
        # Ultralytics 8.4.x bug: after predict() moves YOLO to CUDA, a later
        # set_classes() hits a cuda/cpu mismatch because CLIP tokenizer emits
        # CPU tensors while the cached CLIP model is now on CUDA. Workaround:
        # move the model (and the cached CLIP) to CPU before set_classes; the
        # next predict() call will move it back to self.device automatically.
        try:
            inner = getattr(self.model, "model", None)
            if inner is not None and hasattr(inner, "to"):
                inner.to("cpu")
                cached = getattr(inner, "clip_model", None)
                if cached is not None and hasattr(cached, "to"):
                    cached.to("cpu")
            self.model.set_classes(classes)
            if inner is not None and hasattr(inner, "to"):
                inner.to(self.device)
                cached = getattr(inner, "clip_model", None)
                if cached is not None and hasattr(cached, "to"):
                    cached.to(self.device)
        except Exception:
            # Fall back to the direct call; may fail on device mismatch but
            # lets the user see the underlying error instead of hiding it.
            self.model.set_classes(classes)
        self._prompt = prompt

    def infer(self, image_bgr: np.ndarray) -> Optional[Detection2D]:
        if self._prompt is None:
            return None
        res = self.model.predict(
            image_bgr,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return None
        boxes = res[0].boxes
        scores = boxes.conf.cpu().numpy()
        i = int(scores.argmax())
        xyxy = boxes.xyxy.cpu().numpy()[i].astype(int)
        H, W = image_bgr.shape[:2]
        x1 = max(0, xyxy[0]); y1 = max(0, xyxy[1])
        x2 = min(W - 1, xyxy[2]); y2 = min(H - 1, xyxy[3])
        mask = np.zeros((H, W), dtype=bool)
        mask[y1:y2 + 1, x1:x2 + 1] = True
        return Detection2D(
            score=float(scores[i]),
            bbox_xyxy=np.array([x1, y1, x2, y2], dtype=int),
            mask=mask,
            prompt=self._prompt,
        )
