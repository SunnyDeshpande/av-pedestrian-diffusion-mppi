"""Text-promptable segmentation: LangSAM (py>=3.10) or GroundingDINO+SAM1 fallback (py3.8)."""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

from .yolo_detector import Detection2D


def _default_models_root() -> str:
    """Pick a sensible default for where weights live.

    Order of preference:
      1. ``$GEM_PERCEPTION_MODELS`` env var (explicit override).
      2. ``~/host/gem_perception_models`` if it exists (docker-on-host pattern).
      3. ``~/gem_perception_models`` (standard real-car / non-docker path).
    """
    env = os.environ.get("GEM_PERCEPTION_MODELS")
    if env:
        return os.path.expanduser(env)
    docker_path = os.path.expanduser("~/host/gem_perception_models")
    if os.path.isdir(docker_path):
        return docker_path
    return os.path.expanduser("~/gem_perception_models")


class LangSamDetector:
    """Auto-selects backend based on Python version.

    - Python >= 3.10: uses the `lang-sam` package (SAM2 + Grounding-DINO).
    - Python  < 3.10: falls back to `groundingdino-py` + `segment-anything` (SAM1).

    Both paths expose the same :py:meth:`infer` / :py:meth:`set_prompt` API.
    """

    def __init__(self, device: str = "cuda", sam_type: str = "sam2.1_hiera_small",
                 box_threshold: float = 0.25, text_threshold: float = 0.20,
                 models_root: Optional[str] = None):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._prompt: Optional[str] = None
        self.models_root = models_root or _default_models_root()

        os.environ.setdefault("HF_HOME", os.path.join(self.models_root, "huggingface"))
        os.environ.setdefault("TORCH_HOME", os.path.join(self.models_root, "torch"))

        if os.environ.get("GEM_SAM_BACKEND", "sam1").lower() == "langsam":
            from lang_sam import LangSAM
            self._backend = "langsam"
            self._model = LangSAM(sam_type=sam_type)
        else:
            # Python 3.8 fallback: GroundingDINO (detect) + SAM1 (segment)
            import torch
            from groundingdino.util.inference import load_model, predict as gd_predict
            from groundingdino.util import box_ops
            from segment_anything import SamPredictor, sam_model_registry
            self._backend = "sam1"
            self._torch = torch
            self._gd_predict = gd_predict
            self._box_ops = box_ops

            gd_cfg = os.path.join(self.models_root, "GroundingDINO_SwinT_OGC.py")
            gd_ckpt = os.path.join(self.models_root, "groundingdino_swint_ogc.pth")
            sam_ckpt = os.path.join(self.models_root, "sam_vit_b_01ec64.pth")
            if not (os.path.exists(gd_cfg) and os.path.exists(gd_ckpt) and os.path.exists(sam_ckpt)):
                raise FileNotFoundError(
                    f"Missing model files in {self.models_root}. Run scripts/download_models.py first."
                )
            self._gd = load_model(gd_cfg, gd_ckpt, device=device)
            self._sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt).to(device)
            self._sam_pred = SamPredictor(self._sam)

    def set_prompt(self, prompt: str) -> None:
        prompt = (prompt or "").strip()
        if prompt:
            self._prompt = prompt

    def infer(self, image_bgr: np.ndarray) -> Optional[Detection2D]:
        if self._prompt is None:
            return None
        if self._backend == "langsam":
            return self._infer_langsam(image_bgr)
        return self._infer_sam1(image_bgr)

    # ── LangSAM (SAM2) path ───────────────────────────────────────────────
    def _infer_langsam(self, image_bgr: np.ndarray) -> Optional[Detection2D]:
        import cv2
        from PIL import Image as PILImage
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        res = self._model.predict([pil], [self._prompt],
                                  box_threshold=self.box_threshold,
                                  text_threshold=self.text_threshold)
        if not res:
            return None
        out = res[0]
        boxes, masks, scores = out.get("boxes"), out.get("masks"), out.get("scores")
        if boxes is None or len(boxes) == 0:
            return None
        scores = np.asarray(scores)
        i = int(scores.argmax())
        return self._pack_detection(image_bgr, boxes[i], masks[i] if masks is not None else None, scores[i])

    # ── GroundingDINO + SAM1 path (py3.8) ─────────────────────────────────
    def _infer_sam1(self, image_bgr: np.ndarray) -> Optional[Detection2D]:
        import cv2
        H, W = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        import groundingdino.datasets.transforms as T
        from PIL import Image as PILImage
        pil = PILImage.fromarray(rgb)
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_tensor, _ = transform(pil, None)

        boxes_cxcywh, logits, phrases = self._gd_predict(
            model=self._gd, image=img_tensor, caption=self._prompt,
            box_threshold=self.box_threshold, text_threshold=self.text_threshold,
            device=self.device,
        )
        if len(boxes_cxcywh) == 0:
            return None

        scores = logits.cpu().numpy()
        i = int(scores.argmax())
        # Convert cxcywh (normalised) → xyxy (pixels)
        cx, cy, w, h = boxes_cxcywh[i].cpu().numpy()
        x1 = max(0, int((cx - w / 2) * W))
        y1 = max(0, int((cy - h / 2) * H))
        x2 = min(W - 1, int((cx + w / 2) * W))
        y2 = min(H - 1, int((cy + h / 2) * H))
        box_xyxy = np.array([x1, y1, x2, y2], dtype=int)

        # SAM segmentation
        self._sam_pred.set_image(rgb)
        masks_sam, _, _ = self._sam_pred.predict(box=box_xyxy, multimask_output=False)
        mask = masks_sam[0].astype(bool) if masks_sam is not None else None

        return self._pack_detection(image_bgr, box_xyxy, mask, float(scores[i]))

    def _pack_detection(self, image_bgr, box, mask, score) -> Detection2D:
        import cv2
        H, W = image_bgr.shape[:2]
        box = np.asarray(box).astype(int).reshape(-1)
        x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
        x2 = min(W - 1, int(box[2])); y2 = min(H - 1, int(box[3]))
        if mask is None:
            m = np.zeros((H, W), dtype=bool)
            m[y1:y2 + 1, x1:x2 + 1] = True
        else:
            m = np.asarray(mask).astype(bool)
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        return Detection2D(
            score=float(score),
            bbox_xyxy=np.array([x1, y1, x2, y2], dtype=int),
            mask=m,
            prompt=self._prompt,
        )
