#!/usr/bin/env python3
"""Download model weights into a persistent host-mounted dir.

Inside docker, ~/host resolves to the host's /home/acrl (bind mount). Weights
go to ~/host/gem_perception_models/ so container restarts don't re-download.

Covers:
 - YOLO-World (small) — always
 - LangSAM (HF cache)  — py>=3.10 only
 - GroundingDINO + SAM1 — py<3.10 fallback
"""
import os
import pathlib
import sys
import urllib.request


ROOT = pathlib.Path(os.path.expanduser("~/host/gem_perception_models"))

YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt"
YOLO_DST = ROOT / "yolov8s-worldv2.pt"

GD_CFG_URL = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_CFG_DST = ROOT / "GroundingDINO_SwinT_OGC.py"
GD_CKPT_URL = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
GD_CKPT_DST = ROOT / "groundingdino_swint_ogc.pth"

SAM_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CKPT_DST = ROOT / "sam_vit_b_01ec64.pth"


def _dl(url, dst):
    if dst.exists() and dst.stat().st_size > 1024:
        print(f"[skip] {dst.name}")
        return
    print(f"[get]  {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"       -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    _dl(YOLO_URL, YOLO_DST)

    if sys.version_info >= (3, 10):
        os.environ["HF_HOME"] = str(ROOT / "huggingface")
        os.environ["TORCH_HOME"] = str(ROOT / "torch")
        (ROOT / "huggingface").mkdir(parents=True, exist_ok=True)
        (ROOT / "torch").mkdir(parents=True, exist_ok=True)
        print("[langsam] triggering HF download…")
        try:
            from lang_sam import LangSAM
            LangSAM(sam_type="sam2.1_hiera_small")
            print("[langsam] done")
        except Exception as e:
            print(f"[langsam] failed: {e}")
    else:
        print("[sam1] downloading GroundingDINO + SAM1 fallback weights…")
        _dl(GD_CFG_URL, GD_CFG_DST)
        _dl(GD_CKPT_URL, GD_CKPT_DST)
        _dl(SAM_CKPT_URL, SAM_CKPT_DST)


if __name__ == "__main__":
    main()
