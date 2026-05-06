#!/usr/bin/env python3
"""Download model weights into a persistent dir.

Resolution order (matches sam_detector._default_models_root):
  1. $GEM_PERCEPTION_MODELS env var (explicit override).
  2. ~/host/gem_perception_models (docker on host with /home/acrl bind-mounted).
  3. ~/gem_perception_models (real-car / non-docker).

Covers:
 - YOLO-World (small) — always
 - LangSAM (HF cache)  — py>=3.10 only
 - GroundingDINO + SAM1 — py<3.10 fallback
"""
import argparse
import os
import pathlib
import sys
import urllib.request


def _default_root() -> pathlib.Path:
    env = os.environ.get("GEM_PERCEPTION_MODELS")
    if env:
        return pathlib.Path(os.path.expanduser(env))
    docker_path = pathlib.Path(os.path.expanduser("~/host/gem_perception_models"))
    if docker_path.is_dir():
        return docker_path
    return pathlib.Path(os.path.expanduser("~/gem_perception_models"))


ROOT = _default_root()

# (url, dst, min_bytes) — min_bytes is a sanity floor; files below this are
# treated as incomplete/corrupt and re-downloaded.
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt"
YOLO_DST = ROOT / "yolov8s-worldv2.pt"
YOLO_MIN = 20 * 1024 * 1024  # ~26 MB real

GD_CFG_URL = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_CFG_DST = ROOT / "GroundingDINO_SwinT_OGC.py"
GD_CFG_MIN = 500  # tiny .py file
GD_CKPT_URL = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
GD_CKPT_DST = ROOT / "groundingdino_swint_ogc.pth"
GD_CKPT_MIN = 600 * 1024 * 1024  # ~694 MB real

SAM_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CKPT_DST = ROOT / "sam_vit_b_01ec64.pth"
SAM_CKPT_MIN = 350 * 1024 * 1024  # ~375 MB real


def _dl(url, dst, min_bytes, force=False):
    if dst.exists() and not force:
        size = dst.stat().st_size
        if size >= min_bytes:
            print(f"[skip] {dst.name} ({size / 1e6:.1f} MB)")
            return
        print(f"[redo] {dst.name} is {size / 1e6:.1f} MB, expected >= {min_bytes / 1e6:.1f} MB — re-downloading")
    elif dst.exists() and force:
        print(f"[force] re-downloading {dst.name}")
    print(f"[get]  {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)
    print(f"       -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description="Download GEM perception model weights.")
    ap.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    args, _ = ap.parse_known_args()
    force = args.force

    print(f"[root] {ROOT}")
    ROOT.mkdir(parents=True, exist_ok=True)
    _dl(YOLO_URL, YOLO_DST, YOLO_MIN, force=force)

    if os.environ.get("GEM_SAM_BACKEND", "sam1").lower() == "langsam":
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
        _dl(GD_CFG_URL, GD_CFG_DST, GD_CFG_MIN, force=force)
        _dl(GD_CKPT_URL, GD_CKPT_DST, GD_CKPT_MIN, force=force)
        _dl(SAM_CKPT_URL, SAM_CKPT_DST, SAM_CKPT_MIN, force=force)


if __name__ == "__main__":
    main()
