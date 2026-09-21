#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_seg.py — YOLO-seg 检测 + 实例分割（轻量，用于替代 SAM 的那条路径）

权重由离线训练产出（用 SAM 掩码当伪标签，零人工标注）。
推理时只需 ultralytics，不需要 SAM，速度快一个数量级。
"""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

from .imageutil import b64_to_bgr


def _find_yolo_weight(weights_dir: str) -> Optional[str]:
    if not weights_dir or not os.path.isdir(weights_dir):
        return None
    cands = []
    for pat in ("*yolo*seg*.pt", "*seg*.pt", "*yolo*.pt"):
        cands += sorted(glob.glob(os.path.join(weights_dir, pat)))
    cands += sorted(glob.glob(os.path.join(weights_dir, "**", "best.pt"), recursive=True))
    return cands[0] if cands else None


class YoloSegDetector:
    def __init__(self, weights_dir: str = "/app/weights", device: str = "cpu",
                 default_weight: Optional[str] = None) -> None:
        if device.startswith("npu"):
            import torch_npu  # noqa: F401
        from ultralytics import YOLO

        from .device import get_policy
        self.policy = get_policy(device)
        self.policy.apply()

        self.device = device
        w = _find_yolo_weight(weights_dir) or os.environ.get("YOLO_WEIGHTS") or default_weight
        if not w:
            raise FileNotFoundError(
                f"未找到 YOLO-seg 权重（在 {weights_dir} 下搜 *seg*.pt / best.pt）；"
                f"可用 YOLO_WEIGHTS 环境变量指定")
        self.source = w
        self.model = YOLO(w)

    def run(self, image_b64: str, conf: float = 0.25, iou: float = 0.45,
            max_det: int = 300, return_polygons: bool = True, **_: Any) -> Dict[str, Any]:
        bgr = b64_to_bgr(image_b64)
        H, W = bgr.shape[:2]
        dev = "cpu" if self.device.startswith("cpu") else self.device
        res = self.model.predict(bgr, conf=conf, iou=iou, max_det=max_det,
                                 device=dev, verbose=False)[0]

        items: List[Dict[str, Any]] = []
        names = getattr(res, "names", {}) or {}
        boxes = getattr(res, "boxes", None)
        masks = getattr(res, "masks", None)
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            cfs = boxes.conf.cpu().numpy()
            polys_all = masks.xy if (masks is not None) else None
            for i in range(len(xyxy)):
                x0, y0, x1, y1 = [float(v) for v in xyxy[i]]
                item: Dict[str, Any] = {
                    "index": i, "box": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                    "cls": int(clss[i]), "label": names.get(int(clss[i]), str(clss[i])),
                    "conf": round(float(cfs[i]), 4),
                }
                if return_polygons and polys_all is not None and i < len(polys_all):
                    p = polys_all[i]
                    item["polygons"] = [[round(float(x), 2), round(float(y), 2)] for x, y in p]
                    item["polygon_normalized"] = [
                        [round(float(x) / W, 6), round(float(y) / H, 6)] for x, y in p]
                items.append(item)
        return {"size": [W, H], "detections": items}
