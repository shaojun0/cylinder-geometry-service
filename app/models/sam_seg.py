#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sam_seg.py — SAM 框提示分割

用 transformers 的 SamModel/SamProcessor。给一个框，返回最佳候选掩码。
下游用途：
  * 得到干净掩码 -> 只对掩码内三维点做圆柱拟合（框里常混入墙面等背景，会把拟合带偏）
  * 掩码多边形可直接导出为 YOLO-seg 标签，实现零人工标注
"""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .imageutil import b64_to_bgr, mask_to_polygons

DEFAULT_REPO = "facebook/sam-vit-base"


def _find_sam_weight(weights_dir: str) -> Optional[str]:
    """
    找 SAM 权重。transformers 的 `from_pretrained` 接受**目录**或 repo id，
    所以这里返回一个「含配置文件与权重」的目录。

    兼容：
      a) 扁平目录：   weights/sam-vit-base/  (config.json + model.safetensors)
      b) HF 缓存：    weights/models--facebook--sam-vit-base/snapshots/<sha>/
      c) 单个权重文件：weights/sam_vit_b.pth  -> 返回其所在目录
    """
    if not weights_dir or not os.path.isdir(weights_dir):
        return None
    dirs: List[str] = []
    loose: List[str] = []
    for root, _dirs, files in os.walk(weights_dir):
        low = (root + "/" + " ".join(files)).lower()
        if "sam" not in low:
            continue
        has_cfg = any(f == "config.json" for f in files)
        has_w = any(f.endswith((".safetensors", ".bin", ".pt", ".pth")) for f in files)
        if has_cfg and has_w:
            dirs.append(root)
        for f in files:
            if "sam" in f.lower() and f.endswith((".pt", ".pth", ".safetensors")):
                loose.append(os.path.join(root, f))
    if dirs:
        dirs.sort(key=len)          # 越浅越好（扁平目录优先于 HF 缓存深层路径）
        return dirs[0]
    if loose:
        return os.path.dirname(loose[0])
    return None


class SamSegmenter:
    def __init__(self, weights_dir: str = "/app/weights", device: str = "cpu",
                 repo: str = DEFAULT_REPO) -> None:
        if device.startswith("npu"):
            import torch_npu  # noqa: F401
        import torch
        from transformers import SamModel, SamProcessor

        from .device import get_policy
        self.policy = get_policy(device)
        self.policy.apply()

        self.torch = torch
        self.device = device
        # 显式环境变量优先于启发式查找；失效的路径配置回退（理由见 weight_paths.py）
        from .weight_paths import resolve_weight
        local = resolve_weight("SAM_WEIGHTS",
                               lambda: _find_sam_weight(weights_dir))
        src = local if local else repo
        self.source = src
        self.proc = SamProcessor.from_pretrained(src)
        self.model = SamModel.from_pretrained(src).to(device).eval()
        if self.policy.dtype in ("float16", "bfloat16"):
            self.model = self.model.to(self.policy.torch_dtype())

    def run(self, image_b64: str, boxes: List[List[float]],
            return_polygons: bool = True, **_: Any) -> Dict[str, Any]:
        import cv2

        if not boxes:
            raise ValueError("segment 需要 boxes=[(x0,y0,x1,y1), ...]")
        bgr = b64_to_bgr(image_b64)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = bgr.shape[:2]
        t = self.torch

        inputs = self.proc(rgb, input_boxes=[[list(map(float, b)) for b in boxes]],
                           return_tensors="pt").to(self.device)
        with t.no_grad():
            out = self.model(**inputs)
        masks = self.proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())[0]        # (n_box, 3, H, W)
        scores = out.iou_scores.cpu().numpy()[0]             # (n_box, 3)

        items: List[Dict[str, Any]] = []
        for i, box in enumerate(boxes):
            k = int(np.argmax(scores[i]))
            m = masks[i, k].numpy() > 0
            area = int(m.sum())
            item: Dict[str, Any] = {
                "index": i, "box": [round(float(v), 2) for v in box],
                "score": round(float(scores[i][k]), 4),
                "area_px": area,
                "area_ratio": round(area / float(H * W), 5),
            }
            if return_polygons:
                polys = mask_to_polygons(m)
                item["polygons"] = [[[round(float(x), 2), round(float(y), 2)] for x, y in p]
                                    for p in polys]
                item["polygon_normalized"] = [
                    [[round(float(x) / W, 6), round(float(y) / H, 6)] for x, y in p] for p in polys]
            items.append(item)

        return {"size": [W, H], "segments": items}
