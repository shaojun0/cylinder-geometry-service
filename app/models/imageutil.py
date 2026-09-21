#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图像编解码小工具（不依赖 torch）。"""
from __future__ import annotations

import base64
from typing import Any, Optional, Tuple

import numpy as np


def b64_to_bgr(image_b64: str) -> np.ndarray:
    """base64(可含 data:image/...;base64, 前缀) -> BGR uint8 HxWx3。"""
    import cv2

    if not image_b64:
        raise ValueError("image_b64 为空")
    if "," in image_b64[:64] and image_b64.lstrip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("base64 解码后不是有效图片")
    return img


def bgr_to_b64(img: np.ndarray, ext: str = ".png") -> str:
    import cv2

    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def resize_max_side(bgr: np.ndarray, max_side: Optional[int]) -> Tuple[np.ndarray, float]:
    """按最长边缩放，返回 (图像, 缩放系数)。max_side 为 None/0 时不缩放。"""
    import cv2

    if not max_side:
        return bgr, 1.0
    H, W = bgr.shape[:2]
    sc = min(1.0, float(max_side) / max(H, W))
    if sc >= 1.0:
        return bgr, 1.0
    out = cv2.resize(bgr, (int(round(W * sc)), int(round(H * sc))), interpolation=cv2.INTER_AREA)
    return out, sc


def mask_to_polygons(mask: np.ndarray, min_area_frac: float = 0.0005,
                     eps_frac: float = 0.002) -> list:
    """二值掩码 -> 多边形列表（Nx2 float，图像像素坐标）。"""
    import cv2

    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape[:2]
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area_frac * H * W:
            continue
        ap = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(ap) >= 3:
            out.append(ap.astype(float))
    return out
