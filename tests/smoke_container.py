#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_container.py — 在容器内跑「真实推理路径」的自检（不需要权重/GPU）

CPU 精简镜像的冒烟测试只做了 `import service, model_hub`，**没有碰 cv2**。
而真实推理路径大量依赖 OpenCV（imdecode/cvtColor/resize/findContours/approxPolyDP）。
requirements 里 `opencv-python-headless>=4.8` 实测会解析到 5.0.0.x，
所以必须显式验证 OpenCV 5.x 下这些调用仍然正确。

用法（容器内）：
    docker run --rm -v $PWD/tests:/t -w /app cylinder-geom:cpu python /t/smoke_container.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "/app")

FAILED = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILED.append(msg)


def main() -> int:
    import cv2
    print(f"=== OpenCV {cv2.__version__} 在真实推理路径上的自检 ===\n")

    from models.imageutil import (b64_to_bgr, bgr_to_b64, mask_to_polygons,
                                  resize_max_side)

    img = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    cv2.rectangle(img, (80, 40), (160, 200), (200, 180, 60), -1)

    # --- base64 编解码（HTTP 入口的第一步）---
    back = b64_to_bgr(bgr_to_b64(img))
    check(back.shape == img.shape and back.dtype == np.uint8, f"b64 往返形状/类型 {back.shape}")
    check(np.array_equal(img, back), "b64 往返像素完全一致")
    try:
        b64_to_bgr("data:image/png;base64," + bgr_to_b64(img))
        check(True, "支持 data:image/...;base64, 前缀")
    except Exception as e:                                   # noqa: BLE001
        check(False, f"data: 前缀解析失败 {e}")

    # --- 缩放（MoGe 前处理）---
    r, sc = resize_max_side(img, 200)
    check(max(r.shape[:2]) == 200 and 0 < sc < 1, f"resize_max_side 生效 {img.shape}->{r.shape} sc={sc:.3f}")
    r2, sc2 = resize_max_side(img, 1000)
    check(r2.shape == img.shape and sc2 == 1.0, "超过原尺寸时不放大")

    # --- 颜色空间（MoGe 要求 BGR->RGB）---
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    check(rgb[0, 0].tolist() == img[0, 0][::-1].tolist(), "cvtColor BGR2RGB 通道正确")

    # --- 掩码 -> 多边形（YOLO-seg 标签导出路径）---
    m = np.zeros((240, 320), np.uint8)
    cv2.rectangle(m, (80, 40), (160, 200), 255, -1)
    polys = mask_to_polygons(m)
    check(len(polys) == 1, f"findContours+approxPolyDP 得到 {len(polys)} 个多边形")
    check(len(polys[0]) >= 3, f"多边形顶点数 {len(polys[0])}")
    poly = polys[0]
    check(poly[:, 0].min() >= 78 and poly[:, 0].max() <= 162 and
          poly[:, 1].min() >= 38 and poly[:, 1].max() <= 202,
          f"多边形落在矩形范围内 x[{poly[:,0].min():.0f},{poly[:,0].max():.0f}] "
          f"y[{poly[:,1].min():.0f},{poly[:,1].max():.0f}]")

    # --- 常量存在性（跨版本易变）---
    for name in ("INTER_AREA", "INTER_NEAREST", "COLOR_BGR2RGB", "COLOR_BGR2GRAY",
                 "RETR_EXTERNAL", "CHAIN_APPROX_SIMPLE", "IMREAD_COLOR", "COLORMAP_TURBO"):
        check(hasattr(cv2, name), f"cv2.{name} 存在")

    # --- 几何链路（纯 numpy，不需要 torch）---
    from models.core import estimate_world_frame, fit_cylinder

    rng = np.random.default_rng(0)
    # 造一个正立的圆柱点云：轴沿 +y(图像下方向的反向)，半径 0.15
    t = rng.uniform(-0.7, 0.7, 4000)
    th = rng.uniform(0, 2 * np.pi, 4000)
    Pc = np.stack([0.15 * np.cos(th), t, 3.0 + 0.15 * np.sin(th)], axis=1)
    fit = fit_cylinder(Pc)
    check(fit is not None and abs(fit["radius"] - 0.15) < 0.05,
          f"圆柱拟合 radius={fit['radius']:.3f} ≈ 0.15" if fit else "圆柱拟合失败")
    check(fit is not None and abs(fit["length"] - 1.4) < 0.2,
          f"拟合长度 {fit['length']:.3f} ≈ 1.4" if fit else "-")

    # 重力估计在「无有效法向」时必须安全回退
    f = estimate_world_frame(None, np.ones((20, 20), bool), points=None)
    check(f["info"]["source"] == "camera_fallback", "无法向 -> 回退相机坐标系不崩")

    print("\n" + "=" * 52)
    if FAILED:
        print(f"失败 {len(FAILED)} 项:")
        for x in FAILED:
            print("  -", x)
        return 1
    print("容器内真实路径自检全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
