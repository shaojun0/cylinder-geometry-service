#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
moge_geom.py — MoGe-2 单目几何 + 重力世界系 + 圆柱拟合

一次前向得到：米制点云 points(H,W,3) + 表面法向 normal(H,W,3)
然后：
  1. estimate_world_frame 从法向估计重力方向（一次标定后可作为常量复用）
  2. 对每个区域（框/点/掩码）取三维点做鲁棒圆柱拟合
  3. 输出朝向 pitch、几何中心（轴线中点）、离地高度、半径、长度

注意：几何中心 ≠ 物理重心。物理重心取决于内部质量分布（LPG 液位），视觉不可见。
"""
from __future__ import annotations

import glob
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .core import (estimate_world_frame, fit_cylinder, height_above_plane,
                   pitch_deg, resolve_box)
from .imageutil import b64_to_bgr, resize_max_side

DEFAULT_REPO = "Ruicheng/moge-2-vitl-normal"


def _find_moge_weight(weights_dir: str) -> Optional[str]:
    """
    找 MoGe 权重，返回 **.pt 文件路径**（不是目录）。

    注意：MoGe 的 `from_pretrained` 实现是
        if Path(name).exists(): checkpoint = name   # 直接 torch.load(name)
        else: hf_hub_download(repo_id=name, filename="model.pt")
    所以传**目录**会走 `torch.load(目录)` 而失败——必须给到 `.pt` 文件。

    兼容两种布局：
      a) 扁平：   weights/moge-2-vitl-normal.pt
      b) HF 缓存：weights/models--Ruicheng--moge-2-vitl-normal/snapshots/<sha>/model.pt
      c) 快照目录：weights/moge-2-vitl-normal/  (内含 model.pt)
    """
    if not weights_dir or not os.path.isdir(weights_dir):
        return None
    hinted: List[str] = []      # 路径/文件名里带 moge
    generic: List[str] = []     # 只是叫 model.pt
    for root, _dirs, files in os.walk(weights_dir):
        path_hint = "moge" in root.lower()
        for f in files:
            if not f.endswith((".pt", ".pth")):
                continue
            p = os.path.join(root, f)
            if path_hint or "moge" in f.lower():
                hinted.append(p)
            elif f == "model.pt":
                generic.append(p)
    pool = hinted or generic
    if not pool:
        return None
    pool.sort(key=lambda p: (os.path.basename(p) != "model.pt", len(p)))
    return pool[0]


class MogeGeometry:
    def __init__(self, weights_dir: str = "/app/weights", device: str = "cpu",
                 repo: str = DEFAULT_REPO, resolution_level: int = 9) -> None:
        if device.startswith("npu"):
            import torch_npu  # noqa: F401  （让 torch.npu 命名空间生效）
        import torch
        from moge.model.v2 import MoGeModel

        from .device import get_policy
        self.policy = get_policy(device)
        self.policy.apply()

        self.torch = torch
        self.device = device
        self.resolution_level = resolution_level
        local = _find_moge_weight(weights_dir) or os.environ.get("MOGE_WEIGHTS")
        src = local if local else repo
        self.source = src
        self.model = MoGeModel.from_pretrained(src).to(device).eval()
        # 按策略落精度（310P 必须 fp16；CPU 强制 fp32）
        if self.policy.dtype in ("float16", "bfloat16"):
            self.model = self.model.to(self.policy.torch_dtype())
        self.use_fp16 = self.policy.dtype == "float16"

    # ------------------------------------------------------------ 内部

    def _infer(self, bgr: np.ndarray) -> Dict[str, Any]:
        import cv2
        import torch

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        with torch.no_grad():
            out = self.model.infer(t, resolution_level=self.resolution_level,
                                   use_fp16=self.use_fp16)

        def np_(v):
            if v is None:
                return None
            if torch.is_tensor(v):
                v = v.detach().float().cpu().numpy()
            a = np.asarray(v)
            return a[0] if a.ndim == 4 else a

        res = {"points": np_(out["points"]), "depth": np_(out["depth"]),
               "mask": np_(out["mask"]).astype(bool), "normal": np_(out.get("normal")),
               "intrinsics": np_(out["intrinsics"])}
        if res["normal"] is not None and res["normal"].ndim == 4:
            res["normal"] = res["normal"][0]
        return res

    # ------------------------------------------------------------ 对外

    def run(
        self,
        image_b64: str,
        regions: Optional[List[Dict[str, Any]]] = None,
        max_side: int = 1024,
        refine_gravity: bool = True,
        return_gravity: bool = True,
        **_: Any,
    ) -> Dict[str, Any]:
        bgr = b64_to_bgr(image_b64)
        H0, W0 = bgr.shape[:2]
        bgr, sc = resize_max_side(bgr, max_side)
        H, W = bgr.shape[:2]

        r = self._infer(bgr)
        pts, valid = r["points"], r["mask"]
        valid = valid & np.isfinite(r["depth"]) & (r["depth"] > 0)
        frame = estimate_world_frame(r["normal"], valid, points=pts, seed=0,
                                     refine=refine_gravity)
        up = frame["up"]

        K = r["intrinsics"]
        fx = float(K[0, 0]) * W if K is not None else float("nan")
        fov_x = 2 * math.degrees(math.atan(0.5 * W / fx)) if (fx and fx > 0) else None

        result: Dict[str, Any] = {
            "size": [W, H], "orig_size": [W0, H0], "scale": round(sc, 4),
            "fov_x_deg": round(fov_x, 2) if fov_x else None,
            "valid_ratio": round(float(valid.mean()), 4),
            "median_depth_m": round(float(np.median(r["depth"][valid])), 4) if valid.any() else None,
            "regions": [],
        }
        if return_gravity:
            result["gravity"] = frame["info"]

        for i, reg in enumerate(regions or []):
            item: Dict[str, Any] = {"index": i}
            try:
                if "mask_b64" in reg:
                    from .imageutil import b64_to_bgr as _d
                    import cv2
                    m = cv2.cvtColor(_d(reg["mask_b64"]), cv2.COLOR_BGR2GRAY)
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127
                    sel = pts[valid & m]
                    sel = sel[np.isfinite(sel).all(axis=1)]
                    box_px, vis_px = int(m.sum()), int((valid & m).sum())
                else:
                    box = reg.get("box")
                    if box is None and "point" in reg:
                        x, y = reg["point"]; rad = int(reg.get("radius", 8))
                        box = [x - rad, y - rad, x + rad, y + rad]
                    if box is None:
                        raise ValueError("region 需要 box / point / mask_b64 之一")
                    box = [v * sc for v in box]
                    sel, box_px, vis_px = resolve_box(pts, valid, box)
                item["n_points"] = int(len(sel))
                item["visible_ratio"] = round(vis_px / box_px, 4) if box_px else None

                fit = fit_cylinder(sel)
                if fit is None:
                    item.update({"ok": False, "error": "圆柱拟合失败（有效点不足）"})
                    result["regions"].append(item)
                    continue

                ctr = fit["center"]
                h = height_above_plane(ctr, frame)
                item.update({
                    "ok": True,
                    "pitch_deg": round(pitch_deg(fit["axis"], up), 2),
                    "axis_world": [round(float(v), 5) for v in
                                   np.stack([fit["axis"] @ frame["right"],
                                             fit["axis"] @ up,
                                             fit["axis"] @ frame["forward"]])],
                    "axis_cam": [round(float(v), 5) for v in fit["axis"]],
                    "center_xyz": [round(float(v), 4) for v in ctr],
                    "center_height_m": round(h, 4) if h is not None else None,
                    "radius_m": round(fit["radius"], 4),
                    "length_m": round(fit["length"], 4),
                    "length_over_radius": round(fit["length"] / max(fit["radius"], 1e-9), 2),
                    "elongation": round(fit["elongation"], 2),
                    "is_fallen": bool(abs(pitch_deg(fit["axis"], up)) < 45.0),
                })
            except Exception as e:                      # noqa: BLE001
                item.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
            result["regions"].append(item)

        return result
