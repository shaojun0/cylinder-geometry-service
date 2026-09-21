#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
moge_geom.py — MoGe（v2 / v3 可配）单目几何 + 重力世界系 + 圆柱拟合

一次前向得到：米制点云 points(H,W,3) + 表面法向 normal(H,W,3)
然后：
  1. estimate_world_frame 从法向估计重力方向（一次标定后可作为常量复用）
  2. 对每个区域（框/点/掩码）取三维点做鲁棒圆柱拟合
  3. 输出朝向 pitch、几何中心（轴线中点）、离地高度、半径、长度

⚠️ 每个 region 都带 `gravity_reliable`，与 result["gravity"]["gravity_reliable"] 同源。
   只有它为 true 时，pitch_deg / is_fallen 才是用从场景中**量**出来的重力算的；
   为 false 时 up 退化成「假设相机水平」，相机俯仰/横滚会给 pitch 引入同样大小的
   系统性偏差。**安全告警必须在消费 is_fallen 前检查这个字段。**

注意：几何中心 ≠ 物理重心。物理重心取决于内部质量分布（LPG 液位），视觉不可见。
"""
from __future__ import annotations

import glob
import logging
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .core import (estimate_world_frame, fit_cylinder, height_above_plane,
                   pitch_deg, project_world_axes, resolve_box)
from .imageutil import b64_to_bgr, resize_max_side

log = logging.getLogger(__name__)

# 版本可配：v2（原线上默认）与 v3（MoGe-3，依赖 flex_gemm / Triton）。
# 默认仍是 v2 —— 换模型必须是**显式**动作，不能让没配环境变量的部署悄悄换掉，
# 更不能让"权重目录里恰好有个 v3 的 .pt"来决定线上跑哪个模型。
DEFAULT_VERSION = "v2"
SUPPORTED_VERSIONS = ("v2", "v3")
REPO_BY_VERSION = {
    "v2": "Ruicheng/moge-2-vitl-normal",
    "v3": "Ruicheng/moge-3-vitl",
}
DEFAULT_REPO = REPO_BY_VERSION[DEFAULT_VERSION]
DEFAULT_REFINE_STEPS = 3


def normalize_version(value: Any) -> Optional[str]:
    """把 "v3" / "3" / "V3 " 归一化成 "v3"；不认识则返回 None。"""
    s = str(value or "").strip().lower()
    if s.startswith("v"):
        s = s[1:]
    return f"v{s}" if s in ("2", "3") else None


def version_tag_in(path: Any) -> Optional[str]:
    """从路径/仓库名里认出 "moge-2" / "moge-3" 这样的版本标签；认不出返回 None。"""
    lp = str(path or "").lower()
    for v in ("2", "3"):
        if f"moge-{v}" in lp:
            return f"v{v}"
    return None


def check_version_consistency(src: Any, version: str) -> Optional[str]:
    """
    权重来源与所选版本是否自相矛盾；矛盾返回错误串，否则 None。

    部署目录下 v2/v3 两套权重会**同时存在**。此时一旦 MOGE_WEIGHTS 失效回退到
    启发式查找、或有人手滑指错路径，用 v3 的模型结构去加载 v2 的 checkpoint
    **不会报错**，只会静默给出错误的几何量 —— 对一个用来判断"气瓶是否倒伏"的
    服务来说这是最坏的失败模式，所以这里宁可启动失败。
    """
    tag = version_tag_in(src)
    if tag is not None and tag != version:
        return (f"MOGE_VERSION={version} 与权重来源 {src}（看起来是 {tag}）不一致；"
                f"请把 MOGE_WEIGHTS 指到 {version} 的 checkpoint，或改回 MOGE_VERSION={tag}")
    return None


def _find_moge_weight(weights_dir: str, version: str = DEFAULT_VERSION) -> Optional[str]:
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

    `version` 决定**优先挑哪个版本**（v2/v3 的权重可能同处一个目录）。

    ⚠️ 选择规则见下面 `_rank`：**不能只按路径长度排序**。
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

    ver = normalize_version(version) or DEFAULT_VERSION
    tag = f"moge-{ver[1:]}"          # "moge-2" / "moge-3"

    def _rank(p: str) -> Tuple[int, int, int, int]:
        """
        按**证据强度**排序，最后才用长度兜底。

        坑：`path_hint = "moge" in root` 会匹配**任意祖先目录**。只要权重目录本身
        位于某个叫 `moge` 的路径下（例如 `/root/autodl-tmp/moge/weights/`，而这是
        最自然的命名），该目录里**所有** .pt 都会被当成 moge 候选——包括 YOLO 的
        `best.pt`。此时若按路径长度取最短，就会挑中 YOLO 权重，而 MoGe 的
        `from_pretrained` 会把它丢给 `torch.load(weights_only=True)`，报出
        `Unsupported global: ultralytics.nn.tasks.SegmentationModel`。
        实测踩到过（nmb1 部署）。
        所以顺序是：版本标签匹配 > 文件名里带 moge > HF 约定的 model.pt > 路径最短。

        第 0 顺位（版本标签）是 v3 上线时补的：svc_weights 下现在同时有
        `moge-2-vitl-normal/` 和 `moge-3-vitl/`，两者都是 `model.pt`，
        只能靠路径里的 `moge-N` 区分；否则又退化成"路径短的赢"。
        """
        b = os.path.basename(p).lower()
        lp = p.lower()
        return (0 if tag in lp else 1,
                0 if "moge" in b else 1, 0 if b == "model.pt" else 1, len(p))

    pool.sort(key=_rank)
    best = pool[0]
    if tag not in best.lower():
        # 一个版本标签都没匹配上：说明这个目录里没有该版本的权重。
        log.warning("在 %s 里没找到 %s 的权重（tag=%s），退而选了 %s",
                    weights_dir, ver, tag, best)
    return best


class MogeGeometry:
    def __init__(self, weights_dir: str = "/app/weights", device: str = "cpu",
                 repo: Optional[str] = None, resolution_level: int = 9,
                 version: Optional[str] = None,
                 refine_steps: Optional[int] = None) -> None:
        if device.startswith("npu"):
            import torch_npu  # noqa: F401  （让 torch.npu 命名空间生效）
        import torch

        from .device import get_policy
        self.policy = get_policy(device)
        self.policy.apply()

        self.torch = torch
        self.device = device
        self.resolution_level = resolution_level

        # ---- 版本：显式参数 > MOGE_VERSION 环境变量 > 默认 v2 ----
        raw_version = version if version is not None else os.environ.get("MOGE_VERSION")
        self.version = normalize_version(raw_version) or DEFAULT_VERSION
        if raw_version and normalize_version(raw_version) is None:
            log.warning("MOGE_VERSION=%r 不认识（支持 %s），回退 %s",
                        raw_version, list(SUPPORTED_VERSIONS), DEFAULT_VERSION)

        # MoGe-3 的细化后端 flex_gemm 是 CUDA/Triton 实现，昇腾上跑不了 ——
        # 早点把话说清楚，别让人在 NPU 上等一个必然失败的加载。
        if self.version == "v3" and device.startswith("npu"):
            raise ValueError("MoGe-3 依赖 flex_gemm（CUDA/Triton 实现），昇腾 NPU 上不可用；"
                             "NPU 部署请保持 MOGE_VERSION=v2")

        if refine_steps is None:
            raw_steps = os.environ.get("MOGE_REFINE_STEPS")
            try:
                refine_steps = (int(raw_steps) if raw_steps not in (None, "")
                                else DEFAULT_REFINE_STEPS)
            except (TypeError, ValueError):
                log.warning("MOGE_REFINE_STEPS=%r 不是整数，用默认 %d",
                            raw_steps, DEFAULT_REFINE_STEPS)
                refine_steps = DEFAULT_REFINE_STEPS
        self.refine_steps = max(0, int(refine_steps))

        # 环境变量是**显式配置**，优先级高于启发式查找；但失效的路径配置
        # （被移动/删除后 env 还指着旧路径）必须回退，否则 MoGe 会把它当成
        # HF repo id 去联网下载，抛一个与真实原因无关的 HFValidationError。
        from .weight_paths import resolve_weight
        local = resolve_weight("MOGE_WEIGHTS",
                               lambda: _find_moge_weight(weights_dir, self.version))
        src = local if local else (repo or REPO_BY_VERSION[self.version])

        # 版本与权重必须对得上（v3 结构加载 v2 权重不报错、只算错）。
        mismatch = check_version_consistency(src, self.version)
        if mismatch:
            raise ValueError(mismatch)

        self.source = src
        if self.version == "v3":
            from moge.model.v3 import MoGeModel
        else:
            from moge.model.v2 import MoGeModel

        # ---- 精度（版本相关，必须放在加载前后各管一段）----
        # 设备策略在 CUDA 上默认给 float16；MoGe-2 用着没问题，但 **MoGe-3 不行**：
        # 它的 flex_gemm 稀疏细化路径不接受 fp16 权重（实测
        # `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half`）。
        # 而且 fp32 模型 + use_fp16=True 反而**更慢**（实测 2.02s vs 1.06s，
        # autocast 白开销换不到收益）。所以 v3 固定 fp32 + 关 autocast。
        if self.version == "v3":
            if self.policy.dtype in ("float16", "bfloat16"):
                log.warning("MoGe-3 不能用 fp16 权重（flex_gemm 稀疏细化会 dtype 不匹配），"
                            "强制 float32（设备策略=%s）", self.policy.dtype)
            self.dtype = "float32"
            self.use_fp16 = False
        else:
            self.dtype = self.policy.dtype
            self.use_fp16 = self.policy.dtype == "float16"

        log.info("MoGe 版本=%s 权重=%s refine_steps=%d dtype=%s use_fp16=%s device=%s",
                 self.version, src, self.refine_steps, self.dtype, self.use_fp16, device)
        self.model = MoGeModel.from_pretrained(src).to(device).eval()
        # 按策略落精度（310P 必须 fp16；CPU 强制 fp32）—— 只有非 fp32 才需要换。
        if self.dtype == "float16":
            self.model = self.model.to(torch.float16)
        elif self.dtype == "bfloat16":
            self.model = self.model.to(torch.bfloat16)

    # ------------------------------------------------------------ 内部

    def _infer(self, bgr: np.ndarray) -> Dict[str, Any]:
        import cv2
        import torch

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        kwargs: Dict[str, Any] = {"resolution_level": self.resolution_level,
                                  "use_fp16": self.use_fp16}
        if self.version == "v3":
            # v3 的稀疏体素细化步数；v2 的 infer() 不接受这个参数
            kwargs["refine_steps"] = self.refine_steps
        with torch.no_grad():
            out = self.model.infer(t, **kwargs)

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
        draw_axes: bool = True,
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
        gravity_reliable = bool(frame["info"].get("gravity_reliable", False))

        # MoGe 的内参是**归一化**的（cx=cy=0.5），乘宽/高得到工作分辨率下的像素内参。
        # 实测 fx=K[0,0]*W 与 fy=K[1,1]*H 换算后相等，互为验证。
        K = r["intrinsics"]
        fx = float(K[0, 0]) * W if K is not None else float("nan")
        fy = float(K[1, 1]) * H if K is not None else float("nan")
        cx = float(K[0, 2]) * W if K is not None else float("nan")
        cy = float(K[1, 2]) * H if K is not None else float("nan")
        fov_x = 2 * math.degrees(math.atan(0.5 * W / fx)) if (fx and fx > 0) else None

        result: Dict[str, Any] = {
            # 让调用方/排障一眼看出这次结果是哪个版本、什么精度算的
            # （v2 与 v3 的数值与耗时都不一样）
            "moge_version": self.version,
            "moge_dtype": self.dtype,
            "size": [W, H], "orig_size": [W0, H0], "scale": round(sc, 4),
            "fov_x_deg": round(fov_x, 2) if fov_x else None,
            "valid_ratio": round(float(valid.mean()), 4),
            "median_depth_m": round(float(np.median(r["depth"][valid])), 4) if valid.any() else None,
            "regions": [],
        }
        if return_gravity:
            result["gravity"] = frame["info"]

        # 世界系三轴叠加（前端画红箭头坐标系）。坐标换算到**上传图空间**，
        # 前端可直接按显示尺寸等比画出，不必再关心服务内部的工作分辨率。
        if draw_axes:
            if K is None or not all(np.isfinite([fx, fy, cx, cy])):
                result["axes_overlay"] = {"ok": False, "error": "内参不可用，无法投影世界系"}
            else:
                ax = project_world_axes(frame["up"], frame["right"], frame["forward"],
                                        fx, fy, cx, cy, W, H, frame.get("plane_offset"))
                if sc and sc > 0 and sc != 1.0:
                    if ax.get("origin_px"):
                        ax["origin_px"] = [round(v / sc, 2) for v in ax["origin_px"]]
                    for a in ax.get("axes", []):
                        if a.get("tip_px"):
                            a["tip_px"] = [round(v / sc, 2) for v in a["tip_px"]]
                        a["length_px"] = round(a.get("length_px", 0.0) / sc, 2)
                    ax["space"] = "orig"
                ax["ok"] = True
                result["axes_overlay"] = ax

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
                    "gravity_reliable": gravity_reliable,
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
