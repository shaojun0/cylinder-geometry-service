#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core.py — 与框架无关的纯数值部分（只用 numpy）

包含两块：
  1) 重力/世界系估计：逐像素法向 -> RANSAC 候选轴 -> 物理硬过滤 -> 可靠性闸门
     -> 平面拟合精修
  2) 圆柱拟合：迭代鲁棒 PCA -> 垂直面圆拟合 -> 轴中点即几何中心

这两块的物理含义与踩坑记录见 docs/ASCEND.md 与仓库根目录的说明。
关键点：
  * 重力轴必须通过「与相机 up 夹角 <= 60°」的硬过滤，否则会选中面积最大的墙面。
    但该闸门只挡得住明显离谱的候选：相机俯仰越大，墙面越靠近闸门边界。
  * 闸门放行的候选仍然可能是墙。只有「地板先验」胜出才认为重力是量出来的，
    否则退回相机 up 并标记 gravity_reliable=False（见 estimate_world_frame）。
  * 直接对全部法向对齐点做平面拟合会得到 rms 20~30cm 的「混合平面」，
    必须先沿法向做 offset 直方图隔离出单一物理平面。
  * 直接用可见表面质心当重心是有偏的，会朝相机偏约 (2/pi)R；取轴线中点可消除。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ================================================================ 重力 / 世界系


def dominant_axis(n: np.ndarray, rng: np.random.Generator,
                  iters: int = 600, thresh_deg: float = 7.0) -> Optional[np.ndarray]:
    """RANSAC：找一个方向 a 使尽量多的 |n·a| ≈ 1（用绝对值，地板/天花板共享同一轴）。"""
    if len(n) < 50:
        return None
    cos_t = math.cos(math.radians(thresh_deg))
    best_a, best_c = None, -1
    for _ in range(iters):
        a = n[rng.integers(len(n))]
        c = int((np.abs(n @ a) > cos_t).sum())
        if c > best_c:
            best_c, best_a = c, a
    if best_a is None or best_c < 50:
        return None
    a = best_a
    for _ in range(3):                       # 符号对齐后求均值精修
        inl = np.abs(n @ a) > cos_t
        if inl.sum() < 30:
            break
        s = np.sign(n[inl] @ a)
        s[s == 0] = 1.0
        v = (n[inl] * s[:, None]).mean(axis=0)
        nv = np.linalg.norm(v)
        if nv < 1e-8:
            break
        a = v / nv
    return a


def refine_up_with_plane(points: np.ndarray, normal: np.ndarray, mask: np.ndarray,
                         up0: np.ndarray, sel_deg: float = 8.0, bin_cm: float = 2.0,
                         win_cm: float = 6.0, iters: int = 6) -> Dict[str, Any]:
    """
    用「法向对齐 + 沿法向的 offset 直方图」隔离最大的水平面，再做总体最小二乘拟合。

    为什么需要直方图：室内有地板/桌面/柜顶多个同法向但不同高度的平面，
    直接拟合会得到无意义的「混合平面」。offset(= p·up) 会把它们分成不同的峰。
    返回的 rms 是极好的置信度指标：rms 小说明确实锁定了单一物理平面。
    """
    sel = mask & (np.abs(normal @ up0) > math.cos(math.radians(sel_deg)))
    if int(sel.sum()) < 500:
        return {"ok": False, "reason": "对齐像素太少"}
    Q = points[sel].astype(np.float64)
    t = Q @ up0
    lo, hi = np.percentile(t, [0.5, 99.5])
    if hi - lo < 1e-6:
        return {"ok": False, "reason": "offset 范围过小"}
    nb = max(4, int((hi - lo) / (bin_cm / 100.0)))
    hist, edges = np.histogram(t, bins=nb, range=(lo, hi))
    k = int(np.argmax(hist))
    tc = 0.5 * (edges[k] + edges[k + 1])
    Q = Q[np.abs(t - tc) < (win_cm / 100.0)]
    if len(Q) < 500:
        return {"ok": False, "reason": "峰值附近点太少"}

    nrm = up0.astype(np.float64)
    rms = float("nan")
    for _ in range(iters):
        c = Q.mean(axis=0)
        Qc = Q - c
        _, _, Vt = np.linalg.svd(Qc, full_matrices=False)
        nrm = Vt[-1]                          # 最小奇异向量 = 平面法向
        d = Qc @ nrm
        mad = 1.4826 * float(np.median(np.abs(d)))
        keep = np.abs(d) < max(2.5 * mad, 0.004)
        if int(keep.sum()) < 300:
            break
        rms = float(np.sqrt((d[keep] ** 2).mean()))
        Q = Q[keep]
    if nrm @ up0 < 0:
        nrm = -nrm
    nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
    return {"ok": True, "up": nrm, "rms_m": rms,
            "n_inliers": int(len(Q)), "plane_offset_m": float(np.median(Q @ nrm))}


def estimate_world_frame(normal: Optional[np.ndarray], mask: np.ndarray,
                         points: Optional[np.ndarray] = None, seed: int = 0,
                         max_samples: int = 40000, max_axes: int = 5,
                         align_max_deg: float = 60.0, bottom_frac: float = 0.15,
                         refine: bool = True, plane_rms_max_cm: float = 3.0,
                         require_floor_prior: bool = True
                         ) -> Dict[str, Any]:
    """
    估计重力方向，构造世界系 (right, up, forward)。

    四层：
      1. 迭代 RANSAC 提取若干个主方向（Manhattan 假设），外加「地板在画面下方」先验
      2. 硬过滤：重力轴必须落在相机 up 的 align_max_deg 以内 —— 这一步是关键，
         纯按支撑率选一定会选中面积最大的墙面
      3. 可靠性闸门：只有「地板先验」胜出时才认为重力是从场景里**量**出来的；
         泛化 RANSAC 主方向胜出时一律退回相机 up，并标记 gravity_reliable=False
         （见下方注释的实测数据）
      4. offset 直方图 + 平面拟合精修，并给出 plane_rms 作为置信度

    返回的 info 中用四个字段区分「候选从哪来」与「up 实际用了什么」：
      source            原始候选来源（诊断用，保留全部候选细节）
      up_from           实际决定 up 的来源
      gravity_reliable  up 是不是量出来的（只有地板先验算）
      degraded_from     仅当丢弃了某个候选时出现

    require_floor_prior=False 可关闭第 3 层（仅供实验/对照，服务默认不关）。
    """
    e_up = np.array([0.0, -1.0, 0.0])         # OpenCV 相机系：y 向下
    e_fwd = np.array([0.0, 0.0, 1.0])
    info: Dict[str, Any] = {"source": "camera_fallback", "confidence": 0.0}
    up = e_up

    n_all = None
    if normal is not None and normal.shape[:2] == mask.shape:
        v = normal[mask]
        v = v[np.isfinite(v).all(axis=1)]
        nrm = np.linalg.norm(v, axis=1, keepdims=True)
        n_all = v[nrm[:, 0] > 1e-6] / nrm[nrm[:, 0] > 1e-6]

    if n_all is not None and len(n_all) > 200:
        rng = np.random.default_rng(seed)
        n = n_all
        if len(n) > max_samples:
            n = n[rng.choice(len(n), max_samples, replace=False)]

        cands: List[Tuple[np.ndarray, float, str]] = []
        rest = n
        for _ in range(max_axes):
            if len(rest) < 200:
                break
            a = dominant_axis(rest, rng)
            if a is None:
                break
            sup = float((np.abs(n @ a) > math.cos(math.radians(15.0))).mean())
            cands.append((a, sup, "ransac"))
            rest = rest[np.abs(rest @ a) < math.cos(math.radians(15.0))]

        if bottom_frac > 0:                   # 地板先验
            H, W = mask.shape[:2]
            y0 = int((1.0 - bottom_frac) * H)
            bv = normal[y0:H][mask[y0:H]]
            bv = bv[np.isfinite(bv).all(axis=1)]
            bn = np.linalg.norm(bv, axis=1, keepdims=True)
            bv = bv[bn[:, 0] > 1e-6] / bn[bn[:, 0] > 1e-6]
            if len(bv) > 100:
                ab = dominant_axis(bv, rng, iters=400)
                if ab is not None:
                    sup = float((np.abs(n @ ab) > math.cos(math.radians(15.0))).mean())
                    cands.append((ab, sup, "bottom_band"))

        scored = []
        for a, sup, src in cands:
            ang = math.degrees(math.acos(min(1.0, abs(float(a @ e_up)))))
            scored.append((a, sup, src, ang))
        valid = [s for s in scored if s[3] <= align_max_deg]

        if valid:
            def _key(s):
                return s[1] * (1.25 if s[2] == "bottom_band" else 1.0)
            best_a, best_sup, best_src, best_ang = max(valid, key=_key)
            up = best_a / (np.linalg.norm(best_a) + 1e-12)
            if up @ e_up < 0:
                up = -up
            info = {
                "source": f"normal_{best_src}",
                "confidence": round(best_sup, 4),
                "angle_to_camera_up_deg": round(best_ang, 2),
                "n_candidates": len(scored),
                "candidates": [
                    {"axis": [round(float(x), 4) for x in s[0]], "support": round(s[1], 4),
                     "origin": s[2], "angle_to_camera_up_deg": round(s[3], 2),
                     "accepted": s[3] <= align_max_deg} for s in scored],
            }
        else:
            info = {"source": "camera_fallback", "confidence": 0.0,
                    "reason": f"没有候选轴落在相机上方向 {align_max_deg:.0f}° 以内",
                    "n_candidates": len(scored)}

    # ---- 可靠性闸门：泛化 RANSAC 候选不可信，一律退回相机 up ---------------
    # 实测（857 个有弱真值的实例，frame_source 分组的判定准确率）：
    #     normal_bottom_band  674 例  97.8%   <- 地板先验，可用
    #     camera_fallback     120 例 100.0%   <- 本来就假设相机水平
    #     normal_ransac        63 例  46.0%   <- 泛化 RANSAC，不可用
    # 同一批 normal_ransac 样本若改用相机 up，准确率 46.0% -> 95.2%。
    # 根因：泛化 RANSAC 主方向无法区分「地板」与「墙」。align_max_deg 闸门只挡得住
    # 明显离谱的候选；相机俯仰一大，竖直墙的法向与相机 up 的夹角就是 90°-倾斜角，
    # 俯仰 30° 时恰好落到 60° 边界上，而墙的支撑率通常是地板的两倍以上（实测
    # 0.386 vs 0.179），1.25 倍的地板加成也救不回来。
    # 危险之处在于它**看起来是成功的**：平面精修照常跑通（plane_offset_m 有值），
    # 顶层 confidence 也不报警（分支中位数 0.226 vs 0.238）。
    # 注：若改成按同一分支内的对错拆分，frame_conf 反而有区分度且方向相反
    # （判错 0.365 vs 判对 0.111，即「越自信越错」）—— 该信号尚未独立复核
    # （弱真值、n=63），**不作为告警依据**，此处只用最保守的处置。
    # 因此这里不再信任任何非地板先验的候选 —— 宁可退回「假设相机水平」并显式标记。
    info["gravity_reliable"] = bool(info["source"] == "normal_bottom_band")
    info["up_from"] = info["source"]
    if require_floor_prior and not info["gravity_reliable"]:
        if info["source"] != "camera_fallback":     # camera_fallback 本来就是相机 up
            info["degraded_from"] = info["source"]
        info["up_from"] = "camera_up"
        up = e_up

    if refine and points is not None and points.shape[:2] == mask.shape:
        pr = refine_up_with_plane(points, normal, mask, up)
        if pr.get("ok"):
            rms_cm = pr["rms_m"] * 100.0
            info["plane_rms_cm"] = round(rms_cm, 2)
            info["plane_inliers"] = pr["n_inliers"]
            if rms_cm <= plane_rms_max_cm:
                up = pr["up"]
                info["plane_offset_m"] = round(pr["plane_offset_m"], 4)
                info["refined_by"] = "plane_fit"
            else:
                info["refined_by"] = "rejected_high_rms"
        else:
            info["refined_by"] = f"skipped({pr.get('reason', '?')})"

    fwd = e_fwd - float(e_fwd @ up) * up
    nf = np.linalg.norm(fwd)
    fwd = e_fwd if nf < 1e-6 else fwd / nf
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, fwd)
    up = up / (np.linalg.norm(up) + 1e-12)
    if up @ e_up < 0:
        up, right = -up, -right

    info["camera_tilt_deg"] = round(
        math.degrees(math.acos(max(-1.0, min(1.0, float(up @ e_up))))), 2)
    return {"right": right, "up": up, "forward": fwd,
            "plane_offset": info.get("plane_offset_m"), "info": info}


def height_above_plane(p: np.ndarray, frame: Dict[str, Any]) -> Optional[float]:
    t = frame.get("plane_offset")
    if t is None:
        return None
    return float(np.asarray(p, dtype=np.float64) @ frame["up"] - t)


def project_world_axes(up: np.ndarray, right: np.ndarray, forward: np.ndarray,
                       fx: float, fy: float, cx: float, cy: float,
                       width: int, height: int,
                       plane_offset: Optional[float] = None,
                       anchor_v_frac: float = 0.30, target_frac: float = 0.20,
                       min_len_m: float = 0.15, max_len_m: float = 2.5
                       ) -> Dict[str, Any]:
    """
    把世界系三轴投影成图像上的 2D 箭头端点（供前端叠加「红箭头坐标系」）。

    为什么在后端算：前端既没有世界系向量、也没有相机内参，自己没法投影。
    前端只负责拿返回的端点画箭头（颜色/线宽/标签都归前端）。

    锚点：
      · 优先取「过画面下方一点的视线」与**地平面**的交点 —— 坐标系踩在地上，
        最直观地表达重力方向；
      · 无可用地平面（没有平面 / 交点在相机后方）时退化为该视线上固定 2 m 深处
        的一点（anchor="ray"）。

    三轴**共用**一个 3D 长度 `scale_m`，由「投影后最长的那根约占屏幕对角线的
    target_frac」反解、再夹到 [min_len_m, max_len_m]。共用长度是刻意的：给每根轴
    单独缩放到等屏幕长度会破坏透视关系、看不出真实朝向；代价是某根轴若几乎与
    视线平行，投影会很短甚至缩成一点，此时用 `degenerate` 标记出来，
    让前端别画一个假箭头。

    返回的像素坐标处于**传入的 width/height 空间**；要换算到原图空间，
    调用方自行除以缩放系数。
    """
    up = np.asarray(up, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)
    width, height = int(width), int(height)
    target_px = max(8.0, float(target_frac) * math.hypot(width, height))

    def _project(p: np.ndarray) -> Optional[np.ndarray]:
        p = np.asarray(p, dtype=np.float64)
        if not np.isfinite(p).all() or p[2] <= 1e-6:
            return None                       # 落在相机背后 / 退化，不能投影
        return np.array([fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy])

    # ---- 锚点：过 (cx, cy + anchor_v_frac*H) 的视线 ----------------------
    v0 = cy + float(anchor_v_frac) * height
    ray = np.array([0.0, (v0 - cy) / fy, 1.0])
    origin: Optional[np.ndarray] = None
    anchor = "ray"
    if plane_offset is not None and abs(float(plane_offset)) > 1e-9:
        denom = float(ray @ up)
        if abs(denom) > 1e-6:                 # 地平线：平面 {p·up = plane_offset}
            s = float(plane_offset) / denom
            if 0.05 < s < 100.0:              # 交点必须在相机前方且不离谱
                origin = s * ray
                anchor = "ground"
    if origin is None:
        origin = ray * (2.0 / ray[2])         # 退化：固定 2 m 深

    o_px = _project(origin)
    if o_px is None:
        return {"space": "working", "anchor": anchor, "origin_px": None,
                "scale_m": 0.0, "axes": [], "reason": "锚点落在相机背后"}

    # ---- 共用的 3D 长度：按「1 m 时最长的那根」反解 ----------------------
    axes = (("up", up), ("right", right), ("forward", forward))
    per_m = []
    for _name, a in axes:
        q = _project(origin + a)              # 先量 1 m 能投出多少像素
        per_m.append(float(np.linalg.norm(q - o_px)) if q is not None else 0.0)
    longest = max(per_m) if per_m else 0.0
    scale_m = 1.0 if longest <= 1e-9 else target_px / longest
    scale_m = float(min(max(scale_m, min_len_m), max_len_m))

    out_axes: List[Dict[str, Any]] = []
    for name, a in axes:
        tip = _project(origin + scale_m * a)
        length_px = float(np.linalg.norm(tip - o_px)) if tip is not None else 0.0
        out_axes.append({
            "name": name,
            "tip_px": None if tip is None else [round(float(tip[0]), 2), round(float(tip[1]), 2)],
            "length_px": round(length_px, 2),
            "degenerate": tip is None or length_px < 6.0,
        })
    return {
        "space": "working",
        "anchor": anchor,
        "origin_px": [round(float(o_px[0]), 2), round(float(o_px[1]), 2)],
        "scale_m": round(scale_m, 4),
        "axes": out_axes,
    }


# ================================================================ 圆柱拟合


def basis_perp(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ a)) > 0.9:
        a = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1); e2 /= np.linalg.norm(e2)
    return e1, e2


def _kasa_circle(u: np.ndarray, v: np.ndarray) -> Tuple[float, float, float]:
    """代数最小二乘圆拟合。"""
    A = np.stack([2 * u, 2 * v, np.ones_like(u)], axis=1)
    b = u * u + v * v
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, c = sol
    return float(a), float(bb), math.sqrt(max(float(c) + a * a + bb * bb, 0.0))


def fit_cylinder(P: np.ndarray, iters: int = 6, max_radius: float = 0.30,
                 z_mad: float = 2.0) -> Optional[Dict[str, Any]]:
    """
    迭代鲁棒圆柱拟合。返回 axis / radius / length / center（轴中点 = 几何中心）。

    先按深度 MAD 剔离群（不同深度的背景），再迭代：
      主轴(PCA) -> 垂直面圆拟合 -> 按「到圆柱面距离」剔点 -> 重来
    """
    if P is None or len(P) < 60:
        return None
    Q = P.astype(np.float64)
    z = Q[:, 2]
    med = np.median(z)
    mad = 1.4826 * np.median(np.abs(z - med)) + 1e-9
    Q = Q[np.abs(z - med) < z_mad * mad]
    if len(Q) < 60:
        return None

    axis = None
    a_ = b_ = r = 0.0
    S = np.zeros(3)
    for _ in range(iters):
        c = Q.mean(axis=0)
        _, S, Vt = np.linalg.svd(Q - c, full_matrices=False)
        axis = Vt[0]
        e1, e2 = basis_perp(axis)
        rel = Q - c
        u, v = rel @ e1, rel @ e2
        a_, b_, r = _kasa_circle(u, v)
        rad = np.sqrt((u - a_) ** 2 + (v - b_) ** 2)
        keep = (np.abs(rad - r) < max(0.6 * r, 0.03)) & (rad < max_radius)
        if keep.sum() < 40:
            break
        Q = Q[keep]

    if axis is None or len(Q) < 40:
        return None
    e1, e2 = basis_perp(axis)
    rel = Q - Q.mean(axis=0)
    t = rel @ axis
    t0, t1 = float(t.min()), float(t.max())
    center = Q.mean(axis=0) + (a_ * e1 + b_ * e2) + 0.5 * (t0 + t1) * axis
    return {
        "axis": axis, "radius": float(r), "length": float(t1 - t0), "center": center,
        "surface_centroid": Q.mean(axis=0), "n_points": int(len(Q)),
        "elongation": float(S[0] / max(S[1], 1e-9)) if S[1] > 0 else float("inf"),
    }


def pitch_deg(axis: np.ndarray, up: np.ndarray) -> float:
    """瓶身轴与重力的夹角：直立 ≈ +90°，倒伏 ≈ 0°。"""
    a = axis.copy()
    if float(a @ up) < 0:
        a = -a
    return math.degrees(math.asin(max(-1.0, min(1.0, float(a @ up)))))


def resolve_box(pts: np.ndarray, mask: np.ndarray, box: List[float],
                pad: int = 0) -> Tuple[np.ndarray, int, int]:
    """框内有效三维点。返回 (points, 框像素数, 有效像素数)。"""
    H, W = mask.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, x1 = max(0, min(x0, x1) - pad), min(W, max(x0, x1) + pad)
    y0, y1 = max(0, min(y0, y1) - pad), min(H, max(y0, y1) + pad)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 3)), 0, 0
    sub = pts[y0:y1, x0:x1].reshape(-1, 3)
    m = mask[y0:y1, x0:x1].reshape(-1)
    sel = sub[m] if m.sum() else sub
    sel = sel[np.isfinite(sel).all(axis=1)]
    return sel, int((x1 - x0) * (y1 - y0)), int(m.sum())
