#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core.py — 与框架无关的纯数值部分（只用 numpy）

包含两块：
  1) 重力/世界系估计：逐像素法向 -> RANSAC 候选轴 -> 物理硬过滤 -> 平面拟合精修
  2) 圆柱拟合：迭代鲁棒 PCA -> 垂直面圆拟合 -> 轴中点即几何中心

这两块的物理含义与踩坑记录见 docs/ASCEND.md 与仓库根目录的说明。
关键点：
  * 重力轴必须通过「与相机 up 夹角 <= 60°」的硬过滤，否则会选中面积最大的墙面。
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
                         refine: bool = True, plane_rms_max_cm: float = 3.0
                         ) -> Dict[str, Any]:
    """
    估计重力方向，构造世界系 (right, up, forward)。

    三层：
      1. 迭代 RANSAC 提取若干个主方向（Manhattan 假设），外加「地板在画面下方」先验
      2. 硬过滤：重力轴必须落在相机 up 的 align_max_deg 以内 —— 这一步是关键，
         纯按支撑率选一定会选中面积最大的墙面
      3. offset 直方图 + 平面拟合精修，并给出 plane_rms 作为置信度
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
