#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_masks.py -- 全数据集「零样本 SAM 掩码 + YOLO-seg 标签 + 圆柱几何量测」批量生成

链路 (per image, MoGe 只跑一次):
    image
      |-- MoGe-2  -> points / depth / normal / valid      (工作分辨率 MAXSIDE=1024)
      |     `-- estimate_world_frame -> up / forward / right / plane_offset (重力对齐世界系)
      |-- 每个已有的框 (来自 gas_pose_instances.csv, 像素坐标)
      |     `-- SAM (box prompt, 3 候选取 iou 最高) -> 二值掩码
      |           |-- 掩码 PNG (工作分辨率)
      |           |-- YOLO-seg polygon 标签 (还原到原图尺寸后归一化)
      |           `-- 掩码内三维点 -> fit_cylinder -> axis/radius/length/center
      |                                                   `-- height_above_plane -> 离地高度

输出 (OUT=):
    moge_cache/<image>.npz     MoGe 点云/法向/世界系 缓存 (供后续复用, 免二次推理)
    masks/<image>__g<gid>.png  每实例掩码 (0/255)
    labels/<image>.txt         YOLO-seg 标签, class 0 = cylinder
    parts/<image>.json         每图逐实例结果 (含几何量测)
    done/<image>.done          断点续跑标记
    failed/<image>.fail        失败标记(含 traceback)
    batch.log (由 nohup 重定向)

用法:
    python batch_masks.py --images /root/autodl-tmp/gasdata/images \
                          --csv /root/autodl-tmp/gasdata/gas_pose_instances.csv \
                          --out /root/autodl-tmp/moge/gaspipe/out [--limit N] [--shard i/N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import cv2
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/moge")
sys.path.insert(0, "/root/autodl-tmp/moge/pitch_test")

from spatial_relations import (  # noqa: E402
    load_model, run_moge, estimate_world_frame, height_above_plane,
)
from fit_cylinder import fit_cylinder  # noqa: E402

MAXSIDE = 1024
MOGE_REPO = "Ruicheng/moge-2-vitl-normal"
SAM_REPO = "facebook/sam-vit-base"
CLS_CYL = 0


# ------------------------------------------------------------------ SAM

class SamBox:
    def __init__(self, repo=SAM_REPO, device="cuda"):
        import torch
        from transformers import SamModel, SamProcessor
        self.torch = torch
        self.device = device
        self.proc = SamProcessor.from_pretrained(repo)
        self.model = SamModel.from_pretrained(repo).to(device).eval()

    def mask(self, rgb: np.ndarray, box):
        """box=(x0,y0,x1,y1) 像素 -> (bool mask HxW at rgb resolution, best_iou, all_iou list)"""
        t = self.torch
        x0, y0, x1, y1 = [float(v) for v in box]
        inputs = self.proc(rgb, input_boxes=[[[x0, y0, x1, y1]]], return_tensors="pt").to(self.device)
        with t.no_grad():
            out = self.model(**inputs)
        m = self.proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]                                        # (1, 3, H, W)
        scores = np.asarray(out.iou_scores.cpu().numpy(), dtype=np.float64)
        scores = scores.reshape(-1, scores.shape[-1])[0]     # (3,)
        k = int(np.argmax(scores))
        mm = np.asarray(m[0, k]) > 0
        return mm, float(scores[k]), [round(float(s), 4) for s in scores]


# ------------------------------------------------------------------ 工具

def imread_u(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def mask_to_polygons(mask, eps_frac=0.002, min_area_frac=2e-4, rel_keep=0.15):
    """掩码 -> 多边形列表 (工作分辨率坐标)

    保留最大连通域, 以及面积 >= max(min_area_frac*H*W, rel_keep*最大面积) 的其它连通域,
    以剔除 SAM 掩码上常见的碎斑 (碎斑会污染 YOLO-seg 标签)。
    """
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape
    if not cnts:
        return []
    areas = [cv2.contourArea(c) for c in cnts]
    amax = max(areas)
    thr = max(min_area_frac * H * W, rel_keep * amax)
    polys = []
    for c, ar in zip(cnts, areas):
        if ar < thr:
            continue
        eps = eps_frac * cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) >= 3:
            polys.append(ap.astype(np.float64))
    return polys


def write_yolo_seg(path, polys, W, H, cls=CLS_CYL):
    with open(path, "w") as f:
        for p in polys:
            q = np.clip(p, [0.0, 0.0], [W - 1e-6, H - 1e-6])
            xy = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in q)
            f.write(f"{cls} {xy}\n")


def pitch_of(axis, up):
    """瓶身轴与重力夹角: 直立 ~ +90, 倒伏 ~ 0"""
    a = np.asarray(axis, dtype=np.float64)
    if float(a @ up) < 0:
        a = -a
    return math.degrees(math.asin(max(-1.0, min(1.0, float(a @ up)))))


# ------------------------------------------------------------------ 主流程

def process_image(img_stem, rows_img, moge_model, sam, img_root, out, storage=True):
    """rows_img: DataFrame 子集 (同一图的所有实例)"""
    path = os.path.join(img_root, img_stem + ".jpg")
    bgr = imread_u(path)
    if bgr is None:
        raise RuntimeError(f"imread failed: {path}")
    H0, W0 = bgr.shape[:2]
    sc = min(1.0, MAXSIDE / max(H0, W0))
    if sc < 1.0:
        work = cv2.resize(bgr, (int(round(W0 * sc)), int(round(H0 * sc))), interpolation=cv2.INTER_AREA)
    else:
        work = bgr
    H, W = work.shape[:2]
    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)

    # ---- MoGe (每图一次) ----
    t0 = time.time()
    r = run_moge(moge_model, work, "cuda", use_fp16=True)
    pts = np.asarray(r["points"], dtype=np.float32)
    valid = np.asarray(r["mask"]).astype(bool) & np.isfinite(r["depth"]) & (np.asarray(r["depth"]) > 0)
    normal = r["normal"]
    frame = estimate_world_frame(normal, valid, points=pts, seed=0)
    up = np.asarray(frame["up"], dtype=np.float64)
    t_moge = time.time() - t0

    if storage:
        np.savez_compressed(
            os.path.join(out, "moge_cache", img_stem + ".npz"),
            points=pts.astype(np.float32),
            valid=valid,
            intrinsics=np.asarray(r["intrinsics"], dtype=np.float32),
            up=np.asarray(frame["up"], dtype=np.float32),
            right=np.asarray(frame["right"], dtype=np.float32),
            forward=np.asarray(frame["forward"], dtype=np.float32),
            plane_offset=np.float32(frame["plane_offset"]) if frame.get("plane_offset") is not None
            else np.float32(np.nan),
            scale=sc, H0=H0, W0=W0,
        )

    # ---- 逐框 SAM ----
    parts = []
    t_sam = 0.0
    for k_inst, (_, row) in enumerate(rows_img.iterrows()):
        x0, y0, x1, y1 = float(row.x0) * sc, float(row.y0) * sc, float(row.x1) * sc, float(row.y1) * sc
        bx = [x0, y0, x1, y1]
        # 实例表里有 2 行的 group_id 是 NaN -> 用 900+序号 兜底, 保证唯一
        gid = int(row.group_id) if pd.notna(row.group_id) else 900 + k_inst
        # 越界裁剪到工作分辨率
        cx0 = int(max(0, math.floor(x0))); cy0 = int(max(0, math.floor(y0)))
        cx1 = int(min(W, math.ceil(x1)));  cy1 = int(min(H, math.ceil(y1)))
        rec = dict(
            image=img_stem, source=row.source, group_id=gid,
            W0=W0, H0=H0, W=W, H=H,
            box_x0=round(x0, 2), box_y0=round(y0, 2), box_x1=round(x1, 2), box_y1=round(y1, 2),
            box_px=int(max(0, cx1 - cx0) * max(0, cy1 - cy0)),
            truth_axis_tilt_deg=float(row.axis_tilt_deg),
            has_top=int(row.has_top), has_base=int(row.has_base),
            up=[round(float(v), 6) for v in up],
            frame_source=str(frame.get("info", {}).get("source")),
            frame_conf=round(float(frame.get("info", {}).get("confidence", 0.0)), 4),
            plane_offset_m=(None if frame.get("plane_offset") is None
                            else round(float(frame["plane_offset"]), 5)),
            t_moge=round(t_moge, 2),
        )
        try:
            ts = time.time()
            mask, iou, all_iou = sam.mask(rgb, bx)
            t_sam += time.time() - ts
            mk = int(mask.sum())
            rec["sam_iou"] = round(iou, 4)
            rec["sam_iou_all"] = all_iou
            rec["mask_px"] = mk
            rec["visible_ratio"] = round(mk / max(1, rec["box_px"]), 4)
            if mk < 20:
                rec["status"] = "empty_mask"
                parts.append(rec)
                continue

            # 掩码 PNG
            cv2.imwrite(os.path.join(out, "masks", f"{img_stem}__g{rec['group_id']}.png"),
                        (mask * 255).astype(np.uint8))

            # YOLO-seg 标签 (累加)
            polys = mask_to_polygons(mask)
            rec["n_poly"] = len(polys)
            rec["n_poly_pts"] = int(sum(len(p) for p in polys))
            if not polys:
                rec["status"] = "no_polygon"
                parts.append(rec)
                continue
            rec["_polys"] = [p / sc for p in polys]   # 还原到原图尺寸

            # ---- 几何: 掩码内(且 MoGe 有效)三维点 -> 圆柱拟合 ----
            sel = mask & valid if mask.shape == valid.shape else mask
            P = pts[sel]
            P = P[np.isfinite(P).all(axis=1)]
            rec["n_pts"] = int(len(P))
            fit = fit_cylinder(P) if len(P) >= 60 else None
            if fit is None:
                rec["status"] = "fit_failed"
            else:
                axc = np.asarray(fit["axis"], dtype=np.float64)
                if float(axc @ up) < 0:                 # 按 axis·up 定向 (保证 axis_y >= 0)
                    axc = -axc
                axw = np.array([axc @ frame["right"], axc @ up, axc @ frame["forward"]])
                ctr = np.asarray(fit["center"], dtype=np.float64)
                ctw = np.array([ctr @ frame["right"], ctr @ up, ctr @ frame["forward"]])
                rec["pitch_deg"] = round(math.degrees(math.asin(max(-1.0, min(1.0, float(axw[1]))))), 3)
                rec["axis_xyz"] = [round(float(v), 6) for v in axw]        # 世界系
                rec["axis_cam"] = [round(float(v), 6) for v in axc]
                rec["center_xyz"] = [round(float(v), 6) for v in ctr]      # 相机系, 米
                rec["center_world_xyz"] = [round(float(v), 6) for v in ctw]
                rec["radius_m"] = round(float(fit["radius"]), 5)
                rec["length_m"] = round(float(fit["length"]), 5)
                rec["aspect_ratio"] = round(float(fit["length"]) / max(float(fit["radius"]), 1e-9), 3)
                ch = height_above_plane(ctr, frame)
                rec["center_height_m"] = None if ch is None else round(float(ch), 5)
                rec["status"] = "ok"
            parts.append(rec)
        except Exception as e:                                   # noqa: BLE001
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"
            parts.append(rec)

    # ---- 写该图的 YOLO 标签文件 ----
    all_polys = []
    for p in parts:
        all_polys.extend(p.pop("_polys", []))
    if all_polys:
        write_yolo_seg(os.path.join(out, "labels", img_stem + ".txt"), all_polys, W0, H0)

    for p in parts:
        p["t_sam"] = round(t_sam, 2)
    with open(os.path.join(out, "parts", img_stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(parts, f, ensure_ascii=False)
    return parts, t_moge, t_sam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/root/autodl-tmp/gasdata/images")
    ap.add_argument("--csv", default="/root/autodl-tmp/gasdata/gas_pose_instances.csv")
    ap.add_argument("--out", default="/root/autodl-tmp/moge/gaspipe/out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--match", default="", help="逗号分隔的子串过滤, 只处理名字命中的图 (调试用)")
    ap.add_argument("--shard", default="0/1", help="i/N 分片")
    ap.add_argument("--no-cache", action="store_true", help="不存 MoGe npz 缓存")
    ap.add_argument("--min-interval", type=float, default=0.0)
    a = ap.parse_args()

    for sub in ("moge_cache", "masks", "labels", "parts", "done", "failed"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    df = pd.read_csv(a.csv)
    # 只保留真正存在 jpg 的图片
    have = set(os.path.splitext(f)[0] for f in os.listdir(a.images) if f.lower().endswith(".jpg"))
    imgs = [i for i in sorted(df["image"].unique()) if i in have]
    missing = [i for i in sorted(df["image"].unique()) if i not in have]
    if missing:
        print(f"[warn] {len(missing)} 张图在实例表中但没有 jpg, 跳过: {missing}", flush=True)

    si, sn = (int(v) for v in a.shard.split("/"))
    if sn > 1:
        imgs = [im for k, im in enumerate(imgs) if k % sn == si]
    if a.match:
        keys = [s for s in a.match.split(",") if s]
        imgs = [im for im in imgs if any(k in im for k in keys)]
    if a.limit:
        imgs = imgs[: a.limit]

    todo = [im for im in imgs if not os.path.exists(os.path.join(a.out, "done", im + ".done"))]
    print(f"[info] 总图 {len(imgs)} | 待处理 {len(todo)} | 已跳过 {len(imgs) - len(todo)}", flush=True)
    if not todo:
        print("[info] 没有待处理图片", flush=True)
        return

    print("[info] loading MoGe-2 ...", flush=True)
    moge = load_model(MOGE_REPO, "cuda")
    print("[info] loading SAM ...", flush=True)
    sam = SamBox()
    print("[info] models ready", flush=True)

    grp = {k: v for k, v in df.groupby("image")}
    t_start = time.time()
    n_inst = 0
    for k, im in enumerate(todo):
        t0 = time.time()
        try:
            parts, t_moge, t_sam = process_image(im, grp[im], moge, sam, a.images, a.out,
                                                 storage=not a.no_cache)
            n_inst += len(parts)
            with open(os.path.join(a.out, "done", im + ".done"), "w") as f:
                f.write(str(len(parts)))
            ok = sum(1 for p in parts if p.get("status") == "ok")
            el = time.time() - t0
            eta = (len(todo) - k - 1) * el / 60.0
            print(f"[{k + 1}/{len(todo)}] {im[:60]} n={len(parts)} ok={ok} "
                  f"moge={t_moge:.1f}s sam={t_sam:.1f}s tot={el:.1f}s ETA={eta:.1f}min", flush=True)
        except Exception as e:                                   # noqa: BLE001
            os.makedirs(os.path.join(a.out, "failed"), exist_ok=True)
            with open(os.path.join(a.out, "failed", im + ".fail"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            print(f"[{k + 1}/{len(todo)}] FAILED {im[:60]}: {type(e).__name__}: {e}", flush=True)
        if a.min_interval:
            dt = time.time() - t0
            if dt < a.min_interval:
                time.sleep(a.min_interval - dt)

    el = (time.time() - t_start) / 60.0
    print(f"[done] {len(todo)} 图 / {n_inst} 实例, 用时 {el:.1f} min", flush=True)


if __name__ == "__main__":
    main()
