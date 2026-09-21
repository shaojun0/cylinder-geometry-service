#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_full_infer.py -- (可选第 4 步) 用训好的 YOLO-seg 跑全量推理, 与 SAM 掩码的几何量测对比

对每张图:
  YOLO-seg predict -> 实例掩码 (conf 阈值)
  与同图 SAM 实例按「掩码 IoU / 框 IoU」贪心匹配
  用缓存好的 MoGe 点云 (+ 世界系) 对 YOLO 掩码做同样的 fit_cylinder
输出:
  yolo_vs_sam.csv   逐匹配实例的 mask_iou / pitch/r/length 差异
  yolo_vs_sam.json  汇总统计
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, "/root/autodl-tmp/moge")
sys.path.insert(0, "/root/autodl-tmp/moge/pitch_test")
from fit_cylinder import fit_cylinder  # noqa: E402

MAXSIDE = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="/root/autodl-tmp/moge/gaspipe/ds")
    ap.add_argument("--out", default="/root/autodl-tmp/moge/gaspipe/out")
    ap.add_argument("--images", default="/root/autodl-tmp/gasdata/images")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--csv", default="/root/autodl-tmp/moge/gaspipe/yolo_vs_sam.csv")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)

    m = pd.read_csv(os.path.join(a.ds, "dataset_map.csv"))
    idx = pd.read_csv(os.path.join(a.out, "masks_index.csv"))
    idx = idx[idx.status == "ok"]
    if a.limit:
        m = m.head(a.limit)

    rows = []
    for _, rec in m.iterrows():
        stem, ascii_name = rec.image, rec.ascii_name
        imgp = os.path.join(a.images, stem + ".jpg")
        bgr = cv2.imdecode(np.fromfile(imgp, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        H0, W0 = bgr.shape[:2]
        sc = min(1.0, MAXSIDE / max(H0, W0))
        W, H = int(round(W0 * sc)), int(round(H0 * sc))

        npz_p = os.path.join(a.out, "moge_cache", stem + ".npz")
        if not os.path.exists(npz_p):
            continue
        z = np.load(npz_p)
        pts = z["points"]
        valid = z["valid"]
        up = z["up"].astype(np.float64)
        right = z["right"].astype(np.float64)
        fwd = z["forward"].astype(np.float64)
        plane_offset = float(z["plane_offset"]) if not np.isnan(z["plane_offset"]) else None

        res = model.predict(bgr, imgsz=a.imgsz, conf=a.conf, retina_masks=True, verbose=False)[0]
        ymasks = []
        if res.masks is not None and res.masks.data is not None and len(res.masks.data):
            for k in range(len(res.masks.data)):
                mk = res.masks.data[k].cpu().numpy() > 0.5
                if mk.shape != (H0, W0):
                    mk = cv2.resize(mk.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
                ymasks.append((mk, float(res.boxes.conf[k].cpu())))
        # SAM 实例掩码 (工作分辨率)
        sam = idx[idx.image == stem]
        smasks = []
        for _, r in sam.iterrows():
            mp = os.path.join(a.out, "masks", f"{stem}__g{int(r.group_id)}.png")
            s = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if s is None:
                continue
            if s.shape != (H, W):
                s = cv2.resize(s, (W, H), interpolation=cv2.INTER_NEAREST)
            smasks.append((s > 127, r))
        if not smasks:
            continue

        def geo(mask_work):
            sel = mask_work & valid
            P = pts[sel]
            P = P[np.isfinite(P).all(axis=1)]
            if len(P) < 60:
                return None
            f = fit_cylinder(P)
            if f is None:
                return None
            ax = np.asarray(f["axis"], dtype=np.float64)
            if float(ax @ up) < 0:
                ax = -ax
            aw = np.array([ax @ right, ax @ up, ax @ fwd])
            ctr = np.asarray(f["center"], dtype=np.float64)
            ch = None if plane_offset is None else float(ctr @ up - plane_offset)
            return dict(pitch=math.degrees(math.asin(max(-1, min(1, float(aw[1]))))),
                        r=float(f["radius"]), L=float(f["length"]),
                        center=[float(v) for v in ctr], height=ch, n=int(len(P)))

        used = set()
        for ym, yconf in ymasks:
            yw = cv2.resize(ym.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            best, bj = 0.0, -1
            for j, (sm, r) in enumerate(smasks):
                if j in used:
                    continue
                inter = int((yw & sm).sum()); union = int((yw | sm).sum())
                iou = inter / max(1, union)
                if iou > best:
                    best, bj = iou, j
            if bj < 0 or best < 0.05:
                continue
            used.add(bj)
            sm, r = smasks[bj]
            gs = geo(sm)
            gy = geo(yw)
            rows.append(dict(
                image=stem, source=r.source, group_id=int(r.group_id),
                yolo_conf=round(yconf, 4), mask_iou=round(best, 4),
                sam_iou=float(r.sam_iou), visible_ratio=float(r.visible_ratio),
                area_sam=int(sm.sum()), area_yolo=int(yw.sum()),
                area_ratio=round(float(yw.sum()) / max(1, int(sm.sum())), 4),
                pitch_sam=(None if gs is None else round(gs["pitch"], 3)),
                pitch_yolo=(None if gy is None else round(gy["pitch"], 3)),
                r_sam=(None if gs is None else round(gs["r"], 5)),
                r_yolo=(None if gy is None else round(gy["r"], 5)),
                L_sam=(None if gs is None else round(gs["L"], 5)),
                L_yolo=(None if gy is None else round(gy["L"], 5)),
                h_sam=(None if gs is None or gs["height"] is None else round(gs["height"], 5)),
                h_yolo=(None if gy is None or gy["height"] is None else round(gy["height"], 5)),
                n_det=len(ymasks), n_sam=len(smasks),
            ))
        print(f"[{rec.name + 1}/{len(m)}] {stem[:50]} det={len(ymasks)} sam={len(smasks)} "
              f"matched={sum(1 for x in rows if x['image'] == stem)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.csv, index=False)
    s = dict(n_matched=int(len(df)), n_images=int(df.image.nunique()) if len(df) else 0)
    if len(df):
        g = df.dropna(subset=["pitch_sam", "pitch_yolo"])
        s["median_mask_iou"] = float(df.mask_iou.median())
        s["median_area_ratio"] = float(df.area_ratio.median())
        s["recall"] = float(len(df) / max(1, int(idx.shape[0])))
        s["pitch_absdiff_median"] = float((g.pitch_sam - g.pitch_yolo).abs().median())
        s["lying_agreement"] = float(((g.pitch_sam < 45) == (g.pitch_yolo < 45)).mean())
        for k in ("r", "L", "h"):
            d = (df[f"{k}_sam"].astype(float) - df[f"{k}_yolo"].astype(float)).dropna()
            s[f"{k}_median_absdiff"] = float(d.abs().median()) if len(d) else None
            s[f"{k}_median_rel_diff"] = float((d.abs() / df[f"{k}_sam"].astype(float).dropna().abs().replace(0, np.nan)).median()) if len(d) else None
    json.dump(s, open(os.path.splitext(a.csv)[0] + ".json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(s, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
