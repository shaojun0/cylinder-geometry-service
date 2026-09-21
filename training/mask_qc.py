#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask_qc.py -- 从批量掩码里随机抽 N 个实例, 拼成 contact sheet 供人工抽查。

每个 tile: 原图 + 已有框(红) + SAM 掩码(绿半透明) + 标题(image / source / gid / iou / vis)

用法:
  python mask_qc.py --out OUTDIR --images IMGDIR --n 42 --cols 6 --png mask_qc.png [--seed 0]
"""
from __future__ import annotations
import argparse, glob, json, os, random, sys
import numpy as np
import cv2

sys.path.insert(0, "/root/autodl-tmp/moge")
from spatial_relations import imread_unicode  # noqa: E402


def put(img, text, org, scale, color, th=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=42)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tile", type=int, default=340)
    ap.add_argument("--png", default="mask_qc.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=0, help=">0 时按排序等间隔抽样而非随机")
    ap.add_argument("--require-status", default="ok")
    a = ap.parse_args()

    rows = []
    for f in glob.glob(os.path.join(a.out, "parts", "*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            if a.require_status and r.get("status") != a.require_status:
                continue
            if r.get("mask_px", 0) < 50:
                continue
            rows.append(r)
    rows.sort(key=lambda r: (r["image"], r["group_id"]))
    print(f"[qc] 候选实例 {len(rows)}")
    if not rows:
        raise SystemExit("没有可用实例")

    if a.stride > 0:
        idx = list(range(0, len(rows), max(1, len(rows) // a.n)))[: a.n]
    else:
        random.Random(a.seed).shuffle(rows)
        idx = range(min(a.n, len(rows)))
        rows = rows
    sel = [rows[i] for i in idx] if a.stride > 0 else rows[: min(a.n, len(rows))]
    # 尽量跨 source 覆盖
    if a.stride == 0:
        by_src = {}
        for r in rows:
            by_src.setdefault(r["source"], []).append(r)
        for k in by_src:
            random.Random(a.seed).shuffle(by_src[k])
        sel, srcs = [], sorted(by_src)
        while len(sel) < min(a.n, len(rows)) and any(by_src.values()):
            for s in srcs:
                if by_src[s] and len(sel) < min(a.n, len(rows)):
                    sel.append(by_src[s].pop())

    cols = a.cols
    nrow = (len(sel) + cols - 1) // cols
    T = a.tile
    sheet = np.full((nrow * T, cols * T, 3), 30, np.uint8)
    for i, r in enumerate(sel):
        stem = r["image"]
        bgr = imread_unicode(os.path.join(a.images, stem + ".jpg"))
        H0, W0 = bgr.shape[:2]
        mp = os.path.join(a.out, "masks", f"{stem}__g{r['group_id']}.png")
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (H0, W0):
            m = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST)
        sc = T / max(H0, W0)
        vis = cv2.resize(bgr, (int(round(W0 * sc)), int(round(H0 * sc))), interpolation=cv2.INTER_AREA)
        mm = cv2.resize(m, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
        vis[mm] = (0.45 * vis[mm] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
        bx = [r["box_x0"] * sc, r["box_y0"] * sc, r["box_x1"] * sc, r["box_y1"] * sc]
        cv2.rectangle(vis, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 0, 255), 2)
        tile = np.full((T, T, 3), 15, np.uint8)
        y0 = (T - vis.shape[0]) // 2
        x0 = (T - vis.shape[1]) // 2
        tile[max(0, y0):max(0, y0) + vis.shape[0], max(0, x0):max(0, x0) + vis.shape[1]] = vis
        put(tile, f"{r['source']} g{r['group_id']}", (4, 16), 0.42, (255, 255, 0), 1)
        lab = f"iou={r.get('sam_iou', float('nan')):.2f} vis={r.get('visible_ratio', float('nan')):.2f}"
        put(tile, lab, (4, 32), 0.42, (0, 255, 255), 1)
        lab2 = f"r={r.get('radius_m', float('nan')):.3f} L={r.get('length_m', float('nan')):.2f} p={r.get('pitch_deg', float('nan')):.0f}"
        put(tile, lab2, (4, T - 8), 0.42, (255, 200, 200), 1)
        rr, cc = divmod(i, cols)
        sheet[rr * T:(rr + 1) * T, cc * T:(cc + 1) * T] = tile

    cv2.imwrite(a.png, sheet)
    print(f"[qc] 写入 {a.png}  尺寸 {sheet.shape[1]}x{sheet.shape[0]}  tile={len(sel)}")
    print("[qc] sources:", {s: sum(1 for r in sel if r['source'] == s) for s in sorted(set(r['source'] for r in sel))})


if __name__ == "__main__":
    main()
