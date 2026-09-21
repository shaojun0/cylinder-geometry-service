#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate.py -- 汇总 batch_masks.py 的 parts/*.json, 产出

  OUT/masks_index.csv        掩码索引 (image, source, group_id, box, sam_iou, mask_px, box_px, visible_ratio, ...)
  OUT/geometry.csv           逐实例几何量测 (pitch/axis/center/height/radius/length/...)
  OUT/aggregate_summary.json 统计摘要

用法:
  python aggregate.py --out /root/autodl-tmp/moge/gaspipe/out \
                      --csv /root/autodl-tmp/gasdata/gas_pose_instances.csv \
                      --geom /root/autodl-tmp/moge/gaspipe/geometry.csv
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys
import numpy as np
import pandas as pd

REC_COLS = [
    "image", "source", "group_id", "status", "W0", "H0", "W", "H",
    "box_x0", "box_y0", "box_x1", "box_y1", "box_px", "mask_px", "visible_ratio",
    "sam_iou", "n_poly", "n_poly_pts", "n_pts",
    "pitch_deg", "axis_xyz", "axis_cam", "center_xyz", "center_world_xyz",
    "radius_m", "length_m", "aspect_ratio", "center_height_m",
    "truth_axis_tilt_deg", "has_top", "has_base",
    "frame_source", "frame_conf", "plane_offset_m", "t_moge", "t_sam",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--geom", required=True)
    ap.add_argument("--index-csv", default="")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.out, "parts", "*.json")))
    rows = []
    for f in files:
        try:
            rows.extend(json.load(open(f, encoding="utf-8")))
        except Exception as e:                                   # noqa: BLE001
            print(f"[warn] 坏 part 文件 {f}: {e}", file=sys.stderr)
    print(f"[agg] parts 文件 {len(files)} 个, 实例记录 {len(rows)}")

    df = pd.DataFrame(rows)
    for c in REC_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # 向量列展开
    for i, ax in enumerate("xyz"):
        df[f"axis_{ax}"] = df["axis_xyz"].apply(
            lambda v, i=i: (float(v[i]) if isinstance(v, (list, tuple)) and len(v) == 3 else np.nan))
        df[f"center_{ax}"] = df["center_xyz"].apply(
            lambda v, i=i: (float(v[i]) if isinstance(v, (list, tuple)) and len(v) == 3 else np.nan))

    # ---- masks_index.csv ----
    idx = df[["image", "source", "group_id", "status", "W0", "H0", "box_x0", "box_y0",
              "box_x1", "box_y1", "box_px", "mask_px", "visible_ratio", "sam_iou",
              "n_poly", "n_poly_pts", "radius_m", "length_m", "pitch_deg"]].copy()
    idx["box"] = df.apply(lambda r: f"[{r.box_x0:.1f},{r.box_y0:.1f},{r.box_x1:.1f},{r.box_y1:.1f}]", axis=1)
    idx = idx.sort_values(["source", "image", "group_id"]).reset_index(drop=True)
    ipath = a.index_csv or os.path.join(a.out, "masks_index.csv")
    idx.to_csv(ipath, index=False)
    print(f"[agg] {ipath}  {idx.shape}")

    # ---- geometry.csv ----
    g = pd.DataFrame({
        "image": df["image"], "source": df["source"], "group_id": df["group_id"],
        "status": df["status"],
        "pitch_deg": df["pitch_deg"],
        "axis_x": df["axis_x"], "axis_y": df["axis_y"], "axis_z": df["axis_z"],
        "center_x": df["center_x"], "center_y": df["center_y"], "center_z": df["center_z"],
        "center_height_m": df["center_height_m"],
        "radius_m": df["radius_m"], "length_m": df["length_m"],
        "aspect_ratio": df["aspect_ratio"],
        "visible_ratio": df["visible_ratio"], "sam_iou": df["sam_iou"],
        "mask_px": df["mask_px"], "box_px": df["box_px"], "n_pts": df["n_pts"],
        "axis_tilt_deg": df["truth_axis_tilt_deg"],
        "has_top": df["has_top"], "has_base": df["has_base"],
        "frame_source": df["frame_source"], "frame_conf": df["frame_conf"],
        "plane_offset_m": df["plane_offset_m"],
        "t_moge": df["t_moge"], "t_sam": df["t_sam"],
    })
    for i, ax in enumerate("xyz"):
        g[f"axis_cam_{ax}"] = df["axis_cam"].apply(
            lambda v, i=i: (float(v[i]) if isinstance(v, (list, tuple)) and len(v) == 3 else np.nan))
        g[f"center_world_{ax}"] = df["center_world_xyz"].apply(
            lambda v, i=i: (float(v[i]) if isinstance(v, (list, tuple)) and len(v) == 3 else np.nan))
    okk = (g["status"] == "ok").to_numpy()
    tilt = pd.to_numeric(g["axis_tilt_deg"], errors="coerce").to_numpy(dtype=float)
    has_truth = np.isfinite(tilt)
    # 弱真值只在标注里同时有 top/base (即 axis_tilt_deg 有值) 时才存在; 否则 truth_lying = NaN
    g["truth_lying"] = np.where(okk & has_truth, np.abs(tilt) > 45, np.nan)
    g["pred_lying"] = np.where(okk & g["pitch_deg"].notna().to_numpy(),
                               (g["pitch_deg"].abs() < 45).to_numpy(), np.nan)
    g["has_weak_truth"] = (okk & has_truth)
    g = g.sort_values(["source", "image", "group_id"]).reset_index(drop=True)
    g.to_csv(a.geom, index=False)
    print(f"[agg] {a.geom}  {g.shape}")

    # ---- 摘要 ----
    for c in ("truth_lying", "pred_lying"):
        g[c] = pd.to_numeric(g[c], errors="coerce").astype(float)
    ok = g[g.status == "ok"].copy()
    s = {}
    s["n_records_total"] = int(len(g))
    s["n_ok"] = int(len(ok))
    s["status_counts"] = g["status"].value_counts().to_dict()
    s["sources"] = g["source"].value_counts().to_dict()

    def acc_block(sub):
        v = sub.dropna(subset=["truth_lying", "pred_lying"])
        if not len(v):
            return dict(n_eval=0, acc=None, acc_upright=None, acc_lying=None)
        tl = v["truth_lying"].to_numpy(dtype=float) > 0.5
        pl = v["pred_lying"].to_numpy(dtype=float) > 0.5
        return dict(
            n_eval=int(len(v)),
            acc=float((tl == pl).mean()),
            acc_upright=float((~pl[~tl]).mean()) if (~tl).any() else None,
            acc_lying=float(pl[tl].mean()) if tl.any() else None,
            n_truth_upright=int((~tl).sum()), n_truth_lying=int(tl.sum()),
        )

    if len(ok):
        s.update(acc_block(ok))
        s["n_with_weak_truth"] = int(ok["has_weak_truth"].sum())
        s["n_without_weak_truth"] = int((~ok["has_weak_truth"]).sum())
        s["rmse_pitch_deg_vs_90_minus_abs_tilt"] = float(
            np.sqrt(((ok.loc[ok.has_weak_truth, "pitch_deg"]
                      - (90.0 - ok.loc[ok.has_weak_truth, "axis_tilt_deg"].abs())) ** 2).mean()))
        s["pitch_vs_tilt_spearman"] = float(
            ok.loc[ok.has_weak_truth, ["pitch_deg", "axis_tilt_deg"]]
            .assign(at=lambda d: d.axis_tilt_deg.abs())
            .corr(method="spearman").loc["pitch_deg", "at"])
        s["median_radius_m"] = float(ok.radius_m.median())
        s["median_length_m"] = float(ok.length_m.median())
        s["median_aspect"] = float(ok.aspect_ratio.median())
        s["median_sam_iou"] = float(ok.sam_iou.median())
        s["median_visible_ratio"] = float(ok.visible_ratio.median())
        s["median_center_height_m"] = float(ok.center_height_m.median())
        s["frame_source_counts"] = ok["frame_source"].value_counts().to_dict()
        per = {}
        for src, sub in ok.groupby("source"):
            d = acc_block(sub)
            d.update(
                n=int(len(sub)),
                r_med=(float(sub.radius_m.median()) if sub.radius_m.notna().any() else None),
                L_med=(float(sub.length_m.median()) if sub.length_m.notna().any() else None),
                aspect=(float(sub.aspect_ratio.median()) if sub.aspect_ratio.notna().any() else None),
                pitch_upright=(float(sub.loc[sub.truth_lying < 0.5, "pitch_deg"].median())
                               if (sub.truth_lying < 0.5).any() else None),
                pitch_lying=(float(sub.loc[sub.truth_lying > 0.5, "pitch_deg"].median())
                             if (sub.truth_lying > 0.5).any() else None),
                sam_iou=float(sub.sam_iou.median()),
                vis_med=float(sub.visible_ratio.median()),
            )
            per[src] = d
        s["per_source"] = per
        # 干净子集 (SAM 置信 & 可见度较高) 的尺度统计, 用于对照参考值
        for tag, sub in (("clean_iou95", ok[ok.sam_iou >= 0.95]),
                         ("clean_iou95_vis50", ok[(ok.sam_iou >= 0.95) & (ok.visible_ratio >= 0.5)])):
            if len(sub):
                s[f"{tag}_n"] = int(len(sub))
                s[f"{tag}_r_med"] = float(sub.radius_m.median())
                s[f"{tag}_L_med"] = float(sub.length_m.median())
                s[f"{tag}_aspect_med"] = float(sub.aspect_ratio.median())
    json.dump(s, open(os.path.join(a.out, "aggregate_summary.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(s, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
