#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_yolo.py -- 用零样本 SAM 生成的 YOLO-seg 标签训练轻量分割模型

训练集/验证集 = 组内切分; 测试集 = 完整留出 source (跨域泛化真实指标)

用法:
  python train_yolo.py --ds DS --model yolo11n-seg.pt --imgsz 1024 --batch 16 --epochs 100
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="/root/autodl-tmp/moge/gaspipe/ds")
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="gas_yolo11n_seg")
    ap.add_argument("--project", default="/root/autodl-tmp/moge/gaspipe/runs")
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", default="", help="给定 best.pt, 只做评估")
    a = ap.parse_args()

    from ultralytics import YOLO

    data = os.path.join(a.ds, "data.yaml")
    weights = a.eval_only or a.model
    model = YOLO(weights)

    if not a.eval_only:
        model.train(
            data=data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
            project=a.project, name=a.name, seed=a.seed, workers=a.workers,
            patience=a.patience, exist_ok=True, plots=True, val=True,
            pretrained=True, optimizer="auto", cos_lr=True, close_mosaic=10,
            resume=a.resume, deterministic=True,
        )
        best = os.path.join(a.project, a.name, "weights", "best.pt")
    else:
        best = a.eval_only

    out = {"weights": best}
    for split in ("val", "test"):
        try:
            r = YOLO(best).val(data=data, split=split, imgsz=a.imgsz, batch=a.batch,
                               device=a.device, project=a.project,
                               name=f"{a.name}_{split}", exist_ok=True, plots=True)
            out[split] = {
                "box_map50": float(r.box.map50), "box_map": float(r.box.map),
                "mask_map50": float(r.seg.map50), "mask_map": float(r.seg.map),
                "mask_precision": float(r.seg.mp), "mask_recall": float(r.seg.mr),
                "box_precision": float(r.box.mp), "box_recall": float(r.box.mr),
                "fitness": float(r.fitness),
                "speed_ms": {k: float(v) for k, v in (r.speed or {}).items()},
            }
        except Exception as e:                                       # noqa: BLE001
            out[split] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[warn] {split} eval 失败: {e}", file=sys.stderr)

    p = os.path.join(a.project, a.name, "metrics_summary.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
