#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_app.py — 不依赖 pytest 的自检脚本（用假模型，不需要真实权重/GPU）

覆盖：
  1. core.py 数值回归（用真实 npy 数据，如有）
  2. ModelHub：懒加载 / 单模型驻留 / 驱逐 / flag 派发 / 并发安全 / 缺权重降级
  3. FastAPI 各端点与错误路径

用法：
    cd app && python3 ../tests/test_app.py
    cd app && python3 ../tests/test_app.py --data /path/to/np_dir   # 加跑数值回归
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, APP_DIR)

FAILED = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILED.append(msg)


# ---------------------------------------------------------------- 1. core 回归


def test_core(data_dir: str | None) -> None:
    print("\n=== 1. core.py 数值回归 ===")
    import numpy as np
    from models.core import estimate_world_frame, fit_cylinder, pitch_deg

    # 退化输入必须安全
    check(fit_cylinder(np.zeros((0, 3))) is None, "空点云 -> None")
    check(fit_cylinder(np.random.rand(5, 3)) is None, "点数不足 -> None")
    f = estimate_world_frame(None, np.ones((10, 10), bool), points=None)
    check(f["info"]["source"] == "camera_fallback", "无法向 -> 回退相机坐标系")
    rng = np.random.default_rng(0)
    n = rng.normal(size=(5000, 3)); n /= np.linalg.norm(n, axis=1, keepdims=True)
    P = np.tile(np.array([0.0, 1.0, 3.0]), (5000, 1))
    f2 = estimate_world_frame(n, np.ones((50, 100), bool)[:, :50] & True, points=None)
    check(isinstance(f2["up"], np.ndarray), "纯随机法向不崩")

    if not data_dir:
        print("  (未提供 --data，跳过真实数据回归)")
        return
    need = ["02_Office_normal.npy", "02_Office_points.npy", "02_Office_mask.npy"]
    if not all(os.path.exists(os.path.join(data_dir, x)) for x in need):
        print(f"  (在 {data_dir} 找不到 {need}，跳过)")
        return
    N = np.load(os.path.join(data_dir, "02_Office_normal.npy"))
    P = np.load(os.path.join(data_dir, "02_Office_points.npy"))
    M = np.load(os.path.join(data_dir, "02_Office_mask.npy")).astype(bool)
    fr = estimate_world_frame(N, M, points=P, seed=0)
    i = fr["info"]
    check(abs(i["camera_tilt_deg"] - 17.97) < 0.1, f"camera_tilt={i['camera_tilt_deg']} ≈ 17.97")
    check(abs(i.get("plane_rms_cm", 9) - 0.40) < 0.05, f"plane_rms={i.get('plane_rms_cm')} ≈ 0.40")
    check(abs(abs(fr["plane_offset"]) - 1.887) < 0.01, f"相机离地={abs(fr['plane_offset']):.3f} ≈ 1.887")

    def med(b):
        return np.median(P[b[1]:b[3], b[0]:b[2]].reshape(-1, 3), axis=0)
    d = med((250, 562, 330, 594)) - med((60, 520, 200, 570))
    check(abs(float(d @ fr["up"])) < 0.02, f"两块地板 Δup={float(d @ fr['up'])*100:+.2f}cm ≈ 0")


# ---------------------------------------------------------------- 2. ModelHub


def _mock_hub(weights_dir: str, max_resident: int = 1):
    from model_hub import ModelHub, TaskSpec
    alive: set = set()
    events: list = []

    def mk(name):
        def load():
            events.append(f"LOAD {name}"); alive.add(name); return {"name": name}

        def run(model, kw):
            assert model["name"] == name, f"{name} 被驱逐后仍在使用"
            events.append(f"RUN {name}"); time.sleep(0.01)
            return {"by": name, "fields": sorted(kw)}
        return load, run

    h = ModelHub(device="cpu", weights_dir=weights_dir, max_resident=max_resident)
    h.register(TaskSpec("geometry", "fake", *mk("geometry"), ("moge",)))
    h.register(TaskSpec("segment", "fake", *mk("segment"), ("sam",)))
    h.register(TaskSpec("detect", "fake", *mk("detect"), ("yolo",), aliases=("det", "yolo")))
    return h, events


def test_hub() -> None:
    print("\n=== 2. ModelHub ===")
    wd = tempfile.mkdtemp()
    for n in ("moge_x.pt", "sam_x.pt", "yolo_seg_x.pt"):
        open(os.path.join(wd, n), "w").close()

    h, events = _mock_hub(wd, max_resident=1)
    check(all(v["weights_found"] for v in h.available().values()), "available(): 权重全部就绪")
    check(h.stats()["resident"] == [], "懒加载：调用前无模型驻留")

    r = h.forward("geometry", image_b64="x", regions=[], junk=1)
    check(r["ok"] and r["task"] == "geometry", "forward('geometry') 成功")
    check(h.stats()["resident"] == ["geometry"], "单模型驻留：只有 geometry")

    h.forward("segment", image_b64="x", boxes=[[0, 0, 1, 1]])
    check(h.stats()["resident"] == ["segment"], "切换到 segment 后 geometry 被驱逐")

    r = h.forward("yolo", image_b64="x")
    check(r["task"] == "detect", "别名 'yolo' -> 'detect'")
    r = h.forward("geom", image_b64="x")
    check(r["task"] == "geometry", "别名 'geom' -> 'geometry'")

    # 并发：运行期间模型不得被换掉
    errors: list = []

    def worker(t):
        for _ in range(5):
            res = h.forward(t, image_b64="x")
            if not res["ok"]:
                errors.append((t, res["error"]))

    ths = [threading.Thread(target=worker, args=(t,))
           for t in ["geometry", "segment", "detect"] * 2]
    [t.start() for t in ths]; [t.join() for t in ths]
    check(not errors, f"6 线程并发无错误 (errors={errors[:2]})")

    h2, _ = _mock_hub("/nonexistent-dir-xyz")
    r = h2.forward("geometry")
    check(r["ok"] is False and r["error"].startswith("weights_missing"),
          "缺权重 -> 结构化错误而非抛异常")
    try:
        h.forward("nonexistent")
        check(False, "未知 task 应报错")
    except KeyError:
        check(True, "未知 task -> KeyError")

    out = h.unload()
    check(h.stats()["resident"] == [], f"unload() 清空常驻 (unloaded={out['unloaded']})")


# ---------------------------------------------------------------- 3. HTTP


def test_http() -> None:
    print("\n=== 3. FastAPI ===")
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("  (未安装 fastapi/httpx，跳过)")
        return
    wd = tempfile.mkdtemp()
    for n in ("moge_x.pt", "sam_x.pt", "yolo_seg_x.pt"):
        open(os.path.join(wd, n), "w").close()
    h, _ = _mock_hub(wd, max_resident=1)

    import model_hub
    model_hub._HUB = h
    import service
    c = TestClient(service.app)

    r = c.get("/healthz")
    check(r.status_code == 200 and r.json()["ok"], f"GET /healthz -> {r.status_code}")
    r = c.get("/v1/tasks")
    check(r.status_code == 200 and len(r.json()["tasks"]) == 3, "GET /v1/tasks -> 3 个任务")

    for task, extra in [("geometry", {"regions": [], "max_side": 1024}),
                        ("segment", {"boxes": [[0, 0, 9, 9]]}),
                        ("detect", {"conf": 0.3}),
                        ("yolo", {})]:
        r = c.post("/v1/infer", json={"task": task, "image_b64": "x", **extra})
        check(r.status_code == 200 and r.json()["ok"], f"POST /v1/infer task={task} -> 200")

    r = c.post("/v1/infer", json={"task": "geometry", "image_b64": "x", "junk": 1})
    check(r.json().get("ignored_fields") == ["junk"], "白名单：未知字段被忽略并回报")

    check(c.post("/v1/infer", json={}).status_code == 400, "缺 task -> 400")
    check(c.post("/v1/infer", json={"task": "nope"}).status_code == 400, "未知 task -> 400")

    h3, _ = _mock_hub("/nonexistent-dir-xyz")
    model_hub._HUB = h3
    check(c.post("/v1/infer", json={"task": "geometry", "image_b64": "x"}).status_code == 404,
          "缺权重 -> 404")
    model_hub._HUB = h
    check(c.post("/v1/unload").status_code == 200, "POST /v1/unload -> 200")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="含 02_Office_*.npy 的目录，用于数值回归")
    a = ap.parse_args()
    test_core(a.data)
    test_hub()
    test_http()
    print("\n" + "=" * 56)
    if FAILED:
        print(f"失败 {len(FAILED)} 项:")
        for f in FAILED:
            print("  -", f)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
