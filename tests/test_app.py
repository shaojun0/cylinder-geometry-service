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

    # ---- 重力可靠性闸门（回归：泛化 RANSAC 分支必须降级）----------------
    # 构造三个确定性场景，分别命中三条分支，见 estimate_world_frame 的注释。
    import math

    def _plane(a_deg, flip=False):
        """与相机 up 夹角 a_deg 的单位法向（flip 换到另一侧）。"""
        r = math.radians(a_deg)
        s = 1.0 if flip else -1.0
        return np.array([0.0, -math.cos(r), s * math.sin(r)])

    def _scene(top, bottom, split=42, H=50, W=100):
        N = np.zeros((H, W, 3))
        N[:split] = top
        N[split:] = bottom
        return N, np.ones((H, W), bool)

    A, B = _plane(30), _plane(50, flip=True)      # A 占大部分 -> RANSAC 胜出

    # 场景 1：RANSAC 分支。默认必须丢候选、退回相机 up、标记不可靠
    N1, M1 = _scene(A, B)
    fr = estimate_world_frame(N1, M1, points=None, seed=0)
    i = fr["info"]
    check(i["source"] == "normal_ransac", f"场景1 命中 normal_ransac (source={i['source']})")
    check(i["gravity_reliable"] is False, "normal_ransac -> gravity_reliable=False")
    check(i["up_from"] == "camera_up", f"normal_ransac -> 退回相机 up (up_from={i['up_from']})")
    check(i.get("degraded_from") == "normal_ransac", "normal_ransac -> 记录 degraded_from")
    check(abs(i["camera_tilt_deg"]) < 1e-6, "退回后 camera_tilt=0（=假设相机水平）")
    check(abs(float(fr["up"] @ np.array([0.0, -1.0, 0.0])) - 1.0) < 1e-9,
          "退回后 up == 相机 up")

    # 逃生舱：关闭闸门应恢复原始行为（30° 的错误估计）
    fr_off = estimate_world_frame(N1, M1, points=None, seed=0, require_floor_prior=False)
    io = fr_off["info"]
    check(io["up_from"] == "normal_ransac" and "degraded_from" not in io,
          "require_floor_prior=False -> 不降级")
    check(abs(io["camera_tilt_deg"] - 30.0) < 1.0,
          f"关闭闸门后 tilt≈30° (实测 {io['camera_tilt_deg']}°)")
    check(io["gravity_reliable"] is False,
          "关闭闸门不改变「不可靠」这个事实判断")

    # 场景 2：地板先验胜出 -> 可靠
    N2, M2 = _scene(A, A)
    i2 = estimate_world_frame(N2, M2, points=None, seed=0)["info"]
    check(i2["source"] == "normal_bottom_band" and i2["gravity_reliable"] is True
          and i2["up_from"] == "normal_bottom_band",
          f"地板先验 -> 可靠且不降级 (source={i2['source']})")

    # 场景 3：无候选 -> camera_fallback，本就不该记 degraded_from
    N3, M3 = _scene(_plane(90, flip=True), _plane(90, flip=True))
    i3 = estimate_world_frame(N3, M3, points=None, seed=0)["info"]
    check(i3["source"] == "camera_fallback" and i3["gravity_reliable"] is False
          and i3["up_from"] == "camera_up" and "degraded_from" not in i3,
          "camera_fallback -> 不可靠但不是降级")

    # ---- 权重查找：路径含 "moge" 时不得误选 YOLO 权重 -------------------
    # 实测踩到过：部署目录 /root/autodl-tmp/moge/weights/ 让 "moge" 出现在**任意
    # 祖先目录**里，于是该目录下所有 .pt 都成了 MoGe 候选，再按路径长度取最短就
    # 选中了 yolo-cylinder-seg.pt；MoGe 的 from_pretrained 拿它去 torch.load 后报
    # `Unsupported global: ultralytics.nn.tasks.SegmentationModel`。
    import tempfile as _tf
    from models.moge_geom import _find_moge_weight
    with _tf.TemporaryDirectory() as td:
        # 场景 A —— **实测踩到的原始故障**：祖先目录含 moge + 扁平布局。
        # moge-2-vitl-normal.pt(49) 比 yolo-cylinder-seg.pt(47) 长，旧的
        # "按路径长度取最短"会选中 YOLO 权重。这一例必须能抓到旧实现。
        flat = os.path.join(td, "moge", "svc_weights")
        os.makedirs(flat)
        open(os.path.join(flat, "moge-2-vitl-normal.pt"), "w").close()
        open(os.path.join(flat, "yolo-cylinder-seg.pt"), "w").close()
        got = _find_moge_weight(flat)
        check(got is not None and "moge-2-vitl-normal.pt" in got,
              f"祖先目录含 moge + 扁平布局：不误选 YOLO (got={os.path.basename(str(got))})")

        # 场景 B：HF 快照子目录布局（内含 model.pt），同样不能被 YOLO 抢走
        hf = os.path.join(td, "moge", "hf")
        os.makedirs(os.path.join(hf, "moge-2-vitl-normal"))
        os.makedirs(os.path.join(hf, "sam-vit-base"))
        open(os.path.join(hf, "moge-2-vitl-normal", "model.pt"), "w").close()
        open(os.path.join(hf, "yolo-cylinder-seg.pt"), "w").close()
        got2 = _find_moge_weight(hf)
        check(got2 is not None and got2.endswith(os.path.join("moge-2-vitl-normal", "model.pt")),
              f"HF 快照布局选对 model.pt (got={os.path.basename(str(got2))})")

        # 场景 C：真实 HF 缓存深层布局 models--*/snapshots/<sha>/model.pt
        hub = os.path.join(td, "moge", "hub")
        snap = os.path.join(hub, "models--Ruicheng--moge-2-vitl-normal", "snapshots", "abc123")
        os.makedirs(snap)
        open(os.path.join(snap, "model.pt"), "w").close()
        got3 = _find_moge_weight(hub)
        check(got3 is not None and got3.endswith("model.pt"),
              "HF 缓存深层布局仍能找到 model.pt")

        empty = os.path.join(td, "moge", "empty")
        os.makedirs(empty)
        check(_find_moge_weight(empty) is None, "无 .pt -> 返回 None")

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
    check(i["gravity_reliable"] is True and i["up_from"] == "normal_bottom_band",
          "Office 走地板先验 -> gravity_reliable=True（闸门不得误伤）")

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

    # ---- 前端静态页：挂在 "/" 但绝不能抢走 API 路由 ----------------------
    index = os.path.join(APP_DIR, os.pardir, "web", "index.html")
    if os.path.isfile(index):
        r = c.get("/")
        check(r.status_code == 200 and "text/html" in r.headers.get("content-type", ""),
              f"GET / -> 200 text/html（前端已挂载）")
        check("gravity_reliable" in r.text, "前端 HTML 暴露 gravity_reliable（安全判据可见）")
    else:
        print(f"  (未找到 {os.path.normpath(index)}，跳过前端挂载检查)")
    check(c.get("/healthz").status_code == 200, "挂载静态目录后 /healthz 不受影响")
    check(c.get("/v1/tasks").status_code == 200, "挂载静态目录后 /v1/tasks 不受影响")
    check(c.post("/v1/infer", json={"task": "geometry", "image_b64": "x",
                                    "regions": []}).status_code == 200,
          "挂载静态目录后 /v1/infer 不受影响")


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
