#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
service.py — 气瓶几何量测 HTTP 服务

单进程、单入口、按 flag 派发到不同模型：

    GET  /healthz        存活 + 设备 + 当前驻留模型
    GET  /v1/tasks       可用任务列表（含权重是否就绪）
    POST /v1/infer       {"task": "detect"|"segment"|"geometry", ...}
    POST /v1/unload      卸载常驻模型，释放显存

为什么只有一个 infer 入口：昇腾 NPU 显存有限，同一时刻只驻留一个模型，
由 ModelHub 统一做懒加载与驱逐。详见 model_hub.py 的模块注释。

启动：
    uvicorn service:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict

from fastapi import Body, FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from model_hub import get_hub

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("service")

app = FastAPI(
    title="Cylinder Geometry Service",
    version="1.0.0",
    description=(
        "单目气瓶几何量测：检测 / 分割 / 朝向与几何中心。\n\n"
        "所有模型通过 ModelHub 托管，同一时刻只有一个模型驻留设备，"
        "由 `task` 字段（flag）派发。"
    ),
)

# 各 task 允许透传给模型的字段（白名单，避免把无关字段灌进去）
_TASK_FIELDS: Dict[str, set] = {
    "detect": {"image_b64", "conf", "iou", "max_det", "return_polygons"},
    "segment": {"image_b64", "boxes", "return_polygons"},
    "geometry": {"image_b64", "regions", "max_side", "refine_gravity", "return_gravity"},
}


def _wrap(task: str, out: Dict[str, Any], t0: float) -> Dict[str, Any]:
    out = dict(out or {})
    out["latency_ms"] = int((time.time() - t0) * 1000)
    out.setdefault("task", task)
    return out


@app.on_event("startup")
def _startup() -> None:
    hub = get_hub()
    log.info("服务启动 | device=%s weights_dir=%s max_resident=%d",
             hub.device, hub.weights_dir, hub.max_resident)
    # 设备策略必须在任何模型加载之前落地（310P 的 eager/fp16 尤其如此），
    # 并且要在启动日志里可见 —— 上机时一眼能看出是否真的生效。
    try:
        from models.device import get_policy
        pol = get_policy(hub.device)
        pol.apply()
        log.info("设备策略 | %s", pol.describe())
        if pol.forced_reason:
            log.warning("设备策略被强制覆盖：%s", pol.forced_reason)
    except Exception:                                       # noqa: BLE001
        log.exception("设备策略初始化失败（不阻断启动，但请排查）")
    avail = hub.available()
    for name, meta in avail.items():
        log.info("  task=%-8s weights_found=%-5s %s",
                 name, meta["weights_found"], meta["weights"])


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    hub = get_hub()
    return {"ok": True, "status": "healthy", **hub.stats()}


@app.get("/v1/tasks")
def tasks() -> Dict[str, Any]:
    hub = get_hub()
    return {"ok": True, "device": hub.device, "tasks": hub.available()}


@app.post("/v1/infer")
def infer(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    t0 = time.time()
    task = payload.get("task")
    if not task:
        return JSONResponse(status_code=400, content={
            "ok": False, "task": None, "result": None,
            "error": "缺少 task 字段（detect | segment | geometry）",
            "latency_ms": int((time.time() - t0) * 1000)})

    hub = get_hub()
    try:
        canonical = hub._resolve(task)                       # noqa: SLF001
    except KeyError as e:
        return JSONResponse(status_code=400, content={
            "ok": False, "task": task, "device": hub.device, "result": None,
            "error": str(e), "available": sorted(hub.available()),
            "latency_ms": int((time.time() - t0) * 1000)})

    allowed = _TASK_FIELDS.get(canonical, set())
    kwargs = {k: v for k, v in payload.items() if k != "task" and k in allowed}
    ignored = [k for k in payload if k != "task" and k not in allowed]

    out = hub.forward(canonical, **kwargs)
    if ignored:
        out["ignored_fields"] = ignored
    code = 200 if out.get("ok") else (
        404 if str(out.get("error", "")).startswith("weights_missing") else 500)
    return JSONResponse(status_code=code, content=_wrap(canonical, out, t0))


@app.post("/v1/unload")
def unload(task: str = Query(default="", description="留空则卸载全部")) -> Dict[str, Any]:
    hub = get_hub()
    return {"ok": True, **hub.unload(task or None)}


# ---------------------------------------------------------------- 前端静态页
# 单文件前端（web/index.html）与 API 同源托管，方便直接用浏览器测试。
# 必须**最后**挂载到 "/"：Starlette 按注册顺序匹配，先注册的
# /healthz、/v1/tasks、/v1/infer、/v1/unload（以及 /docs、/openapi.json）
# 优先命中，静态目录只兜底其余路径。
_WEB_DIR_CANDIDATES = (
    # 1) 仓库布局： <repo>/web/   （service.py 在 <repo>/app/ 下）
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web"),
    # 2) 容器布局： /web/         （Dockerfile 把 web/ 拷到 /app 旁边）
    os.path.join(os.sep, "web"),
    # 3) 退化布局： <app>/web/
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "web"),
)


def find_web_dir() -> "str | None":
    """返回第一个含 index.html 的候选目录；都找不到返回 None。"""
    for d in _WEB_DIR_CANDIDATES:
        if os.path.isfile(os.path.join(d, "index.html")):
            return d
    return None


_web_dir = find_web_dir()
if _web_dir:
    app.mount("/", StaticFiles(directory=_web_dir, html=True), name="web")
    log.info("前端已挂载: %s -> /", _web_dir)
else:
    # 降级而非崩溃：前端只是可选的人机界面，缺了不能影响 API。
    log.warning("未找到 web/index.html（候选 %s），跳过前端挂载；"
                "API 端点不受影响", list(_WEB_DIR_CANDIDATES))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",
                port=int(os.environ.get("PORT", "8000")))
