#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_hub.py — 单进程多模型托管，按 flag 派发

为什么需要这个类
----------------
昇腾 NPU 显存有限，实际部署中往往**同一时刻只能驻留一个模型**。
如果每个模型各起一个服务，显存会被同时占满；如果全部一次性加载，直接 OOM。

所以这里把「模型生命周期」集中管理：

    hub.forward("geometry", image_b64=...)   # 需要 MoGe -> 加载, 驱逐其他
    hub.forward("segment",  image_b64=..., boxes=[...])   # 需要 SAM -> 换入 SAM
    hub.forward("detect",   image_b64=...)   # 需要 YOLO -> 换入 YOLO

对外只有一个 `forward(task, **kwargs)` 入口，任务名就是 flag。

设计要点
--------
1. **懒加载**：第一次调用某 task 才加载权重。
2. **单模型驻留**：`max_resident` 默认 1。加载新模型前先驱逐最久未用的，
   并显式释放设备显存（`torch.cuda.empty_cache()` / `torch.npu.empty_cache()`）。
3. **线程安全**：FastAPI 会并发调用。`forward` 全程持锁——因为单 NPU 上
   本来也无法并行跑两个模型，串行化是正确语义，同时避免"跑到一半被驱逐"。
4. **零硬依赖**：torch / transformers / ultralytics 全部在 loader 内部才 import。
   这样纯 CPU 的精简镜像（只做结构校验）也能 `import model_hub` 而不报错。
5. **缺权重不崩**：`available()` 只列可用项；调用不可用 task 返回结构化错误，
   明确指出缺哪个文件，而不是抛 500。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("model_hub")

DEFAULT_WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/app/weights")

# 各 task 需要的权重文件名（用于「可用性」检查，不含扩展名匹配）
_WEIGHT_HINTS: Dict[str, Tuple[str, ...]] = {
    "geometry": ("moge",),          # 目录或 .pt 文件里含 "moge"
    "segment": ("sam",),
    "detect": ("yolo",),
}


# ---------------------------------------------------------------- 设备


def detect_device() -> str:
    """优先昇腾 NPU，其次 CUDA，最后 CPU。"""
    for mod, dev, probe in (
        ("torch_npu", "npu:0", "npu"),
        (None, "cuda:0", "cuda"),
    ):
        try:
            import torch
            if mod:
                try:
                    __import__(mod)          # 让 torch.npu 命名空间生效
                except Exception:
                    continue
            if getattr(torch, probe).is_available():
                return dev
        except Exception:
            continue
    return "cpu"


def _device_family(device: str) -> str:
    if device.startswith("npu"):
        return "npu"
    if device.startswith("cuda"):
        return "cuda"
    return "cpu"


def _empty_cache(device: str) -> None:
    try:
        import torch
        fam = _device_family(device)
        if fam == "npu":
            torch.npu.empty_cache()
        elif fam == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def _mem_info(device: str) -> Dict[str, Any]:
    """设备显存占用（MB）。拿不到就返回空。"""
    try:
        import torch
        fam = _device_family(device)
        if fam == "npu":
            alloc = torch.npu.memory_allocated()
            total = torch.npu.get_device_properties(0).total_memory
        elif fam == "cuda":
            alloc = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
        else:
            return {}
        return {"allocated_mb": round(alloc / 2 ** 20, 1),
                "total_mb": round(total / 2 ** 20, 1)}
    except Exception:
        return {}


# ---------------------------------------------------------------- 任务注册


@dataclass
class TaskSpec:
    name: str
    description: str
    loader: Callable[[], Any]
    runner: Callable[[Any, Dict[str, Any]], Any]
    weight_hints: Tuple[str, ...] = ()
    est_mem_mb: int = 0
    aliases: Tuple[str, ...] = field(default_factory=tuple)


class ModelHub:
    """单进程内的模型托管器：懒加载 + 单模型驻留 + flag 派发。"""

    def __init__(
        self,
        device: Optional[str] = None,
        weights_dir: Optional[str] = None,
        max_resident: int = 1,
        warmup: bool = False,
    ) -> None:
        self.device = device or os.environ.get("MODEL_DEVICE") or detect_device()
        self.weights_dir = weights_dir or DEFAULT_WEIGHTS_DIR
        self.max_resident = max(1, int(max_resident))

        self._lock = threading.RLock()          # 保护加载/驱逐/推理
        self._reg_lock = threading.RLock()      # 只保护注册表（静态）
        self._tasks: Dict[str, TaskSpec] = {}
        self._alias: Dict[str, str] = {}
        self._loaded: "OrderedDict[str, Any]" = OrderedDict()   # LRU：尾部最新
        self._load_error: Dict[str, str] = {}
        self._call_count: Dict[str, int] = {}

        self._register_defaults()
        if warmup:
            self.warmup()

    # ------------------------------------------------------------ 注册

    def register(self, spec: TaskSpec) -> None:
        with self._reg_lock:
            self._tasks[spec.name] = spec
            for a in spec.aliases:
                self._alias[a] = spec.name

    def _resolve(self, task: str) -> str:
        if task in self._tasks:
            return task
        if task in self._alias:
            return self._alias[task]
        raise KeyError(f"未知 task: {task!r}，可用: {sorted(self._tasks)}")

    def _register_defaults(self) -> None:
        self.register(TaskSpec(
            name="geometry",
            description=("MoGe 单目重建（版本由 MOGE_VERSION 选：v2 / v3）+ 重力世界系 + 圆柱拟合"
                         " -> 朝向/几何中心/离地高度/半径/长度"),
            loader=self._load_geometry,
            runner=lambda m, kw: m.run(**kw),
            weight_hints=_WEIGHT_HINTS["geometry"],
            # 实测峰值（1024 max_side, CUDA）：v2 fp16 ≈ 1.3G，v3 fp32 ≈ 2.5G。
            # 取 v3 的量级，宁大勿小。
            est_mem_mb=3000,
            aliases=("geom", "moge"),
        ))
        self.register(TaskSpec(
            name="segment",
            description="SAM 框提示分割 -> 气瓶掩码多边形",
            loader=self._load_segment,
            runner=lambda m, kw: m.run(**kw),
            weight_hints=_WEIGHT_HINTS["segment"],
            est_mem_mb=1800,
            aliases=("seg", "sam"),
        ))
        self.register(TaskSpec(
            name="detect",
            description="YOLO-seg 检测气瓶（自带分割头，通常与 segment 交替使用）",
            loader=self._load_detect,
            runner=lambda m, kw: m.run(**kw),
            weight_hints=_WEIGHT_HINTS["detect"],
            est_mem_mb=800,
            aliases=("det", "yolo"),
        ))

    # ------------------------------------------------------------ loader（内部才 import）

    def _load_geometry(self):
        from models.moge_geom import MogeGeometry
        return MogeGeometry(weights_dir=self.weights_dir, device=self.device)

    def _load_segment(self):
        from models.sam_seg import SamSegmenter
        return SamSegmenter(weights_dir=self.weights_dir, device=self.device)

    def _load_detect(self):
        from models.yolo_seg import YoloSegDetector
        return YoloSegDetector(weights_dir=self.weights_dir, device=self.device)

    # ------------------------------------------------------------ 权重可用性

    def _weights_present(self, hints: Tuple[str, ...]) -> Tuple[bool, List[str]]:
        """检查 weights_dir 下有没有匹配 hint 的文件/目录。"""
        if not hints:
            return True, []
        if not os.path.isdir(self.weights_dir):
            return False, [f"权重目录不存在: {self.weights_dir}"]
        entries = [e.lower() for e in os.listdir(self.weights_dir)]
        found = []
        for h in hints:
            hit = any(h in e for e in entries)
            found.append(h if hit else f"{h}* (缺失)")
        ok = all("缺失" not in f for f in found)
        return ok, found

    def available(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        with self._reg_lock:
            specs = list(self._tasks.values())
        for s in specs:
            ok, files = self._weights_present(s.weight_hints)
            err = self._load_error.get(s.name)
            out[s.name] = {
                "description": s.description,
                "weights_dir": self.weights_dir,
                "weights_found": ok and err is None,
                "weights": files,
                "resident": s.name in self._loaded,
                "est_mem_mb": s.est_mem_mb,
                "load_error": err,
            }
        return out

    # ------------------------------------------------------------ 加载 / 驱逐

    def _evict_until(self, need: int) -> List[str]:
        """驱逐最久未使用的模型，直到常驻数 < need。调用方须持锁。"""
        evicted: List[str] = []
        while len(self._loaded) >= need and self._loaded:
            name, _ = self._loaded.popitem(last=False)   # 弹出最旧的
            evicted.append(name)
            log.info("驱逐模型: %s", name)
        if evicted:
            _empty_cache(self.device)
        return evicted

    def _acquire(self, name: str) -> Any:
        """拿到已加载的模型；不在则加载（必要时先驱逐）。调用方须持锁。"""
        if name in self._loaded:
            self._loaded.move_to_end(name)
            return self._loaded[name]

        spec = self._tasks[name]
        ok, files = self._weights_present(spec.weight_hints)
        if not ok:
            raise FileNotFoundError(
                f"task={name} 缺少权重，期望 {spec.weight_hints}，实际 {files}；"
                f"WEIGHTS_DIR={self.weights_dir}"
            )
        self._evict_until(self.max_resident)
        t0 = time.time()
        log.info("加载模型: %s -> %s", name, self.device)
        model = spec.loader()
        self._loaded[name] = model
        self._load_error.pop(name, None)
        log.info("加载完成: %s (%.1fs)", name, time.time() - t0)
        return model

    # ------------------------------------------------------------ 对外入口

    def forward(self, task: str, **kwargs: Any) -> Dict[str, Any]:
        """按 flag 派发。全程持锁：单设备上并行跑两个模型没有意义，且防止半途被驱逐。"""
        name = self._resolve(task)
        with self._lock:
            try:
                model = self._acquire(name)
                result = self._tasks[name].runner(model, kwargs)
                self._call_count[name] = self._call_count.get(name, 0) + 1
                return {"ok": True, "task": name, "device": self.device, "result": result}
            except FileNotFoundError as e:
                return {"ok": False, "task": name, "device": self.device,
                        "result": None, "error": f"weights_missing: {e}"}
            except Exception as e:                      # noqa: BLE001
                log.exception("task=%s 执行失败", name)
                return {"ok": False, "task": name, "device": self.device,
                        "result": None, "error": f"{type(e).__name__}: {e}"}

    def unload(self, task: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if task is None:
                names = list(self._loaded)
                self._loaded.clear()
            else:
                name = self._resolve(task)
                names = [name] if name in self._loaded else []
                self._loaded.pop(name, None)
            _empty_cache(self.device)
            return {"unloaded": names, "resident": list(self._loaded), **_mem_info(self.device)}

    def warmup(self) -> Dict[str, Any]:
        """预加载所有可用 task（会互相驱逐，最终只剩最后一个）。"""
        done = {}
        for name in list(self._tasks):
            with self._lock:
                try:
                    self._acquire(name)
                    done[name] = "ok"
                except Exception as e:              # noqa: BLE001
                    done[name] = f"{type(e).__name__}: {e}"
        return done

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            out = {
                "device": self.device,
                "weights_dir": self.weights_dir,
                "max_resident": self.max_resident,
                "resident": list(self._loaded),
                "calls": dict(self._call_count),
                "mem": _mem_info(self.device),
            }
            try:                       # 策略信息对排障很重要，但绝不能让它拖垮 /healthz
                from models.device import get_policy
                out["policy"] = get_policy(self.device).report()
            except Exception as e:                          # noqa: BLE001
                out["policy"] = {"error": f"{type(e).__name__}: {e}"}
            return out


# ---------------------------------------------------------------- 单例

_HUB: Optional[ModelHub] = None
_HUB_LOCK = threading.Lock()


def get_hub() -> ModelHub:
    """进程级单例，供 FastAPI 复用。"""
    global _HUB
    if _HUB is None:
        with _HUB_LOCK:
            if _HUB is None:
                _HUB = ModelHub(
                    device=os.environ.get("MODEL_DEVICE"),
                    weights_dir=os.environ.get("WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR),
                    max_resident=int(os.environ.get("MAX_RESIDENT_MODELS", "1")),
                )
    return _HUB
