#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
device.py — 设备与计算策略

为什么需要它
------------
昇腾不同型号能力不同，尤其是 **Atlas 300I Duo (310P)**：
  * 不支持图模式（graph mode），必须走 eager
  * 不支持 bfloat16，只能用 float16
如果在 310P 上按默认（可能启用图模式 / bf16）跑，会直接报错或出静默错误。

这些开关由环境变量控制，与 `scripts/run_ascend.sh` 传参一致：
    CYLINDER_DTYPE = auto | float16 | bfloat16 | float32   (默认 auto)
    CYLINDER_EAGER = 1 | 0                                  (默认 0)

auto 语义：按设备自动选。检测到 310P 时自动改成 float16 + eager。

诚实声明
--------
**本模块的昇腾部分没有在真机上验证过**（我们没有昇腾机器）。
torch_npu 的 API 名称在不同版本间变过，所以这里对「关图模式」做了多种 API 的
best-effort 尝试，并把**实际生效了什么**记录在 `policy.report()` 里，
通过 `/healthz` 暴露出来 —— 上机后一眼就能看出是否真的生效，而不是靠猜。
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger("device")

_VALID_DTYPES = ("auto", "float16", "bfloat16", "float32")


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class DevicePolicy:
    """解析并（尽力）应用设备计算策略。"""

    def __init__(self, device: str) -> None:
        self.device = device
        self.family = ("npu" if device.startswith("npu")
                       else "cuda" if device.startswith("cuda") else "cpu")

        raw_dtype = (os.environ.get("CYLINDER_DTYPE") or "auto").strip().lower()
        if raw_dtype not in _VALID_DTYPES:
            log.warning("CYLINDER_DTYPE=%r 非法，回退 auto（可选 %s）", raw_dtype, _VALID_DTYPES)
            raw_dtype = "auto"
        self.requested_dtype = raw_dtype
        self.requested_eager = _env_bool("CYLINDER_EAGER", False)

        self.device_name = self._device_name()
        self.is_310p = self._looks_like_310p(self.device_name)

        # 解析最终策略
        if self.family == "cpu":
            # CPU 上 fp16 无加速甚至更慢（且部分算子在 CPU 上不支持 fp16），统一 fp32
            self.dtype = "float32"
            self.eager = False
            self.forced_reason = ("CPU 设备：强制 float32（fp16 在 CPU 上无收益且部分算子不支持）"
                                  if raw_dtype in ("auto", "float16", "bfloat16") else None)
        elif self.is_310p:
            # 310P: 不支持图模式与 bf16 —— 强制覆盖，并记录原因
            self.dtype = "float16" if raw_dtype in ("auto", "bfloat16") else raw_dtype
            self.eager = True
            self.forced_reason = ("检测到 310P：不支持图模式与 bf16，"
                                  f"已强制 eager + {self.dtype}")
        else:
            self.dtype = "float16" if raw_dtype == "auto" else raw_dtype
            self.eager = self.requested_eager
            self.forced_reason = None

        self._applied: Dict[str, Any] = {}

    # ------------------------------------------------------------ 探测

    def _device_name(self) -> str:
        try:
            import torch
            if self.family == "npu":
                import torch_npu  # noqa: F401
                return str(torch.npu.get_device_name(0))
            if self.family == "cuda":
                return str(torch.cuda.get_device_name(0))
        except Exception as e:                              # noqa: BLE001
            return f"unknown ({type(e).__name__})"
        return "cpu"

    @staticmethod
    def _looks_like_310p(name: str) -> bool:
        n = (name or "").lower().replace(" ", "")
        # 芯片名 Ascend310P3，或产品名 Atlas 300I Duo（基于 310P）
        return "310p" in n or "310" in n or "300i" in n or "300iduo" in n

    # ------------------------------------------------------------ 应用

    def torch_dtype(self):
        import torch
        return {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}[self.dtype]

    def _try_call(self, dotted: str, **kwargs) -> str:
        """尽力调用一个可能存在的 API，返回结果描述。"""
        try:
            mod_path, _, attr = dotted.rpartition(".")
            mod = importlib.import_module(mod_path)
            fn = getattr(mod, attr, None)
            if fn is None:
                return "absent"
            fn(**kwargs)
            return "ok"
        except Exception as e:                              # noqa: BLE001
            return f"{type(e).__name__}: {e}"

    def apply(self) -> Dict[str, Any]:
        """把策略真正作用到运行时。可重复调用（幂等）。"""
        self._applied = {"device": self.device, "family": self.family,
                         "device_name": self.device_name,
                         "dtype": self.dtype, "eager": self.eager,
                         "is_310p": self.is_310p}
        if self.forced_reason:
            self._applied["forced_reason"] = self.forced_reason

        if self.family != "npu":
            self._applied["eager_api"] = "skipped (non-npu)"
            return self._applied

        if not self.eager:
            self._applied["eager_api"] = "not_requested"
            return self._applied

        # 关图模式：torch_npu 的 API 名在版本间变过，逐个试
        for spec in (
            "torch.npu.set_compile_mode",
            "torch_npu.npu.set_compile_mode",
            "torch_npu.utils.set_compile_mode",
        ):
            r = self._try_call(spec, jit_compile=False)
            if r == "ok":
                self._applied["eager_api"] = spec
                log.info("已关闭图模式: %s(jit_compile=False)", spec)
                return self._applied
        for spec in ("torch.npu.set_graph_mode", "torch_npu.npu.set_graph_mode"):
            r = self._try_call(spec, enable=False)
            if r == "ok":
                self._applied["eager_api"] = spec
                log.info("已关闭图模式: %s(enable=False)", spec)
                return self._applied

        self._applied["eager_api"] = "FAILED: 未找到可用的关图模式 API"
        log.warning("请求了 eager 但没找到可用的 torch_npu API —— "
                    "310P 上可能仍然跑在图模式，请上机核对")
        return self._applied

    # ------------------------------------------------------------ 报告

    def report(self) -> Dict[str, Any]:
        d = dict(self._applied) if self._applied else self.apply()
        d["requested_dtype"] = self.requested_dtype
        d["requested_eager"] = self.requested_eager
        return d

    def describe(self) -> str:
        r = self.report()
        s = f"{self.device} ({self.device_name}) dtype={self.dtype} eager={self.eager}"
        if self.forced_reason:
            s += f"  [{self.forced_reason}]"
        if self.family == "npu":
            s += f"  eager_api={r.get('eager_api')}"
        return s


_POLICY: Optional[DevicePolicy] = None


def get_policy(device: Optional[str] = None) -> DevicePolicy:
    """进程级单例。"""
    global _POLICY
    if _POLICY is None or (device and device != _POLICY.device):
        _POLICY = DevicePolicy(device or "cpu")
    return _POLICY
