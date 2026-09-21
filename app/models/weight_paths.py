#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weight_paths.py — 权重来源的「显式配置 vs 自动查找」仲裁

环境变量（`MOGE_WEIGHTS` / `SAM_WEIGHTS` / `YOLO_WEIGHTS`）代表**显式配置**，
优先级高于在 `WEIGHTS_DIR` 里启发式查找 —— 否则运维明明设了也会被
「找到了别的 .pt」悄悄盖掉。

实测踩到过的两个方向都会出事，所以两边都要防：

1. **查找器赢过显式配置**（已被 env 优先修掉）：
   权重目录位于含 `moge` 的路径下时，该目录里所有 `.pt` 都成了 MoGe 候选，
   按路径长度排序会选中 `yolo-cylinder-seg.pt`。

2. **显式配置失效后硬顶**（本模块负责）：
   权重被移动/改名/删除后环境变量还指着旧路径。此时不能硬顶，否则报错与
   真实原因无关 —— MoGe 的 `from_pretrained` 会把不存在的路径当成 HF repo id
   去联网下载，抛 `HFValidationError: Repo id must be in the form ...`；
   ultralytics 的 `YOLO(路径)` 也会尝试按名字联网拉取。

所以规则是：**看起来是绝对路径但不存在**的值一律忽略并告警，交回查找器兜底；
其余值（存在的路径、或 HF repo id 这种非路径形式）按显式配置直接用。
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

log = logging.getLogger(__name__)


def resolve_weight(env_var: str, finder: Callable[[], Optional[str]]) -> Optional[str]:
    """
    决定该用哪个权重来源。

    :param env_var: 环境变量名，如 "MOGE_WEIGHTS"
    :param finder:  无参兜底查找函数，返回路径或 None
    :return: 环境变量值，或 finder() 的结果
    """
    val = os.environ.get(env_var)
    if val:
        if val.startswith(os.sep) and not os.path.exists(val):
            log.warning("%s=%s 不存在（配置已失效？），回退到自动查找", env_var, val)
        else:
            return val
    return finder()
