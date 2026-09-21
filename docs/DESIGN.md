# 设计契约（Design Contract）

> 本文最初是「Python 应用侧」与「Docker 打包侧」之间的接口契约，用于并行开发时
> 约束双方不改动对方文件。现在保留下来作为**架构与接口的设计文档**。
> 下文所有路径均相对仓库根目录。



---

## 1. 目录布局（固定，不要改动结构）

```

├── Dockerfile                 # [Docker 侧] 多阶段；目标: cpu(本地校验) / ascend(交付)
├── docker-compose.yml         # [Docker 侧]
├── .dockerignore              # [Docker 侧]
├── scripts/
│   ├── build_local_cpu.sh     # [Docker 侧] 本地可行性校验构建
│   ├── build_ascend.sh        # [Docker 侧] 昇腾机上的正式构建
│   ├── run_ascend.sh          # [Docker 侧] docker run，含全部 --device 挂载
│   └── preflight_ascend.sh    # [Docker 侧] 上机自检(npu-smi / 驱动 / 设备节点)
├── docs/
│   └── ASCEND.md              # [Docker 侧] 昇腾部署说明
├── app/                       # [Python 侧] —— Docker 侧只可 COPY，不可改内容
│   ├── model_hub.py           # ModelHub：懒加载 + 单模型驻留 + flag 派发
│   ├── service.py             # FastAPI 服务
│   ├── models/
│   │   ├── __init__.py
│   │   ├── moge_geom.py       # MoGe-2 单目几何 + 重力世界系
│   │   ├── sam_seg.py         # SAM 框提示分割
│   │   └── yolo_seg.py        # YOLO-seg 检测+分割（替代 SAM 的轻量路径）
│   └── requirements.txt       # [Python 侧]
└── weights/                   # 模型权重（可被 .dockerignore 排除，用挂载或构建时拉取）
```

---

## 2. Dockerfile 契约（Docker 侧必须满足）

### 2.1 阶段划分

必须提供**两个构建目标（target）**：

| target | 基础镜像 | 用途 | 体积要求 |
|---|---|---|---|
| `cpu` | `python:3.10-slim` 之类轻量镜像 | **本地可行性校验**：只装 app 的最小依赖，跑一个 import 冒烟测试 | 必须 < 1.5 GB |
| `ascend` | 昇腾 CANN + PyTorch/torch_npu 官方镜像 | 正式交付 | 不限 |

`cpu` 目标存在的唯一原因是：**本地构建机只有 ~4.6 GB 可用磁盘**，昇腾基础镜像 20 GB+ 装不下。
所以 `cpu` 目标必须能在 4.6 GB 内完成构建，用来验证 Dockerfile 语法、COPY 路径、依赖清单、入口命令是否正确。

`ascend` 目标必须：
- 参考 MinerU 的昇腾适配方式（<https://github.com/opendatalab/MinerU>，文件 `docker/china/npu.Dockerfile`，
  文档 <https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/Ascend/>）
- 安装 **torch_npu**（注意：**不需要 vllm/lmdeploy**，MinerU 用它们是因为跑 VLM，本服务只有 CNN/ViT 类小模型）
- 在构建时把权重放进镜像（用户要求「包括模型」），同时**允许运行时用挂载覆盖**
- 设置环境变量：`ASCEND_RT_VISIBLE_DEVICES`、`WEIGHTS_DIR=/app/weights`、`PYTHONUNBUFFERED=1`

### 2.2 入口契约（不可变）

```dockerfile
WORKDIR /app
# 必须存在: /app/service.py, /app/model_hub.py, /app/models/
EXPOSE 8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
```

- 应用代码在**构建上下文根目录的 `app/`**，必须 `COPY app/ /app/`
- 权重放在 `/app/weights/`
- 环境变量 `WEIGHTS_DIR` 默认 `/app/weights`

### 2.3 .dockerignore

必须排除：`weights/*.pt`、`weights/*.pth`、`*.tar`、`.git`、`__pycache__`、`*.pyc`、`docs/`。
（权重通过构建阶段单独拉取或挂载，避免把 2 GB 权重塞进构建上下文。）

---

## 3. 昇腾运行时契约（Docker 侧必须满足）

`scripts/run_ascend.sh` 必须包含 MinerU 那套设备挂载。至少：

```bash
docker run -u root --name cylinder-geom --privileged=true \
  --ipc=host --network=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /var/log/npu/:/usr/slog \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e WEIGHTS_DIR=/app/weights \
  -p 8000:8000 \
  cylinder-geom:ascend
```

要点：
- 设备号与 `ASCEND_RT_VISIBLE_DEVICES` 都要**可参数化**（脚本用环境变量/参数传入，不要写死）
- 如果用户机器是 Atlas 300I Duo（310P），要提示追加 `--enforce-eager --dtype float16`（310P 不支持图模式与 bf16）
- `preflight_ascend.sh` 要检查：`npu-smi info` 是否可用、`/dev/davinci*` 是否存在、驱动版本、
  `/usr/local/Ascend/driver` 是否可读，并给出明确的失败提示

---

## 4. Python 应用契约（Python 侧必须满足，Docker 侧依赖它）

### 4.1 ModelHub

`app/model_hub.py` 提供：

```python
class ModelHub:
    def __init__(self, device: str | None = None,
                 weights_dir: str | None = None,
                 max_resident: int = 1): ...
    def available(self) -> dict[str, str]: ...      # task -> 描述
    def forward(self, task: str, **kwargs): ...      # ← 通过 flag 派发
    def unload(self, task: str | None = None) -> None: ...
    def stats(self) -> dict: ...
```

- `task` 取值：`"detect"` | `"segment"` | `"geometry"`（后续可扩展）
- **核心约束（用户明确要求）**：昇腾上显存有限，**默认同一时刻只驻留一个模型**。
  `forward()` 在需要时懒加载目标模型，并**驱逐**其他常驻模型（释放显存）。
- 必须**线程安全**（一把锁），因为 FastAPI 会并发调用。
- 设备自动探测：能 `import torch_npu` 则用 `npu`，否则 `cuda`，否则 `cpu`。

### 4.2 HTTP 接口

`app/service.py` 暴露（FastAPI）：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 存活 + 当前设备 + 已驻留模型 |
| GET | `/v1/tasks` | 可用 task 列表（来自 `ModelHub.available()`）|
| POST | `/v1/infer` | 主入口，body 里 `{"task": "...", ...}`，**按 task 派发** |
| POST | `/v1/unload` | 卸载常驻模型，释放显存 |

`/v1/infer` 请求体（每个 task 的字段）：
- `detect`：`{"task":"detect","image_b64":"..."}` → 返回框 + 类别 + 置信度
- `segment`：`{"task":"segment","image_b64":"...","boxes":[[x0,y0,x1,y1],...]}` → 返回多边形与面积
- `geometry`：`{"task":"geometry","image_b64":"...","regions":[{"box":[...]} | {"point":[x,y]}]}` → 返回每区域的
  `pitch_deg / center_xyz / center_height_m / radius_m / length_m / visible_ratio`

统一响应包裹：`{"ok":bool,"task":str,"device":str,"result":...,"error":str|null,"latency_ms":int}`

### 4.3 权重与降级

- 权重目录由 `WEIGHTS_DIR` 决定，默认 `/app/weights`
- 找不到权重时：`/v1/tasks` 只列出可用的；调用不可用 task 返回 `ok:false` 且**明确说明缺哪个文件**，不要抛 500
- `geometry` task 依赖 MoGe 权重；`segment` 依赖 SAM；`detect` 依赖 YOLO-seg
- 支持从 `WEIGHTS_DIR` 加载本地权重，也支持 HF repo id 回退（离线环境优先本地）

---

## 5. 本地校验（Docker 侧需执行并报告）

```bash
bash scripts/build_local_cpu.sh       # 必须成功
docker run --rm cylinder-geom:cpu python -c "import service, model_hub; print('import ok')"
```

在 CPU 目标上**不要求模型能真的推理**（权重不在），但要求：
- 镜像能构建成功
- `import service` / `import model_hub` 不报错
- `uvicorn service:app` 能起来，`GET /healthz` 返回 200

把构建输出、镜像大小、healthz 响应粘到 `docs/ASCEND.md` 里作为证据。

---

## 6. 交付时必须在 docs/ASCEND.md 里写清

1. 用了哪个**昇腾基础镜像 tag**，为什么选它，对应哪些硬件（A2 / A3 / 300I Duo）
2. 构建命令、运行命令（含全部设备挂载）
3. 310P 的特殊参数
4. 已知未验证项（我们没有昇腾机器，**所有昇腾相关结论都是未实测的**，必须显式声明）
5. 本地 `cpu` 目标的构建证据
