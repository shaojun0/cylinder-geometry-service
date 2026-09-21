# 气瓶几何量测服务 —— 昇腾（Ascend NPU）部署说明

> 本文是打包侧的交付文档，满足 [DESIGN.md](DESIGN.md) 第 6 节的全部要求。
>
> **⚠ 最重要的一句话（先说清楚）**
> 我们手上**没有任何昇腾机器**（本机 aarch64 Debian 11，`/dev/davinci*`、`npu-smi`、
> `/usr/local/Ascend/driver` 全部不存在 —— 见第 7 节实测).
> 因此本文中**一切与昇腾真机相关的结论都是「未实测的推断」**：
> 基础镜像可用性、torch_npu 兼容性、设备挂载是否充分、CANN/驱动版本匹配、性能与显存，
> **全部需要上机验证**。本文用 ✅ 表示「本地已实测」，用 ❓ 表示「未验证/推断」。

---

## 1. 交付物与用法速查

```
ascend/
├── Dockerfile              # 多阶段：cpu(本地校验) / weights(构建期抓权重) / ascend(交付)
├── docker-compose.yml      # 昇腾 + cpu 两条编排（设备挂载 / 选卡全参数化）
├── .dockerignore           # 排除权重与缓存，构建上下文仅 ~124 kB
├── scripts/
│   ├── build_local_cpu.sh  # 本地可行性校验构建 + import 冒烟 + /healthz 验证
│   ├── build_ascend.sh     # 昇腾机正式构建（按型号选基础镜像 / 构建期 bake 权重）
│   ├── run_ascend.sh       # docker run，含 设计契约 §3 要求的全部 --device / -v 挂载
│   └── preflight_ascend.sh # 上机自检（npu-smi / 驱动 / 设备节点 / 磁盘）
└── docs/ASCEND.md          # 本文
```

最短路径：

```bash
# ── 本地（x86/arm 通用，不需要 NPU）✅ 已实测 ───────────────────────────────
bash scripts/build_local_cpu.sh                 # 构建 + import 冒烟 + curl /healthz
bash scripts/build_local_cpu.sh --ascend-lint   # 额外校验 ascend 段落指令

# ── 昇腾机器上 ❓ 未实测 ────────────────────────────────────────────────────
bash scripts/preflight_ascend.sh                # 先自检
bash scripts/build_ascend.sh --chip 910b        # A2；A3 用 --chip a3；300I Duo 用 --chip 310p
bash scripts/run_ascend.sh --devices 0          # 起服务
curl -fsS http://127.0.0.1:8000/healthz

# ── 或者用 compose ❓ 未实测 ───────────────────────────────────────────────
docker compose up -d --build                    # 昇腾服务
docker compose --profile cpu up --build cylinder-geom-cpu   # 本地校验（✅ 语法已校验）
```

---

## 2. 基础镜像选型（设计契约 §6.1）

### 2.1 选定的 tag ✅ 已核验存在

| 硬件型号 | 基础镜像 tag | `--chip` |
|---|---|---|
| **Atlas A2 系列**（800T A2 / 800I A2 / 900 A2 PoD / 200T A2 Box16 / 300T A2） | `quay.io/ascend/torch-npu:2.7.1.post4-910b-ubuntu22.04-py3.11` **（默认）** | `910b` / `a2` |
| **Atlas A3 系列**（800T A3 / 800I A3 / 900 A3 SuperPoD / 9000 A3 SuperPoD） | `quay.io/ascend/torch-npu:2.7.1.post4-a3-ubuntu22.04-py3.11` | `a3` |
| **Atlas 300I Duo（310P 推理卡）** | `quay.io/ascend/torch-npu:2.7.1.post4-310p-ubuntu22.04-py3.11` | `310p` / `300i` |

型号差别只在 tag 里的芯片代号，三个 tag 打包的软件栈版本一致，所以换硬件 = 换一个 tag。

**国内镜像（已实测与 quay 同 digest）**：

```bash
swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:2.7.1.post4-910b-ubuntu22.04-py3.11
# 实测 arm64 manifest digest 与 quay 完全一致：
#   quay  arm64 digest  sha256:eb6a005bcd54de586ea78780ce5406eb92663d1a0ce8c27ca4085855c27b31b3
#   swr   arm64 digest  sha256:eb6a005bcd54de586ea78780ce5406eb92663d1a0ce8c27ca4085855c27b31b3  ✅
```

反例（避免踩坑）：`quay.m.daocloud.io/ascend/torch-npu:<tag>` **不可用** ——
实测返回 `denied: Read Only`。DaoCloud 只镜像了 `vllm-ascend`（MinerU 用的那个），没有 `torch-npu`。

### 2.2 为什么是 `torch-npu` 而不是 MinerU 的 `ascend-vllm`

用户明确要求参考 MinerU 的昇腾适配（[`docker/china/npu.Dockerfile`](https://github.com/opendatalab/MinerU)、
[官方文档](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/Ascend/)）。
MinerU 的基础镜像是 `quay.m.daocloud.io/ascend/vllm-ascend:v0.11.0`（A2）/ `v0.11.0-a3`（A3）/
`v0.10.0rc1-310p`（300I Duo）。**我们不用它**，原因：

| | MinerU | 本服务 |
|---|---|---|
| 模型 | VLM 大语言模型（需要 PagedAttention / 连续批处理） | ViT/CNN 小模型（MoGe ~331M、SAM ~94M、YOLO-seg ~3M） |
| 推理栈 | `ascend-vllm` 或 `ascend-lmdeploy` | 只要 `torch` + `torch_npu`（普通 `model.forward()`） |
| 基础镜像 | `ascend-vllm:v0.11.0` | **`ascend/torch-npu:2.7.1.post4-*`** |

[DESIGN.md](DESIGN.md) 第 2.1 节也明确写了「**不需要 vllm/lmdeploy**」。
选 `torch-npu` 的额外好处：镜像里已经装好**同一套** `torch` 与 `torch_npu`，
不用自己 pip 装 `torch_npu`（那是昇腾适配里最容易翻车的一步：torch 与 torch_npu 版本必须严格配对，
装错直接 NPU 不可用）。

> 关于 `ascend-pytorch` 系列：`quay.io/ascend/pytorch` 这个仓库**存在**（有 README），
> 但实测其 tag 列表为**空**（`/api/v1/repository/ascend/pytorch/tag/` 返回 `{"tags": []}`）。
> 华为 AscendHub 门户（`hiascend.com/developer/ascendhub`）里的 `ascend-pytorch` 镜像需要登录才能看 tag，
> 我们**无法在无人值守环境下核验**，因此标注为 ❓ 未验证，不作为交付基础镜像。
> 最终选择的 `quay.io/ascend/torch-npu` 是同一批官方仓库里**确实可匿名拉取**的 PyTorch 适配镜像。

### 2.3 镜像里到底装了什么 ✅ 从 registry config blob 实测读出

以 `2.7.1.post4-910b-ubuntu22.04-py3.11`（arm64）为例：

| 项 | 值 | 依据 |
|---|---|---|
| OS | Ubuntu 22.04 | 镜像 label `org.opencontainers.image.version=22.04` |
| CANN | **9.0.0**，装于 `/usr/local/Ascend/cann-9.0.0` | 镜像 `ENV ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` |
| Python | **3.11.15**，装于 `/usr/local/python3.11.15` | 镜像 `ENV PATH` |
| torch | **2.7.1**（`+cpu` wheel，来自 `download.pytorch.org/whl/cpu`） | 镜像 build history 的 `wget torch-2.7.1%2Bcpu...` |
| torch_npu | **2.7.1.post4**（源码 tag `v26.0.0-pytorch2.7.1`） | build history 的 `TORCH_NPU_PATCH_TAG=2.7.1.post4` |
| 额外组件 | ATB / NNAL（`/usr/local/Ascend/nnal/atb`） | `ENV ATB_HOME_PATH` |
| 平台 | 多架构 manifest：**arm64 + amd64** ✅ | `docker manifest inspect` |
| 体积 | **压缩 4.62 GB**（arm64 层大小之和，实测）；展开后 ❓ 预计 15–20 GB+（设计契约说的「20 GB+」，本地无空间实测） | registry manifest layers |

镜像自带 `ENTRYPOINT ["/bin/bash","-c","source /usr/local/Ascend/ascend-toolkit/set_env.sh && ... && exec \"$@\"","--"]`，
即容器启动时会自动 source CANN 环境再执行我们的 `CMD`，所以本 Dockerfile 的
`CMD ["uvicorn", ...]`（设计契约 §2.2 入口契约）可以直接工作 ✅（在替代基础镜像上验证了继承行为，
真机上的 source 步骤 ❓ 未验证）。

**更激进/更保守的备选 tag**（同样已核验存在，供驱动版本不匹配时切换）：
`2.12.0.post2-cann9.1.0-910b-ubuntu22.04-py3.12`、`2.10.0.post6-cann9.1.0-910b-ubuntu22.04-py3.12`、
`2.9.0.post8-cann9.1.0-310p-ubuntu22.04-py3.12` 等。tag 里显式带 CANN 版本，便于对齐驱动版本。
CANN 越新通常要求驱动越新 —— 驱动兼容性见 §7。

---

## 3. 构建（设计契约 §6.2）

### 3.1 `cpu` 目标（本地可行性校验）✅ 已实测成功

```bash
bash scripts/build_local_cpu.sh
# 等价于：
# docker build --file Dockerfile --target cpu --tag cylinder-geom:cpu \
#   --build-arg PYTHON_VERSION=3.11 --build-arg SMOKE_TEST=1 --network host .
```

`cpu` 目标存在的原因就是本机磁盘只有 ~4.6 GB（`/userdata/docker` 90% 已用），
装不下 4.62 GB 压缩 / 20 GB+ 展开的昇腾基础镜像。
它验证的是：**Dockerfile 语法、COPY 路径、依赖清单、入口命令、健康检查**。
**它不做真实推理**（不装 torch，不装权重），见 §6.1 的能力边界。

- 基础镜像：`python:3.11-slim`（刻意对齐昇腾镜像的 Python 3.11.15，避免「cpu 能跑、昇腾不能跑」的版本差异）
- 依赖：`app/requirements.txt` 里纯 Python / 有 aarch64 wheel 的那部分
  （`fastapi` `uvicorn[standard]` `pydantic` `python-multipart` `numpy` `pillow` `opencv-python-headless`）
- 不装：`torch` / `torch_npu` / `transformers` / `ultralytics` / `moge`
  （app 侧所有重型 import 都是延迟的，`import model_hub` / `import service` 不需要它们）

### 3.2 `ascend` 目标（正式交付）❓ 未实测

```bash
bash scripts/build_ascend.sh --chip 910b
# 等价于（可由 --dry-run 打印，✅ 已实测 dry-run 输出正确）：
docker build --network host -f Dockerfile --target ascend -t cylinder-geom:ascend \
  --build-arg ASCEND_BASE_IMAGE=quay.io/ascend/torch-npu:2.7.1.post4-910b-ubuntu22.04-py3.11 \
  --build-arg FETCH_WEIGHTS=1 --build-arg STRICT_WEIGHTS=0 ...
```

参考 MinerU 的做法：同样用官方昇腾基础镜像按硬件改 tag，同样 `--network=host` 构建。

**磁盘要求：预留 ≥ 45 GB**（`build_ascend.sh` 会检查，`preflight_ascend.sh` 也会查）。
构成估算：基础镜像展开 20 GB+ + 权重 1.7 GB + pip 依赖（transformers/ultralytics/…）1.5 GB + 构建缓存。

常用选项：

| 需求 | 命令 |
|---|---|
| 权重不进镜像，运行时挂载 | `--no-weights` |
| 要求必须抓到权重，否则构建失败 | `--strict-weights` |
| 国内 HF 镜像 | `--hf-endpoint https://hf-mirror.com` |
| 内网 PyPI | `--pip-index-url https://pypi.tuna.tsinghua.edu.cn/simple` |
| 内网/私有基础镜像 | `--base-image <内网仓库>/ascend-torch-npu:<tag>` |
| 只打印命令不构建 | `--dry-run` |
| 构建后立刻验证 NPU | `--verify --npu-devices 0` |

### 3.3 权重怎么进镜像（三种方式，按需选）

[DESIGN.md](DESIGN.md) 2.1 要求「构建时把权重放进镜像，同时允许运行时用挂载覆盖」。
`.dockerignore` 排除了 `weights/*.pt` 等大文件（设计契约 §2.3），所以权重**不走构建上下文**，
而是在 Dockerfile 的 `weights` 阶段（`debian:bullseye-slim`，不是那个 20 GB 的基础镜像）构建期拉取：

| 方式 | 做法 | 说明 |
|---|---|---|
| **A. HF 拉取（默认）** | `FETCH_WEIGHTS=1`（默认），从 `Ruicheng/moge-2-vitl-normal` 与 `facebook/sam-vit-base` 下载 | MoGe `model.pt` 1.3 GB + SAM 358 MB ❓ 未实测 |
| **B. 直链 / tarball** | `--yolo-url <URL>`、`--weights-tarball <tar.gz>` | YOLO-seg 是训练产出，通常没有公开直链 |
| **C. 运行时挂载（覆盖镜像内容）** | `run_ascend.sh --weights /data/cyl/weights` 或 compose 里打开 `WEIGHTS_HOST_DIR` | 挂载优先，镜像里的权重被盖掉 |

镜像内的权重布局（对齐 `app/models/moge_geom.py::_find_moge_weight` 与
`app/models/sam_seg.py::_find_sam_weight`，两种布局都兼容，这里用扁平布局）：

```
/app/weights/
├── moge-2-vitl-normal.pt     # ⚠ 必须是 .pt **文件路径**：
│                             #   MoGe 的 from_pretrained 是 `if Path(name).exists(): torch.load(name)`，
│                             #   传目录会直接 torch.load(目录) 崩掉
├── sam-vit-base/             # ⚠ 必须是**目录**（transformers 要 config.json + 权重）
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
└── yolo11n-seg.pt
```

权重总量约 **1.7 GB**，构建期会 `chmod -R a+rX`，日志里会打印文件清单与大小。
`STRICT_WEIGHTS=1` 时若一个文件都没抓到，构建直接失败（正式交付建议开）。

> 为什么 `moge` 用 `pip install --no-deps` 而不是完整装：MoGe 上游 `main` 现在是 v3.0.0，
> `pyproject.toml` 声明了 `flex-gemm`（CUDA 专用）、`pipeline`、`gradio` 等依赖，
> 在昇腾上装 CUDA 依赖既无用又可能污染环境。本服务只要 `moge.model.v2`。
> **但有个坑**：`moge/model/v2.py` 在**模块顶层**就 `import utils3d_moge`（回退 `import utils3d`），
> 所以 `--no-deps` 必须**补装** `utils3d_moge`
> （钉 MoGe `pyproject.toml` 里的同一 commit `62f09d5...`）。
> Dockerfile 里用 `MOGE_EXTRA_PIP_SPECS` 表达，并在构建期做
> `from moge.model.v2 import MoGeModel` 的 import 校验（纯 CPU 可跑），把这类问题拦在构建期。
> ❓ 上述 commit/依赖关系是**静态分析 + 上游文件核对**得出的，未在昇腾机器上实跑。

#### 3.3.1 一个必须防的坑：装模型侧依赖时 `torch` 被偷偷升级

`torch_npu` 与 `torch` 是**严格配对**的（镜像里是 `torch 2.7.1` + `torch_npu 2.7.1.post4`）。
但 `ultralytics` 依赖 `torchvision`，而新版 `torchvision` 会 pin 一个更新的 `torch` ——
如果直接 `pip install ultralytics`，pip 的依赖解析**有可能把 torch 升级掉**，
结果是镜像能 `import torch`，但 `torch.npu.is_available()` 永远是 False。

所以 ascend 阶段做了两件事：

1. 先读出现有 torch 版本，写成 pip 的 **constraint 文件**再安装：
   `printf 'torch==%s\n' "${torch_before%%+*}" > /tmp/torch-constraint.txt` →
   `pip install -c /tmp/torch-constraint.txt transformers ultralytics`
2. 装完**再读一次** torch 版本比对，一旦变了就**让构建失败**并提示改版本，
   而不是留一个「能 import 但 NPU 用不了」的镜像。

这段逻辑用假 `torch` / 假 `pip` 在本地验证过三种分支 ✅：
跳过（`INSTALL_MODEL_PIP=0`）、版本未变（exit 0）、版本被改动（exit 1 + 明确报错）。
❓ 真机上的实际依赖解析结果（哪个 torchvision 版本会被选中）仍未验证。

---

## 4. 运行（设计契约 §6.2 / 6.3 / 3）

### 4.1 完整运行命令

```bash
bash scripts/run_ascend.sh --devices 0
```

等价于（✅ dry-run 输出已实测；真机执行 ❓ 未验证）：

```bash
docker run --name cylinder-geom -u root -d \
  --privileged=true --ipc=host --network host \
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
  -e PYTHONUNBUFFERED=1 \
  -e LOG_LEVEL=INFO \
  -e MAX_RESIDENT_MODELS=1 \
  cylinder-geom:ascend
```

设备号与 `ASCEND_RT_VISIBLE_DEVICES` **都参数化**，没有写死：

```bash
# 换卡
bash scripts/run_ascend.sh --devices 3 --rt-visible-devices 3
# 多卡（挂 2,3 两张，容器内可见 0,1 两张）
bash scripts/run_ascend.sh --devices 2,3 --rt-visible-devices 0,1
# 权重用宿主机目录覆盖镜像里的
bash scripts/run_ascend.sh --weights /data/cylinder/weights --hf-cache /data/hf-cache
# bridge 网络 + 端口映射（默认 host 网络下 -p 无意义，脚本会自动不加）
bash scripts/run_ascend.sh --network bridge --port 8000
```

> **`-p` 的说明**：设计契约 §3 的样例同时给了 `--network=host` 和 `-p 8000:8000`。
> host 网络下 `-p` 是被忽略的（docker 会警告）。脚本的做法是：
> host 模式不加 `-p`（容器直接占宿主 8000），需要端口映射时用 `--network bridge --port N`。

### 4.2 310P（Atlas 300I Duo）特殊参数（设计契约 §6.3）

**310P 不支持 bf16，也不支持图模式。** 这是硬件限制，不是软件配置问题。

MinerU 的对应做法是在 vllm 命令行追加 `--enforce-eager --dtype float16`
（见 [MinerU Ascend 文档](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/Ascend/) 的注释）。

**⚠ 本服务没有 vllm，所以不存在等价的 CLI flag** —— 这是与 MinerU 的**关键差异**：

| MinerU（vllm 后端） | 本服务（uvicorn + torch_npu 裸推理） |
|---|---|
| `mineru-openai-server --enforce-eager --dtype float16` | 没有对应 flag，`uvicorn` 只管 HTTP |
| 由 vllm 引擎读这两个参数 | **控制点在模型加载处**（`app/models/*.py`） |

因此我们把要求**降级为提示 + 环境变量约定**：

```bash
bash scripts/run_ascend.sh --310p        # 打印强制提示，并额外传：
#   -e CYLINDER_DTYPE=float16
#   -e CYLINDER_EAGER=1
```

并且必须在**模型侧**落实（❓ 当前 `app/` 代码**并未读取**这两个变量，属未验证项）：

1. **精度**：权重与推理用 `float16`；不要 `bfloat16`，不要 `autocast(bfloat16)`。
2. **图模式**：走 eager；不要 `torch.compile` / 图模式 / `jit_compile`。

`build_ascend.sh --chip 310p` 也会在构建前打印同一段提示。
`docker-compose.yml` 里对应 `CYLINDER_DTYPE` / `CYLINDER_EAGER` 两个环境变量。

### 4.3 上机自检（`preflight_ascend.sh`）

```bash
bash scripts/preflight_ascend.sh                 # 退出码 1 = 有 FAIL
bash scripts/preflight_ascend.sh --report-only    # 只出报告，永远 exit 0
bash scripts/preflight_ascend.sh --image cylinder-geom:ascend
```

检查项：架构、docker 可用性与版本、docker root 剩余空间、`npu-smi info` 可执行性与输出、
驱动 `version.info`、`/usr/local/Ascend/driver`（含 `lib64`）可读、`/usr/local/dcmi`、
`/dev/davinci*` 与 `/dev/davinci_manager` `/dev/devmm_svm` `/dev/hisi_hdc`、
`/var/log/npu`、davinci 内核模块，每项都有 PASS/WARN/FAIL 与明确的补救提示。

---

## 5. 本地 `cpu` 目标构建证据（设计契约 §6.5）✅ 全部实测

### 5.1 构建环境（实测）

```
架构    : aarch64 (Kunpeng 系，Linux 5.10.160)
系统    : Debian GNU/Linux 11 (bullseye)
Docker  : 20.10.5+dfsg1  (Experimental: false -> 用的经典 builder，非 BuildKit)
Python  : 3.9.2 (宿主)
Docker Root Dir : /userdata/docker   -> 可用 4.6 GB（构建前）/ 4.4 GB（构建后）
```

### 5.2 构建命令与结果 ✅

```console
$ bash scripts/build_local_cpu.sh --tag cylinder-geom:cpu --port 18000
[build_local_cpu] 构建上下文: <repo>
[ OK ] app/ 关键文件齐全
[build_local_cpu] Docker Root Dir: /userdata/docker（可用 4 GB）
[build_local_cpu] 命令: docker build --file .../Dockerfile --target cpu --tag cylinder-geom:cpu \
                   --build-arg PYTHON_VERSION=3.11 --build-arg SMOKE_TEST=1 --network host ...
----------------------------------------------------------------------
Sending build context to Docker daemon  125.4kB       <-- .dockerignore 生效（权重/缓存未进上下文）
Step 1/16 : ARG PYTHON_VERSION=3.11
Step 2/16 : ARG WEIGHTS_BASE_IMAGE=python:3.11-slim
Step 3/16 : ARG ASCEND_BASE_IMAGE=quay.io/ascend/torch-npu:2.7.1.post4-910b-ubuntu22.04-py3.11
Step 4/16 : FROM python:${PYTHON_VERSION}-slim AS cpu
Step 8/16 : RUN apt-get install -y --no-install-recommends libglib2.0-0 libgomp1
Step 10/16 : RUN python -m pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" ...
Successfully installed annotated-doc-0.0.5 annotated-types-0.8.0 anyio-4.15.1 click-8.5.0
  fastapi-0.141.1 h11-0.16.0 httptools-0.8.0 idna-3.20 numpy-2.4.6
  opencv-python-headless-5.0.0.93 pillow-12.3.0 pydantic-2.13.5 pydantic-core-2.46.5
  python-dotenv-1.2.3 python-multipart-0.0.32 pyyaml-6.0.3 starlette-1.6.0
  typing-extensions-4.16.0 typing-inspection-0.4.4 uvicorn-0.53.0 uvloop-0.22.1
  watchfiles-1.2.0 websockets-17.1
Step 11/16 : COPY app/ /app/
Step 13/16 : RUN ... python -c "import model_hub, service; ..."
[cpu] smoke test OK: import model_hub, service
Step 16/16 : CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
Successfully built 78de406f2686
Successfully tagged cylinder-geom:cpu
----------------------------------------------------------------------
[ OK ] cpu 目标构建成功
[ OK ] 满足 设计契约 §2.1 的体积要求（< 1.5 GB）
```

> 这是**首次全量构建**的输出（首次慢在下载 wheel，约 5–6 分钟）。
> 用同样的 build-arg 复跑会命中 layer cache（最后一次复跑 16 步里有 12 步 `Using cache`，秒级完成）
> 并产出**同一个镜像 ID** `78de406f2686` ✅ —— 构建是确定性可复现的。

### 5.3 镜像体积 ✅

```console
$ docker inspect -f '{{.Size}}' cylinder-geom:cpu
386719470                      # 368.8 MB
$ docker images cylinder-geom:cpu
REPOSITORY        TAG    IMAGE ID       CREATED         SIZE
cylinder-geom     cpu    78de406f2686   ...             387MB
```

**368.8 MB < 设计契约 §2.1 要求的 1.5 GB** ✅（约为上限的 1/4）。

镜像配置（实测）符合 设计契约 §2.2 入口契约：

```
WorkDir   = /app
ExposedPorts = 8000/tcp
Cmd       = [uvicorn service:app --host 0.0.0.0 --port 8000]
Healthcheck = CMD python -c "import urllib.request,sys; sys.exit(0 if ... /healthz ...)"
```

### 5.4 设计契约 §5 冒烟 ✅

```console
$ docker run --rm cylinder-geom:cpu python -c "import service, model_hub; print('import ok')"
import ok
```

### 5.5 `GET /healthz` 实测响应 ✅

容器真起起来（`docker run -d -p 127.0.0.1:18000:8000 cylinder-geom:cpu`）后：

```console
$ curl -fsS http://127.0.0.1:18000/healthz
{"ok":true,"status":"healthy","device":"cpu","weights_dir":"/app/weights","max_resident":1,"resident":[],"calls":{},"mem":{}}
```

HTTP 200，`status=healthy`，`device=cpu`（cpu 镜像里 `MODEL_DEVICE=cpu`），
`resident=[]`（还没有加载任何模型 —— 符合「懒加载 + 单模型驻留」的设计）。
镜像内置 `HEALTHCHECK` 也已生效（验证时状态为 `starting`，`start-period` 内属正常）。

> 注：`app/service.py` 与 `app/model_hub.py` 是 Python 侧交付的**真实文件**（非桩），
> 本次构建/冒烟/healthz 都是打在真实代码上的。

### 5.6 `ascend` 阶段的指令级校验（替代方案，不是真机构建）✅

因为没有昇腾基础镜像（4.62 GB 压缩 / 20 GB+ 展开）也没有 45 GB 磁盘，
`ascend` 目标**无法在本地构建**。为了尽量把 Dockerfile 的问题提前暴露，
`build_local_cpu.sh --ascend-lint` 会**用一个很小的镜像顶替昇腾基础镜像**，
把 `ascend` 阶段的指令真跑一遍：

```bash
# 等价于：
docker build --target ascend -t cylinder-geom:ascend-lint \
  --build-arg ASCEND_BASE_IMAGE=python:3.11-slim \
  --build-arg FETCH_WEIGHTS=0 --build-arg INSTALL_MOGE=0 \
  --build-arg INSTALL_MODEL_PIP=0 --build-arg SMOKE_TEST=1 .
```

它验证了：`ascend` 阶段的 apt 依赖、`app/requirements.txt` 的 torch 过滤逻辑、
`COPY app/`、`COPY --from=weights /out/weights/ /app/weights/`、
`compileall` 语法检查、`import model_hub, service` 冒烟、
以及 `weights` 阶段的 apt 安装。
**它不能证明**：昇腾基础镜像能拉、CANN 环境正确、torch_npu 可用、NPU 上能推理。

#### 5.6.1 第一次尝试失败：暴露出一个真实的可移植性问题 ❌ → ✅

第一次跑 `--ascend-lint` 时，**`weights` 阶段就失败了**（不是 ascend 阶段）：

```console
Step 16/62 : FROM debian:bullseye-slim AS weights
Step 19/62 : RUN apt-get install -y --no-install-recommends ca-certificates curl python3 python3-pip
E: Failed to fetch http://deb.debian.org/debian-security/pool/updates/main/s/setuptools/
   python3-setuptools_52.0.0-4+deb11u2_all.deb  404  Not Found
E: Failed to fetch http://deb.debian.org/debian-security/pool/updates/main/p/python-pip/
   python3-pip_20.3.4-4+deb11u2_all.deb  404  Not Found
[FAIL] ascend 阶段指令级校验失败 —— 说明 Dockerfile 的 ascend 段落有问题，必须修。
```

原因：**`debian:bullseye-slim` 已经 EOL**，上面那两个包已从 apt 源移走；
而且 bullseye 自带的 `pip 20.3.4` 太老，也装不动当前版本的 `huggingface_hub`。
已改为 `python:3.11-slim`（Debian 13 trixie，自带 Python 3.11.16 + pip 24.0 + ca-certificates），
weights 阶段只需要再 `apt-get install curl` 一个包。
这正是 `--ascend-lint` 的价值：**把只在构建期才会炸的问题在本地提前暴露出来**。

#### 5.6.2 第二次尝试 ✅ 通过

```console
$ bash scripts/build_local_cpu.sh --ascend-lint --no-smoke --no-health
[build_local_cpu] ascend 阶段指令级校验：用 python:3.11-slim 顶替昇腾基础镜像
[WARN] 这只验证 Dockerfile 语法/路径/指令，**不能**证明昇腾可用性（我们没有昇腾机器）。
Step 17/63 : FROM ${WEIGHTS_BASE_IMAGE} AS weights          <-- 全局 ARG 替换生效
Step 31/63 : RUN apt-get install -y --no-install-recommends curl
Step 33/63 : RUN ... WEIGHTS_TARBALL_URL 判断 ...
[weights] 未设置 WEIGHTS_TARBALL_URL，跳过
Step 34/63 : RUN ... YOLO_WEIGHTS_URL 判断 ...
[weights] 未设置 YOLO_WEIGHTS_URL（训练产出，通常靠挂载），跳过
Step 35/63 : RUN ... FETCH_WEIGHTS 判断 ...
[weights] FETCH_WEIGHTS=0，跳过硬权重下载（改用 -v 挂载 /app/weights）
Step 36/63 : RUN ... chmod / 清单 / STRICT 校验 ...
[weights] 权重文件清单（0 个）:
Step 37/63 : FROM ${ASCEND_BASE_IMAGE} AS ascend            <-- 全局 ARG 替换生效
Step 50/63 : RUN apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 git ca-certificates
Step 52/63 : COPY app/ /app/
Step 53/63 : RUN grep -viE '<torch 过滤正则>' /app/requirements.txt > /tmp/requirements.ascend.txt
[ascend] 过滤 torch* 后的依赖清单:
        fastapi>=0.110 / uvicorn[standard]>=0.27 / pydantic>=2.0 / python-multipart>=0.0.9
        numpy>=1.24 / pillow>=10.0 / opencv-python-headless>=4.8      <-- torch* 已被过滤掉
Step 56/63 : COPY --from=weights /out/weights/ /app/weights/          <-- 跨阶段 COPY 生效
Step 57/63 : RUN chmod -R a+rX /app/weights && ls -la /app/weights
Step 58/63 : RUN python -m compileall -q /app/model_hub.py /app/service.py /app/models
[ascend] 语法检查通过
Step 59/63 : RUN ... 版本清单 + moge/transformers import 校验 ...
[ascend] 已安装版本（MISSING = 未安装）:
     torch            MISSING     <-- 本次 lint 用 --build-arg 把 INSTALL_* 全关，属预期
     torch_npu        MISSING
     transformers     MISSING
     huggingface_hub  MISSING
     ultralytics      MISSING
     moge             MISSING
     utils3d_moge     MISSING
Step 60/63 : RUN python -c "import model_hub, service; ..."
[ascend] smoke test OK: import model_hub, service
Step 63/63 : CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
Successfully built 0242d3e75a74
Successfully tagged cylinder-geom:ascend-lint
[ OK ] ascend 阶段指令级校验通过（镜像 tag cylinder-geom:ascend-lint）
```

校验镜像体积：**473.0 MB**（`cylinder-geom:ascend-lint`，用 `python:3.11-slim` 顶替后的产物）。

> 上面这段是「第二次尝试」的原始输出。之后又为 §3.3.1 的 torch 钉版保护加了一条指令并**重跑了一次**：
> `Step 1–53` 全部 `Using cache`，`Step 54–63` 重跑，`Successfully built 80f8f8140bd5`，同样 exit 0 ✅。
> 也就是说当前 Dockerfile 的这一段处于**最后一次重跑后**的状态。
> 校验完成后 `cylinder-geom:ascend-lint` 镜像已删除以释放磁盘（本机 `/userdata` 只剩几 GB），
> 需要时用 `bash scripts/build_local_cpu.sh --ascend-lint` 随时重建。

**这一段验证了**：`weights` 与 `ascend` 两个阶段的所有 Dockerfile 指令、全局 `ARG` 在
`FROM` 行里的替换、`COPY app/`、`COPY --from=weights`、`requirements.txt` 的 torch 过滤、
`compileall` 语法检查、`import model_hub, service` 冒烟，以及 `--target ascend` 的阶段选择。

**这一段没有验证**：昇腾基础镜像能否拉取/启动、CANN 环境变量是否正常、
镜像里真正的 `torch`/`torch_npu` 版本、权重能否下载、NPU 上能否推理
（即 §7 里的所有 ❓ 项）。

---

## 6. 与 `app/` 的接口约定

| 项 | 值 | 来源 |
|---|---|---|
| 代码位置 | 构建上下文 `app/` → 容器 `/app/` | 设计契约 §2.2 |
| 入口 | `uvicorn service:app --host 0.0.0.0 --port 8000` | 设计契约 §2.2 |
| 权重目录 | `WEIGHTS_DIR`，默认 `/app/weights` | 设计契约 §2.1 |
| 选卡 | `ASCEND_RT_VISIBLE_DEVICES`（默认 `0`，脚本/compose 可覆盖） | 设计契约 §3 |
| 设备探测 | 镜像**不设** `MODEL_DEVICE`，让 `model_hub.detect_device()` 自己探测 `npu → cuda → cpu` | `app/model_hub.py` |
| 单模型驻留 | `MAX_RESIDENT_MODELS`，默认 `1` | `app/model_hub.py` |
| 日志级别 | `LOG_LEVEL`，默认 `INFO` | `app/service.py` |
| torch 归属 | **torch / torch_npu 由基础镜像提供**。`ascend` 阶段会**过滤掉** `app/requirements.txt` 里的 `torch*` 行，避免覆盖镜像里与 CANN 配对的版本 | — |
| 模型侧依赖 | `transformers>=4.40`、`huggingface-hub>=0.23`、`ultralytics>=8.2`（`app/requirements.txt` 里刻意注释掉、由昇腾镜像装） | `app/requirements.txt` |
| MoGe 本体 | `pip install --no-deps <MoGe@commit>` + `utils3d_moge`（见 §3.3 的坑） | `app/models/moge_geom.py` |

### 6.1 `cpu` 目标的能力边界（**重要**）

`cpu` 目标**只做结构校验，不做真实推理**。明确不支持的：

- ❌ 不装 `torch` / `torch_npu` / `transformers` / `ultralytics` / `moge`
- ❌ `/app/weights` 是空目录，`/v1/tasks` 会报告所有 task `weights_found=false`
- ❌ 调 `/v1/infer` 会返回 `ok:false` + `weights_missing`（这正是 `app` 侧设计的结构化降级）

`cpu` 目标能证明的：镜像能构建、依赖清单可解、`import service, model_hub` 通过、
`uvicorn service:app` 能起来、`GET /healthz` 返回 200、入口契约与健康检查配置正确。

#### ✅ 补充：OpenCV 运行期路径已被覆盖（`tests/smoke_container.py`）

原先 `cpu` 冒烟不 import `cv2`，OpenCV 5.x 的兼容性没有证据。现已补一个**容器内真实路径自检**：

```bash
docker run --rm -v "$PWD/tests:/t:ro" -w /app cylinder-geom:cpu python /t/smoke_container.py
```

实测输出（OpenCV **5.0.0**）：

```
=== OpenCV 5.0.0 在真实推理路径上的自检 ===
  [PASS] b64 往返形状/类型 (240, 320, 3)
  [PASS] b64 往返像素完全一致
  [PASS] 支持 data:image/...;base64, 前缀
  [PASS] resize_max_side 生效 (240, 320, 3)->(150, 200, 3) sc=0.625
  [PASS] 超过原尺寸时不放大
  [PASS] cvtColor BGR2RGB 通道正确
  [PASS] findContours+approxPolyDP 得到 1 个多边形
  [PASS] 多边形落在矩形范围内 x[80,160] y[40,200]
  [PASS] cv2.INTER_AREA / INTER_NEAREST / COLOR_*/RETR_EXTERNAL / CHAIN_APPROX_SIMPLE /
         IMREAD_COLOR / COLORMAP_TURBO 全部存在
  [PASS] 圆柱拟合 radius=0.150 ≈ 0.15
  [PASS] 拟合长度 1.400 ≈ 1.4
  [PASS] 无法向 -> 回退相机坐标系不崩
容器内真实路径自检全部通过 ✅
```

即：`imdecode` / `imencode` / `cvtColor` / `resize` / `findContours` / `approxPolyDP`
以及几何链路在 OpenCV 5.0.0 上均正确。**注意这仍是在 x86/aarch64 CPU 上验的**，
昇腾上的算子实现差异不在覆盖范围内。

---

## 7. 已知未验证项（设计契约 §6.4）❓

> **我们手上没有昇腾机器。** 下面是本机实测：`/dev/davinci*` 不存在、`npu-smi` 不存在、
> `/usr/local/Ascend/driver` 不存在、`/dev/davinci_manager` `/dev/devmm_svm` `/dev/hisi_hdc`
> 全部不存在（`preflight_ascend.sh` 在本机跑会得到 6 项 FAIL —— 这正是预期）。

| # | 未验证项 | 风险 | 上机怎么验 |
|---|---|---|---|
| 1 | **昇腾基础镜像能否拉取/启动** | 中 | `docker pull quay.io/ascend/torch-npu:2.7.1.post4-910b-ubuntu22.04-py3.11`；失败就换 `swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:<tag>` |
| 2 | **CANN 9.0.0 与宿主驱动的版本匹配** | **高** | `preflight_ascend.sh` 打印驱动版本；对照华为 CANN/驱动兼容表。不匹配就换 tag（如 `2.12.0.post2-cann9.1.0-*` 或更老的） |
| 3 | **`torch_npu` 在 910B/A3/310P 上是否可用** | **高** | `bash scripts/build_ascend.sh --chip 910b --verify` 会跑 `torch.npu.is_available()` / `device_count()` |
| 4 | **设备挂载是否充分** | 中 | `docker exec <容器> npu-smi info`；本方案照抄 MinerU 的挂载清单（davinci* + davinci_manager + devmm_svm + hisi_hdc + dcmi + npu-smi + driver + /var/log/npu） |
| 5 | **权重能否在构建期成功下载**（HF / 镜像站 / 内网） | 中 | 构建日志会打印 `[weights] 权重文件清单`；用 `--strict-weights` 强制校验 |
| 6 | **MoGe `model.pt` 能否在 NPU 上加载并推理** | **高** | 上机后 `curl -X POST /v1/infer -d '{"task":"geometry",...}'`。MoGe 是 ViT + DPT 类结构，某些算子可能没有昇腾实现，需要 fallback |
| 7 | **SAM(`transformers`) / YOLO(`ultralytics`) 在 NPU 上可用** | 中高 | 同上，分别调 `segment` / `detect` task |
| 8 | **310P 的 fp16/eager 是否真的生效** | 中 | ✅ **app 侧已落实**（`app/models/device.py`）：读 `CYLINDER_DTYPE`/`CYLINDER_EAGER`，检测到 310P 时**强制** float16+eager；`torch_npu` 关图模式的 API 名在版本间变过，做了多 API best-effort，**实际生效了哪个会记在 `/healthz` 的 `policy.eager_api`**。❓ 仍未验的是：在真机上 `eager_api` 到底解析成哪个（若显示 `FAILED:` 说明该版本 API 名不同，需按日志补） |
| 9 | **显存是否够 + 单模型驻留/驱逐是否真的生效** | 中 | `curl /v1/unload` 前后看 `npu-smi info` 的显存；`/healthz` 的 `resident` / `mem` 字段 |
| 10 | **性能（延迟/吞吐）** | 中 | 无任何基准数据。MoGe-2 ViT-L 单帧在 NPU 上的耗时完全未知 |
| 11 | **`quay.io/ascend/pytorch`（AscendHub 的 ascend-pytorch 系列）是否更合适** | 低 | 我们没有华为账号，无法列出该仓库/门户的 tag；已在 §2.2 标注未验证 |
| 12 | ~~`opencv-python-headless` 解析到 5.0.0.93，运行期未覆盖~~ | ~~中~~ | ✅ **已覆盖**：`tests/smoke_container.py` 在容器内跑通全部真实路径 cv2 调用（见 §6.1）。昇腾上的算子差异仍不在覆盖范围 |
| 13 | **`import ultralytics` 运行期行为**（首次运行可能联网检查/下载字体） | 中 | 离线环境建议设 `YOLO_OFFLINE=1` 或 `ULTRALYTICS_OFFLINE=1`，并在模型侧关掉自动更新 |
| 14 | **Dockerfile 在昇腾机器的**老版本 docker（如 20.10.12）**上能否构建 | 低 | 本 Dockerfile 刻意只用经典 builder 也支持的语法（无 heredoc / 无 `RUN --mount` / RUN 里只用 POSIX sh），已在本地 docker 20.10.5 上验证通过 ✅ |

### 7.1 `preflight_ascend.sh` 在本机的实测输出 ✅（证明自检脚本真的会拦住）

```console
$ bash scripts/preflight_ascend.sh --report-only
 主机: <host>   架构: aarch64   内核: 5.10.160
== 架构 ==
[PASS] 架构 aarch64（Kunpeng / Ascend 服务器常见架构）
== Docker ==
[PASS] docker 命令存在：/usr/bin/docker
[PASS] docker 守护进程可用，Server 版本 20.10.5+dfsg1
[PASS] docker 版本满足多阶段构建 + --device 挂载（>= 19.03）
[FAIL] Docker Root Dir /userdata/docker 只有 3 GB 可用，低于要求的 45 GB。
[PASS] 存储驱动：overlay2
== 昇腾驱动与 npu-smi ==
[FAIL] 找不到 npu-smi（npu-smi 与 /usr/local/bin/npu-smi 都不存在）。
[FAIL] 找不到 /usr/local/Ascend/driver。昇腾驱动未安装或安装路径不同。
[WARN] 找不到 /usr/local/dcmi（npu-smi 的部分功能依赖它）。
== 设备节点 ==
[FAIL] 没有任何 /dev/davinci* 节点。这台机器要么不是昇腾机器，要么驱动没加载。
[FAIL] /dev/davinci_manager 不存在。MinerU 与昇腾官方推荐的容器运行方式都会挂载它。
[FAIL] /dev/devmm_svm 不存在。...
[FAIL] /dev/hisi_hdc 不存在。...
[WARN] /var/log/npu 不存在，挂载 /var/log/npu/:/usr/slog 会失败。
== 其他 ==
[WARN] 没看到 davinci/devdrv 相关内核模块
[PASS] 宿主机没有 ascend-toolkit —— 正常，本方案把 CANN 放在镜像里，宿主只需要驱动
[PASS] 找到项目 Dockerfile：<repo>/Dockerfile
======================================================================
 自检结果:  PASS 7   WARN 3   FAIL 7
======================================================================
存在 7 项 FAIL：这台机器现在还不能跑昇腾容器。
$ echo $?
1
```

这个「7 项 FAIL」就是本文所有昇腾结论都标 ❓ 的直接证据：
**本机既没有昇腾驱动、没有设备节点，也没有 45 GB 磁盘，无法做任何昇腾真机验证。**

`run_ascend.sh` 也会在启动前拦住同样的问题（实测）：

```console
$ bash scripts/run_ascend.sh --image hello-world:latest --devices 0
[FAIL] 宿主机缺少昇腾设备节点：/dev/davinci0 /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
   这说明当前机器不是（或没装好）昇腾机器。请：
     1. bash scripts/preflight_ascend.sh      # 完整自检
     2. 确认已安装昇腾驱动并加载：lsmod | grep -E 'davinci|drv_davinci'
     3. 容器/虚拟机内请确认设备已透传
   确实要跳过检查（例如先看命令长什么样）：--skip-device-check
$ echo $?
1
```

---

## 8. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `docker build` 报 `no space left on device` | 磁盘不足（昇腾镜像展开 20 GB+） | `df -h $(docker info -f '{{.DockerRootDir}}')`；清理**自己的**无用镜像/缓存。**不要** `docker system prune -a`（会误删别人的镜像） |
| `docker pull` 基础镜像超时 | 国际网络 | 换 `swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:<tag>` 或 `--base-image` 指向内网仓库 |
| 容器里 `npu-smi info` 报错 | 设备节点/driver 未挂进去 或 权限 | `preflight_ascend.sh`；确认 `--privileged` 与 4 个 `-v` 挂载 |
| `weights` 阶段 `apt-get install` 报 404 Not Found | 用了 EOL 的 debian 发行版做基础镜像（bullseye）| 已改用 `python:3.11-slim`；如需自定义用 `--build-arg WEIGHTS_BASE_IMAGE=debian:bookworm-slim` |
| `import torch_npu` 失败 / `torch.npu.is_available()` False | 驱动版本与镜像内 CANN 不匹配（最常见） | 看 `preflight_ascend.sh` 打印的驱动版本；换一个 CANN 版本匹配的 tag |
| 构建时报 `ERROR: 装模型侧依赖时 torch 被改动了` | pip 想把 torch 升级掉（torch_npu 会失配） | 这是**保护性失败**。把对应包降到你 torch 版本兼容的版本：`--model-packages 'transformers==4.x ultralytics==8.y'` |
| `from moge.model.v2 import MoGeModel` ImportError | 漏了 `utils3d_moge` | 已修：`MOGE_EXTRA_PIP_SPECS`（见 §3.3）。构建期 `CHECK_MODEL_IMPORTS=1` 会先报错 |
| `torch.load(目录)` 报 `Is a directory` | MoGe 权重给了目录 | MoGe 必须是**扁平 `.pt` 文件**；SAM 必须是**目录**。见 §3.3 布局 |
| `/v1/infer` 返回 `weights_missing` | `/app/weights` 里没有对应权重 | `docker exec <容器> ls -la /app/weights`；或 `--weights <宿主目录>` 挂载 |
| 310P 上精度异常 / bf16 报错 | 310P 不支持 bf16 与图模式 | 见 §4.2：float16 + eager，`run_ascend.sh --310p` |
| 容器日志 `Address already in use` | host 网络下 8000 被占 | 换端口：`run_ascend.sh --network bridge --port 18000` 或释放宿主 8000 |

---

## 9. 参考资料

- MinerU 昇腾适配（用户指定参考）
  - Dockerfile：<https://github.com/opendatalab/MinerU> → `docker/china/npu.Dockerfile`
  - 文档：<https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/Ascend/>
- 昇腾 PyTorch 适配（`torch-npu` 镜像来源）：<https://quay.io/repository/ascend/torch-npu> ·
  <https://gitcode.com/Ascend/pytorch>
- 华为 AscendHub 镜像门户：<https://www.hiascend.com/developer/ascendhub>
- CANN 官方镜像（备选，只有 CANN 没有 torch）：<https://quay.io/repository/ascend/cann>
- MoGe：<https://github.com/microsoft/MoGe>（本服务用 `moge.model.v2`，权重 `Ruicheng/moge-2-vitl-normal`）
- `utils3d_moge`：<https://github.com/EasternJournalist/utils3d-moge>
- SAM：<https://huggingface.co/facebook/sam-vit-base>（transformers）
- `ASCEND_RT_VISIBLE_DEVICES` 说明：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/maintenref/envvar/envref_07_0028.html>

---

## 10. 变更与决策记录

| 决策 | 理由 |
|---|---|
| 用 `quay.io/ascend/torch-npu` 而不是 MinerU 的 `ascend-vllm` | 本服务无 LLM，不需要 vllm/lmdeploy；`torch-npu` 镜像已配对好 torch + torch_npu + CANN |
| `cpu` 目标用 `python:3.11-slim` 而不是 3.10 | 对齐昇腾镜像的 Python 3.11.15，减少「本地能跑、昇腾不能跑」的差异 |
| Dockerfile 只用经典 builder 语法（无 heredoc / `RUN --mount`） | 昇腾机器 docker 可能很老（MinerU 文档的测试机是 20.10.12），本地是 20.10.5 且 BuildKit 未开 |
| `ascend` 阶段过滤 `requirements.txt` 里的 `torch*` | 基础镜像里是配好 CANN 的那一套，重装会破坏 NPU 支持 |
| 镜像不设 `MODEL_DEVICE` | 保留 `app` 侧 `detect_device()` 的自动探测；写死 `npu:0` 在换卡/无卡时会让服务起不来 |
| 权重放独立的 `weights` 阶段（`debian:bullseye-slim`） | 权重不进构建上下文（设计契约 §2.3），单独成层可命中 cache，避免重建应用层时重下 1.7 GB |
| ascend 阶段用 pip constraint 钉住 torch | `ultralytics → torchvision → torch` 的依赖链可能在安装时升级 torch，而 torch_npu 与 torch 严格配对，升级即 NPU 不可用。钉版 + 装完比对，失败就报错 |
| 310P 用环境变量而不是 CLI flag | 本服务没有 vllm，`--enforce-eager --dtype float16` 无对应物；真正的控制点在模型加载处（见 §4.2） |
| `.dockerignore` 额外排除 `*.safetensors` / `*.bin` | 设计契约只点名 `*.pt`/`*.pth`，但 SAM 的 `model.safetensors` 有 358 MB，一起排掉才真正达到「不让大权重进上下文」的目的 |
| `weights` 阶段改用 `python:3.11-slim`（而不是 `debian:bullseye-slim`） | **实测发现的坑**：bullseye 已 EOL，apt 源上的 `python3-pip` / `python3-setuptools` 已被移走，构建直接 404 失败；而且 bullseye 自带 pip 20.3.4 太老，装不动当前 `huggingface_hub`。换成 `python:3.11-slim` 后只需要再装 curl（pip/证书自带） |
