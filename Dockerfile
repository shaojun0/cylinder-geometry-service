# =============================================================================
#  气瓶几何量测服务 —— 昇腾（Ascend NPU）镜像
#
#  接口契约见 SPEC.md。提供两个构建目标：
#
#    --target cpu      本地可行性校验镜像（python:3.11-slim）
#                      用途：验证 Dockerfile 语法 / COPY 路径 / 依赖清单 / 入口命令。
#                      要求在 ~4.6 GB 可用磁盘内构建成功，体积 < 1.5 GB。
#                      不做真实推理（不装 torch，不装权重）。
#
#    --target ascend   正式交付镜像（昇腾 CANN + torch_npu 官方基础镜像）
#                      本机无昇腾硬件 + 磁盘不足，未实际构建过，见 docs/ASCEND.md。
#
#  另外提供一个中间阶段：
#
#    weights           构建期权重抓取（可在 ascend 构建时复用 build cache）
#
#  ---------------------------------------------------------------------------
#  兼容性说明（重要）
#    本文件刻意只使用「经典 builder（非 BuildKit）」也支持的语法：
#      * 不用 heredoc（RUN <<EOF 需要 dockerfile:1.4 frontend / BuildKit）
#      * 不用 RUN --mount=...
#      * RUN 里只用 POSIX sh（dash）语法，不用 set -o pipefail / [[ ]] / 数组
#    原因：昇腾机器上的 docker 版本可能较老（MinerU 官方文档的测试机是 20.10.12），
#    而本地校验机是 20.10.5，BuildKit 未必可用。可移植性优先。
#
#  构建（本地校验）：
#    bash scripts/build_local_cpu.sh
#  构建（昇腾机器）：
#    bash scripts/build_ascend.sh --chip 910b
# =============================================================================

# 全局 ARG：只用于 FROM 行。默认值可用 --build-arg 覆盖。
ARG PYTHON_VERSION=3.11
# 构建期抓权重那个阶段的基础镜像（只需 python3+pip+curl，越轻越好）
ARG WEIGHTS_BASE_IMAGE=python:3.11-slim

# 昇腾基础镜像。三个硬件型号对应三个 tag，型号不同只需换这一个值：
#   A2  / Atlas 800T A2, 800I A2, 900 A2 PoD      -> 2.7.1.post4-910b-ubuntu22.04-py3.11
#   A3  / Atlas 800T A3, 800I A3, 900 A3 SuperPoD -> 2.7.1.post4-a3-ubuntu22.04-py3.11
#   300I Duo (310P)                                -> 2.7.1.post4-310p-ubuntu22.04-py3.11
# 这些 tag 已在 quay.io 上核对存在（含 arm64 manifest）；镜像内自带
# CANN 9.0.0 + torch 2.7.1 + torch_npu 2.7.1.post4 + Python 3.11.15。
# 详见 docs/ASCEND.md「基础镜像选型」。
ARG ASCEND_BASE_IMAGE=quay.io/ascend/torch-npu:2.7.1.post4-910b-ubuntu22.04-py3.11


# =============================================================================
#  阶段 cpu —— 本地可行性校验镜像
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS cpu

ARG DEBIAN_FRONTEND=noninteractive
# 冒烟测试开关：--build-arg SMOKE_TEST=0 可跳过 import 检查（app/ 未就绪时用）
ARG SMOKE_TEST=1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WEIGHTS_DIR=/app/weights \
    MODEL_DEVICE=cpu \
    LOG_LEVEL=INFO

# opencv-python-headless 只去掉 libGL/X11 依赖，仍然需要 libglib2.0-0；
# libgomp1 是 numpy/opencv 的 OpenMP 运行时。这两个包合计 ~10 MB。
RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends libglib2.0-0 libgomp1; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# cpu 目标的最小依赖子集。
# 思路：只装 app/requirements.txt 里「纯 Python 或带 aarch64 wheel」的运行依赖，
# 明确不装 torch / torch_npu / transformers / ultralytics / moge：
#   * torch 的 aarch64 wheel 约 100+ MB（CN 镜像体积会翻 3 倍），
#   * 而且 app 侧所有重型 import 都是延迟的（见 model_hub.py 模块注释第 4 条），
#     `import model_hub` / `import service` 不需要它们。
# 这段清单与 app/requirements.txt 的「CPU 相关」部分保持一致。
RUN set -eu; \
    python -m pip install --no-cache-dir \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "pydantic>=2.0" \
        "python-multipart>=0.0.9" \
        "numpy>=1.24" \
        "pillow>=10.0" \
        "opencv-python-headless>=4.8"

# 应用代码：契约要求构建上下文根目录下的 app/ 映射到 /app
# （.dockerignore 已排除 __pycache__ / *.pyc，不会把宿主机的 .pyc 带进来）
COPY app/ /app/

# 单文件前端：按 service.py 的探测约定，容器里放在 /app 旁边的 /web。
# 由 StaticFiles 挂在 "/"（注册在全部 API 路由之后），与 API 同源托管。
COPY web/ /web/

RUN mkdir -p /app/weights

# 结构冒烟测试：不加载模型、不碰权重，只验证依赖齐 + 模块可 import。
RUN set -eu; \
    if [ "${SMOKE_TEST}" = "1" ]; then \
        cd /app; \
        python -c "import model_hub, service; print('[cpu] smoke test OK: import model_hub, service')"; \
    else \
        echo "[cpu] SMOKE_TEST=0，跳过 import 冒烟测试"; \
    fi

EXPOSE 8000

# 健康检查用 python 而不是 curl：python:3.11-slim 里没有 curl。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"]

# 入口契约（SPEC 2.2，不可变）
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]


# =============================================================================
#  阶段 weights —— 构建期把模型权重抓进镜像
#
#  为什么单独一个阶段：
#    * SPEC 2.3 要求 .dockerignore 排除 weights/*.pt 等大文件，权重不进构建上下文；
#    * 用户要求「把模型一起打进镜像」，所以要在构建期拉取；
#    * 单独成阶段后，只要 build-arg 不变就能命中 cache，重建应用层时不会重复下 1.7 GB。
#
#  基础镜像刻意选很小的 python:3.11-slim（约 50 MB 压缩），
#  避免为「下载」再拉一份 4.6 GB 的昇腾基础镜像。
#  它自带 python3 / pip 24 / ca-certificates，只需要再装一个 curl。
#  ⚠ 不要用 debian:bullseye-slim：bullseye 已 EOL，apt 源上的
#    python3-pip / python3-setuptools 包已被移走，实测 apt 报 404 装不上；
#    而且它自带的 pip 20.3.4 太老，装不动当前版本的 huggingface_hub。
#
#  权重布局（与 app/models/*.py 的查找逻辑对齐，两种布局都兼容，这里用扁平布局）：
#    /app/weights/moge-2-vitl-normal.pt     <- 必须是 .pt 文件路径，不能是目录
#    /app/weights/sam-vit-base/             <- 必须是目录（config.json + 权重）
#    /app/weights/yolo11n-seg.pt            <- 训练产出，默认没有公开 URL
#
#  权重体积约 1.7 GB（MoGe model.pt 1.3 GB + SAM 358 MB + YOLO ~6 MB）。
# =============================================================================
FROM ${WEIGHTS_BASE_IMAGE} AS weights

ARG DEBIAN_FRONTEND=noninteractive

# 总开关。0 = 光构建不下载（之后用 -v 挂载 /app/weights）。
ARG FETCH_WEIGHTS=1
# 抓不到任何权重时是否让构建失败。1 = 严格模式（推荐用于正式交付）。
ARG STRICT_WEIGHTS=0

# HF 端点。国内可设为 https://hf-mirror.com
ARG HF_ENDPOINT=https://huggingface.co
# MoGe-2：HF repo + 文件名 + 镜像内扁平文件名
ARG MOGE_HF_REPO=Ruicheng/moge-2-vitl-normal
ARG MOGE_HF_FILE=model.pt
ARG MOGE_TARGET=moge-2-vitl-normal.pt
# SAM：HF repo + 镜像内目录名（transformers 需要目录）
ARG SAM_HF_REPO=facebook/sam-vit-base
ARG SAM_TARGET_DIR=sam-vit-base
# YOLO-seg：训练产出，通常没有公开直链，可给 URL 或用 tarball 兜底
ARG YOLO_WEIGHTS_URL=
ARG YOLO_TARGET=yolo11n-seg.pt
# 兜底：一个包含全部权重的 .tar / .tar.gz
ARG WEIGHTS_TARBALL_URL=

ENV FETCH_WEIGHTS=${FETCH_WEIGHTS} \
    STRICT_WEIGHTS=${STRICT_WEIGHTS} \
    HF_ENDPOINT=${HF_ENDPOINT} \
    MOGE_HF_REPO=${MOGE_HF_REPO} \
    MOGE_HF_FILE=${MOGE_HF_FILE} \
    MOGE_TARGET=${MOGE_TARGET} \
    SAM_HF_REPO=${SAM_HF_REPO} \
    SAM_TARGET_DIR=${SAM_TARGET_DIR} \
    YOLO_TARGET=${YOLO_TARGET}

RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl; \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /out/weights

# ---- 1) 可选：私有 tarball 兜底（离线交付常用） --------------------------------
RUN set -eu; \
    if [ -n "${WEIGHTS_TARBALL_URL}" ]; then \
        echo "[weights] 下载 tarball: ${WEIGHTS_TARBALL_URL}"; \
        curl -fL --retry 3 --retry-delay 2 -o /tmp/weights.tar "${WEIGHTS_TARBALL_URL}"; \
        tar -xf /tmp/weights.tar -C /out/weights; \
        rm -f /tmp/weights.tar; \
    else \
        echo "[weights] 未设置 WEIGHTS_TARBALL_URL，跳过"; \
    fi

# ---- 2) 可选：YOLO-seg 直链 -----------------------------------------------
RUN set -eu; \
    if [ -n "${YOLO_WEIGHTS_URL}" ]; then \
        echo "[weights] 下载 YOLO: ${YOLO_WEIGHTS_URL}"; \
        curl -fL --retry 3 --retry-delay 2 -o "/out/weights/${YOLO_TARGET}" "${YOLO_WEIGHTS_URL}"; \
    else \
        echo "[weights] 未设置 YOLO_WEIGHTS_URL（训练产出，通常靠挂载），跳过"; \
    fi

# ---- 3) MoGe + SAM：从 HuggingFace 拉取 -----------------------------------
# huggingface_hub 同时提供 huggingface-cli（旧）与 hf（新）两个入口，这里自动探测。
RUN set -eu; \
    if [ "${FETCH_WEIGHTS}" = "1" ]; then \
        python3 -m pip install --no-cache-dir -q "huggingface_hub>=0.23"; \
        if command -v huggingface-cli >/dev/null 2>&1; then HF_CLI=huggingface-cli; \
        elif command -v hf >/dev/null 2>&1; then HF_CLI=hf; \
        else echo "ERROR: 没找到 huggingface-cli / hf 命令"; exit 1; fi; \
        echo "[weights] 使用 ${HF_CLI}，HF_ENDPOINT=${HF_ENDPOINT}"; \
        echo "[weights] MoGe: ${MOGE_HF_REPO}/${MOGE_HF_FILE} -> ${MOGE_TARGET}"; \
        "${HF_CLI}" download "${MOGE_HF_REPO}" "${MOGE_HF_FILE}" --local-dir /tmp/moge-dl; \
        mv "/tmp/moge-dl/${MOGE_HF_FILE}" "/out/weights/${MOGE_TARGET}"; \
        rm -rf /tmp/moge-dl; \
        echo "[weights] SAM:  ${SAM_HF_REPO} -> ${SAM_TARGET_DIR}/"; \
        "${HF_CLI}" download "${SAM_HF_REPO}" --local-dir "/out/weights/${SAM_TARGET_DIR}"; \
        rm -rf "/out/weights/${SAM_TARGET_DIR}/.cache"; \
    else \
        echo "[weights] FETCH_WEIGHTS=0，跳过硬权重下载（改用 -v 挂载 /app/weights）"; \
    fi

# ---- 4) 权限 + 清单 + 严格模式校验 ----------------------------------------
# app 以 root 跑，但 SPEC/运维约定要求权重 a+r，方便换非 root 用户或只读挂载。
RUN set -eu; \
    chmod -R a+rX /out/weights; \
    count=$(find /out/weights -type f | wc -l); \
    echo "[weights] 权重文件清单（${count} 个）:"; \
    find /out/weights -type f -exec ls -l {} \; | awk '{printf "    %10d  %s\n", $5, $NF}' | sort -k2; \
    if [ "${STRICT_WEIGHTS}" = "1" ] && [ "${count}" -eq 0 ]; then \
        echo "ERROR: STRICT_WEIGHTS=1 但没抓到任何权重。"; \
        echo "       请设置 MOGE_HF_REPO / SAM_HF_REPO / WEIGHTS_TARBALL_URL，"; \
        echo "       或去掉 --strict-weights 并改用 -v 挂载 /app/weights。"; \
        exit 1; \
    fi


# =============================================================================
#  阶段 ascend —— 正式交付镜像（昇腾 CANN + torch_npu）
# =============================================================================
FROM ${ASCEND_BASE_IMAGE} AS ascend

ARG DEBIAN_FRONTEND=noninteractive

# 是否安装 app/requirements.txt 里的其余依赖（torch* 会被过滤掉）
ARG INSTALL_APP_REQUIREMENTS=1
# 模型侧依赖：app/requirements.txt 里刻意注释掉、留给昇腾镜像装的那几个
ARG MODEL_PIP_PACKAGES=transformers>=4.40 huggingface-hub>=0.23 ultralytics>=8.2
ARG INSTALL_MODEL_PIP=1
# MoGe 本体：上游 microsoft/MoGe 没有发布 tag，这里按 commit 固定。
# --no-deps 是刻意的：MoGe main 现在是 v3.0.0，pyproject 里声明了
# flex-gemm（CUDA 专用）/ pipeline（v3 用）/ gradio 等重依赖，
# 而本服务只用 `moge.model.v2`。
ARG MOGE_PIP_SPEC=git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a
# ⚠ 关键：moge/model/v2.py 在**模块顶层**就 `import utils3d_moge as utils3d`
#   （except ImportError 后回退 `import utils3d`），所以光装 moge --no-deps 会让
#   `from moge.model.v2 import MoGeModel` 直接 ImportError，geometry task 全废。
#   必须补上 utils3d 这一支。这里钉 MoGe pyproject 里声明的同一个 commit。
#   utils3d_moge 的依赖只有 moderngl / numpy / scipy，都是轻量且有 aarch64 wheel。
ARG MOGE_EXTRA_PIP_SPECS=git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37
ARG INSTALL_MOGE=1
# 构建期就验证 MoGe / 模型侧依赖的 import 链（不需要 NPU，纯 CPU 也能查出来）
ARG CHECK_MODEL_IMPORTS=1
# 可选：PyPI 镜像（例如 https://pypi.tuna.tsinghua.edu.cn/simple）
ARG PIP_INDEX_URL=
# 同 cpu 阶段：1 = 做 import 冒烟；0 = 只做语法检查
ARG SMOKE_TEST=1

# SPEC 2.1 要求的环境变量。ASCEND_RT_VISIBLE_DEVICES 只是默认值，
# run_ascend.sh / docker-compose.yml 会用 -e 覆盖成实际卡号。
# 刻意不设 MODEL_DEVICE：让 app 侧的 detect_device() 自己探测 npu/cuda/cpu，
# 强制写死 npu:0 会在换卡或没有卡时直接把服务带崩。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ASCEND_RT_VISIBLE_DEVICES=0 \
    WEIGHTS_DIR=/app/weights \
    LOG_LEVEL=INFO

WORKDIR /app

# opencv 运行库 + git（pip 装 git+https:// 依赖需要 git 与 ca-certificates）。
# requirements.txt 用的是 opencv-python-headless，所以不装 libgl1（省 ~100 MB）；
# 若以后换成 opencv-python，把 libgl1 加回来即可。
RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 git ca-certificates; \
    rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    if [ -n "${PIP_INDEX_URL}" ]; then \
        echo "[ascend] 使用 PyPI 镜像: ${PIP_INDEX_URL}"; \
        python -m pip config set global.index-url "${PIP_INDEX_URL}"; \
    fi

COPY app/ /app/
# 前端静态页：容器布局下 /app 旁边是 /web（与 service.py 的候选路径一致）。
COPY web/ /web/

# app/requirements.txt。
# 必须过滤掉 torch / torch_npu / torchvision / torchaudio：基础镜像里已经装好
# 与 CANN 9.0.0 匹配的那一套，用 requirements 里的版本重装会直接破坏 NPU 支持。
RUN set -eu; \
    if [ "${INSTALL_APP_REQUIREMENTS}" = "1" ] && [ -f /app/requirements.txt ]; then \
        echo "[ascend] 过滤 torch* 后的依赖清单:"; \
        grep -viE '^[[:space:]]*(torch|torch_npu|torch-npu|torchvision|torchaudio)([<>=!~;[:space:]].*)?$' \
            /app/requirements.txt > /tmp/requirements.ascend.txt; \
        cat /tmp/requirements.ascend.txt; \
        python -m pip install --no-cache-dir -r /tmp/requirements.ascend.txt; \
    else \
        echo "[ascend] INSTALL_APP_REQUIREMENTS=${INSTALL_APP_REQUIREMENTS}，跳过 app/requirements.txt"; \
    fi

# 模型侧依赖（SAM 用 transformers，YOLO-seg 用 ultralytics，权重下载用 huggingface-hub）
#
# ⚠ 这里必须把镜像里已有的 torch 版本**钉住**（pip constraint），否则
#   transformers / ultralytics 的依赖解析可能顺手把 torch 升级掉
#   （例如 ultralytics 依赖 torchvision，而新版 torchvision 会 pin 一个更新的 torch）。
#   torch_npu 是跟 torch **严格配对**的，torch 被升级 = NPU 直接不可用。
#   装完再校验一次，一旦 torch 变了就立刻失败，而不是留一个「能 import 但 NPU 用不了」的镜像。
RUN set -eu; \
    if [ "${INSTALL_MODEL_PIP}" = "1" ] && [ -n "${MODEL_PIP_PACKAGES}" ]; then \
        torch_before="$(python -c 'import torch; print(torch.__version__)')"; \
        printf 'torch==%s\n' "${torch_before%%+*}" > /tmp/torch-constraint.txt; \
        echo "[ascend] 冻结 torch（constraint）: ${torch_before} -> $(cat /tmp/torch-constraint.txt)"; \
        echo "[ascend] 安装模型侧依赖: ${MODEL_PIP_PACKAGES}"; \
        python -m pip install --no-cache-dir -c /tmp/torch-constraint.txt ${MODEL_PIP_PACKAGES}; \
        torch_after="$(python -c 'import torch; print(torch.__version__)')"; \
        if [ "${torch_before}" != "${torch_after}" ]; then \
            echo "ERROR: 装模型侧依赖时 torch 被改动了：${torch_before} -> ${torch_after}"; \
            echo "       torch_npu 与 torch 严格配对，这样会让 NPU 不可用。"; \
            echo "       请把对应包装成与 torch ${torch_before} 兼容的版本：--model-packages '...'"; \
            exit 1; \
        fi; \
        echo "[ascend] torch 版本未被改动: ${torch_after}"; \
    else \
        echo "[ascend] 跳过模型侧依赖"; \
    fi

# MoGe 本体（app/models/moge_geom.py 里 `from moge.model.v2 import MoGeModel`）
# --no-deps 跳过 MoGe 声明的 CUDA/桌面依赖（flex-gemm / pipeline / gradio），
# 但 utils3d 是 v2.py 顶层 import 的硬依赖，必须单独补装（见上面的注释）。
RUN set -eu; \
    if [ "${INSTALL_MOGE}" = "1" ]; then \
        echo "[ascend] 安装 MoGe: ${MOGE_PIP_SPEC} (--no-deps)"; \
        python -m pip install --no-cache-dir --no-deps "${MOGE_PIP_SPEC}"; \
        echo "[ascend] 安装 MoGe 的 utils3d 依赖: ${MOGE_EXTRA_PIP_SPECS}"; \
        python -m pip install --no-cache-dir ${MOGE_EXTRA_PIP_SPECS}; \
    else \
        echo "[ascend] 跳过 MoGe 安装"; \
    fi

# 构建期把权重放进镜像（用户要求「包括模型」）。
# 运行时仍然可以用 -v /path/to/weights:/app/weights 覆盖（挂载优先于镜像内容）。
# 若 weights 阶段没抓到任何东西，这里只是复制一个空目录，不会报错。
COPY --from=weights /out/weights/ /app/weights/
RUN set -eu; \
    mkdir -p /app/weights; \
    chmod -R a+rX /app/weights; \
    echo "[ascend] /app/weights 内容:"; \
    ls -la /app/weights || true

# 构建期自检。
# 注意：这里**刻意只做语法检查 + 无设备依赖的 import 冒烟**，
# 不加载任何模型（模型加载需要 NPU，且 1.7 GB 权重不该在构建机上跑）。
RUN set -eu; \
    python -m compileall -q /app/model_hub.py /app/service.py /app/models; \
    echo "[ascend] 语法检查通过"

# 模型侧 import 链校验：这几步纯 CPU 就能跑，把「装上了但 import 不了」的问题
# 拦在构建期，而不是等上机第一次调用才炸。
RUN set -eu; \
    if [ "${CHECK_MODEL_IMPORTS}" = "1" ]; then \
        echo "[ascend] 已安装版本（MISSING = 未安装）:"; \
        for pkg in torch torch_npu transformers huggingface_hub ultralytics moge utils3d_moge; do \
            printf '     %-16s ' "${pkg}"; \
            python -c "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))" "${pkg}" 2>/dev/null || echo "MISSING"; \
        done; \
        if [ "${INSTALL_MOGE}" = "1" ]; then \
            python -c "from moge.model.v2 import MoGeModel; print('[ascend] moge.model.v2 import OK')"; \
        fi; \
        if [ "${INSTALL_MODEL_PIP}" = "1" ]; then \
            python -c "import transformers; print('[ascend] transformers', transformers.__version__)"; \
        fi; \
    else \
        echo "[ascend] CHECK_MODEL_IMPORTS=0，跳过模型侧 import 校验"; \
    fi

RUN set -eu; \
    if [ "${SMOKE_TEST}" = "1" ]; then \
        cd /app; \
        python -c "import model_hub, service; print('[ascend] smoke test OK: import model_hub, service')"; \
    else \
        echo "[ascend] SMOKE_TEST=0，跳过 import 冒烟测试"; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"]

# 入口契约（SPEC 2.2，不可变）。
# 基础镜像自带 ENTRYPOINT ["/bin/bash","-c","source ...set_env.sh... && exec \"$@\"","--"]，
# 所以 CANN 环境变量会在容器启动时被 source，我们的 CMD 会被正确地 exec。
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
