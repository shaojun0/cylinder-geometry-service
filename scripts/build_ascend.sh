#!/usr/bin/env bash
# =============================================================================
#  build_ascend.sh —— 在昇腾机器上构建正式交付镜像
#
#  参考 MinerU 的昇腾适配方式（docker/china/npu.Dockerfile）：
#    * 用昇腾官方基础镜像，按硬件型号换 tag
#    * docker build --network=host
#  差异：本服务只跑 CNN/ViT 小模型，不需要 vllm/lmdeploy，
#        所以基础镜像用 CANN + torch_npu（quay.io/ascend/torch-npu），
#        而不是 ascend-vllm 镜像。理由见 docs/ASCEND.md。
#
#  用法：
#    bash scripts/build_ascend.sh --chip 910b            # A2
#    bash scripts/build_ascend.sh --chip a3              # A3
#    bash scripts/build_ascend.sh --chip 310p --310p     # Atlas 300I Duo
#    bash scripts/build_ascend.sh --chip 910b --dry-run  # 只打印命令
#
#  退出码：0 成功；1 失败；2 参数错误
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- 已在 quay.io 核对存在的官方 tag（含 arm64 manifest）--------------------
# 格式：<TorchNPU版本>-<芯片>-<OS>-<Python版本>，镜像内自带 CANN 9.0.0。
# 换硬件型号 = 换这一个 tag；镜像内容（torch 2.7.1 / torch_npu 2.7.1.post4）一致。
declare -A CHIP_IMAGE=(
  [910b]="quay.io/ascend/torch-npu:2.7.1.post4-910b-ubuntu22.04-py3.11"
  [a2]="quay.io/ascend/torch-npu:2.7.1.post4-910b-ubuntu22.04-py3.11"
  [a3]="quay.io/ascend/torch-npu:2.7.1.post4-a3-ubuntu22.04-py3.11"
  [310p]="quay.io/ascend/torch-npu:2.7.1.post4-310p-ubuntu22.04-py3.11"
  [300i]="quay.io/ascend/torch-npu:2.7.1.post4-310p-ubuntu22.04-py3.11"
  [300iduo]="quay.io/ascend/torch-npu:2.7.1.post4-310p-ubuntu22.04-py3.11"
)
CHIP_IMAGE_HELP="910b(A2) | a3(A3) | 310p(Atlas 300I Duo) | 或 --base-image 直接指定"

TAG="cylinder-geom:ascend"
CHIP=""
BASE_IMAGE=""
NO_CACHE=0
HOST_NETWORK=1
PUSH=0
DRY_RUN=0
VERIFY=0
NPU_DEVICES="${NPU_DEVICES:-0}"

# weights 阶段
FETCH_WEIGHTS="${FETCH_WEIGHTS:-1}"
STRICT_WEIGHTS="${STRICT_WEIGHTS:-0}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
MOGE_HF_REPO="${MOGE_HF_REPO:-Ruicheng/moge-2-vitl-normal}"
SAM_HF_REPO="${SAM_HF_REPO:-facebook/sam-vit-base}"
YOLO_WEIGHTS_URL="${YOLO_WEIGHTS_URL:-}"
WEIGHTS_TARBALL_URL="${WEIGHTS_TARBALL_URL:-}"

# 依赖阶段（需要调参时可覆盖）
INSTALL_APP_REQUIREMENTS="${INSTALL_APP_REQUIREMENTS:-1}"
MODEL_PIP_PACKAGES="${MODEL_PIP_PACKAGES:-transformers>=4.40 huggingface-hub>=0.23 ultralytics>=8.2}"
INSTALL_MODEL_PIP="${INSTALL_MODEL_PIP:-1}"
MOGE_PIP_SPEC="${MOGE_PIP_SPEC:-git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a}"
# moge/model/v2.py 顶层 import utils3d(_moge)，必须补装，否则 geometry task 直接 ImportError
MOGE_EXTRA_PIP_SPECS="${MOGE_EXTRA_PIP_SPECS:-git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37}"
INSTALL_MOGE="${INSTALL_MOGE:-1}"
CHECK_MODEL_IMPORTS="${CHECK_MODEL_IMPORTS:-1}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"

MIN_FREE_GB="${MIN_FREE_GB:-45}"

c_red=$'\033[31m'; c_grn=$'\033[32m'; c_yel=$'\033[33m'; c_bld=$'\033[1m'; c_off=$'\033[0m'
log()  { printf '%s[build_ascend]%s %s\n' "$c_bld" "$c_off" "$*"; }
ok()   { printf '%s[ OK ]%s %s\n'          "$c_grn" "$c_off" "$*"; }
warn() { printf '%s[WARN]%s %s\n'          "$c_yel" "$c_off" "$*" >&2; }
die()  { printf '%s[FAIL]%s %s\n'          "$c_red" "$c_off" "$*" >&2; exit 1; }
usage_die() { usage >&2; printf '\n%s[FAIL]%s %s\n' "$c_red" "$c_off" "$*" >&2; exit 2; }

usage() {
  cat <<'USAGE'
build_ascend.sh —— 昇腾机器上的正式构建

用法：
  bash scripts/build_ascend.sh --chip 910b [选项]
  bash scripts/build_ascend.sh --base-image <完整镜像ref> [选项]

硬件型号（--chip，二选一必填，或改用 --base-image）：
  910b | a2     Atlas A2 系列（800T A2 / 800I A2 / 900 A2 PoD / 300T A2）
  a3            Atlas A3 系列（800T A3 / 800I A3 / 900 A3 SuperPoD）
  310p | 300i | 300iduo   Atlas 300I Duo（310P 推理卡）
  支持标签: 910b(A2) | a3(A3) | 310p(Atlas 300I Duo) | 或 --base-image 直接指定

选项：
  -c, --chip NAME         硬件型号 -> 选基础镜像 tag
      --base-image REF    直接指定基础镜像（覆盖 --chip），例如内网镜像仓库地址
  -t, --tag NAME          产出镜像 tag（默认 cylinder-geom:ascend）
      --no-weights        构建期不下载权重（之后用 -v 挂载 /app/weights）
      --strict-weights    抓不到任何权重就让构建失败（正式交付建议开）
      --weights-tarball U 从一个 .tar/.tar.gz 拉全部权重（离线交付）
      --yolo-url U        YOLO-seg 权重直链
      --hf-endpoint U     HuggingFace 端点（默认 https://huggingface.co；国内可 https://hf-mirror.com）
      --moge-repo REPO    MoGe HF repo（默认 Ruicheng/moge-2-vitl-normal）
      --sam-repo REPO     SAM HF repo（默认 facebook/sam-vit-base）
      --moge-spec SPEC    MoGe 安装源（默认 git+https://github.com/microsoft/MoGe.git@<commit>）
      --moge-extra-spec S MoGe 的 utils3d 依赖源（v2.py 顶层 import 需要，必须装）
      --no-moge           不装 MoGe 本体（geometry task 会不可用）
      --no-check-imports  跳过构建期的 moge/transformers import 校验
      --model-packages "…" 模型侧 pip 包（默认 transformers huggingface-hub ultralytics）
      --no-model-packages 不装模型侧 pip 包
      --pip-index-url U   PyPI 镜像（如 https://pypi.tuna.tsinghua.edu.cn/simple）
      --no-cache          禁用 layer cache
      --no-host-network   不用 --network=host 构建（默认用，MinerU 同款）
      --npu-devices LIST  --verify 用的卡号（默认 0，逗号分隔）
      --verify            构建后跑一次 NPU 可用性自检（需要能访问设备节点）
      --push              构建完成后 docker push
      --min-free-gb N     构建前要求的最小可用磁盘 GB（默认 45）
      --dry-run           只打印将要执行的命令，不构建
  -h, --help              显示本帮助

注意（310P / Atlas 300I Duo）：不支持 bf16、不支持图模式。
  MinerU 是 vllm 后端所以加 `--enforce-eager --dtype float16`；
  本服务没有 vllm，**不存在等价的 CLI flag**，需要在模型侧用 float16 并禁用图模式。
  详见 docs/ASCEND.md「310P 特殊参数」。

退出码：0 成功；1 失败；2 参数错误
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    -c|--chip)              CHIP="${2:?--chip 需要一个值}"; shift 2 ;;
    --chip=*)               CHIP="${1#*=}"; shift ;;
    --base-image)           BASE_IMAGE="${2:?--base-image 需要一个值}"; shift 2 ;;
    --base-image=*)         BASE_IMAGE="${1#*=}"; shift ;;
    -t|--tag)               TAG="${2:?--tag 需要一个值}"; shift 2 ;;
    --tag=*)                TAG="${1#*=}"; shift ;;
    --no-weights)           FETCH_WEIGHTS=0; shift ;;
    --strict-weights)       STRICT_WEIGHTS=1; shift ;;
    --weights-tarball)      WEIGHTS_TARBALL_URL="${2:?}"; shift 2 ;;
    --weights-tarball=*)    WEIGHTS_TARBALL_URL="${1#*=}"; shift ;;
    --yolo-url)             YOLO_WEIGHTS_URL="${2:?}"; shift 2 ;;
    --yolo-url=*)           YOLO_WEIGHTS_URL="${1#*=}"; shift ;;
    --hf-endpoint)          HF_ENDPOINT="${2:?}"; shift 2 ;;
    --hf-endpoint=*)        HF_ENDPOINT="${1#*=}"; shift ;;
    --moge-repo)            MOGE_HF_REPO="${2:?}"; shift 2 ;;
    --moge-repo=*)          MOGE_HF_REPO="${1#*=}"; shift ;;
    --sam-repo)             SAM_HF_REPO="${2:?}"; shift 2 ;;
    --sam-repo=*)           SAM_HF_REPO="${1#*=}"; shift ;;
    --moge-spec)            MOGE_PIP_SPEC="${2:?}"; shift 2 ;;
    --moge-spec=*)          MOGE_PIP_SPEC="${1#*=}"; shift ;;
    --moge-extra-spec)      MOGE_EXTRA_PIP_SPECS="${2:?}"; shift 2 ;;
    --moge-extra-spec=*)    MOGE_EXTRA_PIP_SPECS="${1#*=}"; shift ;;
    --no-moge)              INSTALL_MOGE=0; shift ;;
    --no-check-imports)     CHECK_MODEL_IMPORTS=0; shift ;;
    --model-packages)       MODEL_PIP_PACKAGES="${2:?}"; shift 2 ;;
    --model-packages=*)     MODEL_PIP_PACKAGES="${1#*=}"; shift ;;
    --no-model-packages)    INSTALL_MODEL_PIP=0; MODEL_PIP_PACKAGES=""; shift ;;
    --pip-index-url)        PIP_INDEX_URL="${2:?}"; shift 2 ;;
    --pip-index-url=*)      PIP_INDEX_URL="${1#*=}"; shift ;;
    --no-cache)             NO_CACHE=1; shift ;;
    --no-host-network)      HOST_NETWORK=0; shift ;;
    --npu-devices)          NPU_DEVICES="${2:?}"; shift 2 ;;
    --npu-devices=*)        NPU_DEVICES="${1#*=}"; shift ;;
    --verify)               VERIFY=1; shift ;;
    --push)                 PUSH=1; shift ;;
    --min-free-gb)          MIN_FREE_GB="${2:?}"; shift 2 ;;
    --min-free-gb=*)        MIN_FREE_GB="${1#*=}"; shift ;;
    --dry-run)              DRY_RUN=1; shift ;;
    -h|--help)              usage; exit 0 ;;
    *)                      usage >&2; printf '\n未知参数: %s\n' "$1" >&2; exit 2 ;;
  esac
done

# ------------------------------------------------------------------ 选基础镜像
if [ -z "${BASE_IMAGE}" ]; then
  [ -n "${CHIP}" ] || { usage >&2; printf '\n必须给 --chip 或 --base-image。\n' >&2; exit 2; }
  key="$(printf '%s' "${CHIP}" | tr '[:upper:]' '[:lower:]')"
  BASE_IMAGE="${CHIP_IMAGE[${key}]:-}"
  [ -n "${BASE_IMAGE}" ] || usage_die "未知 --chip '${CHIP}'。支持: ${CHIP_IMAGE_HELP}"
fi

if [ "${CHIP}" = "310p" ] || [ "${CHIP}" = "300i" ] || [ "${CHIP}" = "300iduo" ]; then
  IS_310P=1
else
  IS_310P=0
fi

log "基础镜像: ${BASE_IMAGE}"
log "产出 tag:  ${TAG}"

# ------------------------------------------------------------------ 前置检查
# dry-run 只打印命令，不做任何环境/磁盘/文件检查（本机磁盘必然不够，但 dry-run 依然有用）
ARCH="$(uname -m)"
log "构建机架构: ${ARCH}"
case "${ARCH}" in
  aarch64|arm64) : ;;
  x86_64|amd64)  warn "构建机是 x86_64。昇腾官方镜像是多架构 manifest（arm64+amd64），x86 也能构建，
       但昇腾训练/推理服务器通常是 aarch64（Kunpeng）。如果产物要拷到 Kunpeng 机器上跑，
       必须在同架构机器上构建，否则会得到 amd64 镜像、无法运行。" ;;
  *)             warn "未识别的架构 ${ARCH}，继续但请自行确认镜像 manifest 覆盖该架构。" ;;
esac

if [ "${DRY_RUN}" = "1" ]; then
  log "[dry-run] 跳过 docker / 磁盘 / app 文件检查（只打印命令）"
else
  command -v docker >/dev/null 2>&1 || die "找不到 docker 命令。"
  docker info >/dev/null 2>&1 || die "docker 守护进程不可用（docker info 失败）。"

  DOCKER_ROOT="$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || true)"
  [ -n "${DOCKER_ROOT}" ] || DOCKER_ROOT="/var/lib/docker"
  FREE_GB=$(( $(df -Pk "${DOCKER_ROOT}" | awk 'NR==2{print $4}') / 1024 / 1024 ))
  log "Docker Root Dir: ${DOCKER_ROOT}（可用 ${FREE_GB} GB）"
  # 估算：基础镜像压缩 ~4.6 GB / 展开 20 GB+，权重 ~1.7 GB，pip 依赖 ~1.5 GB
  if [ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]; then
    die "可用磁盘 ${FREE_GB} GB < 要求的 ${MIN_FREE_GB} GB。
   昇腾基础镜像展开后 20 GB+（压缩 4.6 GB），再加 1.7 GB 权重与 pip 依赖。
   * 清理无用镜像/构建缓存后重试（不要动别人的镜像）
   * 或把 docker data-root 迁到大盘
   * 权宜之计：--no-weights 先不 bake 权重（省 1.7 GB），或调低 --min-free-gb
   * 本机（/userdata 只剩 4.6 GB）**不可能**完成该构建，属预期。"
  fi
  [ "${FREE_GB}" -lt 60 ] && warn "可用磁盘 ${FREE_GB} GB，建议预留 60 GB 以上。"

  for f in app/model_hub.py app/service.py app/requirements.txt; do
    [ -e "${PROJECT_DIR}/${f}" ] || die "缺少 ${f}。"
  done
  ok "app/ 关键文件齐全"
fi

# ------------------------------------------------------------------ 组命令
BUILD_ARGS=(
  --file "${PROJECT_DIR}/Dockerfile"
  --target ascend
  --tag "${TAG}"
  --build-arg "ASCEND_BASE_IMAGE=${BASE_IMAGE}"
  --build-arg "FETCH_WEIGHTS=${FETCH_WEIGHTS}"
  --build-arg "STRICT_WEIGHTS=${STRICT_WEIGHTS}"
  --build-arg "HF_ENDPOINT=${HF_ENDPOINT}"
  --build-arg "MOGE_HF_REPO=${MOGE_HF_REPO}"
  --build-arg "SAM_HF_REPO=${SAM_HF_REPO}"
  --build-arg "YOLO_WEIGHTS_URL=${YOLO_WEIGHTS_URL}"
  --build-arg "WEIGHTS_TARBALL_URL=${WEIGHTS_TARBALL_URL}"
  --build-arg "INSTALL_APP_REQUIREMENTS=${INSTALL_APP_REQUIREMENTS}"
  --build-arg "MODEL_PIP_PACKAGES=${MODEL_PIP_PACKAGES}"
  --build-arg "INSTALL_MODEL_PIP=${INSTALL_MODEL_PIP}"
  --build-arg "MOGE_PIP_SPEC=${MOGE_PIP_SPEC}"
  --build-arg "MOGE_EXTRA_PIP_SPECS=${MOGE_EXTRA_PIP_SPECS}"
  --build-arg "INSTALL_MOGE=${INSTALL_MOGE}"
  --build-arg "CHECK_MODEL_IMPORTS=${CHECK_MODEL_IMPORTS}"
  --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}"
)
[ "${HOST_NETWORK}" = "1" ] && BUILD_ARGS+=(--network host)
[ "${NO_CACHE}" = "1" ]    && BUILD_ARGS+=(--no-cache)

if [ "${IS_310P}" = "1" ]; then
  cat >&2 <<'E310P'
----------------------------------------------------------------------
[310P / Atlas 300I Duo 提示]
  310P 不支持 bf16，也不支持图模式。
  MinerU（vllm 后端）的做法是在命令行追加 `--enforce-eager --dtype float16`。
  本服务**没有 vllm**，所以没有等价的 CLI flag：
    * 精度：模型必须用 float16（不要用 bfloat16 / autocast(bf16)）
    * 图模式：必须走 eager（不要用 torch.compile / jit_compile 图模式）
  运行脚本用 `--310p` 会把这两个要求作为 CYLINDER_DTYPE / CYLINDER_EAGER
  传给容器；当前 app 代码尚未读取这两个变量（未验证项，见 docs/ASCEND.md）。
----------------------------------------------------------------------
E310P
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo
  log "[dry-run] 将要执行的命令："
  printf 'docker build'
  printf ' %q' "${BUILD_ARGS[@]}"
  printf ' %q\n' "${PROJECT_DIR}"
  echo
  log "[dry-run] 之后清理构建缓存可执行：docker builder prune（注意：别用 docker system prune -a，会误删别人的镜像）"
  exit 0
fi

# ------------------------------------------------------------------ 构建
log "开始构建（预计下载 4.6 GB 基础镜像 + 1.7 GB 权重，视网络可能很久）"
echo "----------------------------------------------------------------------"
if ! docker build "${BUILD_ARGS[@]}" "${PROJECT_DIR}"; then
  echo "----------------------------------------------------------------------"
  die "ascend 目标构建失败。排查顺序：
   1. 基础镜像能不能拉：docker pull ${BASE_IMAGE}
      （国内可换华为云镜像：swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:<同 tag>）
   2. 磁盘：df -h \$(docker info -f '{{.DockerRootDir}}')
   3. 权重下载：加 --no-weights 先验证镜像本体；或 --hf-endpoint https://hf-mirror.com
   4. PyPI：加 --pip-index-url https://pypi.tuna.tsinghua.edu.cn/simple
   5. MoGe 源码：--no-moge 先跳过，确认别的步骤能过"
fi
echo "----------------------------------------------------------------------"
ok "ascend 目标构建成功"

SIZE_BYTES="$(docker inspect -f '{{.Size}}' "${TAG}" 2>/dev/null || echo 0)"
log "镜像体积: $(docker images --format '{{.Size}}' "${TAG}" | head -1)（$(( SIZE_BYTES / 1024 / 1024 )) MB）"
docker images --format '{{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}' "${TAG}"
log "镜像内基础信息（用于留档）："
docker run --rm --entrypoint /bin/sh "${TAG}" -c \
  'echo "  python: $(python -V 2>&1)"; python -c "import torch,torch_npu;print(\"  torch:\",torch.__version__,\"| torch_npu:\",torch_npu.__version__)" 2>&1 | sed "s/^/  /"; echo "  CANN: ${ASCEND_HOME_PATH:-未设置}"' \
  2>&1 | sed 's/^/  /' || warn "容器内版本自检失败（不影响镜像产物，可稍后手动查）"

# ------------------------------------------------------------------ 可选：NPU 自检
if [ "${VERIFY}" = "1" ]; then
  log "NPU 可用性自检（卡: ${NPU_DEVICES}）"
  DEV_ARGS=()
  IFS=',' read -r -a _cards <<< "${NPU_DEVICES}"
  for c in "${_cards[@]}"; do DEV_ARGS+=("--device=/dev/davinci${c}"); done
  DEV_ARGS+=(--device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc)
  docker run --rm -u root --privileged=true --ipc=host \
    "${DEV_ARGS[@]}" \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /var/log/npu:/usr/slog \
    -e "ASCEND_RT_VISIBLE_DEVICES=${NPU_DEVICES}" \
    "${TAG}" \
    python -c "import torch, torch_npu, os; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu available:', torch.npu.is_available()); print('device count:', torch.npu.device_count())" \
    && ok "NPU 自检通过" \
    || die "NPU 自检失败。先跑 bash scripts/preflight_ascend.sh 看宿主机驱动/设备节点状态。"
fi

# ------------------------------------------------------------------ 可选：push
if [ "${PUSH}" = "1" ]; then
  log "docker push ${TAG}"
  docker push "${TAG}" || die "push 失败。确认已 docker login 到目标仓库，且 tag 前缀与仓库地址匹配。"
  ok "已推送 ${TAG}"
fi

echo
ok "完成。下一步："
printf '  bash scripts/preflight_ascend.sh                          # 上机自检\n'
printf '  bash scripts/run_ascend.sh --image %s --dry-run   # 看运行命令\n' "${TAG}"
printf '  bash scripts/run_ascend.sh --image %s                  # 跑起来\n' "${TAG}"
