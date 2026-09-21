#!/usr/bin/env bash
# =============================================================================
#  run_ascend.sh —— 在昇腾机器上启动服务容器
#
#  设备挂载方式对齐 MinerU 的昇腾运行方式（官方文档的 docker run 清单）：
#    --privileged=true --ipc=host --network=host
#    --device=/dev/davinci<N> --device=/dev/davinci_manager
#    --device=/dev/devmm_svm  --device=/dev/hisi_hdc
#    -v /usr/local/dcmi:/usr/local/dcmi
#    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
#    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
#    -v /var/log/npu/:/usr/slog
#    -e ASCEND_RT_VISIBLE_DEVICES=<卡号>
#
#  设备号与 ASCEND_RT_VISIBLE_DEVICES 都可以参数化，没有写死。
#
#  用法：
#    bash scripts/run_ascend.sh --devices 0
#    bash scripts/run_ascend.sh --devices 3 --rt-visible-devices 3
#    bash scripts/run_ascend.sh --devices 0,1 --weights /data/cylinder/weights
#    bash scripts/run_ascend.sh --310p            # Atlas 300I Duo
#    bash scripts/run_ascend.sh --dry-run         # 只打印命令（本地可验证）
#
#  退出码：0 启动成功；1 失败；2 参数错误
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-cylinder-geom:ascend}"
NAME="${NAME:-cylinder-geom}"
NPU_DEVICES="${NPU_DEVICES:-0}"
RT_VISIBLE="${ASCEND_RT_VISIBLE_DEVICES:-}"
HTTP_PORT="${HTTP_PORT:-8000}"
WEIGHTS_HOST_DIR=""
HF_CACHE_HOST_DIR=""
NETWORK_MODE="${NETWORK_MODE:-host}"
PRIVILEGED=1
DETACH=1
AUTO_REMOVE=0
IS_310P=0
SKIP_DEVICE_CHECK=0
DRY_RUN=0
EXTRA_ARGS=()
WEIGHTS_IN_CONTAINER="/app/weights"

c_red=$'\033[31m'; c_grn=$'\033[32m'; c_yel=$'\033[33m'; c_bld=$'\033[1m'; c_off=$'\033[0m'
log()  { printf '%s[run_ascend]%s %s\n' "$c_bld" "$c_off" "$*"; }
ok()   { printf '%s[ OK ]%s %s\n'        "$c_grn" "$c_off" "$*"; }
warn() { printf '%s[WARN]%s %s\n'        "$c_yel" "$c_off" "$*" >&2; }
die()  { printf '%s[FAIL]%s %s\n'        "$c_red" "$c_off" "$*" >&2; exit 1; }
usage_die() { usage >&2; printf '\n%s[FAIL]%s %s\n' "$c_red" "$c_off" "$*" >&2; exit 2; }

usage() {
  cat <<'USAGE'
run_ascend.sh —— 启动昇腾服务容器

用法：
  bash scripts/run_ascend.sh [选项]

选项：
  -i, --image REF          镜像（默认 cylinder-geom:ascend）
  -n, --name NAME          容器名（默认 cylinder-geom）
      --devices LIST       昇腾卡号，逗号分隔（默认 0）。例如 --devices 0,1,2
                          -> 生成 --device=/dev/davinci0/1/2
      --rt-visible-devices LIST
                           容器内 ASCEND_RT_VISIBLE_DEVICES（默认 = --devices）
      --port N             宿主机端口（默认 8000）。仅在 --network bridge 下生效
      --weights DIR        用宿主机目录覆盖 /app/weights（只读挂载）
      --hf-cache DIR       挂载 HF 缓存目录（权重只读挂载时需要可写缓存）
      --network MODE       host（默认）或 bridge
      --no-privileged      不加 --privileged（默认加，昇腾上通常必须）
      --foreground         前台运行（不加 -d）
      --rm                 容器退出后自动删除
      --310p               Atlas 300I Duo（310P）模式：打印 fp16/eager 要求并传对应变量
      --skip-device-check  跳过宿主机设备节点检查（排障用）
      --extra "ARGS"       追加任意 docker run 参数（会按空格拆分）
      --dry-run            只打印将要执行的 docker run 命令
  -h, --help               显示本帮助

310P（Atlas 300I Duo）注意：
  310P 不支持 bf16、不支持图模式。MinerU 用 vllm，所以加
  `--enforce-eager --dtype float16`；本服务没有 vllm，**没有等价的 CLI flag**。
  本脚本改用环境变量表达这两个要求（CYLINDER_DTYPE / CYLINDER_EAGER），
  但当前 app 代码并未读取它们 —— 真正的控制点在模型加载处（见 docs/ASCEND.md）。

退出码：0 成功；1 失败；2 参数错误
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    -i|--image)              IMAGE="${2:?--image 需要一个值}"; shift 2 ;;
    --image=*)               IMAGE="${1#*=}"; shift ;;
    -n|--name)               NAME="${2:?--name 需要一个值}"; shift 2 ;;
    --name=*)                NAME="${1#*=}"; shift ;;
    --devices)               NPU_DEVICES="${2:?--devices 需要一个值}"; shift 2 ;;
    --devices=*)             NPU_DEVICES="${1#*=}"; shift ;;
    --rt-visible-devices)    RT_VISIBLE="${2:?}"; shift 2 ;;
    --rt-visible-devices=*)  RT_VISIBLE="${1#*=}"; shift ;;
    --port)                  HTTP_PORT="${2:?}"; shift 2 ;;
    --port=*)                HTTP_PORT="${1#*=}"; shift ;;
    --weights)               WEIGHTS_HOST_DIR="${2:?}"; shift 2 ;;
    --weights=*)             WEIGHTS_HOST_DIR="${1#*=}"; shift ;;
    --hf-cache)              HF_CACHE_HOST_DIR="${2:?}"; shift 2 ;;
    --hf-cache=*)            HF_CACHE_HOST_DIR="${1#*=}"; shift ;;
    --network)               NETWORK_MODE="${2:?}"; shift 2 ;;
    --network=*)             NETWORK_MODE="${1#*=}"; shift ;;
    --no-privileged)         PRIVILEGED=0; shift ;;
    --foreground)            DETACH=0; shift ;;
    --rm)                    AUTO_REMOVE=1; shift ;;
    --310p)                  IS_310P=1; shift ;;
    --skip-device-check)     SKIP_DEVICE_CHECK=1; shift ;;
    --extra)                 read -r -a _e <<< "${2:?}"; EXTRA_ARGS+=("${_e[@]}"); shift 2 ;;
    --extra=*)               read -r -a _e <<< "${1#*=}"; EXTRA_ARGS+=("${_e[@]}"); shift ;;
    --dry-run)               DRY_RUN=1; shift ;;
    -h|--help)               usage; exit 0 ;;
    *)                       usage >&2; printf '\n未知参数: %s\n' "$1" >&2; exit 2 ;;
  esac
done

# ------------------------------------------------------------------ 参数规范化
# 卡号列表 -> 数组；去掉空白
NPU_DEVICES="$(printf '%s' "${NPU_DEVICES}" | tr -d '[:space:]')"
[ -n "${NPU_DEVICES}" ] || usage_die "--devices 不能为空。"
IFS=',' read -r -a CARDS <<< "${NPU_DEVICES}"
for c in "${CARDS[@]}"; do
  case "${c}" in
    ''|*[!0-9]*) usage_die "--devices 里 '${c}' 不是合法的卡号（必须是 0-9 的数字，逗号分隔）。示例：--devices 0,1" ;;
  esac
done
# 默认：容器内可见卡 = 挂进来的卡
[ -n "${RT_VISIBLE}" ] || RT_VISIBLE="${NPU_DEVICES}"

case "${NETWORK_MODE}" in
  host|bridge|none|default) : ;;
  *) usage_die "--network '${NETWORK_MODE}' 不支持（host / bridge / none）。" ;;
esac

# ------------------------------------------------------------------ 前置检查
if [ "${DRY_RUN}" != "1" ]; then
  command -v docker >/dev/null 2>&1 || die "找不到 docker 命令。"
  docker info >/dev/null 2>&1 || die "docker 守护进程不可用（docker info 失败）。"
  docker image inspect "${IMAGE}" >/dev/null 2>&1 \
    || die "本地没有镜像 ${IMAGE}。先在昇腾机器上构建：bash scripts/build_ascend.sh --chip 910b"
fi

if [ "${SKIP_DEVICE_CHECK}" != "1" ] && [ "${DRY_RUN}" != "1" ]; then
  MISSING=()
  for c in "${CARDS[@]}"; do
    [ -c "/dev/davinci${c}" ] || MISSING+=("/dev/davinci${c}")
  done
  for d in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    [ -e "${d}" ] || MISSING+=("${d}")
  done
  if [ "${#MISSING[@]}" -gt 0 ]; then
    die "宿主机缺少昇腾设备节点：${MISSING[*]}
   这说明当前机器不是（或没装好）昇腾机器。请：
     1. bash scripts/preflight_ascend.sh      # 完整自检
     2. 确认已安装昇腾驱动并加载：lsmod | grep -E 'davinci|drv_davinci'
     3. 容器/虚拟机内请确认设备已透传
   确实要跳过检查（例如先看命令长什么样）：--skip-device-check"
  fi
  DRIVER_DIR=/usr/local/Ascend/driver
  [ -d "${DRIVER_DIR}" ] || die "宿主机没有 ${DRIVER_DIR}。昇腾驱动未安装或路径不同，请确认后再运行。"
  [ -e /usr/local/bin/npu-smi ] || warn "宿主机没有 /usr/local/bin/npu-smi，容器内将无法执行 npu-smi info。"
  [ -d /usr/local/dcmi ]       || warn "宿主机没有 /usr/local/dcmi，部分 dcmi 能力不可用。"
  [ -d /var/log/npu ]          || warn "宿主机没有 /var/log/npu，挂载 /usr/slog 会失败。可先 sudo mkdir -p /var/log/npu。"
  ok "宿主机设备节点检查通过（卡: ${NPU_DEVICES}）"
fi

# ------------------------------------------------------------------ 310P 提示
if [ "${IS_310P}" = "1" ]; then
  cat >&2 <<'E310P'
----------------------------------------------------------------------
[310P / Atlas 300I Duo 模式]
  310P 不支持 bf16、不支持图模式。
  MinerU（vllm 后端）的方式：追加 `--enforce-eager --dtype float16`。
  本服务**没有 vllm**，uvicorn 也没有这类 flag，所以脚本改传：
      -e CYLINDER_DTYPE=float16
      -e CYLINDER_EAGER=1
  请务必在**模型侧**落实这两点：
      * 权重/推理用 float16，不要 bfloat16、不要 autocast(bfloat16)
      * 走 eager，不要 torch.compile / 图模式
  （当前 app 代码尚未读取这两个变量，属未验证项，见 docs/ASCEND.md）
----------------------------------------------------------------------
E310P
fi

# ------------------------------------------------------------------ 组装 docker run
RUN_ARGS=(--name "${NAME}")
RUN_ARGS+=(-u root)
[ "${DETACH}" = "1" ]     && RUN_ARGS+=(-d)
[ "${AUTO_REMOVE}" = "1" ] && RUN_ARGS+=(--rm)
[ "${PRIVILEGED}" = "1" ] && RUN_ARGS+=(--privileged=true)
RUN_ARGS+=(--ipc=host)
RUN_ARGS+=(--network "${NETWORK_MODE}")

for c in "${CARDS[@]}"; do
  RUN_ARGS+=("--device=/dev/davinci${c}")
done
RUN_ARGS+=(--device=/dev/davinci_manager)
RUN_ARGS+=(--device=/dev/devmm_svm)
RUN_ARGS+=(--device=/dev/hisi_hdc)

RUN_ARGS+=(-v /usr/local/dcmi:/usr/local/dcmi)
RUN_ARGS+=(-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi)
RUN_ARGS+=(-v /usr/local/Ascend/driver:/usr/local/Ascend/driver)
RUN_ARGS+=(-v /var/log/npu/:/usr/slog)

[ -n "${WEIGHTS_HOST_DIR}" ] && RUN_ARGS+=(-v "${WEIGHTS_HOST_DIR}:${WEIGHTS_IN_CONTAINER}:ro")
[ -n "${HF_CACHE_HOST_DIR}" ] && RUN_ARGS+=(-v "${HF_CACHE_HOST_DIR}:/root/.cache/huggingface")

RUN_ARGS+=(-e "ASCEND_RT_VISIBLE_DEVICES=${RT_VISIBLE}")
RUN_ARGS+=(-e "WEIGHTS_DIR=${WEIGHTS_IN_CONTAINER}")
RUN_ARGS+=(-e "PYTHONUNBUFFERED=1")
RUN_ARGS+=(-e "LOG_LEVEL=${LOG_LEVEL:-INFO}")
RUN_ARGS+=(-e "MAX_RESIDENT_MODELS=${MAX_RESIDENT_MODELS:-1}")
if [ "${IS_310P}" = "1" ]; then
  RUN_ARGS+=(-e "CYLINDER_DTYPE=float16")
  RUN_ARGS+=(-e "CYLINDER_EAGER=1")
fi

# host 网络下 -p 无效（docker 会警告），只在 bridge 下加端口映射
if [ "${NETWORK_MODE}" != "host" ]; then
  RUN_ARGS+=(-p "${HTTP_PORT}:8000")
fi

[ "${#EXTRA_ARGS[@]}" -gt 0 ] && RUN_ARGS+=("${EXTRA_ARGS[@]}")

RUN_ARGS+=("${IMAGE}")

# ------------------------------------------------------------------ 执行
log "卡: ${NPU_DEVICES}  容器内可见卡: ASCEND_RT_VISIBLE_DEVICES=${RT_VISIBLE}"
log "网络: ${NETWORK_MODE}  镜像: ${IMAGE}  容器名: ${NAME}"
[ -n "${WEIGHTS_HOST_DIR}" ] && log "权重: 用 ${WEIGHTS_HOST_DIR} 覆盖镜像内 ${WEIGHTS_IN_CONTAINER}（只读）"

if [ "${DRY_RUN}" = "1" ]; then
  echo
  log "[dry-run] 将要执行的命令："
  printf 'docker run'
  printf ' %q' "${RUN_ARGS[@]}"
  printf '\n'
  echo
  log "[dry-run] 未做任何检查、未启动容器。"
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  die "已存在同名容器 ${NAME}。先删除：docker rm -f ${NAME}  或用 --name 换名字。"
fi

if ! docker run "${RUN_ARGS[@]}"; then
  die "docker run 失败。排查：
   * 设备节点权限/存在性：bash scripts/preflight_ascend.sh
   * 端口占用（bridge 模式）：ss -lntp | grep ${HTTP_PORT}
   * 镜像内启动日志：docker logs ${NAME}
   * 驱动版本与镜像 CANN 版本不匹配是最常见原因（见 docs/ASCEND.md 的版本兼容表）"
fi

if [ "${DETACH}" = "1" ]; then
  ok "容器已启动：${NAME}"
  log "端口: ${HTTP_PORT}（host 网络模式下即容器 8000）"
  printf '\n查看日志:   docker logs -f %s\n' "${NAME}"
  printf '健康检查:   curl -fsS http://127.0.0.1:%s/healthz\n' "${HTTP_PORT}"
  printf '任务列表:   curl -fsS http://127.0.0.1:%s/v1/tasks\n' "${HTTP_PORT}"
  printf '卸载模型:   curl -fsS -X POST http://127.0.0.1:%s/v1/unload\n' "${HTTP_PORT}"
  printf '进容器:     docker exec -it %s bash   # 里面可以跑 npu-smi info\n' "${NAME}"
  printf '停掉:       docker rm -f %s\n\n' "${NAME}"
  log "等 20-60 秒让服务起来（MoGe/SAM 首次调用才加载，启动本身很快）"
fi
