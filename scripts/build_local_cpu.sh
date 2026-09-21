#!/usr/bin/env bash
# =============================================================================
#  build_local_cpu.sh —— 本地可行性校验构建（不需要昇腾硬件）
#
#  做三件事：
#    1. 构建 Dockerfile 的 cpu 目标（python:3.11-slim + 最小依赖子集）
#    2. 跑 SPEC 5 要求的 import 冒烟：docker run --rm <tag> python -c "import service, model_hub"
#    3. 真起一个容器，curl GET /healthz（可用 --no-health 跳过）
#
#  可选：--ascend-lint 用一个小基础镜像「顶替」昇腾基础镜像，把 ascend 阶段的
#        Dockerfile 指令（apt / pip 过滤 / COPY --from / compileall）真跑一遍，
#        用来在没有昇腾基础镜像（20 GB+）的情况下尽量做静态+指令级校验。
#        这**不能**证明昇腾可用性，只能证明那段 Dockerfile 语法与路径正确。
#
#  用法：
#    bash scripts/build_local_cpu.sh [选项]
#
#  选项：
#    -t, --tag NAME          cpu 镜像 tag（默认 cylinder-geom:cpu）
#        --port N            /healthz 验证用宿主端口（默认 18000）
#        --no-cache          构建时禁用 layer cache
#        --no-smoke          跳过构建期/运行期 import 冒烟测试
#        --no-health         跳过容器启动 + /healthz 验证
#        --keep-container    验证完不删除容器（排障用）
#        --ascend-lint       额外做 ascend 阶段指令级校验构建
#        --lint-base-image R --ascend-lint 用的替代基础镜像（默认 python:3.11-slim）
#        --no-host-network   构建容器不用 host 网络（默认用，绕开内网 DNS 问题）
#        --min-free-gb N     构建前要求的最小可用磁盘（默认 2）
#    -h, --help             显示本帮助
#
#  退出码：0 成功；1 失败；2 参数错误
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TAG="cylinder-geom:cpu"
LINT_TAG="cylinder-geom:ascend-lint"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
HTTP_PORT="${HTTP_PORT:-18000}"
DO_SMOKE=1
DO_HEALTH=1
DO_ASCEND_LINT=0
KEEP_CONTAINER=0
NO_CACHE=0
HOST_NETWORK=1
MIN_FREE_GB=2
LINT_BASE_IMAGE="python:${PYTHON_VERSION}-slim"

# ------------------------------------------------------------------ 输出
c_red=$'\033[31m'; c_grn=$'\033[32m'; c_yel=$'\033[33m'; c_bld=$'\033[1m'; c_off=$'\033[0m'
log()  { printf '%s[build_local_cpu]%s %s\n'  "$c_bld" "$c_off" "$*"; }
ok()   { printf '%s[ OK ]%s %s\n'             "$c_grn" "$c_off" "$*"; }
warn() { printf '%s[WARN]%s %s\n'             "$c_yel" "$c_off" "$*" >&2; }
die()  { printf '%s[FAIL]%s %s\n'             "$c_red" "$c_off" "$*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
build_local_cpu.sh —— 本地可行性校验构建（不需要昇腾硬件）

做三件事：
  1. 构建 Dockerfile 的 cpu 目标（python:3.11-slim + 最小依赖子集）
  2. 跑 SPEC 5 要求的 import 冒烟：docker run --rm <tag> python -c "import service, model_hub"
  3. 真起一个容器，curl GET /healthz（可用 --no-health 跳过）

可选：--ascend-lint 用一个小基础镜像「顶替」昇腾基础镜像，把 ascend 阶段的
      Dockerfile 指令（apt / pip 过滤 / COPY --from / compileall）真跑一遍。
      这**不能**证明昇腾可用性，只能证明那段 Dockerfile 的语法与路径正确。

用法：
  bash scripts/build_local_cpu.sh [选项]

选项：
  -t, --tag NAME          cpu 镜像 tag（默认 cylinder-geom:cpu）
      --port N            /healthz 验证用宿主端口（默认 18000）
      --no-cache          构建时禁用 layer cache
      --no-smoke          跳过 import 冒烟测试
      --no-health         跳过容器启动 + /healthz 验证
      --keep-container    验证完不删除容器（排障用）
      --ascend-lint       额外做 ascend 阶段指令级校验构建
      --lint-base-image R --ascend-lint 用的替代基础镜像（默认 python:3.11-slim）
      --no-host-network   构建容器不用 host 网络（默认用，绕开内网 DNS 问题）
      --min-free-gb N     构建前要求的最小可用磁盘 GB（默认 2）
  -h, --help              显示本帮助

退出码：0 成功；1 失败；2 参数错误
USAGE
}

# ------------------------------------------------------------------ 参数
while [ $# -gt 0 ]; do
  case "$1" in
    -t|--tag)            TAG="${2:?--tag 需要一个值}"; shift 2 ;;
    --tag=*)             TAG="${1#*=}"; shift ;;
    --port)              HTTP_PORT="${2:?--port 需要一个值}"; shift 2 ;;
    --port=*)            HTTP_PORT="${1#*=}"; shift ;;
    --no-cache)          NO_CACHE=1; shift ;;
    --no-smoke)          DO_SMOKE=0; shift ;;
    --no-health)         DO_HEALTH=0; shift ;;
    --keep-container)    KEEP_CONTAINER=1; shift ;;
    --ascend-lint)       DO_ASCEND_LINT=1; shift ;;
    --lint-base-image)   LINT_BASE_IMAGE="${2:?--lint-base-image 需要一个值}"; shift 2 ;;
    --lint-base-image=*) LINT_BASE_IMAGE="${1#*=}"; shift ;;
    --no-host-network)   HOST_NETWORK=0; shift ;;
    --min-free-gb)       MIN_FREE_GB="${2:?--min-free-gb 需要一个值}"; shift 2 ;;
    --min-free-gb=*)     MIN_FREE_GB="${1#*=}"; shift ;;
    -h|--help)           usage; exit 0 ;;
    *)                   usage >&2; printf '\n未知参数: %s\n' "$1" >&2; exit 2 ;;
  esac
done

# ------------------------------------------------------------------ 前置检查
command -v docker >/dev/null 2>&1 || die "找不到 docker 命令。"
docker info >/dev/null 2>&1 || die "docker 守护进程不可用（docker info 失败）。可能是权限或 daemon 未启动：sudo systemctl status docker"

[ -f "${PROJECT_DIR}/Dockerfile" ] || die "找不到 ${PROJECT_DIR}/Dockerfile"

log "构建上下文: ${PROJECT_DIR}"
for f in app/model_hub.py app/service.py app/requirements.txt app/models; do
  [ -e "${PROJECT_DIR}/${f}" ] || die "缺少 ${f}。app/ 由 Python 侧交付，缺失时无法做 import 冒烟测试。"
done
ok "app/ 关键文件齐全"

DOCKER_ROOT="$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || true)"
[ -n "${DOCKER_ROOT}" ] || DOCKER_ROOT="/var/lib/docker"
FREE_GB=$(( $(df -Pk "${DOCKER_ROOT}" | awk 'NR==2{print $4}') / 1024 / 1024 ))
log "Docker Root Dir: ${DOCKER_ROOT}（可用 ${FREE_GB} GB）"
if [ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]; then
  die "可用磁盘 ${FREE_GB} GB < 要求的 ${MIN_FREE_GB} GB。请先清理无用的构建缓存/镜像，或用 --min-free-gb 调低阈值。"
fi
[ "${FREE_GB}" -lt 4 ] && warn "可用磁盘只剩 ${FREE_GB} GB，构建过程中可能吃紧。"

# ------------------------------------------------------------------ 构建 cpu target
BUILD_ARGS=(
  --file "${PROJECT_DIR}/Dockerfile"
  --target cpu
  --tag "${TAG}"
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
  --build-arg "SMOKE_TEST=${DO_SMOKE}"
)
[ "${HOST_NETWORK}" = "1" ] && BUILD_ARGS+=(--network host)
[ "${NO_CACHE}" = "1" ]    && BUILD_ARGS+=(--no-cache)

log "docker build --target cpu -> ${TAG}"
log "命令: docker build ${BUILD_ARGS[*]} ${PROJECT_DIR}"
echo "----------------------------------------------------------------------"
if ! docker build "${BUILD_ARGS[@]}" "${PROJECT_DIR}"; then
  echo "----------------------------------------------------------------------"
  die "cpu 目标构建失败。常见原因：
   * 磁盘不足（见上面的 df / no space left on device）—— 用 df -h 确认 ${DOCKER_ROOT}
   * PyPI 不可达（内网环境）—— 加 --build-arg 传代理，或改用内网 PyPI
   * app/ 里有语法错误 —— 构建末尾的 import 冒烟会直接暴露
   排障时可加 --no-cache 并去掉 --no-smoke 看完整输出。"
fi
echo "----------------------------------------------------------------------"
ok "cpu 目标构建成功"

# ------------------------------------------------------------------ 体积证据
SIZE_BYTES="$(docker inspect -f '{{.Size}}' "${TAG}" 2>/dev/null || echo 0)"
SIZE_HUMAN="$(docker images --format '{{.Size}}' "${TAG}" 2>/dev/null | head -1 || echo '?')"
SIZE_MB=$(( SIZE_BYTES / 1024 / 1024 ))
log "镜像体积: ${SIZE_HUMAN}（${SIZE_MB} MB，${SIZE_BYTES} bytes）"
if [ "${SIZE_MB}" -ge 1536 ]; then
  warn "SPEC 2.1 要求 cpu 目标 < 1.5 GB，当前 ${SIZE_MB} MB 超出。"
else
  ok "满足 SPEC 2.1 的体积要求（< 1.5 GB）"
fi
docker images --format '{{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}  (created {{.CreatedSince}})' "${TAG}"

# ------------------------------------------------------------------ SPEC 5 冒烟
if [ "${DO_SMOKE}" = "1" ]; then
  log "SPEC 5 冒烟: docker run --rm ${TAG} python -c \"import service, model_hub; print('import ok')\""
  if docker run --rm "${TAG}" python -c "import service, model_hub; print('import ok')"; then
    ok "import 冒烟通过"
  else
    die "import 冒烟失败：cpu 镜像里 service / model_hub 无法 import。
   排查：docker run --rm -it ${TAG} bash，然后逐个 import 看 traceback。"
  fi
else
  warn "已跳过 import 冒烟测试（--no-smoke）"
fi

# ------------------------------------------------------------------ /healthz 验证
CONTAINER_NAME="cylinder-geom-cpu-verify-$$"
if [ "${DO_HEALTH}" = "1" ]; then
  command -v curl >/dev/null 2>&1 || warn "宿主机没有 curl，将用 python3 代替（功能等价）"
  log "启动容器做 /healthz 验证：宿主端口 ${HTTP_PORT} -> 容器 8000"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker run -d --name "${CONTAINER_NAME}" -p "127.0.0.1:${HTTP_PORT}:8000" "${TAG}" >/dev/null

  HEALTH_OK=0
  for i in $(seq 1 30); do
    if command -v curl >/dev/null 2>&1; then
      BODY="$(curl -fsS --max-time 3 "http://127.0.0.1:${HTTP_PORT}/healthz" 2>/dev/null || true)"
    else
      BODY="$(python3 -c "
import urllib.request,sys
try:
    print(urllib.request.urlopen('http://127.0.0.1:${HTTP_PORT}/healthz', timeout=3).read().decode())
except Exception:
    sys.exit(1)
" 2>/dev/null || true)"
    fi
    if [ -n "${BODY}" ]; then HEALTH_OK=1; break; fi
    sleep 1
  done

  if [ "${HEALTH_OK}" = "1" ]; then
    ok "GET /healthz -> 200"
    printf '响应体: %s\n' "${BODY}"
    printf '镜像内 HEALTHCHECK 状态: %s\n' \
      "$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}(未定义){{end}}' "${CONTAINER_NAME}" 2>/dev/null || echo '?')"
  else
    warn "30 秒内没能从 /healthz 拿到响应。容器日志："
    docker logs --tail 40 "${CONTAINER_NAME}" 2>&1 || true
    if [ "${KEEP_CONTAINER}" != "1" ]; then docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true; fi
    die "/healthz 验证失败。若 app/service.py 正在被改动，可加 --no-health 跳过并把原因写进文档。"
  fi

  if [ "${KEEP_CONTAINER}" = "1" ]; then
    warn "容器保留：${CONTAINER_NAME}（清理：docker rm -f ${CONTAINER_NAME}）"
  else
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    log "已删除验证容器 ${CONTAINER_NAME}"
  fi
else
  warn "已跳过 /healthz 验证（--no-health）"
fi

# ------------------------------------------------------------------ ascend 指令级 lint
if [ "${DO_ASCEND_LINT}" = "1" ]; then
  log "ascend 阶段指令级校验：用 ${LINT_BASE_IMAGE} 顶替昇腾基础镜像"
  warn "这只验证 Dockerfile 语法/路径/指令，**不能**证明昇腾可用性（我们没有昇腾机器）。"
  LINT_ARGS=(
    --file "${PROJECT_DIR}/Dockerfile"
    --target ascend
    --tag "${LINT_TAG}"
    --build-arg "ASCEND_BASE_IMAGE=${LINT_BASE_IMAGE}"
    --build-arg "FETCH_WEIGHTS=0"
    --build-arg "INSTALL_MOGE=0"
    --build-arg "INSTALL_MODEL_PIP=0"
    --build-arg "SMOKE_TEST=1"
  )
  [ "${HOST_NETWORK}" = "1" ] && LINT_ARGS+=(--network host)
  [ "${NO_CACHE}" = "1" ]    && LINT_ARGS+=(--no-cache)
  if docker build "${LINT_ARGS[@]}" "${PROJECT_DIR}"; then
    ok "ascend 阶段指令级校验通过（镜像 tag ${LINT_TAG}）"
  else
    die "ascend 阶段指令级校验失败 —— 说明 Dockerfile 的 ascend 段落有问题，必须修。"
  fi
fi

echo
ok "本地校验完成。下一步："
printf '  docker run --rm %s python -c "import service, model_hub; print(\x27import ok\x27)"\n' "${TAG}"
printf '  bash scripts/build_ascend.sh --chip 910b --dry-run     # 看昇腾机的正式构建命令\n'
