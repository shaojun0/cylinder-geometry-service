#!/usr/bin/env bash
# =============================================================================
#  preflight_ascend.sh —— 昇腾上机自检
#
#  在**昇腾机器**上、构建或启动容器之前跑这个脚本，确认：
#    * npu-smi info 可用、能看到卡和驱动版本
#    * /dev/davinci*、/dev/davinci_manager、/dev/devmm_svm、/dev/hisi_hdc 存在
#    * /usr/local/Ascend/driver、/usr/local/dcmi、/usr/local/bin/npu-smi 可读可用
#    * docker 可用、docker 版本够、docker root 剩余空间够装 20 GB+ 的昇腾镜像
#
#  每一项都有 PASS / WARN / FAIL 和明确的补救提示。
#
#  用法：
#    bash scripts/preflight_ascend.sh
#    bash scripts/preflight_ascend.sh --image cylinder-geom:ascend
#    bash scripts/preflight_ascend.sh --report-only   # 只出报告，永远 exit 0
#
#  退出码：0 全部通过（允许有 WARN）；1 存在 FAIL；2 参数错误
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE=""
REPORT_ONLY=0
STRICT=0
MIN_FREE_GB="${MIN_FREE_GB:-45}"

PASS_N=0; WARN_N=0; FAIL_N=0
HINTS=()

c_red=$'\033[31m'; c_grn=$'\033[32m'; c_yel=$'\033[33m'; c_bld=$'\033[1m'; c_off=$'\033[0m'
p()  { printf '%s[PASS]%s %s\n' "$c_grn" "$c_off" "$*"; PASS_N=$((PASS_N+1)); }
w()  { printf '%s[WARN]%s %s\n' "$c_yel" "$c_off" "$*"; WARN_N=$((WARN_N+1)); HINTS+=("WARN: $*"); }
f()  { printf '%s[FAIL]%s %s\n' "$c_red" "$c_off" "$*"; FAIL_N=$((FAIL_N+1)); HINTS+=("FAIL: $*"); }
sec(){ printf '\n%s== %s ==%s\n' "$c_bld" "$*" "$c_off"; }

usage() {
  cat <<'USAGE'
preflight_ascend.sh —— 昇腾上机自检

用法：
  bash scripts/preflight_ascend.sh [选项]

选项：
      --image REF        额外检查本地是否存在该镜像
      --report-only      只出报告，无论 FAIL 与否都 exit 0
      --strict           把 WARN 也当失败（exit 1）
      --min-free-gb N    docker root 需要的最小可用空间 GB（默认 45）
  -h, --help             显示本帮助

退出码：0 通过；1 有 FAIL；2 参数错误
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --image)             IMAGE="${2:?}"; shift 2 ;;
    --image=*)           IMAGE="${1#*=}"; shift ;;
    --report-only)       REPORT_ONLY=1; shift ;;
    --strict)            STRICT=1; shift ;;
    --min-free-gb)       MIN_FREE_GB="${2:?}"; shift 2 ;;
    --min-free-gb=*)     MIN_FREE_GB="${1#*=}"; shift ;;
    -h|--help)           usage; exit 0 ;;
    *)                   usage >&2; printf '\n未知参数: %s\n' "$1" >&2; exit 2 ;;
  esac
done

printf '%s' "$c_bld"
echo "======================================================================"
echo " 昇腾上机自检  ($(date '+%F %T'))"
echo " 主机: $(hostname)   架构: $(uname -m)   内核: $(uname -r)"
echo "======================================================================"
printf '%s' "$c_off"

# ------------------------------------------------------------------ 架构
sec "架构"
ARCH="$(uname -m)"
case "${ARCH}" in
  aarch64|arm64)
    p "架构 ${ARCH}（Kunpeng / Ascend 服务器常见架构）" ;;
  x86_64|amd64)
    w "架构 ${ARCH}。昇腾官方镜像有多架构 manifest，x86 也能跑，但绝大多数 Atlas 服务器是 aarch64；
       如果容器里 torch_npu 报 'no kernel image' 之类错误，先确认架构一致。" ;;
  *)
    w "架构 ${ARCH} 未在预期列表里，请自行确认昇腾镜像 manifest 支持该架构。" ;;
esac

# ------------------------------------------------------------------ Docker
sec "Docker"
if command -v docker >/dev/null 2>&1; then
  p "docker 命令存在：$(command -v docker)"
  if docker info >/dev/null 2>&1; then
    DOCKER_VER="$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo '?')"
    p "docker 守护进程可用，Server 版本 ${DOCKER_VER}"
    # --device 很老就有；--privileged / --ipc=host 同理。这里只要求别太旧。
    MAJOR="${DOCKER_VER%%.*}"
    if [ "${MAJOR}" -ge 19 ] 2>/dev/null; then
      p "docker 版本满足多阶段构建 + --device 挂载（>= 19.03）"
    else
      w "docker 版本 ${DOCKER_VER} 偏旧，多阶段构建的 COPY --from / --target 需要 >= 17.05，请确认。"
    fi
    DOCKER_ROOT="$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo '')"
    if [ -n "${DOCKER_ROOT}" ]; then
      FREE_GB=$(( $(df -Pk "${DOCKER_ROOT}" | awk 'NR==2{print $4}') / 1024 / 1024 ))
      if [ "${FREE_GB}" -ge "${MIN_FREE_GB}" ]; then
        p "Docker Root Dir ${DOCKER_ROOT} 可用 ${FREE_GB} GB（要求 >= ${MIN_FREE_GB} GB）"
      else
        f "Docker Root Dir ${DOCKER_ROOT} 只有 ${FREE_GB} GB 可用，低于要求的 ${MIN_FREE_GB} GB。
       昇腾基础镜像展开后 20 GB+（压缩 4.6 GB），还要 1.7 GB 权重 + pip 依赖。
       补救：清理无用镜像/构建缓存（**别用 docker system prune -a**，会误删别人的镜像），
             或迁移 data-root：/etc/docker/daemon.json 里 \"data-root\": \"/大盘/docker\" 后重启 docker，
             或先用 --no-weights 构建。"
      fi
    fi
    if docker info -f '{{.Driver}}' >/dev/null 2>&1; then
      p "存储驱动：$(docker info -f '{{.Driver}}' 2>/dev/null)"
    fi
    for img in "${IMAGE:-}"; do
      [ -n "${img}" ] || continue
      if docker image inspect "${img}" >/dev/null 2>&1; then
        p "本地存在镜像 ${img}（$(docker images --format '{{.Size}}' "${img}" | head -1)）"
      else
        w "本地没有镜像 ${img}，先跑 bash scripts/build_ascend.sh --chip <型号>"
      fi
    done
  else
    f "docker 守护进程不可用（docker info 失败）。
       补救：sudo systemctl status docker / sudo systemctl start docker；
             若当前用户不在 docker 组：sudo usermod -aG docker \$USER 后重新登录。"
  fi
else
  f "找不到 docker 命令。宿主机需要安装 docker（昇腾容器必须有 docker 才能跑）。"
fi

# ------------------------------------------------------------------ 驱动 / npu-smi
sec "昇腾驱动与 npu-smi"
NPU_SMI=""
if command -v npu-smi >/dev/null 2>&1; then
  NPU_SMI="$(command -v npu-smi)"
elif [ -x /usr/local/bin/npu-smi ]; then
  NPU_SMI=/usr/local/bin/npu-smi
fi
if [ -n "${NPU_SMI}" ]; then
  p "npu-smi 存在：${NPU_SMI}"
  NPU_OUT="$(timeout 25 "${NPU_SMI}" info 2>&1 || true)"
  if [ -n "${NPU_OUT}" ]; then
    if printf '%s' "${NPU_OUT}" | grep -qiE 'error|fail|not found|no such|permission'; then
      f "npu-smi info 输出里有错误字样：
$(printf '%s' "${NPU_OUT}" | head -15 | sed 's/^/       /')
       补救：确认驱动已加载（lsmod | grep -E 'davinci|drv_davinci'）；
             确认当前用户在设备节点的访问权限（ls -l /dev/davinci*）；
             或需要 root：sudo ${NPU_SMI} info。"
    else
      p "npu-smi info 可执行并返回内容（摘录）："
      printf '%s\n' "${NPU_OUT}" | head -20 | sed 's/^/       /'
      CARDS_SEEN="$(printf '%s' "${NPU_OUT}" | grep -cE '^\|?[[:space:]]*[0-9]+[[:space:]]' || true)"
      [ "${CARDS_SEEN}" -gt 0 ] && p "npu-smi 报告了 ${CARDS_SEEN} 行卡信息"
    fi
  else
    f "npu-smi info 没有任何输出。驱动可能未加载，或当前用户无权限。"
  fi
else
  f "找不到 npu-smi（npu-smi 与 /usr/local/bin/npu-smi 都不存在）。
     说明这台机器没装昇腾驱动，或不是昇腾机器。
     补救：安装 Atlas 驱动 + 固件（Ascend HDK），然后确认 npu-smi info 能看到卡。"
fi

DRIVER_DIR=/usr/local/Ascend/driver
if [ -d "${DRIVER_DIR}" ]; then
  if [ -r "${DRIVER_DIR}" ]; then
    p "${DRIVER_DIR} 存在且可读"
  else
    f "${DRIVER_DIR} 存在但当前用户不可读（容器需要 -v 挂进去）。
       补救：sudo chmod -R a+rX ${DRIVER_DIR} 或用 root 运行 docker。"
  fi
  if [ -f "${DRIVER_DIR}/version.info" ]; then
    VER_LINE="$(head -5 "${DRIVER_DIR}/version.info" 2>/dev/null | tr '\n' ' ' || true)"
    if [ -r "${DRIVER_DIR}/version.info" ]; then
      p "驱动版本信息：${VER_LINE}"
    else
      w "驱动版本文件存在但不可读：${DRIVER_DIR}/version.info（sudo 才能读？）"
    fi
  else
    w "没找到 ${DRIVER_DIR}/version.info，无法确认驱动版本。
       请人工确认驱动版本与镜像内 CANN 版本匹配（见 docs/ASCEND.md 兼容性表）。"
  fi
  [ -d "${DRIVER_DIR}/lib64" ] && p "${DRIVER_DIR}/lib64 存在（LD_LIBRARY_PATH 需要它）" \
                               || w "缺少 ${DRIVER_DIR}/lib64，容器内可能找不到 libascend_hal.so。"
else
  f "找不到 ${DRIVER_DIR}。昇腾驱动未安装或安装路径不同。
     补救：安装 Ascend HDK 驱动；装完后确认 ls ${DRIVER_DIR}/lib64 有内容。
     如果驱动装在别的路径，请调整 run_ascend.sh / docker-compose.yml 里的挂载源。"
fi

if [ -d /usr/local/dcmi ]; then
  p "/usr/local/dcmi 存在"
  [ -e /usr/local/dcmi/libdcmi.so ] && p "/usr/local/dcmi/libdcmi.so 存在" \
                                    || w "/usr/local/dcmi 下没有 libdcmi.so，DCMI 相关能力可能不可用。"
else
  w "找不到 /usr/local/dcmi（npu-smi 的部分功能依赖它）。容器挂载时会失败，请确认路径。"
fi

# ------------------------------------------------------------------ 设备节点
sec "设备节点"
SHOWN="$(ls -1 /dev/davinci* 2>/dev/null | head -20 || true)"
if [ -n "${SHOWN}" ]; then
  p "发现 davinci 设备节点："
  printf '%s\n' "${SHOWN}" | sed 's/^/       /'
  IDS="$(ls -1 /dev/davinci[0-9]* 2>/dev/null | sed 's#.*/dev/davinci##' | sort -n | tr '\n' ',' | sed 's/,$//' || true)"
  if [ -n "${IDS}" ]; then
    p "可用卡号：${IDS}（run_ascend.sh --devices ${IDS%%,*} 取第一张，或 --devices ${IDS}）"
  else
    w "只看到 /dev/davinci* 但没有 /dev/davinci<数字>。确认驱动是否正常加载。"
  fi
else
  f "没有任何 /dev/davinci* 节点。这台机器要么不是昇腾机器，要么驱动没加载。
     补救：lsmod | grep -E 'davinci|drv_davinci'；dmesg | tail -50 看驱动加载错误；
           容器/虚机里请确认设备已透传。"
fi

NODE_HINT_davinci_manager="davinci_manager 由驱动提供，缺失说明驱动未装好/未加载；它负责设备管理，缺了容器里 torch_npu 初始化会失败。"
NODE_HINT_devmm_svm="devmm_svm 是设备内存管理节点，缺失通常意味着驱动/固件没装完整。"
NODE_HINT_hisi_hdc="hisi_hdc 是 host-device 通信节点；310P/Atlas 推理卡上缺失会导致设备无法被容器识别。"
for d in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  if [ -e "${d}" ]; then
    p "${d} 存在 ($(ls -l "${d}" 2>/dev/null | awk '{print $1, $5" bytes"}' || true))"
  else
    case "${d}" in
      */davinci_manager) hint="${NODE_HINT_davinci_manager}" ;;
      */devmm_svm)       hint="${NODE_HINT_devmm_svm}" ;;
      */hisi_hdc)        hint="${NODE_HINT_hisi_hdc}" ;;
      *)                 hint="" ;;
    esac
    f "${d} 不存在。MinerU 与昇腾官方推荐的容器运行方式都会挂载它。
       ${hint}
       补救：确认 Ascend HDK 驱动 + 固件完整安装并已加载；lsmod | grep -E 'davinci|drv_davinci'。"
  fi
done

if [ -d /var/log/npu ]; then
  p "/var/log/npu 存在（容器挂载到 /usr/slog）"
else
  w "/var/log/npu 不存在，挂载 /var/log/npu/:/usr/slog 会失败。
     补救：sudo mkdir -p /var/log/npu && sudo chmod 755 /var/log/npu"
fi

# ------------------------------------------------------------------ 内核模块 / 其他
sec "其他"
if command -v lsmod >/dev/null 2>&1; then
  MODS="$(lsmod 2>/dev/null | awk '{print $1}' | grep -iE 'davinci|drv_davinci|devdrv|hisi' | tr '\n' ' ' || true)"
  if [ -n "${MODS}" ]; then
    p "已加载相关内核模块：${MODS}"
  else
    w "没看到 davinci/devdrv 相关内核模块（若 npu-smi 正常则可忽略，部分版本模块名不同）。"
  fi
fi
if [ -d /usr/local/Ascend/ascend-toolkit ]; then
  p "宿主机装了 ascend-toolkit（非必需，镜像里自带 CANN）"
else
  p "宿主机没有 ascend-toolkit —— 正常，本方案把 CANN 放在镜像里，宿主只需要驱动"
fi
if [ -f "${PROJECT_DIR}/Dockerfile" ]; then
  p "找到项目 Dockerfile：${PROJECT_DIR}/Dockerfile"
else
  f "找不到 ${PROJECT_DIR}/Dockerfile，构建会失败。"
fi

# ------------------------------------------------------------------ 汇总
printf '\n%s======================================================================%s\n' "$c_bld" "$c_off"
printf ' 自检结果:  %sPASS %d%s   %sWARN %d%s   %sFAIL %d%s\n' \
  "$c_grn" "${PASS_N}" "$c_off" "$c_yel" "${WARN_N}" "$c_off" "$c_red" "${FAIL_N}" "$c_off"
printf '%s======================================================================%s\n' "$c_bld" "$c_off"

if [ "${#HINTS[@]}" -gt 0 ]; then
  echo
  echo "需要处理的问题："
  for h in "${HINTS[@]}"; do printf '  - %s\n' "${h}"; done
fi

if [ "${FAIL_N}" -gt 0 ]; then
  echo
  printf '%s存在 %d 项 FAIL：这台机器现在还不能跑昇腾容器。%s\n' "$c_red" "${FAIL_N}" "$c_off"
  if [ "${REPORT_ONLY}" = "1" ]; then exit 0; fi
  exit 1
fi
if [ "${STRICT}" = "1" ] && [ "${WARN_N}" -gt 0 ]; then
  echo
  printf '%s--strict：存在 %d 项 WARN，按失败处理。%s\n' "$c_red" "${WARN_N}" "$c_off"
  [ "${REPORT_ONLY}" = "1" ] && exit 0
  exit 1
fi

echo
printf '%s自检通过，可以开始构建/启动容器：%s\n' "$c_grn" "$c_off"
printf '  bash scripts/build_ascend.sh --chip 910b\n'
printf '  bash scripts/run_ascend.sh --devices 0\n'
exit 0
