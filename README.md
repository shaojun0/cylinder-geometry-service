# 气瓶几何量测服务（Cylinder Geometry Service）

单张图像的**气瓶检测 / 分割 / 三维朝向与几何中心**估计，单进程多模型托管，
支持 NVIDIA GPU 与**华为昇腾 NPU** 部署。

核心思路：把「气瓶是否倒伏」从**分类问题**降级为**几何量测问题**——
倒伏是可量测的物理量（瓶身轴与重力的夹角），因此**不需要倒伏正样本训练分类器**。

```
elev3d = asin( (p_base − p_top) · up / ‖p_base − p_top‖ )
          直立 ≈ ±90°        倒伏 ≈ 0°
```

其中 `up` 为重力方向。该量**尺度无关**（分子分母同乘缩放因子会约掉），
所以单目深度模型的绝对尺度误差不影响判定。

---

## ✨ 特性

- **单入口多模型**：`ModelHub` 按 flag 派发，同一时刻只驻留一个模型，
  适配 NPU 显存受限场景（懒加载 + LRU 驱逐 + 线程安全）。
- **零样本分割**：SAM 用已有的检测框当 prompt 生成掩码，
  **无需像素级人工标注**；掩码多边形可直接导出为 YOLO-seg 训练标签。
- **重力对齐世界系**：从单目表面法向估计重力方向，
  并在 `/healthz` 暴露 `plane_rms` 作为置信度（rms 小 = 确实锁定单一物理平面）。
- **几何中心 ≠ 物理重心**：拟合圆柱后取轴线中点，消除「可见表面质心朝相机偏 `(2/π)R`」的系统偏差。
- **昇腾适配**：`CYLINDER_DTYPE` / `CYLINDER_EAGER` 策略，310P 自动强制 fp16 + eager。
- **可降级**：权重缺失时返回结构化 `weights_missing` 错误，不会 500。

---

## 🏗 架构

```
                    ┌──────────────────────────────────────┐
   HTTP  ──────────►│  service.py  (FastAPI)               │
                    │    /healthz  /v1/tasks /v1/infer ... │
                    └───────────────┬──────────────────────┘
                                    │  task = "geometry" | "segment" | "detect"
                                    ▼
                    ┌──────────────────────────────────────┐
                    │  ModelHub  (model_hub.py)            │
                    │   · 懒加载                            │
                    │   · max_resident=1 → 换入即驱逐        │
                    │   · 全局锁，推理期间不会被换掉          │
                    └───────────────┬──────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  moge_geom.py                sam_seg.py                  yolo_seg.py
  MoGe-2 单目重建              SAM 框提示分割               YOLO-seg 检测
  + 重力世界系                 → 掩码多边形                 （离线训练产出）
  + 圆柱拟合
        │
        ▼
   core.py（纯 numpy）
   · estimate_world_frame   RANSAC 候选轴 → 物理硬过滤 → offset 直方图平面拟合
   · fit_cylinder           迭代鲁棒 PCA → 垂直面圆拟合 → 轴中点 = 几何中心
```

### 三个关键的工程细节

**1. 重力轴必须做物理硬过滤，不能只看支撑率。**
室内最大的表面通常是墙面，纯按支撑率选**一定会选中墙**。实测某张图：
背墙支撑率 0.386（角度 75.8°）、地板支撑率仅 0.179（角度 18.1°）。
所以必须加「重力轴与相机 up 夹角 ≤ 60°」的过滤，支撑率只做 tie-breaker。

**2. 平面拟合前要用 offset 直方图隔离单一平面。**
室内有地板/桌面/柜顶等**同法向但不同高度**的平面，直接拟合会得到 rms 20~30cm 的
「混合平面」。沿法向取 `p·up` 做直方图能分开它们，之后拟合的 **rms 是极好的置信度指标**
（实测 0.15~2.87 cm）。

**3. 可见表面质心不能当重心。**
单目只能看到朝向相机的半边，质心朝相机偏约 `(2/π)R`。实测拟合半径 0.291 m 时
垂直偏置 0.147 m，与理论 0.185 m 同量级。取轴线中点可消除该偏差。

---

## 🚀 快速开始

### 本地（CPU，仅验证结构，不做真实推理）

```bash
bash scripts/build_local_cpu.sh
docker run --rm cylinder-geom:cpu python -c "import service, model_hub; print('import ok')"

# 容器内真实路径自检（OpenCV / 编解码 / 掩码转多边形 / 几何链路）
docker run --rm -v "$PWD/tests:/t:ro" -w /app cylinder-geom:cpu python /t/smoke_container.py
```

### 昇腾 NPU

```bash
# 1. 上机自检（驱动 / 设备节点 / npu-smi）
bash scripts/preflight_ascend.sh

# 2. 构建（按硬件换 tag：910b / a3 / 310p）
bash scripts/build_ascend.sh --chip 910b

# 3. 运行（含全部设备挂载，设备号可参数化）
ASCEND_DEVICES=0 PORT=8000 bash scripts/run_ascend.sh
```

详见 **[docs/ASCEND.md](docs/ASCEND.md)**（含基础镜像选型依据与未验证风险表）。

### 不用 Docker

```bash
cd app
pip install -r requirements.txt
export WEIGHTS_DIR=/path/to/weights
uvicorn service:app --host 0.0.0.0 --port 8000
```

---

## 📡 API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 存活 + 设备 + **计算策略** + 当前驻留模型 |
| GET | `/v1/tasks` | 任务列表（含权重是否就绪） |
| POST | `/v1/infer` | 主入口，按 `task` 派发 |
| POST | `/v1/unload` | 卸载常驻模型，释放显存 |

### `task` 取值

```bash
# 检测
curl -X POST localhost:8000/v1/infer -H 'Content-Type: application/json' \
  -d '{"task":"detect","image_b64":"..."}'

# 分割（给框，出掩码多边形）
curl -X POST localhost:8000/v1/infer -H 'Content-Type: application/json' \
  -d '{"task":"segment","image_b64":"...","boxes":[[100,80,220,600]]}'

# 几何：出朝向角 / 几何中心 / 离地高度 / 半径 / 长度
curl -X POST localhost:8000/v1/infer -H 'Content-Type: application/json' \
  -d '{"task":"geometry","image_b64":"...","regions":[{"box":[100,80,220,600]}]}'
```

`geometry` 返回示例：

```json
{
  "ok": true, "task": "geometry", "device": "npu:0", "latency_ms": 1830,
  "result": {
    "fov_x_deg": 65.03,
    "gravity": {"source": "normal_bottom_band", "camera_tilt_deg": 17.97,
                "plane_rms_cm": 0.4, "plane_offset_m": -1.887},
    "regions": [{
      "ok": true,
      "pitch_deg": 87.57,          // 直立≈±90, 倒伏≈0
      "center_xyz": [0.42, -0.31, 3.05],
      "center_height_m": 0.79,
      "radius_m": 0.115,
      "length_m": 1.442,
      "length_over_radius": 12.5,
      "is_fallen": false
    }]
  }
}
```

### 设备策略环境变量

| 变量 | 取值 | 说明 |
|---|---|---|
| `CYLINDER_DTYPE` | `auto`(默认) / `float16` / `bfloat16` / `float32` | 检测到 **310P 时强制 float16** |
| `CYLINDER_EAGER` | `1` / `0` | 关闭图模式；**310P 强制开启** |
| `WEIGHTS_DIR` | 路径 | 默认 `/app/weights` |
| `MAX_RESIDENT_MODELS` | 整数 | 默认 `1` |

实际生效的策略会打印在启动日志并出现在 `/healthz` 的 `policy` 字段，
**上机后一眼可核对是否真的生效**。

---

## 🧪 测试

```bash
cd app && python3 ../tests/test_app.py --data /path/to/np_dir
```

覆盖三层：数值回归（用真实点云数据校验重力/平面拟合并复现已验证的数字）、
ModelHub（懒加载/驱逐/flag 派发/并发/降级）、HTTP（含 400/404 错误路径）。
无需 GPU 与真实权重，用假模型即可。

---

## 📦 权重

| 模型 | 用途 | 体积 | 放置 |
|---|---|---|---|
| MoGe-2 `Ruicheng/moge-2-vitl-normal` | 单目几何 | 1.3 GB | `weights/moge-2-vitl-normal.pt`（**必须是 .pt 文件**） |
| SAM `facebook/sam-vit-base` | 框提示分割 | 358 MB | `weights/sam-vit-base/`（**必须是目录**） |
| YOLO-seg | 检测（离线训练产出） | ~6 MB | `weights/*seg*.pt` |

> ⚠️ MoGe 的 `from_pretrained` 对**目录**会直接 `torch.load(目录)` 而失败，
> 必须给 `.pt` 文件路径；而 transformers 的 SAM 需要**目录**。两者要求相反，
> 代码同时兼容扁平布局与 HF 缓存布局。

---

## ⚠️ 已知局限

1. **几何中心 ≠ 物理重心。** 物理重心取决于内部质量分布（LPG 液位），
   视觉不可见。本服务输出的是**几何中心**（拟合圆柱的轴线中点）。
   如需「会不会倒」的稳定性预测，还需要充装量信息。
2. **圆柱只有 5 个自由度。** 绕自身轴的自转不可观测，也无物理意义。
   需要知道阀门朝向时必须单独检测阀门。
3. **绝对尺度依赖单目深度模型的先验**，非测量值。
   但朝向角是尺度无关的，不受影响。
4. **重力估计依赖场景中存在水平面。** 无水平面时会正确回退到相机坐标系，
   此时「上下」退化为「画面的上下」。
5. **昇腾部分未在真机验证过**——详见 [docs/ASCEND.md](docs/ASCEND.md) 的风险表。

---

## 📄 许可

尚未指定许可证。若需开源请补充 `LICENSE`；未指定时默认保留所有权利。
