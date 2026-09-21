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
- **重力对齐世界系（带可靠性闸门）**：从单目表面法向估计重力方向。
  只有「地板先验」胜出时才把 up 当作**从场景里量出来的**结果，否则退回相机 up
  并置 `gravity_reliable: false`；`plane_rms_cm`（rms 小 = 确实锁定单一物理平面）
  随 `geometry` 响应返回，作为平面精修的置信度。
- **世界系坐标轴叠加**：`geometry` 返回 `axes_overlay` —— 世界系三轴按相机内参投影成
  图像上的端点（三轴**共用**同一 3D 轴长，锚点取视线与**地平面**的交点，所以坐标系是踩在地上的）。
  前端据此画红箭头坐标系；某根轴若几乎沿视线会被标 `degenerate`，前端画圈标注而不是画一根假箭头。
  请求可带 `draw_axes: false` 关掉。
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
所以必须加「重力轴与相机 up 夹角 ≤ 60°」的过滤，支撑率只做 tie-breaker
（地板候选另有 1.25 倍加成）。

⚠️ 但该闸门只挡得住**明显离谱**的候选，它的余量会被相机俯仰吃掉：
竖直墙的法向与相机 up 的夹角为 `arccos(cos(roll)·sin(pitch))`，横向不歪时就是 `90° − pitch`。
**俯仰 30° 时墙恰好落在 60.0° 边界**（代码是 `<= 60`，进不进由浮点与法向噪声决定）；
俯仰 ≥ 45° 时墙与地板夹角相同，闸门彻底失效。这正是「泛化 RANSAC」分支的来源——
所以闸门之外还需要一道**可靠性闸门**（见下）。

**2. 平面拟合前要用 offset 直方图隔离单一平面。**
室内有地板/桌面/柜顶等**同法向但不同高度**的平面，直接拟合会得到 rms 20~30cm 的
「混合平面」。沿法向取 `p·up` 做直方图能分开它们，之后拟合的 **rms 是极好的置信度指标**
（实测 0.15~2.87 cm）。

**3. 可见表面质心不能当重心。**
单目只能看到朝向相机的半边，质心朝相机偏约 `(2/π)R`。实测拟合半径 0.291 m 时
垂直偏置 0.147 m，与理论 0.185 m 同量级。取轴线中点可消除该偏差。

**4. 只有地板先验胜出时，重力才算「量」出来的。**
泛化 RANSAC 主方向**无法区分地板与墙**（撑过 60° 闸门的候选仍可能是墙），
而它的失败**看起来像成功**：平面精修照常跑通（`plane_offset_m` 有值），
顶层 `confidence` 也不报警（分支中位数 0.226 vs 0.238）——
即「自信地给错答案」。因此 `estimate_world_frame` 增加一道可靠性闸门：
非地板先验的候选一律丢弃、退回相机 up，并在响应里标记 `gravity_reliable: false`
与 `degraded_from`。**安全告警必须在消费 `is_fallen` 前检查该字段。**

> 关于 `confidence` 的补充：上面那个 0.226 vs 0.238 是拿**分支**的中位数在比。
> 若改成按**同一分支内的对错**拆分，`frame_conf` 反而有区分度且方向相反
> （判错中位 0.365 vs 判对 0.111，即「越自信越错」），阈值 0.3 可在该分支标出 25 例、
> 其中 24 例确实是错的。该信号基于弱真值与 n=63，**尚未独立复核，不作为告警依据**，
> 但它说明「无区分度」这个结论依赖于分组口径，值得单独查证。

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
    "gravity": {"source": "normal_bottom_band",   // 原始候选来源（诊断用）
                "up_from": "normal_bottom_band",  // up 实际来自哪
                "gravity_reliable": true,         // false 时 pitch 不可用于告警
                "camera_tilt_deg": 17.97,
                "plane_rms_cm": 0.4, "plane_offset_m": -1.887},
    "axes_overlay": {                            // 世界系三轴的 2D 投影，供前端画红箭头
      "ok": true, "space": "orig", "anchor": "ground",
      "origin_px": [640.0, 576.0],               // 锚点=视线与地平面的交点
      "scale_m": 0.75,                           // 三轴共用的 3D 轴长
      "axes": [
        {"name": "up",      "tip_px": [640.0, 470.2], "length_px": 105.8, "degenerate": false},
        {"name": "right",   "tip_px": [738.4, 578.9], "length_px": 98.5,  "degenerate": false},
        {"name": "forward", "tip_px": [639.1, 543.0], "length_px": 33.0,  "degenerate": false}
      ]},
    "regions": [{
      "ok": true,
      "pitch_deg": 87.57,          // 直立≈±90, 倒伏≈0
      "gravity_reliable": true,    // 与该 region 的计算同源，消费前必查
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
| `MOGE_VERSION` | `v2`(默认) / `v3` | MoGe 版本；**默认 v2，换模型必须显式** |
| `MOGE_WEIGHTS` | 路径 | MoGe checkpoint 的 **`.pt` 文件**（给目录会 `torch.load(目录)` 崩） |
| `MOGE_REFINE_STEPS` | 整数，默认 `3` | 仅 v3：稀疏体素细化步数（越大越细越慢） |

实际生效的策略会打印在启动日志并出现在 `/healthz` 的 `policy` 字段，
**上机后一眼可核对是否真的生效**。

#### MoGe-3（`MOGE_VERSION=v3`）

- 需要 **FlexGEMM**（`flex_gemm`，编译期 `FLEX_GEMM_BUILD_CUDA=1`）——它是
  Triton/CUDA 实现，**昇腾 NPU 上不可用**，服务会直接拒绝启动而不是给你一个坏结果。
- **强制 float32**：flex_gemm 的稀疏细化不接受 fp16 权重（`mat1 Float vs mat2 Half`），
  而 fp32 模型 + `use_fp16=True` 实测反而更慢（2.02s vs 1.06s）。所以 v3 忽略
  `CYLINDER_DTYPE`，并在启动日志里说明原因。
- **版本与权重必须一致**：`svc_weights` 下 v2/v3 同时存在时，用 v3 结构加载 v2 权重
  **不会报错、只会静默算错**，因此启动时会做一致性校验，不一致直接抛错。
- 实测差异（同图同框）：02_Office 平面 rms 0.39→0.25 cm、相机离地 1.887→1.773 m；
  但**气瓶 pitch 会变**，个别实例靠近 45° 判定线时会翻转 `is_fallen` —— 换版本后
  建议重新评估判定阈值，别默认"数值差不多"。

---

## 🧪 测试

```bash
cd app && python3 ../tests/test_app.py --data /path/to/np_dir
```

覆盖三层：数值回归（用真实点云数据校验重力/平面拟合并复现已验证的数字）、
ModelHub（懒加载/驱逐/flag 派发/并发/降级）、HTTP（含 400/404 错误路径）。
无需 GPU 与真实权重，用假模型即可。

重力那一层除了真实数据回归，还用**三个确定性合成场景**锁定可靠性闸门的行为：
命中 `normal_ransac` 必须降级并标记不可靠、命中地板先验必须不降级、
`camera_fallback` 必须不记 `degraded_from`。这样闸门逻辑在没有数据集时也能防回归。

---

## 🏋️ 训练

`training/` 是**离线**部分：零样本 SAM 生成掩码 → 导出 YOLO-seg 标签 → 训练轻量分割模型。
服务只加载训好的 `best.pt`，不依赖 SAM（`detect` 任务的 SAM 仅用于离线打标签）。
逐脚本说明、防泄漏切分规则与复现出的指标见 [`training/README.md`](training/README.md)。

```bash
cd training
python batch_masks.py --images IMGDIR --csv instances.csv --out OUT   # 1. 掩码+几何
python aggregate.py --out OUT --csv instances.csv --geom geometry.csv # 2. 汇总
python make_dataset.py --out OUT --ds DS                              # 3. 防泄漏切分
python train_yolo.py --ds DS --model yolo11n-seg.pt --imgsz 1024      # 4. 训练
python yolo_infer.py --ds DS --weights best.pt --out OUT              # 5. 与 SAM 对比
```

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

## 📊 世界系估计的实测口径

数据集：634 图 / 1315 个可处理实例，其中 **857 个有弱真值**
（`truth_lying` 由 2D 倾角推出，本身含透视缩短等已知问题）。
按 `frame_source` 分组的判定准确率：

| `frame_source` | n | 准确率 | 反事实：改用相机 up |
|---|---|---|---|
| `normal_bottom_band` | 674 | 97.8% | 97.0% |
| `camera_fallback` | 120 | 100.0% | 100.0% |
| **`normal_ransac`** | **63** | **46.0%** ❌ | **95.2%** |
| 合计 | 857 | 94.3% | 97.3% |

**绝对数字意义有限**（弱真值），但 46.0% vs 95.2% 是同一套弱真值下的相对对比，
足以支撑「非地板先验一律退回相机 up」这个决定。复现该表需要原始数据集，不在本仓库内；
`tests/test_app.py` 用三个确定性合成场景锁定了**闸门行为本身**。

> `camera_fallback` 的 100% 有选择偏差：它只在估计器彻底找不到可信平面时触发，
> 而那类照片往往相机确实端平。它说明的是「该认输时认输」，不是「相机 up 万能」。

---

## ⚠️ 已知局限

1. **几何中心 ≠ 物理重心。** 物理重心取决于内部质量分布（LPG 液位），
   视觉不可见。本服务输出的是**几何中心**（拟合圆柱的轴线中点）。
   如需「会不会倒」的稳定性预测，还需要充装量信息。
2. **圆柱只有 5 个自由度。** 绕自身轴的自转不可观测，也无物理意义。
   需要知道阀门朝向时必须单独检测阀门。
3. **绝对尺度依赖单目深度模型的先验**，非测量值。
   但朝向角是尺度无关的，不受影响。
4. **重力估计依赖场景中存在水平面，且只有地板先验胜出时才可信。**
   `gravity_reliable: false` 时 up 退化成「假设相机水平」，「上下」即画面的上下：
   相机俯仰 θ 或横滚 φ 会给 `pitch_deg` 引入 `arccos(cosθ·cosφ)` 的**系统性偏差**
   （同一姿态下所有瓶子一起错），而判直立/倒伏的阈值只有 30°/45°。
   实测泛化 RANSAC 分支仅 46.0% 正确（改用相机 up 为 95.2%，见上表）。
   **消费 `is_fallen` 前必须检查 `gravity_reliable`。**
5. **昇腾部分未在真机验证过**——详见 [docs/ASCEND.md](docs/ASCEND.md) 的风险表。

---

## 📄 许可

尚未指定许可证。若需开源请补充 `LICENSE`；未指定时默认保留所有权利。
