# 训练代码 —— 零样本 SAM 掩码 → YOLO-seg

本目录的 6 个脚本与产出下文数字的版本**逐字节一致**（已用 md5 核对，未做任何改写）。
因此它们能复现结果，但也**默认路径绑定在原来的 AutoDL 容器上**；所有路径都能用
命令行参数覆盖，唯一不可覆盖的是 3 处 `sys.path.insert`，见「可移植性」一节。

## 为什么只训分割，不训倒伏分类

倒伏是**几何量**（瓶身轴与重力的夹角），不是外观类别。所以训练部分只需要解决
「把瓶子分割出来」这一个子问题，而分割标签由**零样本 SAM** 自动生成，无需像素级人工标注。
这绕开了原方案卡住的地方：倒伏正样本只有约 19 个实例，不足以训分类器。

## 前置条件

- **MoGe-2 代码**（`moge` 包）——`batch_masks.py` / `mask_qc.py` / `yolo_infer.py` 会 `import`
- 本项目 `app/models/core.py`——圆柱拟合与重力世界系（`sys.path.insert` 里的 `pitch_test`）
- `ultralytics` + `torch`
- 数据集：`gas_pose_v2_0920.zip` 解出的图，加上 LabelMe JSON 的解析表
  `gas_pose_instances.csv`（每实例含框、`aspect_wh`、`axis_tilt_deg`）

预期的目录布局（原机路径）：

```
/root/autodl-tmp/gasdata/images/                原图（含中文文件名）
/root/autodl-tmp/gasdata/gas_pose_instances.csv 标注解析表
/root/autodl-tmp/moge/                          MoGe 代码（sys.path 依赖这个根）
/root/autodl-tmp/moge/gaspipe/out/              所有中间产物
/root/autodl-tmp/moge/gaspipe/ds/               ultralytics 数据集
```

## 流水线（按顺序执行）

| # | 脚本 | 作用 | 关键输出 |
|---|---|---|---|
| 1 | `batch_masks.py` | 每图只跑一次 MoGe；对每个已有框用 SAM（3 候选取 IoU 最高）出掩码；掩码内三维点做圆柱拟合 | `masks/` `labels/` `parts/` `moge_cache/` |
| 2 | `aggregate.py` | 汇总 `parts/*.json` | `masks_index.csv` `geometry.csv` `aggregate_summary.json` |
| 3 | `make_dataset.py` | 组织 ultralytics 数据集 + **防泄漏切分** | `ds/` `data.yaml` `dataset_map.csv` `split_report.json` |
| 4 | `train_yolo.py` | 训练 YOLO-seg | `runs/<name>/weights/best.pt` |
| 5 | `yolo_infer.py` | 全量推理，与同图 SAM 实例按掩码 IoU 贪心匹配后逐实例对比几何量测 | `yolo_vs_sam.csv/json` |
| — | `mask_qc.py` | 抽 N 个实例拼 contact sheet 供**人工抽查**（可选，但建议做） | `mask_qc*.png` |

`batch_masks.py` 支持 `--shard i/N` 分片与 `done/` 断点续跑标记，`--min-interval` 限速。
原机实测：**634 图 / 1315 实例，0 失败，13.5 分钟**（MoGe 约 0.2–1.5 s/图）。

## 防泄漏切分（硬要求，别改）

`make_dataset.py` 的四条规则缺一不可：

1. **视频连续帧整体不可切**：`r_video`(83) / `r1_video`(56) / `v1..v5` 各自作为一整组
2. **`df__` 同一源图的全部增广归为一组**
3. 再做一次近重复检测（dHash + 缩略图 L1），用 union-find 合并剩余漏网（实测合并 **565 对**）
4. **按 source 整体留出测试集**：`lpg_gas` + `oxygen_tank` + `hash`，完全不参与训练与选型

理由：按图随机切会让同一段视频的相邻帧同时落进 train 和 val，指标虚高。这套规则实测
抓到 **13 张 `gas_clinder2__lpg_gas_*` 其实是 `lpg_gas_*` 的原图副本**。
自检结果（`split_report.json`）：`groups_spanning_splits = 0`、`cross_split_near_dup_pairs = 0`。

## 复现出的数字

跨域 test（`lpg_gas` + `oxygen_tank` + `hash` 整体留出，82 图 / 196 实例）：

| 指标 | 值 |
|---|---|
| mask mAP50 | **0.917** |
| mask mAP50-95 | 0.779 |
| precision / recall | 0.973 / 0.879 |

val（83 图）mask mAP50 = 0.820。

YOLO 与 SAM 的一致性（`yolo_vs_sam.json`，1289 个匹配实例）：SAM 实例被检出率
**0.980**、掩码 IoU 中位 **0.9535**、pitch 差中位 **0.192°**、倒地判定一致率 **0.993**。
即：**蒸馏到轻量模型几乎没有掉几何精度**，且训练/评测在时间上快了三个数量级。

产物中的长径比中位 11.44（真实 40L 瓶 ≈ 12.2）、半径中位 0.095 m、长度中位 1.116 m
——混了多种瓶型，**不是误差**。

## 可移植性

- 所有 `--ds` / `--out` / `--images` / `--csv` 都是 argparse 参数，直接覆盖即可。
- ⚠️ 有 3 处**不可覆盖**的硬编码：
  `batch_masks.py:45-46`、`mask_qc.py:16`、`yolo_infer.py:20-21` 的
  `sys.path.insert(0, "/root/autodl-tmp/moge"...)`。换机器时改这几行，或让
  `MOGE_ROOT` 布局与之对齐。
- `data.yaml` 里的 `path:` 也是绝对路径，由 `make_dataset.py` 生成，换机器会重写。

> 之所以不做可移植化改写：这些脚本的产出一旦改动就**无法在本仓库内重新验证**
> （需要 GPU 与原始数据集）。保持逐字节一致，比提交一份改过但没跑过的版本更可靠。

## 已知局限

1. **没有 ground-truth 掩码。** 上面的 mAP 只代表「像 SAM」，**不代表分割正确**。
   人工抽查 42+12 格时看到梯子/栏杆、阀门、粘连瓶各有若干假阳性。
2. **12.1% 的掩码是多块碎片**，集中在 `home`(46%) 与 `df__12`（整张源图）。
   建议加自动闸门（`sam_iou < 0.7` / `n_poly != 1` / `visible_ratio` 越界 /
   长径比不在 3~30），集约 12% 才需要人工看。
3. **只训了 1 个配置**（`yolo11n-seg` / 1024px / 100 epoch），无 yolov8n、yolo11s、640px 对比。
4. **绝对尺度未标定**：单目深度是学出来的先验。朝向角尺度无关，不受影响。

更完整的记录见 `gaspipe_artifacts/REPORT.md` 与 `QC_FINDINGS.md`（产物，未纳入本仓库）。
