#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dataset.py -- 用 batch_masks.py 产出的 YOLO-seg 标签组织 ultralytics 数据集,
                  并做「防泄漏」切分。

防泄漏规则 (硬要求):
  1. 视频连续帧: r_video(83 帧) / r1_video(56 帧) / v1..v5 各自整体作为一个不可切分组
  2. df__ 的同一源图的全部 aug 归为同一组 (df__<n>)
  3. 再做一次近重复检测(dHash + 缩略图 L1), union-find 合并近重复图, 兜住其余漏网
  4. 留出完整 source 作为 test: lpg_gas + oxygen_tank + hash  (完全不参与训练/选型)
  5. 剩余组按「组」随机切 train/val (val≈15%)

输出:
  DS/images/{train,val,test}/<ascii>.jpg   (硬链接, 不复制)
  DS/labels/{train,val,test}/<ascii>.txt
  DS/data.yaml
  DS/dataset_map.csv       ascii_name, image, source, group_key, split, n_inst
  DS/split_report.json     切分统计 + 泄漏自检结果
"""
from __future__ import annotations
import argparse, json, os, re, shutil, sys
import numpy as np
import pandas as pd
import cv2

TEST_SOURCES = {"lpg_gas", "oxygen_tank", "hash"}
VIDEO_SOURCES = {"r_video", "r1_video", "v1", "v2", "v3", "v4", "v5"}


def base_group(image: str, source: str) -> str:
    if source == "df__":
        m = re.match(r"df__(\d+)_", image)
        return f"df__{m.group(1)}" if m else f"img:{image}"
    if source in VIDEO_SOURCES:
        return f"vid:{source}"
    return f"img:{image}"


def dhash(img, hs=8):
    g = cv2.cvtColor(cv2.resize(img, (hs + 1, hs), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return (g[:, 1:] > g[:, :-1]).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/moge/gaspipe/out")
    ap.add_argument("--images", default="/root/autodl-tmp/gasdata/images")
    ap.add_argument("--ds", default="/root/autodl-tmp/moge/gaspipe/ds")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hash-thr", type=int, default=6)
    ap.add_argument("--skip-near-dup", action="store_true")
    a = ap.parse_args()

    idx = pd.read_csv(os.path.join(a.out, "masks_index.csv"))
    ok = idx[idx.status == "ok"].copy()
    print(f"[ds] 实例 {len(idx)} -> 可用 {len(ok)}")
    imgs = sorted(ok["image"].unique())
    src_of = ok.groupby("image")["source"].first().to_dict()
    n_inst = ok.groupby("image").size().to_dict()
    print(f"[ds] 可用图 {len(imgs)}")

    # ---------- 近重复检测 ----------
    parent = {im: im for im in imgs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    hashes, thumbs = {}, {}
    for im in imgs:
        p = os.path.join(a.images, im + ".jpg")
        buf = np.fromfile(p, dtype=np.uint8)
        b = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if b is None:
            continue
        hashes[im] = dhash(b)
        thumbs[im] = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (32, 32),
                                interpolation=cv2.INTER_AREA).astype(np.int16)
    imgs = [i for i in imgs if i in hashes]
    n_dup_pairs = 0
    if not a.skip_near_dup:
        keys = list(imgs)
        Hm = np.stack([hashes[i] for i in keys])
        Tm = np.stack([thumbs[i] for i in keys]).reshape(len(keys), -1)
        for i in range(len(keys)):
            hd = (Hm[i + 1:] != Hm[i]).sum(axis=1)
            td = np.abs(Tm[i + 1:] - Tm[i]).mean(axis=1) / 255.0
            hit = np.where((hd <= a.hash_thr) | (td < 0.02))[0]
            for j in hit:
                union(keys[i], keys[i + 1 + int(j)])
                n_dup_pairs += 1
    print(f"[ds] 近重复合并对 {n_dup_pairs}")

    # 基础分组 & 最终组
    g0 = {im: base_group(im, src_of[im]) for im in imgs}
    # 把同一基础组的所有图连起来
    for im in imgs:
        for im2 in imgs:
            if im != im2 and g0[im] == g0[im2]:
                union(im, im2)
    group = {im: find(im) for im in imgs}

    # ---------- 切分 ----------
    # 关键: 组是原子的。只要组里有任何一张图属于留出 source, 整组都进 test。
    from collections import defaultdict
    comp = defaultdict(list)
    for im in imgs:
        comp[group[im]].append(im)
    group_has_test = {g: any(src_of[i] in TEST_SOURCES for i in ms) for g, ms in comp.items()}
    test_imgs = [im for im in imgs if group_has_test[group[im]]]
    pool_groups = {g: ms for g, ms in comp.items() if not group_has_test[g]}
    # 两遍贪心: 优先小组成 val, 避免单个大组(如整段视频)独占验证集
    gkeys = sorted(pool_groups)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(gkeys)
    target = a.val_frac * sum(len(v) for v in pool_groups.values())
    val_groups, nval = set(), 0
    for max_sz in (30, 10 ** 9):
        for k in gkeys:
            if k in val_groups or nval >= target:
                continue
            if len(pool_groups[k]) > max_sz:
                continue
            val_groups.add(k)
            nval += len(pool_groups[k])
    split = {}
    for im in imgs:
        if group_has_test[group[im]]:
            split[im] = "test"
        else:
            split[im] = "val" if group[im] in val_groups else "train"

    # ---------- 写盘 ----------
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(a.ds, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(a.ds, "labels", sp), exist_ok=True)
    rows, tag = [], {}
    for k, im in enumerate(sorted(imgs)):
        sp = split[im]
        ascii_name = f"g{k:04d}"
        tag[im] = (ascii_name, sp)
        src = os.path.join(a.images, im + ".jpg")
        dst = os.path.join(a.ds, "images", sp, ascii_name + ".jpg")
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        lp = os.path.join(a.out, "labels", im + ".txt")
        dl = os.path.join(a.ds, "labels", sp, ascii_name + ".txt")
        if os.path.exists(lp):
            shutil.copyfile(lp, dl)
        rows.append(dict(ascii_name=ascii_name, image=im, source=src_of[im],
                         group_key=group[im], split=sp, n_inst=n_inst.get(im, 0)))
    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(a.ds, "dataset_map.csv"), index=False)

    with open(os.path.join(a.ds, "data.yaml"), "w") as f:
        f.write(f"path: {a.ds}\ntrain: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n  0: cylinder\n")

    # ---------- 泄漏自检 ----------
    leak_groups = [g for g in set(m.group_key) if m[m.group_key == g].split.nunique() > 1]
    # 跨 split 近重复检查
    cross_dup = 0
    for sp1, sp2 in (("train", "val"), ("train", "test"), ("val", "test")):
        i1 = m[m.split == sp1].image.tolist(); i2 = m[m.split == sp2].image.tolist()
        if not i1 or not i2:
            continue
        T1 = np.stack([thumbs[i] for i in i1]).reshape(len(i1), -1)
        T2 = np.stack([thumbs[i] for i in i2]).reshape(len(i2), -1)
        H1 = np.stack([hashes[i] for i in i1]); H2 = np.stack([hashes[i] for i in i2])
        for r in range(len(i1)):
            hd = (H2 != H1[r]).sum(axis=1)
            td = np.abs(T2 - T1[r]).mean(axis=1) / 255.0
            cross_dup += int(((hd <= a.hash_thr) | (td < 0.02)).sum())

    rep = dict(
        n_images_total=int(len(m)),
        n_instances_total=int(m.n_inst.sum()),
        split_images=m.split.value_counts().to_dict(),
        split_instances=m.groupby("split").n_inst.sum().to_dict(),
        split_sources={sp: sorted(m[m.split == sp].source.unique().tolist())
                       for sp in ("train", "val", "test")},
        test_sources_held_out=sorted(TEST_SOURCES),
        n_groups=int(m.group_key.nunique()),
        group_size_hist={str(k): int(v) for k, v in
                         pd.Series([len(v) for v in comp.values()]).value_counts().sort_index().items()},
        max_group_size=int(max(len(v) for v in comp.values())),
        groups_spanning_splits=len(leak_groups),
        cross_split_near_dup_pairs=int(cross_dup),
        near_dup_pairs_merged=int(n_dup_pairs),
        val_frac_requested=a.val_frac,
        val_frac_actual=float((m.split == "val").mean()),
    )
    json.dump(rep, open(os.path.join(a.ds, "split_report.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print("\n[ds] per-source x split:")
    print(pd.crosstab(m.source, m.split, margins=True).to_string())


if __name__ == "__main__":
    main()
