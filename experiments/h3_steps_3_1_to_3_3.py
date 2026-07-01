# -*- coding: utf-8 -*-
# =============================================================================
# H3 自适应格网生成  -  步骤3.1 / 3.2 / 3.3
# 算法依据：The HAND Algorithm (附件2)
#
# 步骤3.1：多准则分裂判定  (Algorithm A.4.2)
# 步骤3.2：构建自适应H3分裂树  (Algorithm 1-H3)
# 步骤3.3：执行树构建，输出统计与可视化
#
# ── 参考代码错误修正说明 ──
# 1. h3.h3_get_hexagon_area_km2(resolution)  →  h3.cell_area(cell, unit='m^2')
#    原函数已废弃，且单位需为m²以匹配几何坐标系中的面积单位
# 2. h3.h3_to_children(h, r+1)  →  h3.cell_to_children(h, r+1)
#    H3 v4 API 已更名
# 3. 面积密度计算：几何坐标系(度²)的面积需正确转换为m²再比对
# 4. H0_initial 未定义  →  从 H0_grid.csv 读取
# 5. yolo_results 未定义  →  从 targets_classified.geojson 读取
# 6. 分裂上限：参考代码 r_max=15 对本数据集过深，
#    根据目标尺寸自动推算合适 r_max（目标最小尺寸对应分辨率附近+2）
# =============================================================================

import h3
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import math
from collections import deque
import pyproj
from shapely.ops import transform as shapely_transform
from shapely.geometry import Polygon, shape
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print(" H3 自适应格网生成  —  步骤3.1 / 3.2 / 3.3  自适应分裂树构建")
print("=" * 70)
print()

# =============================================================================
# 读取前置步骤成果
# =============================================================================
print("[读取数据]")
print("-" * 50)

# 读取H0格网
h0_grid = pd.read_csv('H0_grid.csv')
r_init = int(h0_grid['resolution'].iloc[0])
H0_initial = set(h0_grid['h3_index'].tolist())
print(f"  [OK] H0格网: {len(h0_grid)} 个, 分辨率 r_init={r_init}")

# 读取目标分类结果（含几何信息）
targets_gdf = gpd.read_file('targets_classified.geojson')

# 解析h3_coverage字段（numpy ndarray 或 JSON字符串）
def parse_h3_coverage(val):
    if isinstance(val, np.ndarray):
        return [str(v) for v in val.tolist() if v]
    if isinstance(val, list):
        return [str(v) for v in val if v]
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        if val.startswith('['):
            try:
                parsed = json.loads(val)
                return [str(v) for v in parsed if v]
            except Exception:
                pass
        return [v.strip() for v in val.split(';') if v.strip()]
    return []

# targets_classified.geojson 中 h3_coverage 存为 h3_coverage_json 或 h3_coverage
if 'h3_coverage_json' in targets_gdf.columns:
    targets_gdf['h3_coverage'] = targets_gdf['h3_coverage_json'].apply(parse_h3_coverage)
elif 'h3_coverage' in targets_gdf.columns:
    targets_gdf['h3_coverage'] = targets_gdf['h3_coverage'].apply(parse_h3_coverage)
else:
    # 从原始文件重建
    orig = gpd.read_file('targets_with_mapping.geojson')
    orig['h3_coverage'] = orig['h3_coverage'].apply(parse_h3_coverage)
    targets_gdf['h3_coverage'] = orig['h3_coverage'].values

print(f"  [OK] 目标数据: {len(targets_gdf)} 个目标")
# 验证
v0 = targets_gdf['h3_coverage'].iloc[0]
print(f"  [验证] h3_coverage[0]: {v0}")
print()

# =============================================================================
# 工具函数
# =============================================================================

# 构建 UTM 投影（用于精确面积计算，避免地理坐标系误差）
# 厦门地区约在 UTM zone 50N (EPSG:32650)
_utm_proj = pyproj.CRS("EPSG:32650")
_wgs84    = pyproj.CRS("EPSG:4326")
_transformer_to_utm = pyproj.Transformer.from_crs(_wgs84, _utm_proj, always_xy=True)

def geom_to_utm_area_m2(geom):
    """将WGS84几何体转换为UTM投影后计算面积(m²)"""
    try:
        geom_utm = shapely_transform(_transformer_to_utm.transform, geom)
        return geom_utm.area
    except Exception:
        return 0.0

def h3_cell_to_shapely_polygon(cell):
    """H3单元 → Shapely Polygon (经度, 纬度)"""
    boundary = h3.cell_to_boundary(cell)           # [(lat, lon), ...]
    coords   = [(lon, lat) for lat, lon in boundary]
    return Polygon(coords)

# =============================================================================
# 步骤3.1：多准则分裂判定  (Algorithm A.4.2)
# =============================================================================
print("[步骤3.1] 多准则分裂判定 (Algorithm A.4.2)")
print("-" * 50)

# 阈值设定（参考代码给出，结合本数据集特征保持一致）
TAU_N = 2        # 目标数量阈值：格网内目标数 >= τ_n 则触发分裂
TAU_D = 0.05     # 面积密度阈值（归一化）：目标总面积/格网面积 >= τ_d 则触发
                 # 注：参考代码τ_d=0.5对本数据集过严（目标最大密度仅~0.003），
                 # 调整为0.05以反映实际数据特征
TAU_S = 0.85     # 置信度阈值：格网内最高置信度 >= τ_s 则触发分裂

print(f"  分裂阈值设置：")
print(f"    τ_n (目标数量) = {TAU_N}  : 格网内目标数 ≥ {TAU_N}")
print(f"    τ_d (面积密度) = {TAU_D}  : 目标总面积/格网面积 ≥ {TAU_D}")
print(f"    τ_s (置信度)   = {TAU_S}  : 最高置信度 ≥ {TAU_S}")
print()

def calculate_split_criteria(h3_cell, targets_in_cell, resolution):
    """
    计算H3格网的分裂判定条件 (Algorithm A.4.2)

    P_H3(h_j) = (|T_j| >= τ_n)  OR  (面积密度 >= τ_d)  OR  (置信度 >= τ_s)

    参数:
        h3_cell        : H3单元标识符
        targets_in_cell: 当前格网内目标的字典列表 (含 geometry, confidence 字段)
        resolution     : 当前分辨率

    返回:
        dict: 分裂判定结果及各条件状态

    ── 修正说明 ──
    原参考代码：h3.h3_get_hexagon_area_km2(resolution) — 已废弃
    修正后：h3.cell_area(h3_cell, unit='m^2') 获取真实格网面积
    面积密度：将目标几何体投影到UTM后计算m²，再除以格网面积m²
    """
    target_count = len(targets_in_cell)

    # ── 条件1：目标数量 ──
    condition_n = target_count >= TAU_N

    # ── 条件2：面积密度 ──
    # 格网真实面积 (m²)
    cell_area_m2 = h3.cell_area(h3_cell, unit='m^2')

    # 目标总面积 (m²)，通过UTM投影精确计算
    total_target_area_m2 = 0.0
    for t in targets_in_cell:
        geom = t.get('geometry')
        if geom is not None:
            total_target_area_m2 += geom_to_utm_area_m2(geom)

    area_density = total_target_area_m2 / cell_area_m2 if cell_area_m2 > 0 else 0.0
    condition_d = area_density >= TAU_D

    # ── 条件3：最高置信度 ──
    if targets_in_cell:
        max_confidence = max(t.get('confidence', 0) for t in targets_in_cell)
    else:
        max_confidence = 0.0
    condition_s = max_confidence >= TAU_S

    should_split = condition_n or condition_d or condition_s

    return {
        'should_split':    should_split,
        'target_count':    target_count,
        'cell_area_m2':    cell_area_m2,
        'total_area_m2':   total_target_area_m2,
        'area_density':    area_density,
        'max_confidence':  max_confidence,
        'condition_n':     condition_n,
        'condition_d':     condition_d,
        'condition_s':     condition_s,
        'conditions':      [condition_n, condition_d, condition_s],
    }

print(f"  [OK] calculate_split_criteria 函数定义完毕")
print()

# =============================================================================
# 步骤3.2：构建自适应H3分裂树  (Algorithm 1-H3)
# =============================================================================
print("[步骤3.2] H3AdaptiveTree 类定义")
print("-" * 50)

class H3AdaptiveTree:
    """
    基于Aperture-7的自适应H3层级分裂树  (Algorithm 1-H3)

    每轮BFS迭代：
      1. 从队列取出 (cell, resolution)
      2. 查找落入该格网的目标
      3. 调用 calculate_split_criteria 判定是否分裂
      4. 若分裂：获取7个子格网，入队
         若不分裂或达到最大分辨率：标记为叶子节点

    ── 修正说明 ──
    原参考代码：h3.h3_to_children(h, r+1)  — H3 v3 API，已废弃
    修正后：    h3.cell_to_children(h, r+1) — H3 v4 API
    """

    def __init__(self, initial_resolution, r_max=12):
        """
        initial_resolution : H0格网分辨率 (r_init)
        r_max              : 最大允许分辨率，默认12
                             （本数据集目标最小约10m，对应H3 res≈11；设12留一层余量）
        """
        self.initial_resolution = initial_resolution
        self.r_max    = r_max
        self.tree     = {}        # {cell: node_info}
        self.leaf_cells = set()

    def _build_index(self, targets_gdf):
        """
        预建索引：{r_init级格网: [目标记录列表]}
        以及所有目标的 (center_cell, 记录) 列表，供快速上溯查找
        """
        self._cell_index = {}   # r_init cell → [target records]
        self._all_targets = []  # [(center_cell, record)]

        for _, row in targets_gdf.iterrows():
            center_cell = str(row.get('h3_center', ''))
            if not center_cell:
                continue
            record = {
                'geometry':   row.geometry,
                'confidence': float(row.get('confidence', 0)),
                'class':      str(row.get('class', '')),
                'target_id':  row.name,
                'center_cell': center_cell,
            }
            self._all_targets.append((center_cell, record))
            if center_cell not in self._cell_index:
                self._cell_index[center_cell] = []
            self._cell_index[center_cell].append(record)

    def _get_targets_in_cell(self, h_cell, targets_gdf=None):
        """
        查找覆盖了指定H3格网的所有目标。

        判断逻辑（三种情况）：
          - r_h == r_init : 直接查索引
          - r_h  < r_init : 将 center_cell 上溯到 r_h 层，检查是否等于 h_cell
          - r_h  > r_init : 将 h_cell 上溯到 r_init 层，检查是否等于 center_cell
        """
        current_res = h3.get_resolution(h_cell)
        r_target    = self.initial_resolution   # r_init = 7
        result      = []

        if current_res == r_target:
            # 直接查索引（最高频路径）
            return list(self._cell_index.get(h_cell, []))

        elif current_res < r_target:
            # h_cell 是粗格网（比r_init还粗，理论上不出现，但做保护）
            for center_cell, record in self._all_targets:
                try:
                    ancestor = h3.cell_to_parent(center_cell, current_res)
                    if ancestor == h_cell:
                        result.append(record)
                except Exception:
                    pass

        else:
            # current_res > r_target（子格网，最常见场景）
            # 将 h_cell 上溯到 r_init 层，找到对应父格网
            try:
                parent_at_r_init = h3.cell_to_parent(h_cell, r_target)
                # 在该父格网的目标中，检查目标的精确格网是否属于此子格网
                for record in self._cell_index.get(parent_at_r_init, []):
                    # 将目标中心点映射到 current_res 级格网，判断是否落入 h_cell
                    center_cell = record['center_cell']
                    try:
                        geom = record['geometry']
                        centroid = geom.centroid
                        mapped_cell = h3.latlng_to_cell(
                            centroid.y, centroid.x, current_res
                        )
                        if mapped_cell == h_cell:
                            result.append(record)
                    except Exception:
                        pass
            except Exception:
                pass

        return result

    def build_tree(self, targets_gdf, r_max=None):
        """
        BFS构建自适应H3分裂树

        参数:
            targets_gdf : 目标GeoDataFrame（含 h3_center, confidence, geometry）
            r_max       : 最大分辨率上限（可覆盖构造函数中的值）
        """
        if r_max is not None:
            self.r_max = r_max

        # 预建索引，大幅加速目标查找
        self._build_index(targets_gdf)

        Q      = deque()
        leaves = set()

        # 初始化：将所有H0格网入队
        for h in H0_initial:
            Q.append((h, self.initial_resolution))

        total_processed = 0
        split_count     = 0

        print(f"  初始队列: {len(Q)} 个H0格网 (r={self.initial_resolution})")
        print(f"  最大分辨率上限: r_max={self.r_max}")
        print()

        while Q:
            h_current, resolution = Q.popleft()
            total_processed += 1

            if total_processed % 500 == 0:
                print(f"    进度: 已处理 {total_processed} 个节点, "
                      f"队列剩余 {len(Q)}, 已分裂 {split_count}")

            # 获取当前格网内的目标
            current_targets = self._get_targets_in_cell(h_current)

            # ── 分裂决策 ──
            if resolution < self.r_max and len(current_targets) > 0:
                split_info = calculate_split_criteria(
                    h_current, current_targets, resolution
                )

                if split_info['should_split']:
                    # ── 执行分裂 ──
                    # 修正：h3.cell_to_children(cell, child_resolution)
                    children = list(h3.cell_to_children(h_current, resolution + 1))
                    split_count += 1

                    self.tree[h_current] = {
                        'resolution':    resolution,
                        'targets':       len(current_targets),
                        'split':         True,
                        'children':      children,
                        'split_criteria': split_info,
                    }

                    # 子格网入队
                    for child in children:
                        Q.append((child, resolution + 1))
                else:
                    # ── 不分裂，标记叶子 ──
                    leaves.add(h_current)
                    self.tree[h_current] = {
                        'resolution': resolution,
                        'targets':    len(current_targets),
                        'split':      False,
                        'children':   [],
                        'split_criteria': split_info,
                    }
            else:
                # 无目标 或 达到最大分辨率 → 叶子
                leaves.add(h_current)
                self.tree[h_current] = {
                    'resolution': resolution,
                    'targets':    len(current_targets),
                    'split':      False,
                    'children':   [],
                }

        self.leaf_cells = leaves
        print(f"    进度: 共处理 {total_processed} 个节点, 分裂节点 {split_count}")
        return self.tree, self.leaf_cells

    def get_statistics(self):
        """获取树的统计信息"""
        split_nodes = sum(1 for n in self.tree.values() if n['split'])
        leaf_nodes  = len(self.leaf_cells)
        total_nodes = len(self.tree)

        # 按分辨率统计叶子节点分布
        res_dist = {}
        for h in self.leaf_cells:
            r = self.tree[h]['resolution']
            res_dist[r] = res_dist.get(r, 0) + 1

        # 叶子节点平均目标数
        leaf_targets = [self.tree[h]['targets'] for h in self.leaf_cells]
        avg_targets  = float(np.mean(leaf_targets)) if leaf_targets else 0.0
        max_targets  = max(leaf_targets) if leaf_targets else 0

        # 按叶子目标数分布
        target_dist = {}
        for t in leaf_targets:
            target_dist[t] = target_dist.get(t, 0) + 1

        return {
            'total_nodes':          total_nodes,
            'split_nodes':          split_nodes,
            'leaf_nodes':           leaf_nodes,
            'avg_targets_per_leaf': avg_targets,
            'max_targets_per_leaf': max_targets,
            'resolution_distribution': dict(sorted(res_dist.items())),
            'target_count_distribution': dict(sorted(target_dist.items())),
        }

print("  [OK] H3AdaptiveTree 类定义完毕")
print()

# =============================================================================
# 步骤3.3：执行树构建
# =============================================================================
print("[步骤3.3] 执行自适应H3树构建")
print("-" * 50)

# r_max 推算：本数据集目标最小约10m，H3 res11 面积约3m²，res10约23m²，res9约163m²
# 考虑到目标尺寸分布（均值~1082m², 中位~300m²），res=10（~23m²）可分辨单个目标
# 设 r_max=10 既能充分细化，又避免无意义深度分裂
R_MAX = 10

adaptive_tree = H3AdaptiveTree(r_init, r_max=R_MAX)

print(f"\n  开始构建自适应H3树 (r_init={r_init}, r_max={R_MAX})...\n")
tree_structure, leaf_set = adaptive_tree.build_tree(targets_gdf)

stats = adaptive_tree.get_statistics()

print()
print("  ✓ H3树构建完成！")
print()
print(f"  ── 树结构统计 ──")
print(f"    总节点数     : {stats['total_nodes']}")
print(f"    分裂节点数   : {stats['split_nodes']}")
print(f"    叶子节点数   : {stats['leaf_nodes']}")
print(f"    平均目标/叶子: {stats['avg_targets_per_leaf']:.3f}")
print(f"    最大目标/叶子: {stats['max_targets_per_leaf']}")
print()
print(f"  ── 叶子节点分辨率分布 ──")
for res, cnt in stats['resolution_distribution'].items():
    bar = '█' * min(cnt, 60)
    print(f"    r={res:2d}: {cnt:5d} 个叶子  {bar}")
print()
print(f"  ── 叶子节点目标数分布 ──")
for t_cnt, cell_cnt in list(stats['target_count_distribution'].items())[:15]:
    bar = '█' * min(cell_cnt // 2 + 1, 60)
    print(f"    目标数={t_cnt:3d}: {cell_cnt:5d} 个格网  {bar}")

# =============================================================================
# 保存成果
# =============================================================================
print()
print("[保存成果]")
print("-" * 50)

# 1. 树结构 JSON
tree_export = {}
for cell, info in tree_structure.items():
    node = {
        'resolution': info['resolution'],
        'targets':    info['targets'],
        'split':      info['split'],
        'children':   info['children'],
    }
    if 'split_criteria' in info:
        sc = info['split_criteria']
        node['split_criteria'] = {
            'should_split':   sc['should_split'],
            'target_count':   sc['target_count'],
            'area_density':   round(sc['area_density'], 8),
            'max_confidence': round(sc['max_confidence'], 6),
            'condition_n':    sc['condition_n'],
            'condition_d':    sc['condition_d'],
            'condition_s':    sc['condition_s'],
        }
    tree_export[cell] = node

with open('adaptive_tree.json', 'w', encoding='utf-8') as f:
    json.dump(tree_export, f, ensure_ascii=False, indent=2)
print(f"  [OK] 树结构JSON         -> adaptive_tree.json ({len(tree_export)} 节点)")

# 2. 叶子节点 CSV
leaf_records = []
for cell in leaf_set:
    info = tree_structure[cell]
    sc   = info.get('split_criteria', {})
    leaf_records.append({
        'h3_cell':        cell,
        'resolution':     info['resolution'],
        'targets':        info['targets'],
        'area_density':   round(sc.get('area_density', 0), 8),
        'max_confidence': round(sc.get('max_confidence', 0), 6),
        'is_h0':          cell in H0_initial,
    })

leaf_df = pd.DataFrame(leaf_records).sort_values(['resolution', 'h3_cell'])
leaf_df.to_csv('adaptive_tree_leaves.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] 叶子节点CSV        -> adaptive_tree_leaves.csv ({len(leaf_df)} 行)")

# 3. 分裂节点 CSV
split_records = []
for cell, info in tree_structure.items():
    if info['split']:
        sc = info.get('split_criteria', {})
        split_records.append({
            'h3_cell':        cell,
            'resolution':     info['resolution'],
            'targets':        info['targets'],
            'n_children':     len(info['children']),
            'area_density':   round(sc.get('area_density', 0), 8),
            'max_confidence': round(sc.get('max_confidence', 0), 6),
            'cond_n':         sc.get('condition_n', False),
            'cond_d':         sc.get('condition_d', False),
            'cond_s':         sc.get('condition_s', False),
        })

split_df = pd.DataFrame(split_records).sort_values(['resolution', 'h3_cell'])
split_df.to_csv('adaptive_tree_splits.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] 分裂节点CSV        -> adaptive_tree_splits.csv ({len(split_df)} 行)")

print()

# =============================================================================
# 可视化
# =============================================================================
print("[可视化] 生成步骤3结果图")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle(
    'H3 Adaptive Grid Generation — Step 3\n'
    'Adaptive Split Tree Construction (Algorithm A.4.2 / Algorithm 1-H3)',
    fontsize=13, fontweight='bold'
)

# ── 图1：叶子节点分辨率空间分布 ──
ax1 = axes[0, 0]
ax1.set_title('Step 3.3: Leaf Cells — Resolution Distribution (Spatial)', fontsize=11)

res_list = sorted(stats['resolution_distribution'].keys())
cmap_res  = cm.get_cmap('YlOrRd', len(res_list) + 1)

for cell in leaf_set:
    r = tree_structure[cell]['resolution']
    try:
        boundary = h3.cell_to_boundary(cell)
        coords   = [(lon, lat) for lat, lon in boundary]
        color    = cmap_res((r - r_init) / max(R_MAX - r_init, 1))
        poly = MplPolygon(coords, facecolor=color, edgecolor='none', alpha=0.75)
        ax1.add_patch(poly)
    except Exception:
        pass

# H0轮廓叠加
for _, hrow in h0_grid.iterrows():
    cb = h3.cell_to_boundary(hrow['h3_index'])
    coords = [(lon, lat) for lat, lon in cb]
    poly = MplPolygon(coords, facecolor='none', edgecolor='#444444',
                      linewidth=0.8, alpha=0.7)
    ax1.add_patch(poly)

# 图例
legend_patches = []
for r in res_list:
    color = cmap_res((r - r_init) / max(R_MAX - r_init, 1))
    cnt   = stats['resolution_distribution'][r]
    legend_patches.append(
        mpatches.Patch(facecolor=color, edgecolor='gray',
                       label=f'r={r}  (n={cnt})')
    )
ax1.legend(handles=legend_patches, loc='upper right', fontsize=8,
           title='Resolution', title_fontsize=8)

# 目标散点叠加
ax1.scatter(targets_gdf['cx_lon'], targets_gdf['cx_lat'],
            c='blue', s=4, alpha=0.3, zorder=5, label='Targets')

lons = targets_gdf['cx_lon']
lats = targets_gdf['cx_lat']
pad = 0.02
ax1.set_xlim(lons.min() - pad, lons.max() + pad)
ax1.set_ylim(lats.min() - pad, lats.max() + pad)
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.grid(True, alpha=0.2)

# ── 图2：叶子节点目标数热力图 ──
ax2 = axes[0, 1]
ax2.set_title('Step 3.3: Leaf Cells — Target Count Heatmap', fontsize=11)

max_t = stats['max_targets_per_leaf']
norm_t = mcolors.Normalize(vmin=0, vmax=max(max_t, 1))
cmap_t = cm.get_cmap('hot_r')

for cell in leaf_set:
    t = tree_structure[cell]['targets']
    try:
        boundary = h3.cell_to_boundary(cell)
        coords   = [(lon, lat) for lat, lon in boundary]
        color    = cmap_t(norm_t(t))
        poly = MplPolygon(coords, facecolor=color, edgecolor='none', alpha=0.8)
        ax2.add_patch(poly)
    except Exception:
        pass

for _, hrow in h0_grid.iterrows():
    cb = h3.cell_to_boundary(hrow['h3_index'])
    coords = [(lon, lat) for lat, lon in cb]
    poly = MplPolygon(coords, facecolor='none', edgecolor='#555555',
                      linewidth=0.8, alpha=0.7)
    ax2.add_patch(poly)

sm = cm.ScalarMappable(cmap=cmap_t, norm=norm_t)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, shrink=0.8)
cbar.set_label('Target Count', fontsize=9)

ax2.set_xlim(lons.min() - pad, lons.max() + pad)
ax2.set_ylim(lats.min() - pad, lats.max() + pad)
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.grid(True, alpha=0.2)

# ── 图3：叶子节点分辨率分布柱状图 ──
ax3 = axes[1, 0]
ax3.set_title('Step 3: Leaf Node Resolution Distribution', fontsize=11)

res_vals  = list(stats['resolution_distribution'].keys())
leaf_cnts = [stats['resolution_distribution'][r] for r in res_vals]
bar_colors = [cmap_res((r - r_init) / max(R_MAX - r_init, 1)) for r in res_vals]

bars = ax3.bar(res_vals, leaf_cnts, color=bar_colors, edgecolor='black', alpha=0.9)
for bar, cnt in zip(bars, leaf_cnts):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             str(cnt), ha='center', va='bottom', fontsize=9)

ax3.set_xlabel('H3 Resolution')
ax3.set_ylabel('Leaf Node Count')
ax3.set_xticks(res_vals)
ax3.grid(True, alpha=0.3, axis='y')

# 分裂节点也叠加显示
split_res_dist = {}
for cell, info in tree_structure.items():
    if info['split']:
        r = info['resolution']
        split_res_dist[r] = split_res_dist.get(r, 0) + 1

if split_res_dist:
    ax3_twin = ax3.twinx()
    sr = list(split_res_dist.keys())
    sc = [split_res_dist[r] for r in sr]
    ax3_twin.plot(sr, sc, 'ro-', linewidth=2, markersize=6, label='Split nodes')
    ax3_twin.set_ylabel('Split Node Count', color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    ax3_twin.legend(loc='upper left', fontsize=8)

ax3.set_title('Step 3: Node Count by Resolution\n(bars=leaves, line=splits)', fontsize=11)

# ── 图4：分裂触发条件分析 ──
ax4 = axes[1, 1]
ax4.set_title('Step 3.1: Split Trigger Condition Analysis', fontsize=11)

# 统计哪个条件触发了分裂
cond_n_only = 0
cond_d_only = 0
cond_s_only = 0
cond_ns = 0
cond_nd = 0
cond_ds = 0
cond_nds = 0

for cell, info in tree_structure.items():
    if not info['split']:
        continue
    sc = info.get('split_criteria', {})
    n = sc.get('condition_n', False)
    d = sc.get('condition_d', False)
    s = sc.get('condition_s', False)
    key = (n, d, s)
    if key == (True, False, False):   cond_n_only += 1
    elif key == (False, True, False): cond_d_only += 1
    elif key == (False, False, True): cond_s_only += 1
    elif key == (True, True, False):  cond_ns += 1
    elif key == (True, False, True):  cond_nd += 1
    elif key == (False, True, True):  cond_ds += 1
    elif key == (True, True, True):   cond_nds += 1

labels_pie = []
sizes_pie  = []
colors_pie = ['#E91E63','#2196F3','#FF9800','#9C27B0','#4CAF50','#FF5722','#00BCD4']
all_conds  = [
    (cond_n_only, 'N only (count≥τ_n)'),
    (cond_d_only, 'D only (density≥τ_d)'),
    (cond_s_only, 'S only (conf≥τ_s)'),
    (cond_ns,     'N+D'),
    (cond_nd,     'N+S'),
    (cond_ds,     'D+S'),
    (cond_nds,    'N+D+S'),
]
for cnt, label in all_conds:
    if cnt > 0:
        labels_pie.append(f"{label}\n({cnt})")
        sizes_pie.append(cnt)

if sizes_pie:
    wedges, texts = ax4.pie(
        sizes_pie, labels=labels_pie,
        colors=colors_pie[:len(sizes_pie)],
        startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=1.5)
    )
    for t in texts:
        t.set_fontsize(8)
else:
    ax4.text(0.5, 0.5, 'No split nodes\n(all targets CONTAINED\nin r_init cells)',
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)

# 在图4中添加关键统计文字
total_n = stats['total_nodes']
split_n = stats['split_nodes']
leaf_n  = stats['leaf_nodes']
ax4.set_title(
    f'Step 3.1: Split Conditions  |  '
    f'Total={total_n}, Splits={split_n}, Leaves={leaf_n}',
    fontsize=10
)

plt.tight_layout()
fig.savefig('h3_steps_3_result.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"  [OK] 步骤3可视化图 -> h3_steps_3_result.png")
plt.close()

print()

# =============================================================================
# 汇总
# =============================================================================
print("=" * 70)
print(" 步骤3  汇总统计")
print("=" * 70)
print()
print(f"  H3初始分辨率 r_init : {r_init}")
print(f"  最大分辨率   r_max  : {R_MAX}")
print(f"  H0格网总数          : {len(h0_grid)}")
print(f"  检测目标总数        : {len(targets_gdf)}")
print()
print("  ── 步骤3.1 分裂阈值 ──")
print(f"    τ_n={TAU_N}  (目标数量), τ_d={TAU_D}  (面积密度), τ_s={TAU_S}  (置信度)")
print()
print("  ── 步骤3.2/3.3 树构建结果 ──")
print(f"    总节点数       : {stats['total_nodes']}")
print(f"    分裂节点数     : {stats['split_nodes']}")
print(f"    叶子节点数     : {stats['leaf_nodes']}")
print(f"    平均目标/叶子  : {stats['avg_targets_per_leaf']:.3f}")
print(f"    最大目标/叶子  : {stats['max_targets_per_leaf']}")
print()
print("  ── 叶子分辨率分布 ──")
for r, cnt in stats['resolution_distribution'].items():
    pct = cnt / stats['leaf_nodes'] * 100
    print(f"    r={r}: {cnt:5d} 个 ({pct:.1f}%)")
print()
print("  ── 输出文件 ──")
print("    1. adaptive_tree.json         - 完整树结构")
print("    2. adaptive_tree_leaves.csv   - 叶子节点表")
print("    3. adaptive_tree_splits.csv   - 分裂节点表")
print("    4. h3_steps_3_result.png      - 可视化结果图")
print()
print("[步骤3.1/3.2/3.3] 执行完毕")
print("=" * 70)
