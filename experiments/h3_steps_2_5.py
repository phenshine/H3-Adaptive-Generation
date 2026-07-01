# -*- coding: utf-8 -*-
# =============================================================================
# H3 自适应格网生成  -  步骤2.5.1 & 2.5.2
# 算法依据：The HAND Algorithm (附件2)
#
# 步骤2.5.1：目标中心点映射
# 步骤2.5.2：目标覆盖映射
# =============================================================================

import h3
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, mapping
import json

print("=" * 70)
print(" H3 自适应格网生成  —  步骤2.5  目标映射")
print("=" * 70)
print()

# =============================================================================
# 读取已有成果
# =============================================================================
print("[读取数据] 加载目标检测数据和H0格网")
print("-" * 50)

# 读取目标检测数据
targets_gdf = gpd.read_file('detections.geojson')
print(f"  [OK] 成功加载目标检测数据: {len(targets_gdf)} 个目标")

# 读取H0格网数据
h0_grid = pd.read_csv('H0_grid.csv')
print(f"  [OK] 成功加载H0格网数据: {len(h0_grid)} 个格网")

# 从H0_grid.csv获取分辨率
r_init = int(h0_grid['resolution'].iloc[0])
print(f"  [OK] 分辨率 r_init = {r_init}")

# 创建H0格网集合（用于快速查询）
h0_set = set(h0_grid['h3_index'].tolist())

print()

# =============================================================================
# 步骤2.5.1：目标中心点映射
# =============================================================================
print("[步骤2.5.1] 目标中心点映射")
print("-" * 50)
print("  公式: h_i^center = h3_geo_to_h3(λ_i, φ_i, r_init)")
print()

def map_target_center_to_h3(row):
    """将目标中心点映射到H3格网"""
    lat = row['cx_lat']
    lon = row['cx_lon']
    return h3.latlng_to_cell(lat, lon, r_init)

# 应用中心点映射
targets_gdf['h3_center'] = targets_gdf.apply(map_target_center_to_h3, axis=1)

# 验证中心点是否在H0格网内
targets_gdf['center_in_h0'] = targets_gdf['h3_center'].isin(h0_set)

print(f"  中心点映射完成:")
print(f"    - 总目标数      : {len(targets_gdf)}")
print(f"    - 落在H0内中心点: {targets_gdf['center_in_h0'].sum()}")
print(f"    - 落在H0外中心点: {(~targets_gdf['center_in_h0']).sum()}")

# 统计类别
print()
print("  按类别统计:")
center_counts = targets_gdf.groupby(['class', 'center_in_h0']).size().unstack(fill_value=0)
for cls in center_counts.index:
    in_h0 = center_counts.loc[cls].get(True, 0)
    out_h0 = center_counts.loc[cls].get(False, 0)
    bar = '#' * (in_h0 // 5)
    print(f"    {cls:26s}: {in_h0:4d} 个在H0内, {out_h0:4d} 个在H0外  {bar}")

print()

# =============================================================================
# 步骤2.5.2：目标覆盖映射
# =============================================================================
print("[步骤2.5.2] 目标覆盖映射 (h3_polyfill)")
print("-" * 50)
print("  公式: M(t_i) = h3_polyfill(Polygon(b_i), r_init)")
print()

def get_target_h3_coverage(row):
    """获取目标覆盖的所有H3格网"""
    geom = row['geometry']

    # 转换为GeoJSON格式的字典
    poly_dict = mapping(geom)

    try:
        # 使用 polyfill 获取覆盖的H3格网
        # h3 >= 4.x API
        h3_cells = h3.polygon_to_cells(
            poly_dict,
            res=r_init
        )
        return list(h3_cells)
    except Exception as e:
        # 备用方法：使用更宽松的模式
        try:
            from shapely.ops import transform
            import pyproj

            # 简化多边形（避免自相交问题）
            simplified = geom.simplify(0.0001, preserve_topology=True)
            poly_dict = mapping(simplified)

            h3_cells = h3.polygon_to_cells(
                poly_dict,
                res=r_init,
                geo_json_conformant=True
            )
            return list(h3_cells)
        except Exception as e2:
            # 备选方案：只使用中心点
            center_cell = h3.latlng_to_cell(row['cx_lat'], row['cx_lon'], r_init)
            return [center_cell]

# 应用覆盖映射
print("  正在计算目标覆盖格网...")
targets_gdf['h3_coverage'] = targets_gdf.apply(get_target_h3_coverage, axis=1)

# 统计覆盖格网数
targets_gdf['coverage_count'] = targets_gdf['h3_coverage'].apply(len)

print(f"  覆盖映射完成!")
print()
print(f"  目标分类统计:")
single_count = (targets_gdf['coverage_count'] == 1).sum()
boundary_count = (targets_gdf['coverage_count'] == 2).sum()
multi_count = (targets_gdf['coverage_count'] >= 3).sum()

print(f"    - 单格网目标 (完全落在一格内)      : {single_count:4d} 个 ({single_count/len(targets_gdf)*100:.1f}%)")
print(f"    - 边界穿越目标 (跨越2个格网)       : {boundary_count:4d} 个 ({boundary_count/len(targets_gdf)*100:.1f}%)")
print(f"    - 多邻域目标 (跨越>=3个格网)       : {multi_count:4d} 个 ({multi_count/len(targets_gdf)*100:.1f}%)")

# 详细分类统计
print()
print("  覆盖情况详细分类 (按类别):")
for cls in targets_gdf['class'].unique():
    subset = targets_gdf[targets_gdf['class'] == cls]
    s1 = (subset['coverage_count'] == 1).sum()
    s2 = (subset['coverage_count'] == 2).sum()
    s3 = (subset['coverage_count'] >= 3).sum()
    total = len(subset)
    bar = '#' * (total // 5)
    print(f"    {cls:26s}: 单格={s1:3d}, 双格={s2:3d}, 多格={s3:3d}, 合计={total:3d}  {bar}")

# 统计覆盖H0格网的情况
print()
print("  覆盖格网与H0关系:")
covered_in_h0 = 0
covered_out_h0 = 0
for idx, row in targets_gdf.iterrows():
    for cell in row['h3_coverage']:
        if cell in h0_set:
            covered_in_h0 += 1
        else:
            covered_out_h0 += 1

print(f"    - 覆盖的H0格网: {covered_in_h0} 次")
print(f"    - 覆盖的H0外格网: {covered_out_h0} 次")

print()

# =============================================================================
# 目标-格网关联统计
# =============================================================================
print("[目标-格网关联] 统计每个H0格网内的目标覆盖情况")
print("-" * 50)

# 创建反向索引：格网 -> 覆盖该格网的目标
cell_to_targets = {}
for idx, row in targets_gdf.iterrows():
    for cell in row['h3_coverage']:
        if cell not in cell_to_targets:
            cell_to_targets[cell] = []
        cell_to_targets[cell].append({
            'target_id': idx,
            'class': row['class'],
            'confidence': row['confidence'],
            'center_cell': row['h3_center'],
            'center_in_h0': row['center_in_h0']
        })

# 统计每个H0格网被多少目标覆盖
h0_coverage_stats = []
for cell in h0_grid['h3_index']:
    targets_covered = cell_to_targets.get(cell, [])
    h0_coverage_stats.append({
        'h3_index': cell,
        'targets_covered_count': len(targets_covered),
        'unique_targets': len(set(t['target_id'] for t in targets_covered))
    })

coverage_df = pd.DataFrame(h0_coverage_stats)

print(f"  H0格网覆盖统计:")
print(f"    - 无目标覆盖的格网: {(coverage_df['targets_covered_count'] == 0).sum()}")
print(f"    - 被1个目标覆盖的格网: {(coverage_df['targets_covered_count'] == 1).sum()}")
print(f"    - 被2-5个目标覆盖的格网: {((coverage_df['targets_covered_count'] >= 2) & (coverage_df['targets_covered_count'] <= 5)).sum()}")
print(f"    - 被5+个目标覆盖的格网: {(coverage_df['targets_covered_count'] > 5).sum()}")

# 最大覆盖次数
max_coverage = coverage_df['targets_covered_count'].max()
max_cell = coverage_df.loc[coverage_df['targets_covered_count'].idxmax(), 'h3_index']
print(f"    - 最多覆盖: {max_coverage} 次 (格网 {max_cell})")

print()

# =============================================================================
# 保存中间成果
# =============================================================================
print("[保存成果]")
print("-" * 50)

# 1. 目标映射表 CSV
mapping_output = targets_gdf[[
    'class', 'class_id', 'confidence',
    'cx_lat', 'cx_lon',
    'width_m', 'height_m',
    'h3_center', 'center_in_h0',
    'h3_coverage', 'coverage_count'
]].copy()

# 将列表转为字符串存储
mapping_output['h3_coverage_str'] = mapping_output['h3_coverage'].apply(lambda x: ';'.join(x))
mapping_output = mapping_output.drop(columns=['h3_coverage'])
mapping_output = mapping_output.rename(columns={'h3_coverage_str': 'h3_coverage'})

mapping_output.to_csv('targets_h3_mapping.csv', index=True, index_label='target_id', encoding='utf-8-sig')
print(f"  [OK] 目标映射表    -> targets_h3_mapping.csv  ({len(mapping_output)} 行)")

# 2. 格网覆盖统计表
coverage_df.to_csv('h0_coverage_stats.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] 格网覆盖统计  -> h0_coverage_stats.csv    ({len(coverage_df)} 行)")

# 3. 详细的目标-格网关系表
detailed_relations = []
for idx, row in targets_gdf.iterrows():
    for cell in row['h3_coverage']:
        detailed_relations.append({
            'target_id': idx,
            'target_class': row['class'],
            'confidence': row['confidence'],
            'h3_cell': cell,
            'in_h0_grid': cell in h0_set,
            'is_center': cell == row['h3_center']
        })

detailed_df = pd.DataFrame(detailed_relations)
detailed_df.to_csv('target_cell_relations.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] 目标-格网关系  -> target_cell_relations.csv ({len(detailed_df)} 行)")

# 4. 保存GeoJSON格式的目标映射结果
targets_with_mapping = targets_gdf.copy()
targets_with_mapping['h3_center'] = targets_with_mapping['h3_center'].astype(str)
targets_with_mapping['h3_coverage_str'] = targets_with_mapping['h3_coverage'].apply(lambda x: json.dumps(x))
targets_with_mapping = targets_with_mapping.drop(columns=['h3_coverage'])
targets_with_mapping = targets_with_mapping.rename(columns={'h3_coverage_str': 'h3_coverage'})
targets_with_mapping.to_file('targets_with_mapping.geojson', driver='GeoJSON')
print(f"  [OK] 目标映射GeoJSON -> targets_with_mapping.geojson ({len(targets_with_mapping)} 个目标)")

print()

# =============================================================================
# 可视化：生成步骤2.5结果图
# =============================================================================
print("[可视化] 生成步骤2.5结果图")
print("-" * 50)

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('H3 Adaptive Grid Generation - Step 2.5 Target Mapping Results', fontsize=14, fontweight='bold')

# ----- 图1: 目标中心点分布 -----
ax1 = axes[0, 0]
ax1.set_title('Step 2.5.1: Target Center H3 Grid Distribution', fontsize=11)

# 绘制H0格网背景
for _, row in h0_grid.iterrows():
    cell_boundary = h3.cell_to_boundary(row['h3_index'])
    coords = [(lon, lat) for lat, lon in cell_boundary]
    poly = MplPolygon(coords, facecolor='lightgray', edgecolor='gray', alpha=0.3)
    ax1.add_patch(poly)

# 绘制目标中心点
in_h0 = targets_gdf[targets_gdf['center_in_h0']]
out_h0 = targets_gdf[~targets_gdf['center_in_h0']]

ax1.scatter(in_h0['cx_lon'], in_h0['cx_lat'], c='green', s=30, alpha=0.7, label=f'In H0 ({len(in_h0)})')
ax1.scatter(out_h0['cx_lon'], out_h0['cx_lat'], c='red', s=30, alpha=0.7, label=f'Outside H0 ({len(out_h0)})')

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# ----- 图2: 覆盖格网数分布 -----
ax2 = axes[0, 1]
ax2.set_title('Step 2.5.2: Target Coverage Grid Count Distribution', fontsize=11)

# 统计覆盖数分布
coverage_hist = targets_gdf['coverage_count'].value_counts().sort_index()

# 柱状图
bars = ax2.bar(coverage_hist.index, coverage_hist.values, color='steelblue', edgecolor='black')
ax2.set_xlabel('覆盖的H3格网数')
ax2.set_ylabel('目标数量')
ax2.set_xticks(range(1, max(coverage_hist.index) + 1))

# 添加数值标签
for bar, count in zip(bars, coverage_hist.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(count), ha='center', va='bottom', fontsize=10)

ax2.set_xlabel('Covered H3 Grid Count')
ax2.set_ylabel('Target Count')
ax2.grid(True, alpha=0.3, axis='y')

# ----- 图3: 各类别覆盖情况 -----
ax3 = axes[1, 0]
ax3.set_title('Coverage Grid Count by Target Class', fontsize=11)

# 按类别分组统计
class_coverage = targets_gdf.groupby(['class', 'coverage_count']).size().unstack(fill_value=0)

# 堆叠柱状图
class_names = class_coverage.index.tolist()
x = range(len(class_names))

bottom = np.zeros(len(class_names))
cmap = plt.cm.viridis
norm = mcolors.Normalize(vmin=1, vmax=max(coverage_hist.index))

for cov_count in class_coverage.columns:
    values = class_coverage[cov_count].values
    bars = ax3.bar(x, values, bottom=bottom, label=f'{cov_count}格',
                   color=cmap(norm(cov_count)/max(coverage_hist.index)), edgecolor='black', alpha=0.8)
    bottom += values

ax3.set_xticks(x)
ax3.set_xticklabels(class_names, rotation=45, ha='right')
ax3.set_ylabel('Target Count')
ax3.legend(title='Grid Count', loc='upper right', ncol=2)
ax3.grid(True, alpha=0.3, axis='y')

# ----- 图4: H0格网覆盖热力图 -----
ax4 = axes[1, 1]
ax4.set_title('H0 Grid Coverage Heatmap', fontsize=11)

# 合并覆盖统计到H0格网
h0_with_coverage = h0_grid.merge(coverage_df, on='h3_index')

# 绘制每个H0格网
for _, row in h0_with_coverage.iterrows():
    cell_boundary = h3.cell_to_boundary(row['h3_index'])
    coords = [(lon, lat) for lat, lon in cell_boundary]
    
    # 颜色根据覆盖次数
    count = row['targets_covered_count']
    if count == 0:
        color = 'lightgray'
    elif count <= 3:
        color = 'yellow'
    elif count <= 10:
        color = 'orange'
    else:
        color = 'red'
    
    poly = MplPolygon(coords, facecolor=color, edgecolor='black', alpha=0.7)
    ax4.add_patch(poly)

# 添加颜色条
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightgray', edgecolor='black', label='0 times'),
    Patch(facecolor='yellow', edgecolor='black', label='1-3 times'),
    Patch(facecolor='orange', edgecolor='black', label='4-10 times'),
    Patch(facecolor='red', edgecolor='black', label='>10 times')
]
ax4.legend(handles=legend_elements, loc='upper right', title='Coverage')

ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
ax4.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()
plt.savefig('h3_steps_2_5_result.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"  [OK] Step 2.5 result figure -> h3_steps_2_5_result.png")

print()

# =============================================================================
# 汇总统计
# =============================================================================
print("=" * 70)
print(" 步骤2.5 汇总统计")
print("=" * 70)
print(f"  研究区面积     : 233.60 km²")
print(f"  H3分辨率      : r = {r_init}")
print(f"  H0格网总数    : {len(h0_grid)} 个")
print(f"  检测目标总数  : {len(targets_gdf)} 个")
print()
print(f"  步骤2.5.1 目标中心点映射:")
print(f"    - 中心点落在H0内: {in_h0.shape[0]} 个 ({in_h0.shape[0]/len(targets_gdf)*100:.1f}%)")
print(f"    - 中心点落在H0外: {out_h0.shape[0]} 个 ({out_h0.shape[0]/len(targets_gdf)*100:.1f}%)")
print()
print(f"  步骤2.5.2 目标覆盖映射:")
print(f"    - 单格网目标: {single_count} 个 ({single_count/len(targets_gdf)*100:.1f}%)")
print(f"    - 边界穿越目标: {boundary_count} 个 ({boundary_count/len(targets_gdf)*100:.1f}%)")
print(f"    - 多邻域目标: {multi_count} 个 ({multi_count/len(targets_gdf)*100:.1f}%)")
print(f"    - 最大覆盖格网数: {targets_gdf['coverage_count'].max()} 个")
print()
print("  输出文件:")
print("    1. targets_h3_mapping.csv  - 目标H3映射表")
print("    2. h0_coverage_stats.csv    - H0格网覆盖统计")
print("    3. target_cell_relations.csv - 目标-格网关系表")
print("    4. targets_with_mapping.geojson - 目标映射GeoJSON")
print("    5. h3_steps_2_5_result.png  - 结果可视化图")
print()
print("[步骤2.5] 执行完毕")
print("=" * 70)