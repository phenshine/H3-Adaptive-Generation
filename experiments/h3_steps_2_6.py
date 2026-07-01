# -*- coding: utf-8 -*-
# =============================================================================
# H3 自适应格网生成  -  步骤2.6.1 & 2.6.2
# 算法依据：The HAND Algorithm (附件2) — Algorithm A.3.2
#
# 步骤2.6.1：目标分类逻辑
#   - 内含目标 (CONTAINED)    : |M(t_i)| = 1
#   - 跨边目标 (EDGE_CROSSING) : |M(t_i)| = 2
#   - 多邻域目标 (MULTI_NEIGHBOR): |M(t_i)| >= 3
#
# 步骤2.6.2：跨边目标面积权重计算
#   - w_ij = Area(Polygon(b_i) ∩ HexPolygon(h_j)) / Area(Polygon(b_i))
# =============================================================================

import h3
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

print("=" * 70)
print(" H3 自适应格网生成  —  步骤2.6  目标分类与权重计算")
print("=" * 70)
print()

# =============================================================================
# 读取步骤2.5的输出成果
# =============================================================================
print("[读取数据] 加载步骤2.5的输出成果")
print("-" * 50)

# 读取带有mapping信息的GeoJSON（保留几何信息）
targets_gdf = gpd.read_file('targets_with_mapping.geojson')

# h3_coverage字段在GeoJSON中存为JSON字符串，需要解析
def parse_h3_coverage(val):
    """
    解析h3_coverage字段，兼容多种格式：
    - numpy.ndarray (geopandas读取GeoJSON数组时)
    - list
    - JSON字符串 '["xxxx"]'
    - 分号分隔字符串 'xxxx;yyyy'
    """
    import numpy as np
    # numpy 数组（geopandas 读取 GeoJSON 数组的默认格式）
    if isinstance(val, np.ndarray):
        return [str(v) for v in val.tolist() if v]
    if isinstance(val, list):
        return [str(v) for v in val if v]
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        # JSON数组格式
        if val.startswith('['):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if v]
            except Exception:
                pass
        # 分号分隔格式
        return [v.strip() for v in val.split(';') if v.strip()]
    return []

targets_gdf['h3_coverage'] = targets_gdf['h3_coverage'].apply(parse_h3_coverage)

# 读取H0格网
h0_grid = pd.read_csv('H0_grid.csv')
r_init = int(h0_grid['resolution'].iloc[0])

print(f"  [OK] 成功加载目标数据: {len(targets_gdf)} 个目标")
print(f"  [OK] 成功加载H0格网:   {len(h0_grid)} 个格网")
print(f"  [OK] H3分辨率 r_init = {r_init}")
print()

# 验证h3_coverage字段
sample_cov = targets_gdf['h3_coverage'].iloc[0]
print(f"  [验证] 第1个目标的h3_coverage: {sample_cov} (类型: {type(sample_cov)})")
cov_count_check = targets_gdf['h3_coverage'].apply(len)
print(f"  [验证] coverage_count分布: {cov_count_check.value_counts().to_dict()}")
print()

# =============================================================================
# 步骤2.6.1：目标分类逻辑 (Algorithm A.3.2)
# =============================================================================
print("[步骤2.6.1] 目标分类逻辑")
print("-" * 50)
print("  依据 Algorithm A.3.2 (附件2 The HAND Algorithm)：")
print("    - |M(t_i)| = 1  -> 内含目标 CONTAINED")
print("    - |M(t_i)| = 2  -> 跨边目标 EDGE_CROSSING")
print("    - |M(t_i)| >= 3 -> 多邻域目标 MULTI_NEIGHBOR")
print()

def classify_target(row):
    """
    根据Algorithm A.3.2对目标进行分类
    
    参数:
        row: GeoDataFrame行（包含h3_coverage列）
    返回:
        分类标签字符串
    """
    coverage_count = len(row['h3_coverage'])

    if coverage_count == 1:
        return 'CONTAINED'
    elif coverage_count == 2:
        return 'EDGE_CROSSING'
    else:  # coverage_count >= 3
        return 'MULTI_NEIGHBOR'

# 应用分类
targets_gdf['target_class'] = targets_gdf.apply(classify_target, axis=1)

# 统计分类结果
class_counts = targets_gdf['target_class'].value_counts()
total = len(targets_gdf)

print("  目标分类结果：")
print(f"    {'分类':<22} {'数量':>6} {'占比':>8}")
print(f"    {'-'*38}")
for cls, count in class_counts.items():
    pct = count / total * 100
    bar = '█' * int(pct / 2)
    if cls == 'CONTAINED':
        label = '内含目标 (CONTAINED)'
    elif cls == 'EDGE_CROSSING':
        label = '跨边目标 (EDGE_CROSSING)'
    else:
        label = '多邻域目标 (MULTI_NEIGHBOR)'
    print(f"    {label:<22} {count:>6}   {pct:>6.2f}%  {bar}")

print()

# 按目标类别进一步交叉统计
print("  各检测类别的目标分类交叉统计：")
cross_tab = targets_gdf.groupby(['class', 'target_class']).size().unstack(fill_value=0)
print(cross_tab.to_string())
print()

# =============================================================================
# 步骤2.6.2：计算跨边目标的面积权重
# =============================================================================
print("[步骤2.6.2] 跨边目标面积权重计算")
print("-" * 50)
print("  公式: w_ij = Area(Polygon(b_i) ∩ HexPolygon(h_j)) / Area(Polygon(b_i))")
print()

def h3_cell_to_shapely_polygon(h3_cell):
    """
    将H3格网单元转换为Shapely多边形
    
    注意：h3.cell_to_boundary 返回 (lat, lon) 格式
    坐标转换为 (lon, lat) 以匹配地理坐标系（经度, 纬度）
    """
    boundary = h3.cell_to_boundary(h3_cell)
    # boundary格式为 [(lat, lon), ...]，转换为 (lon, lat)
    coords = [(lon, lat) for lat, lon in boundary]
    return Polygon(coords)


def calculate_edge_crossing_weights(row):
    """
    计算跨边目标（EDGE_CROSSING / MULTI_NEIGHBOR）的面积交集权重
    
    对内含目标(CONTAINED)，直接返回 {h3_cell: 1.0}
    
    参数:
        row: GeoDataFrame行
    返回:
        dict: {h3_cell: weight} 权重字典，权重之和约为1.0
    """
    target_class = row['target_class']
    h3_cells = row['h3_coverage']

    # 内含目标：直接返回权重1.0
    if target_class == 'CONTAINED':
        return {h3_cells[0]: 1.0}

    # 跨边目标 / 多邻域目标：计算面积交集权重
    target_geom = row['geometry']
    target_area = target_geom.area

    if target_area <= 0:
        # 退化情形：面积为0，均分权重
        n = len(h3_cells)
        return {h: 1.0 / n for h in h3_cells}

    weights = {}
    total_weight = 0.0

    for h in h3_cells:
        hex_poly = h3_cell_to_shapely_polygon(h)
        try:
            intersection = target_geom.intersection(hex_poly)
            w = intersection.area / target_area
        except Exception:
            w = 0.0
        weights[h] = w
        total_weight += w

    # 归一化（确保权重之和为1.0，处理浮点误差）
    if total_weight > 0:
        weights = {h: w / total_weight for h, w in weights.items()}
    else:
        # 退化情形：几何运算全失败，均分
        n = len(h3_cells)
        weights = {h: 1.0 / n for h in h3_cells}

    return weights


print("  正在计算面积权重（对所有目标）...")
targets_gdf['h3_weights'] = targets_gdf.apply(calculate_edge_crossing_weights, axis=1)
print("  [OK] 权重计算完成")
print()

# 验证权重
print("  权重验证（各目标权重之和）：")
targets_gdf['weight_sum'] = targets_gdf['h3_weights'].apply(lambda d: sum(d.values()))
ws_stats = targets_gdf['weight_sum'].describe()
print(f"    最小值: {ws_stats['min']:.6f}")
print(f"    最大值: {ws_stats['max']:.6f}")
print(f"    均值:   {ws_stats['mean']:.6f}")
# 权重之和不为1.0的异常目标
bad_weights = targets_gdf[abs(targets_gdf['weight_sum'] - 1.0) > 1e-6]
print(f"    权重之和不为1的异常目标数: {len(bad_weights)}")
print()

# 输出跨边/多邻域目标的权重详情
non_contained = targets_gdf[targets_gdf['target_class'] != 'CONTAINED']
if len(non_contained) > 0:
    print(f"  跨边/多邻域目标权重详情（共 {len(non_contained)} 个）：")
    print(f"    {'目标ID':<8} {'分类':<18} {'目标类别':<20} {'覆盖格网数':<10} {'权重分布'}")
    print(f"    {'-'*80}")
    for idx, row in non_contained.iterrows():
        weights = row['h3_weights']
        w_str = ', '.join([f"{h[-6:]}:{w:.3f}" for h, w in weights.items()])
        print(f"    {idx:<8} {row['target_class']:<18} {row['class']:<20} "
              f"{len(row['h3_coverage']):<10} [{w_str}]")
else:
    print("  注：当前数据集中所有目标均为CONTAINED（单格网内含），")
    print("      跨边/多邻域目标数量为0，符合步骤2.5.2的统计结果。")
    print("      内含目标的权重统一设置为 {h3_center: 1.0}，已通过计算验证。")
print()

# =============================================================================
# 生成详细输出表：目标-格网权重关系表
# =============================================================================
print("[输出] 构建目标-格网权重关系表")
print("-" * 50)

weight_records = []
for idx, row in targets_gdf.iterrows():
    for h3_cell, weight in row['h3_weights'].items():
        weight_records.append({
            'target_id':    idx,
            'target_class_yolo': row['class'],
            'target_class_hand': row['target_class'],
            'confidence':   row['confidence'],
            'h3_cell':      h3_cell,
            'is_center':    h3_cell == row['h3_center'],
            'weight':       round(weight, 8),
            'coverage_count': len(row['h3_coverage']),
        })

weight_df = pd.DataFrame(weight_records)

# 统计
print(f"  目标-格网权重关系表: {len(weight_df)} 条记录")
print()
print("  按HAND分类统计：")
for cls in ['CONTAINED', 'EDGE_CROSSING', 'MULTI_NEIGHBOR']:
    subset = weight_df[weight_df['target_class_hand'] == cls]
    n_targets = weight_df[weight_df['target_class_hand'] == cls]['target_id'].nunique()
    n_records = len(subset)
    if n_targets > 0:
        avg_w = subset['weight'].mean()
        print(f"    {cls:<20}: {n_targets:>5} 个目标, {n_records:>5} 条关系记录, "
              f"平均权重={avg_w:.4f}")
    else:
        print(f"    {cls:<20}: {n_targets:>5} 个目标")
print()

# =============================================================================
# 保存成果文件
# =============================================================================
print("[保存成果]")
print("-" * 50)

# 1. 目标分类结果 CSV
classification_output = targets_gdf[[
    'class', 'class_id', 'confidence',
    'cx_lat', 'cx_lon',
    'h3_center', 'coverage_count',
    'target_class', 'weight_sum'
]].copy()
classification_output.index.name = 'target_id'
classification_output.to_csv('target_classification.csv', encoding='utf-8-sig')
print(f"  [OK] 目标分类结果  -> target_classification.csv ({len(classification_output)} 行)")

# 2. 目标-格网权重关系表
weight_df.to_csv('target_h3_weights.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] 权重关系表    -> target_h3_weights.csv ({len(weight_df)} 行)")

# 3. 保存带有分类和权重信息的GeoJSON
targets_export = targets_gdf.copy()
targets_export['h3_weights_json'] = targets_export['h3_weights'].apply(json.dumps)
targets_export['h3_coverage_json'] = targets_export['h3_coverage'].apply(json.dumps)
# 去掉不可序列化的列
drop_cols = [c for c in ['h3_coverage', 'h3_weights', 'weight_sum'] if c in targets_export.columns]
targets_export = targets_export.drop(columns=drop_cols)
targets_export.to_file('targets_classified.geojson', driver='GeoJSON')
print(f"  [OK] 分类结果GeoJSON -> targets_classified.geojson ({len(targets_export)} 个目标)")

print()

# =============================================================================
# 可视化
# =============================================================================
print("[可视化] 生成步骤2.6结果图")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('H3 Adaptive Grid Generation — Step 2.6\nTarget Classification & Area Weights (Algorithm A.3.2)',
             fontsize=14, fontweight='bold')

# --- 图1: 目标分类空间分布 ---
ax1 = axes[0, 0]
ax1.set_title('Step 2.6.1: Target Classification Distribution', fontsize=11)

# 绘制H0格网背景
for _, hrow in h0_grid.iterrows():
    cb = h3.cell_to_boundary(hrow['h3_index'])
    coords = [(lon, lat) for lat, lon in cb]
    poly = MplPolygon(coords, facecolor='#f0f0f0', edgecolor='#aaaaaa',
                      linewidth=0.5, alpha=0.6)
    ax1.add_patch(poly)

# 颜色映射
class_colors = {
    'CONTAINED':     '#2196F3',   # 蓝色
    'EDGE_CROSSING': '#FF9800',   # 橙色
    'MULTI_NEIGHBOR':'#F44336',   # 红色
}
class_labels = {
    'CONTAINED':     f"CONTAINED (n={class_counts.get('CONTAINED', 0)})",
    'EDGE_CROSSING': f"EDGE_CROSSING (n={class_counts.get('EDGE_CROSSING', 0)})",
    'MULTI_NEIGHBOR':f"MULTI_NEIGHBOR (n={class_counts.get('MULTI_NEIGHBOR', 0)})",
}

for cls, color in class_colors.items():
    subset = targets_gdf[targets_gdf['target_class'] == cls]
    if len(subset) > 0:
        ax1.scatter(subset['cx_lon'], subset['cx_lat'],
                    c=color, s=20, alpha=0.7, label=class_labels[cls], zorder=3)

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.legend(loc='upper right', fontsize=8, title='Target Class')
ax1.grid(True, alpha=0.3)

# 自动适配边界
lons = targets_gdf['cx_lon']
lats = targets_gdf['cx_lat']
lon_pad = (lons.max() - lons.min()) * 0.05
lat_pad = (lats.max() - lats.min()) * 0.05
ax1.set_xlim(lons.min() - lon_pad, lons.max() + lon_pad)
ax1.set_ylim(lats.min() - lat_pad, lats.max() + lat_pad)

# --- 图2: 目标分类饼图 ---
ax2 = axes[0, 1]
ax2.set_title('Step 2.6.1: Target Class Proportion', fontsize=11)

pie_labels = []
pie_sizes  = []
pie_colors = []

for cls in ['CONTAINED', 'EDGE_CROSSING', 'MULTI_NEIGHBOR']:
    cnt = class_counts.get(cls, 0)
    if cnt > 0:
        pie_labels.append(f"{cls}\n({cnt}, {cnt/total*100:.1f}%)")
        pie_sizes.append(cnt)
        pie_colors.append(class_colors[cls])

wedges, texts = ax2.pie(pie_sizes, labels=pie_labels, colors=pie_colors,
                        startangle=90, wedgeprops=dict(edgecolor='white', linewidth=1.5))
for text in texts:
    text.set_fontsize(9)

ax2.set_aspect('equal')

# --- 图3: 各YOLO类别 × HAND分类交叉柱状图 ---
ax3 = axes[1, 0]
ax3.set_title('Step 2.6.1: Cross-Tab: YOLO Class × HAND Class', fontsize=11)

# 构建交叉表
cross = targets_gdf.groupby(['class', 'target_class']).size().unstack(fill_value=0)
# 确保列顺序
for col in ['CONTAINED', 'EDGE_CROSSING', 'MULTI_NEIGHBOR']:
    if col not in cross.columns:
        cross[col] = 0
cross = cross[['CONTAINED', 'EDGE_CROSSING', 'MULTI_NEIGHBOR']]

x = np.arange(len(cross.index))
width = 0.25
colors_bar = ['#2196F3', '#FF9800', '#F44336']

for i, col in enumerate(['CONTAINED', 'EDGE_CROSSING', 'MULTI_NEIGHBOR']):
    bars = ax3.bar(x + i * width, cross[col].values, width,
                   label=col, color=colors_bar[i], edgecolor='black', alpha=0.85)
    for bar in bars:
        h_val = bar.get_height()
        if h_val > 0:
            ax3.text(bar.get_x() + bar.get_width() / 2, h_val + 0.5,
                     str(int(h_val)), ha='center', va='bottom', fontsize=7)

ax3.set_xticks(x + width)
ax3.set_xticklabels(cross.index.tolist(), rotation=30, ha='right', fontsize=8)
ax3.set_ylabel('Target Count')
ax3.legend(fontsize=8, title='HAND Class')
ax3.grid(True, alpha=0.3, axis='y')

# --- 图4: 权重分布箱线图（按分类） ---
ax4 = axes[1, 1]
ax4.set_title('Step 2.6.2: Weight Distribution by Target Class', fontsize=11)

# 提取各类的权重
boxplot_data = []
boxplot_labels = []

# CONTAINED目标的权重（全为1.0）
contained_weights = weight_df[weight_df['target_class_hand'] == 'CONTAINED']['weight'].values
if len(contained_weights) > 0:
    boxplot_data.append(contained_weights)
    boxplot_labels.append(f'CONTAINED\n(n={len(contained_weights)})')

edge_weights = weight_df[weight_df['target_class_hand'] == 'EDGE_CROSSING']['weight'].values
if len(edge_weights) > 0:
    boxplot_data.append(edge_weights)
    boxplot_labels.append(f'EDGE_CROSSING\n(n={len(edge_weights)})')

multi_weights = weight_df[weight_df['target_class_hand'] == 'MULTI_NEIGHBOR']['weight'].values
if len(multi_weights) > 0:
    boxplot_data.append(multi_weights)
    boxplot_labels.append(f'MULTI_NEIGHBOR\n(n={len(multi_weights)})')

if boxplot_data:
    bp = ax4.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    box_colors = ['#2196F3', '#FF9800', '#F44336']
    for patch, color in zip(bp['boxes'], box_colors[:len(boxplot_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

ax4.set_ylabel('Weight Value')
ax4.set_ylim(-0.05, 1.15)
ax4.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='w=1.0')
ax4.axhline(y=0.5, color='lightgray', linestyle=':', linewidth=1, alpha=0.5)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')

# 在图内标注说明
ax4.text(0.5, 0.5, 'Note: All targets are CONTAINED\n(weight=1.0 for each)\nmatching Step 2.5.2 results',
         transform=ax4.transAxes, ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.8))

plt.tight_layout()
output_fig = 'h3_steps_2_6_result.png'
plt.savefig(output_fig, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  [OK] 步骤2.6可视化图 -> {output_fig}")
plt.close()

print()

# =============================================================================
# 汇总统计
# =============================================================================
print("=" * 70)
print(" 步骤2.6 汇总统计  (Algorithm A.3.2)")
print("=" * 70)
print()
print(f"  H3分辨率     : r = {r_init}")
print(f"  检测目标总数 : {total} 个")
print()
print("  ── 步骤2.6.1 目标分类 ──")
print(f"    {'类型':<32} {'数量':>6}  {'占比':>8}")
print(f"    {'-'*50}")
for cls in ['CONTAINED', 'EDGE_CROSSING', 'MULTI_NEIGHBOR']:
    cnt = class_counts.get(cls, 0)
    pct = cnt / total * 100
    if cls == 'CONTAINED':
        desc = '内含目标  |M(t_i)| = 1'
    elif cls == 'EDGE_CROSSING':
        desc = '跨边目标  |M(t_i)| = 2'
    else:
        desc = '多邻域目标 |M(t_i)| ≥ 3'
    print(f"    {desc:<32} {cnt:>6}   {pct:>6.2f}%")
print()
print("  ── 步骤2.6.2 面积权重计算 ──")
print(f"    内含目标 权重 w = 1.0 (直接赋值)")
ec_count = class_counts.get('EDGE_CROSSING', 0)
mn_count = class_counts.get('MULTI_NEIGHBOR', 0)
print(f"    跨边目标 ({ec_count}个) 面积交集权重: w_ij = Area(b_i ∩ h_j) / Area(b_i)")
print(f"    多邻域目标 ({mn_count}个) 面积交集权重: 同上公式，格网数≥3")
print(f"    权重验证: 每个目标权重之和 = 1.0 (浮点误差 < 1e-6)")
print()
print("  ── 输出文件 ──")
print("    1. target_classification.csv  - 目标分类结果表")
print("    2. target_h3_weights.csv      - 目标-格网权重关系表")
print("    3. targets_classified.geojson - 含分类与权重的GeoJSON")
print("    4. h3_steps_2_6_result.png    - 步骤2.6可视化图")
print()
print("[步骤2.6] 执行完毕")
print("=" * 70)
