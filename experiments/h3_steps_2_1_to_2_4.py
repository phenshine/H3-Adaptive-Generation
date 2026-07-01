# -*- coding: utf-8 -*-
# =============================================================================
# H3 自适应格网生成  -  步骤2.1 ~ 步骤2.4  【修正版】
# 算法依据：The HAND Algorithm (附件2)
#
# 【修正说明】
# 1. r_init 修正：研究区面积 233 km²，按参考代码阈值应选 r_init = 7
# 2. 外扩一环：采用 Algorithm 1-H3 的边界扩展策略，确保所有目标被完备覆盖
# 3. h3-py 4.x API 适配：polyfill → h3shape_to_cells, h3_to_geo → cell_to_latlng
# =============================================================================

import h3
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box, Polygon as ShapelyPolygon

print("=" * 62)
print(" H3 自适应格网生成  —  步骤2.1 至 步骤2.4  【修正版】")
print("=" * 62)
print(f" h3 版本: {h3.__version__}")
print()

# =============================================================================
# 步骤2.1：H3 库初始化与目标数据导入
# =============================================================================
print("[步骤2.1] H3库初始化与目标数据导入")
print("-" * 42)

# ── 读取目标检测数据 ──
targets_gdf = gpd.read_file('detections.geojson')

print(f"  [OK] 成功导入目标检测数据")
print(f"       要素总数   : {len(targets_gdf)}")
print(f"       坐标系     : {targets_gdf.crs}")
print(f"       目标类别数 : {targets_gdf['class'].nunique()}")
print()
print("  目标类别分布:")
for cls, cnt in targets_gdf['class'].value_counts().items():
    bar = '#' * (cnt // 10)
    print(f"    {cls:26s}: {cnt:4d}  {bar}")

# ── 获取研究区域边界框 ──
bbox = targets_gdf.total_bounds   # [minx, miny, maxx, maxy]  (lon, lat)
print(f"\n  研究区域边界框 (EPSG:4326 / WGS84):")
print(f"    经度范围: {bbox[0]:.6f} ~ {bbox[2]:.6f}")
print(f"    纬度范围: {bbox[1]:.6f} ~ {bbox[3]:.6f}")

# 用 Shapely 构建边界矩形多边形
bbox_polygon_geo = box(bbox[0], bbox[1], bbox[2], bbox[3])

# ── 面积估算（转 UTM 50N 投影）──
targets_utm  = targets_gdf.to_crs('EPSG:32650')
bbox_utm     = targets_utm.total_bounds
bbox_poly_utm = box(bbox_utm[0], bbox_utm[1], bbox_utm[2], bbox_utm[3])
A0_m2  = bbox_poly_utm.area
A0_km2 = A0_m2 / 1e6

print(f"\n  研究区域面积（UTM 50N 投影）:")
print(f"    A0 = {A0_m2:.2f} m²  =  {A0_km2:.4f} km²")
print()

# =============================================================================
# 步骤2.2：初始分辨率计算（Algorithm 1-H3 / Nyquist 原理）
# =============================================================================
print("[步骤2.2] 初始分辨率计算  (Algorithm 1-H3)")
print("-" * 42)

# ── H3 各分辨率参数表 ──
print("  H3 分辨率参考表:")
print(f"  {'r':>3}  {'面积(km²)':>14}  {'边长(m)':>10}  {'适用场景'}")
print(f"  {'-'*3}  {'-'*14}  {'-'*10}  {'-'*20}")
for r in range(4, 12):
    area_km2 = h3.average_hexagon_area(r, unit='km^2')
    edge_m   = h3.average_hexagon_edge_length(r, unit='m')
    if   r == 4: scene = "城市级"
    elif r == 5: scene = "区域级"
    elif r == 6: scene = "大街区级"
    elif r == 7: scene = "街区级"
    elif r == 8: scene = "街道级"
    elif r == 9: scene = "遥感目标级"
    else:        scene = "精细级"
    print(f"  {r:>3}  {area_km2:>14.4f}  {edge_m:>10.1f}  {scene}")
print()

# ── 分辨率判定逻辑（修正版）──
# 参考代码阈值：
#   A0 > 1e4 km² → r=4
#   A0 > 1e2 km² → r=7  
#   else → r=9
# 研究区 A0 ≈ 233 km²，落在 (100, 10000] 区间 → r_init = 7

eta = 0.8   # 面积占有率因子

if A0_km2 > 10000:
    r_init = 4
elif A0_km2 > 100:
    r_init = 7    # 【修正】233 km² → r_init = 7（街区级）
elif A0_km2 > 10:
    r_init = 8
else:
    r_init = 9

hex_area_km2 = h3.average_hexagon_area(r_init, unit='km^2')
hex_edge_m   = h3.average_hexagon_edge_length(r_init, unit='m')
min_tgt_size = targets_gdf[['width_m', 'height_m']].min().min()

print(f"  目标最小尺寸   : {min_tgt_size:.1f} m")
print(f"  面积占有率因子 : eta = {eta}")
print(f"  研究区面积 A0  : {A0_km2:.4f} km²")
print()
print(f"  [OK] 选定初始分辨率  r_init = {r_init}")
print(f"       六边形平均面积 : {hex_area_km2:.4f} km²  ({hex_area_km2*1e6:.0f} m²)")
print(f"       六边形平均边长 : {hex_edge_m:.1f} m")
print()

# =============================================================================
# 步骤2.3：初始格网集合构建 + 外扩一环（完备覆盖策略）
# =============================================================================
print("[步骤2.3] 初始格网集合构建 + 外扩一环")
print("-" * 42)

# ── 3.1 基础 Polyfill ──
exterior_latlng = [(lat, lon) for lon, lat in bbox_polygon_geo.exterior.coords]
h3_polygon = h3.LatLngPoly(exterior_latlng)
H0_base = set(h3.h3shape_to_cells(h3_polygon, res=r_init))

print(f"  基础 Polyfill:")
print(f"    分辨率 r = {r_init}")
print(f"    初始格网数: {len(H0_base)} 个")

# ── 3.2 外扩一环（Ring-1 Expansion）──
# 根据 Algorithm 1-H3，为确保边界目标被完备覆盖，
# 对基础格网集合进行一环扩展（包含所有相邻格网）

H0_expanded = set(H0_base)
for cell in H0_base:
    # grid_ring 返回距离为 k 的所有格网（一环 = 距离1）
    try:
        ring_cells = h3.grid_ring(cell, k=1)
        H0_expanded.update(ring_cells)
    except Exception as e:
        # 五边形或边界格网可能没有完整的一环
        pass

# 统计扩展效果
expansion_count = len(H0_expanded) - len(H0_base)
print(f"\n  外扩一环 (Ring-1):")
print(f"    扩展格网数: +{expansion_count} 个")
print(f"    扩展后总数: {len(H0_expanded)} 个")

# ── 3.3 统计目标覆盖情况 ──
print(f"\n  目标覆盖验证:")

# 获取所有目标所在的格网
target_cells = set()
for _, row in targets_gdf.iterrows():
    cell = h3.latlng_to_cell(row['cx_lat'], row['cx_lon'], r_init)
    target_cells.add(cell)

print(f"    目标所在格网数: {len(target_cells)} 个")

# 基础集合覆盖率
base_covered = target_cells & H0_base
base_coverage = len(base_covered) / len(target_cells) * 100
print(f"    基础集合覆盖 : {len(base_covered)}/{len(target_cells)} ({base_coverage:.1f}%)")

# 扩展后覆盖率
exp_covered = target_cells & H0_expanded
exp_coverage = len(exp_covered) / len(target_cells) * 100
print(f"    扩展后覆盖   : {len(exp_covered)}/{len(target_cells)} ({exp_coverage:.1f}%)")

# 检查是否有遗漏
missing = target_cells - H0_expanded
if missing:
    print(f"    ⚠ 遗漏格网: {missing}")
else:
    print(f"    ✓ 完备覆盖: 所有目标格网均已包含")

H0_initial = sorted(list(H0_expanded))
print()

# =============================================================================
# 步骤2.4：构建初始格网数据结构
# =============================================================================
print("[步骤2.4] 构建初始格网数据结构")
print("-" * 42)

H0_data = []

for h_idx in H0_initial:
    center_latlng = h3.cell_to_latlng(h_idx)
    boundary = h3.cell_to_boundary(h_idx)
    parent = h3.cell_to_parent(h_idx, r_init - 1) if r_init > 0 else None
    hex_type = 'PENTAGON' if h3.is_pentagon(h_idx) else 'HEX'
    cell_area_km2 = h3.cell_area(h_idx, unit='km^2')
    
    # 检查是否含目标
    has_target = h_idx in target_cells
    
    H0_data.append({
        'h3_index'     : h_idx,
        'resolution'   : r_init,
        'center_lat'   : center_latlng[0],
        'center_lon'   : center_latlng[1],
        'boundary'     : boundary,
        'target_list'  : [],
        'parent'       : parent,
        'children'     : [],
        'hex_type'     : hex_type,
        'target_count' : 0,
        'has_target'   : has_target,
        'cell_area_km2': cell_area_km2,
        'is_expansion' : h_idx not in H0_base,  # 标记是否为扩展格网
    })

H0_df = pd.DataFrame(H0_data)

print(f"  [OK] H0 格网数据框构建完成")
print(f"       形状     : {H0_df.shape[0]} 行 x {H0_df.shape[1]} 列")
print(f"       基础格网 : {(~H0_df['is_expansion']).sum()} 个")
print(f"       扩展格网 : {H0_df['is_expansion'].sum()} 个")
print(f"       含目标   : {H0_df['has_target'].sum()} 个")
print()

# =============================================================================
# 保存成果文件
# =============================================================================
print("=" * 62)
print(" 保存成果文件")
print("=" * 62)

# ─ 1. H0 格网属性表 CSV ─
H0_save = H0_df.drop(columns=['boundary', 'target_list', 'children'])
H0_save.to_csv('H0_grid.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] H0格网属性表    -> H0_grid.csv        ({len(H0_save)} 行)")

# ─ 2. H0 格网空间文件 GeoJSON ─
def boundary_to_polygon(boundary):
    coords = [(lon, lat) for lat, lon in boundary]
    return ShapelyPolygon(coords)

H0_df['geometry'] = H0_df['boundary'].apply(boundary_to_polygon)

H0_gdf = gpd.GeoDataFrame(
    H0_df[['h3_index', 'resolution', 'center_lat', 'center_lon',
           'parent', 'hex_type', 'target_count', 'has_target',
           'cell_area_km2', 'is_expansion', 'geometry']],
    geometry='geometry',
    crs='EPSG:4326'
)
H0_gdf.to_file('H0_grid.geojson', driver='GeoJSON')
print(f"  [OK] H0格网空间文件  -> H0_grid.geojson     ({len(H0_gdf)} 个六边形)")

# =============================================================================
# 汇总统计
# =============================================================================
print()
print("=" * 62)
print(" 汇总统计")
print("=" * 62)
print(f"  研究区域     : 厦门海沧湾（福建省厦门市）")
print(f"  数据坐标系   : EPSG:4326  WGS84")
print(f"  研究区面积   : {A0_km2:.4f} km²")
print(f"  选定分辨率   : r_init = {r_init}")
print(f"  六边形边长   : {hex_edge_m:.1f} m")
print(f"  基础格网数   : {len(H0_base)} 个")
print(f"  扩展格网数   : {expansion_count} 个")
print(f"  H0 格网总数  : {len(H0_initial)} 个")
print(f"  目标格网数   : {len(target_cells)} 个")
print(f"  覆盖率       : {exp_coverage:.1f}%")
print(f"  五边形单元数 : {(H0_df['hex_type']=='PENTAGON').sum()} 个")
print(f"  检测目标总数 : {len(targets_gdf)}")
print()
print("[步骤2.1~2.4] 全部执行完毕")
print("  输出文件: H0_grid.csv  |  H0_grid.geojson")
