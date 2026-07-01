# -*- coding: utf-8 -*-
"""
CD-MCAR 六场景批量实验脚本
============================
对6景Sentinel-2 10m分辨率遥感影像的YOLO11n-OBB检测结果，
逐一运行CD-MCAR自适应格网生成完整实验流程。

场景配置:
S01: 舟山海岸带 (Coastal Zone)
S02: 南京城区 (Inland Dense Urban)
S03: 苏州县城 (County/Township)
S04: 江西农田 (Contiguous Farmland)
S05: 武夷山林地 (Hilly/Mountainous Forest)
S06: 宁波港口 (Port/Industrial Zone)

影像分辨率: Sentinel-2, 10m
检测模型: YOLO11n-OBB
对比基线: H3 Resolution 10 均匀覆盖

作者: WorkBuddy AI
日期: 2026-05-15
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

import h3
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MPLPolygon, Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from shapely.geometry import box, Polygon as ShapelyPolygon, shape
from shapely.ops import transform as shapely_transform
import pyproj
from collections import deque
import math

# ============================================================================
# 场景配置
# ============================================================================

SCENES = {
    'S01': {
        'name': 'Coastal Zone',
        'name_cn': '舟山海岸带',
        'file': '01_coastal_Zhoushan_detections.geojson',
        'crs': 'EPSG:32650',  # UTM Zone 50N
        'description': 'Coastal zone with Zhoushan Islands'
    },
    'S02': {
        'name': 'Inland Dense Urban',
        'name_cn': '南京城区',
        'file': '02_urban_Nanjing_detections.geojson',
        'crs': 'EPSG:32650',  # UTM Zone 50N
        'description': 'Nanjing urban area with dense buildings'
    },
    'S03': {
        'name': 'County/Township',
        'name_cn': '苏州县城',
        'file': '03_county_Suzhou_detections.geojson',
        'crs': 'EPSG:32650',  # UTM Zone 50N
        'description': 'Suzhou county town area'
    },
    'S04': {
        'name': 'Contiguous Farmland',
        'name_cn': '江西农田',
        'file': '04_farmland_Jiangxi_detections.geojson',
        'crs': 'EPSG:32650',  # UTM Zone 50N
        'description': 'Jiangxi agricultural farmland'
    },
    'S05': {
        'name': 'Hilly/Mountainous Forest',
        'name_cn': '武夷山林地',
        'file': '05_mountain_Wuyishan_detections.geojson',
        'crs': 'EPSG:32650',  # UTM Zone 50N
        'description': 'Wuyishan mountainous forested area'
    },
    'S06': {
        'name': 'Port/Industrial Zone',
        'name_cn': '宁波港口',
        'file': '06_port_Ningbo_detections.geojson',
        'crs': 'EPSG:32651',  # UTM Zone 51N
        'description': 'Ningbo port and industrial zone'
    }
}

# CD-MCAR 参数配置
TAU_N = 2        # 目标数量阈值
TAU_D = 0.05     # 面积密度阈值
TAU_S = 0.85     # 置信度阈值
R_MAX = 10       # 最大分辨率
R_INIT_BASE = 7  # 基础初始分辨率（可根据面积调整）

# ============================================================================
# 工具函数
# ============================================================================

def setup_utm_projection(crs_str):
    """根据CRS设置UTM投影转换器"""
    return pyproj.Transformer.from_crs(crs_str, 'EPSG:4326', always_xy=True)

def geom_to_utm_area_m2(geom, transformer):
    """将几何体转换为UTM投影后计算面积(m²)"""
    try:
        geom_utm = shapely_transform(transformer.transform, geom)
        return geom_utm.area
    except:
        return 0.0

def geom_to_utm(geom, transformer):
    """将几何体转换为UTM投影"""
    try:
        return shapely_transform(transformer.transform, geom)
    except:
        return geom

def parse_h3_coverage(val):
    """解析h3_coverage字段"""
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
                return [str(v) for v in json.loads(val) if v]
            except:
                pass
        return [v.strip() for v in val.split(';') if v.strip()]
    return []

# ============================================================================
# 步骤2: H0格网生成
# ============================================================================

def step2_generate_h0_grid(scene_id, scene_config, detections_gdf):
    """步骤2: 生成H0初始格网"""
    print(f"\n{'='*60}")
    print(f"  场景 {scene_id}: {scene_config['name']} ({scene_config['name_cn']})")
    print(f"  步骤2: H0格网生成")
    print(f"{'='*60}")

    # 设置坐标转换
    src_crs = scene_config['crs']
    transformer = pyproj.Transformer.from_crs(src_crs, 'EPSG:4326', always_xy=True)
    transformer_to_utm = pyproj.Transformer.from_crs('EPSG:4326', src_crs, always_xy=True)

    # 转换为WGS84
    detections_gdf_wgs84 = detections_gdf.to_crs('EPSG:4326')

    # 计算边界框和面积
    bbox = detections_gdf_wgs84.total_bounds  # [minx, miny, maxx, maxy] (lon, lat)
    bbox_polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])

    # 计算面积 (转换到源CRS以获得精确值)
    detections_utm = detections_gdf.to_crs(src_crs)
    bbox_utm = box(detections_utm.total_bounds[0], detections_utm.total_bounds[1],
                   detections_utm.total_bounds[2], detections_utm.total_bounds[3])
    A0_km2 = bbox_utm.area / 1e6

    print(f"  边界框: [{bbox[0]:.4f}, {bbox[1]:.4f}] ~ [{bbox[2]:.4f}, {bbox[3]:.4f}]")
    print(f"  研究区面积: {A0_km2:.4f} km²")

    # 确定初始分辨率
    if A0_km2 > 10000:
        r_init = 4
    elif A0_km2 > 100:
        r_init = 7
    elif A0_km2 > 10:
        r_init = 8
    else:
        r_init = 9

    hex_area_km2 = h3.average_hexagon_area(r_init, unit='km^2')
    print(f"  选定分辨率: r_init = {r_init} (六边形面积: {hex_area_km2:.4f} km²)")

    # 生成H0格网
    exterior_coords = [(lat, lon) for lon, lat in bbox_polygon.exterior.coords]
    h3_polygon = h3.LatLngPoly(exterior_coords)
    H0_base = set(h3.h3shape_to_cells(h3_polygon, res=r_init))

    # 外扩一环
    H0_expanded = set(H0_base)
    for cell in H0_base:
        try:
            ring_cells = h3.grid_ring(cell, k=1)
            H0_expanded.update(ring_cells)
        except:
            pass

    # 计算目标-格网映射
    target_cells = set()
    for idx, row in detections_gdf_wgs84.iterrows():
        try:
            geom = row.geometry
            if geom.geom_type == 'Polygon':
                centroid = geom.centroid
                cell = h3.latlng_to_cell(centroid.y, centroid.x, r_init)
                target_cells.add(cell)
        except:
            pass

    # 构建H0数据
    H0_data = []
    for h_idx in sorted(list(H0_expanded)):
        center_latlng = h3.cell_to_latlng(h_idx)
        boundary = h3.cell_to_boundary(h_idx)
        cell_area = h3.cell_area(h_idx, unit='km^2')

        H0_data.append({
            'h3_index': h_idx,
            'resolution': r_init,
            'center_lat': center_latlng[0],
            'center_lon': center_latlng[1],
            'boundary': boundary,
            'target_count': 0,
            'has_target': h_idx in target_cells,
            'cell_area_km2': cell_area,
            'is_expansion': h_idx not in H0_base,
        })

    H0_df = pd.DataFrame(H0_data)

    print(f"  H0格网数: {len(H0_df)} (基础: {len(H0_base)}, 扩展: +{len(H0_df)-len(H0_base)})")
    print(f"  目标所在格网: {len(target_cells)}")
    print(f"  含目标格网: {H0_df['has_target'].sum()}")

    # 为检测数据添加必要属性
    detections_gdf_wgs84['cx_lon'] = detections_gdf_wgs84.geometry.centroid.x
    detections_gdf_wgs84['cx_lat'] = detections_gdf_wgs84.geometry.centroid.y
    detections_gdf_wgs84['h3_center'] = detections_gdf_wgs84.apply(
        lambda r: h3.latlng_to_cell(r.geometry.centroid.y, r.geometry.centroid.x, r_init), axis=1
    )

    # 计算像素尺寸（如果是GeoJSON格式的检测结果）
    # 注意：这些数据已经是米为单位的尺寸
    if 'width_px' in detections_gdf.columns and 'height_px' in detections_gdf.columns:
        # 假设10m分辨率，像素尺寸乘以10得到米
        detections_gdf_wgs84['width_m'] = detections_gdf['width_px'] * 10
        detections_gdf_wgs84['height_m'] = detections_gdf['height_px'] * 10
    else:
        # 使用实际几何面积估算
        detections_gdf_wgs84['width_m'] = 100
        detections_gdf_wgs84['height_m'] = 100

    if 'confidence' not in detections_gdf_wgs84.columns:
        detections_gdf_wgs84['confidence'] = 0.8
    if 'class' not in detections_gdf_wgs84.columns:
        detections_gdf_wgs84['class'] = detections_gdf_wgs84.get('class_name', 'unknown')

    # 保存成果
    prefix = f"{scene_id}_"

    # H0格网CSV
    H0_save = H0_df.drop(columns=['boundary'])
    H0_save.to_csv(f'{prefix}H0_grid.csv', index=False, encoding='utf-8-sig')

    # H0格网GeoJSON
    def boundary_to_polygon(boundary):
        coords = [(lon, lat) for lat, lon in boundary]
        return ShapelyPolygon(coords)

    H0_df['geometry'] = H0_df['boundary'].apply(boundary_to_polygon)
    H0_gdf = gpd.GeoDataFrame(
        H0_df[['h3_index', 'resolution', 'center_lat', 'center_lon',
               'target_count', 'has_target', 'cell_area_km2', 'is_expansion', 'geometry']],
        geometry='geometry', crs='EPSG:4326'
    )
    H0_gdf.to_file(f'{prefix}H0_grid.geojson', driver='GeoJSON')

    print(f"  输出: {prefix}H0_grid.csv, {prefix}H0_grid.geojson")

    return {
        'r_init': r_init,
        'A0_km2': A0_km2,
        'H0_df': H0_df,
        'H0_initial': H0_expanded,
        'detections_gdf': detections_gdf_wgs84,
        'bbox': bbox,
        'prefix': prefix,
        'target_cells': target_cells,
        'transformer': transformer,
    }

# ============================================================================
# 步骤3: CD-MCAR自适应分裂树
# ============================================================================

class CDMCARTree:
    """CD-MCAR自适应分裂树"""

    def __init__(self, initial_resolution, r_max=10):
        self.initial_resolution = initial_resolution
        self.r_max = r_max
        self.tree = {}
        self.leaf_cells = set()

    def build_index(self, targets_gdf):
        """构建目标索引"""
        self._cell_index = {}
        self._all_targets = []

        for _, row in targets_gdf.iterrows():
            center_cell = str(row.get('h3_center', ''))
            if not center_cell:
                continue

            record = {
                'geometry': row.geometry,
                'confidence': float(row.get('confidence', 0)),
                'class': str(row.get('class', '')),
                'width_m': float(row.get('width_m', 100)),
                'height_m': float(row.get('height_m', 100)),
                'h3_center': center_cell,
            }
            self._all_targets.append((center_cell, record))
            if center_cell not in self._cell_index:
                self._cell_index[center_cell] = []
            self._cell_index[center_cell].append(record)

    def get_targets_in_cell(self, h_cell):
        """获取格网内的目标"""
        current_res = h3.get_resolution(h_cell)
        r_target = self.initial_resolution

        if current_res == r_target:
            return list(self._cell_index.get(h_cell, []))

        elif current_res < r_target:
            result = []
            for center_cell, record in self._all_targets:
                try:
                    ancestor = h3.cell_to_parent(center_cell, current_res)
                    if ancestor == h_cell:
                        result.append(record)
                except:
                    pass
            return result

        else:
            result = []
            try:
                parent_at_r = h3.cell_to_parent(h_cell, r_target)
                for record in self._cell_index.get(parent_at_r, []):
                    try:
                        centroid = record['geometry'].centroid
                        mapped_cell = h3.latlng_to_cell(centroid.y, centroid.x, current_res)
                        if mapped_cell == h_cell:
                            result.append(record)
                    except:
                        pass
            except:
                pass
            return result

    def calculate_split_criteria(self, h3_cell, targets_in_cell, resolution):
        """计算分裂判定条件"""
        target_count = len(targets_in_cell)
        cell_area_m2 = h3.cell_area(h3_cell, unit='m^2')

        # 目标总面积
        total_area_m2 = sum(
            record.get('width_m', 100) * record.get('height_m', 100)
            for record in targets_in_cell
        )
        area_density = total_area_m2 / cell_area_m2 if cell_area_m2 > 0 else 0.0

        # 最大置信度
        max_conf = max((t.get('confidence', 0) for t in targets_in_cell), default=0.0)

        should_split = (target_count >= TAU_N) or (area_density >= TAU_D) or (max_conf >= TAU_S)

        return {
            'should_split': should_split,
            'target_count': target_count,
            'cell_area_m2': cell_area_m2,
            'total_area_m2': total_area_m2,
            'area_density': area_density,
            'max_confidence': max_conf,
            'condition_n': target_count >= TAU_N,
            'condition_d': area_density >= TAU_D,
            'condition_s': max_conf >= TAU_S,
        }

    def build_tree(self, targets_gdf, H0_initial):
        """构建自适应分裂树"""
        self.build_index(targets_gdf)

        Q = deque()
        for h in H0_initial:
            Q.append((h, self.initial_resolution))

        split_count = 0

        while Q:
            h_current, resolution = Q.popleft()
            current_targets = self.get_targets_in_cell(h_current)

            if resolution < self.r_max and len(current_targets) > 0:
                split_info = self.calculate_split_criteria(h_current, current_targets, resolution)

                if split_info['should_split']:
                    children = list(h3.cell_to_children(h_current, resolution + 1))
                    split_count += 1

                    self.tree[h_current] = {
                        'resolution': resolution,
                        'targets': len(current_targets),
                        'split': True,
                        'children': children,
                        'split_criteria': split_info,
                    }

                    for child in children:
                        Q.append((child, resolution + 1))
                else:
                    self.leaf_cells.add(h_current)
                    self.tree[h_current] = {
                        'resolution': resolution,
                        'targets': len(current_targets),
                        'split': False,
                        'children': [],
                        'split_criteria': split_info,
                    }
            else:
                self.leaf_cells.add(h_current)
                self.tree[h_current] = {
                    'resolution': resolution,
                    'targets': len(current_targets),
                    'split': False,
                    'children': [],
                }

        return self.tree, self.leaf_cells

    def get_statistics(self):
        """获取统计信息"""
        split_nodes = sum(1 for n in self.tree.values() if n['split'])
        leaf_nodes = len(self.leaf_cells)
        total_nodes = len(self.tree)

        res_dist = {}
        for h in self.leaf_cells:
            r = self.tree[h]['resolution']
            res_dist[r] = res_dist.get(r, 0) + 1

        return {
            'total_nodes': total_nodes,
            'split_nodes': split_nodes,
            'leaf_nodes': leaf_nodes,
            'resolution_distribution': dict(sorted(res_dist.items())),
        }


def step3_build_adaptive_tree(step2_result):
    """步骤3: 构建CD-MCAR自适应分裂树"""
    print(f"\n{'-'*50}")
    print(f"  步骤3: CD-MCAR自适应分裂树构建")
    print(f"{'-'*50}")

    r_init = step2_result['r_init']
    H0_initial = step2_result['H0_initial']
    detections_gdf = step2_result['detections_gdf']
    prefix = step2_result['prefix']

    # 构建树
    tree = CDMCARTree(r_init, r_max=R_MAX)
    tree_structure, leaf_set = tree.build_tree(detections_gdf, H0_initial)
    stats = tree.get_statistics()

    print(f"  总节点数: {stats['total_nodes']}")
    print(f"  分裂节点: {stats['split_nodes']}")
    print(f"  叶子节点: {stats['leaf_nodes']}")
    print(f"  分辨率分布:")
    for r, cnt in sorted(stats['resolution_distribution'].items()):
        print(f"    r={r}: {cnt}")

    # 保存树结构
    tree_export = {}
    for cell, info in tree_structure.items():
        node = {
            'resolution': info['resolution'],
            'targets': info['targets'],
            'split': info['split'],
            'children': info['children'],
        }
        if 'split_criteria' in info:
            sc = info['split_criteria']
            node['split_criteria'] = {
                'should_split': sc['should_split'],
                'target_count': sc['target_count'],
                'area_density': round(sc['area_density'], 8),
                'max_confidence': round(sc['max_confidence'], 6),
                'condition_n': sc['condition_n'],
                'condition_d': sc['condition_d'],
                'condition_s': sc['condition_s'],
            }
        tree_export[cell] = node

    with open(f'{prefix}adaptive_tree.json', 'w', encoding='utf-8') as f:
        json.dump(tree_export, f, ensure_ascii=False, indent=2)

    # 叶子节点CSV
    leaf_records = []
    for cell in leaf_set:
        info = tree_structure[cell]
        sc = info.get('split_criteria', {})
        leaf_records.append({
            'h3_cell': cell,
            'resolution': info['resolution'],
            'targets': info['targets'],
            'area_density': round(sc.get('area_density', 0), 8),
            'max_confidence': round(sc.get('max_confidence', 0), 6),
        })

    leaf_df = pd.DataFrame(leaf_records).sort_values(['resolution', 'h3_cell'])
    leaf_df.to_csv(f'{prefix}adaptive_tree_leaves.csv', index=False, encoding='utf-8-sig')

    # 分裂节点CSV
    split_records = []
    for cell, info in tree_structure.items():
        if info['split']:
            sc = info.get('split_criteria', {})
            split_records.append({
                'h3_cell': cell,
                'resolution': info['resolution'],
                'targets': info['targets'],
                'n_children': len(info['children']),
                'area_density': round(sc.get('area_density', 0), 8),
                'max_confidence': round(sc.get('max_confidence', 0), 6),
                'cond_n': sc.get('condition_n', False),
                'cond_d': sc.get('condition_d', False),
                'cond_s': sc.get('condition_s', False),
            })

    split_df = pd.DataFrame(split_records).sort_values(['resolution', 'h3_cell'])
    split_df.to_csv(f'{prefix}adaptive_tree_splits.csv', index=False, encoding='utf-8-sig')

    print(f"  输出: {prefix}adaptive_tree.json, {prefix}adaptive_tree_leaves.csv")

    return {
        'tree_structure': tree_export,
        'leaf_set': leaf_set,
        'stats': stats,
        'prefix': prefix,
    }

# ============================================================================
# 步骤4: 边界编码
# ============================================================================

def step4_boundary_encoding(step2_result, step3_result):
    """步骤4: 边界效应处理与编码"""
    print(f"\n{'-'*50}")
    print(f"  步骤4: 边界编码")
    print(f"{'-'*50}")

    tree_structure = step3_result['tree_structure']
    leaf_set = step3_result['leaf_set']
    prefix = step3_result['prefix']

    # 检测分辨率不连续
    discontinuities = []
    leaf_resolution = {h: tree_structure[h]['resolution'] for h in leaf_set if h in tree_structure}

    for h_leaf in leaf_resolution:
        r_leaf = leaf_resolution[h_leaf]
        try:
            neighbors = h3.grid_disk(h_leaf, 1)
            for neighbor in neighbors:
                if neighbor in leaf_resolution and neighbor != h_leaf:
                    r_neighbor = leaf_resolution[neighbor]
                    if abs(r_leaf - r_neighbor) > 1:
                        discontinuities.append({
                            'cell1': h_leaf, 'cell2': neighbor,
                            'r1': r_leaf, 'r2': r_neighbor,
                            'r_diff': abs(r_leaf - r_neighbor)
                        })
        except:
            pass

    # 编码策略统计
    strategies = {'H3-Ascend': 0, 'H3-Primary-Secondary': 0, 'H3-Multi-Code': 0}
    for cell, info in tree_structure.items():
        if info['targets'] > 0:
            r = info['resolution']
            if r >= 9:
                strategies['H3-Ascend'] += 1
            elif r >= 7:
                strategies['H3-Primary-Secondary'] += 1
            else:
                strategies['H3-Multi-Code'] += 1

    encoding_result = {
        'strategies': strategies,
        'discontinuities': len(discontinuities),
    }

    print(f"  不连续检测: {len(discontinuities)} 处")
    print(f"  编码策略分布: {strategies}")

    # 保存边界编码结果
    with open(f'{prefix}boundary_encoding.json', 'w', encoding='utf-8') as f:
        json.dump(encoding_result, f, indent=2)

    summary_data = [{'strategy': k, 'count': v} for k, v in strategies.items()]
    pd.DataFrame(summary_data).to_csv(f'{prefix}boundary_encoding_summary.csv', index=False)

    print(f"  输出: {prefix}boundary_encoding.json, {prefix}boundary_encoding_summary.csv")

    return {
        'discontinuities': discontinuities,
        'encoding_result': encoding_result,
        'prefix': prefix,
    }

# ============================================================================
# 步骤5: 定量分析
# ============================================================================

def step5_quantitative_analysis(step2_result, step3_result, step4_result):
    """步骤5: 定量分析"""
    print(f"\n{'-'*50}")
    print(f"  步骤5: 定量分析")
    print(f"{'-'*50}")

    tree_structure = step3_result['tree_structure']
    leaf_set = step3_result['leaf_set']
    stats = step3_result['stats']
    bbox = step2_result['bbox']
    A0_km2 = step2_result['A0_km2']
    detections_gdf = step2_result['detections_gdf']
    prefix = step3_result['prefix']

    # 计算各分辨率全覆盖格网数
    region_poly = h3.LatLngPoly([
        (bbox[3], bbox[0]), (bbox[3], bbox[2]),
        (bbox[1], bbox[2]), (bbox[1], bbox[0]),
    ])

    ref_counts = {}
    for res in [7, 8, 9, 10]:
        try:
            cells = h3.h3shape_to_cells(region_poly, res)
            ref_counts[res] = len(cells)
        except:
            avg_area = h3.average_hexagon_area(res, unit='km^2')
            ref_counts[res] = int(A0_km2 / avg_area) + 10

    leaf_count = len(leaf_set)
    res10_count = ref_counts.get(10, 0)

    # 效率指标
    reduction_vs_res10 = (1 - leaf_count / res10_count) * 100 if res10_count > 0 else 0

    # 存储估算
    bytes_per_trad = 32
    bytes_per_tree = 64
    traditional_storage = res10_count * bytes_per_trad
    tree_storage = len(tree_structure) * bytes_per_tree
    storage_savings = (1 - tree_storage / traditional_storage) * 100 if traditional_storage > 0 else 0

    # 查询效能测试
    query_times = {}
    for _ in range(100):
        start = time.perf_counter()
        # 模拟查询所有叶子节点
        _ = list(leaf_set)
        query_times['full'] = (time.perf_counter() - start) * 1000 / 100

    # 构建结果
    metrics = {
        'leaf_count': leaf_count,
        'tree_nodes': len(tree_structure),
        'split_nodes': stats['split_nodes'],
        'ref_counts': ref_counts,
        'reduction_vs_res10': reduction_vs_res10,
        'traditional_storage_mb': traditional_storage / 1024 / 1024,
        'tree_storage_mb': tree_storage / 1024 / 1024,
        'storage_savings': storage_savings,
        'query_time_ms': query_times.get('full', 0),
        'resolution_distribution': stats['resolution_distribution'],
        'A0_km2': A0_km2,
        'target_count': len(detections_gdf),
    }

    print(f"  叶子节点: {leaf_count}")
    print(f"  相比Res-10减少: {reduction_vs_res10:.1f}%")
    print(f"  存储节省: {storage_savings:.1f}%")
    print(f"  全量查询时间: {query_times.get('full', 0):.4f} ms")

    # 保存定量分析结果
    results_data = {
        'scene_id': prefix.rstrip('_'),
        'leaf_count': leaf_count,
        'tree_nodes': len(tree_structure),
        'split_nodes': stats['split_nodes'],
        'ref_res7': ref_counts.get(7, 0),
        'ref_res8': ref_counts.get(8, 0),
        'ref_res9': ref_counts.get(9, 0),
        'ref_res10': ref_counts.get(10, 0),
        'reduction_vs_res10_pct': round(reduction_vs_res10, 2),
        'traditional_storage_mb': round(traditional_storage / 1024 / 1024, 4),
        'tree_storage_mb': round(tree_storage / 1024 / 1024, 4),
        'storage_savings_pct': round(storage_savings, 2),
        'query_time_ms': round(query_times.get('full', 0), 4),
        'A0_km2': round(A0_km2, 4),
        'target_count': len(detections_gdf),
        'resolution_distribution': stats['resolution_distribution'],
    }

    with open(f'{prefix}quantitative_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    return {
        'metrics': metrics,
        'prefix': prefix,
    }

# ============================================================================
# 步骤6: 可视化
# ============================================================================

def step6_visualization(step2_result, step3_result):
    """步骤6: 生成可视化成果图"""
    print(f"\n{'-'*50}")
    print(f"  步骤6: 可视化")
    print(f"{'-'*50}")

    tree_structure = step3_result['tree_structure']
    leaf_set = step3_result['leaf_set']
    stats = step3_result['stats']
    H0_df = step2_result['H0_df']
    detections_gdf = step2_result['detections_gdf']
    bbox = step2_result['bbox']
    r_init = step2_result['r_init']
    prefix = step3_result['prefix']

    # 配色方案
    resolution_colors = {
        7:  ('#2E86AB', '#82D8FF'),
        8:  ('#A23B72', '#FF82D8'),
        9:  ('#F18F01', '#FFD080'),
        10: ('#C73E1D', '#FF8060'),
    }

    class_colors = {
        'soccer ball field': '#FF6B6B',
        'storage tank': '#FFD700',
        'plane': '#00FF7F',
        'ground-track-field': '#FF8C00',
        'basketball court': '#DA70D6',
        'bridge': '#00BFFF',
        'roundabout': '#FF1493',
        'tennis court': '#7FFF00',
        'ship': '#FF4500',
        'harbor': '#20B2AA',
        'large vehicle': '#9370DB',
        'small vehicle': '#F0E68C',
    }

    pad_x = (bbox[2] - bbox[0]) * 0.02
    pad_y = (bbox[3] - bbox[1]) * 0.02
    xlim = (bbox[0] - pad_x, bbox[2] + pad_x)
    ylim = (bbox[1] - pad_y, bbox[3] + pad_y)

    # 图1: 自适应格网最终结果
    fig1, ax1 = plt.subplots(1, 1, figsize=(11, 8))
    fig1.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    leaf_by_res = {}
    for h_leaf in leaf_set:
        r = tree_structure.get(h_leaf, {}).get('resolution', 10)
        leaf_by_res.setdefault(r, []).append(h_leaf)

    for res in sorted(leaf_by_res.keys()):
        fc, ec = resolution_colors.get(res, ('#888888', '#AAAAAA'))
        alpha = 0.55 if res <= 8 else 0.45
        for h_leaf in leaf_by_res[res]:
            try:
                boundary = h3.cell_to_boundary(h_leaf)
                xy = [(lon, lat) for lat, lon in boundary]
                poly = MPLPolygon(xy, fill=True, facecolor=fc, edgecolor=ec,
                                   linewidth=0.4, alpha=alpha)
                ax1.add_patch(poly)
            except:
                pass

    # H0轮廓
    for _, hrow in H0_df.iterrows():
        try:
            cb = h3.cell_to_boundary(hrow['h3_index'])
            xy = [(lon, lat) for lat, lon in cb]
            poly = MPLPolygon(xy, fill=False, edgecolor='#444444',
                             linewidth=0.8, alpha=0.5)
            ax1.add_patch(poly)
        except:
            pass

    # 目标散点
    plotted_classes = set()
    for _, row in detections_gdf.iterrows():
        cls = row.get('class', 'unknown')
        color = class_colors.get(cls, '#AAAAAA')
        ax1.plot(row.geometry.centroid.x, row.geometry.centroid.y,
                 'o', color=color, markersize=3, alpha=0.7)
        plotted_classes.add(cls)

    # 图例
    legend_patches = [
        Patch(facecolor=resolution_colors[r][0], edgecolor=resolution_colors[r][1],
              label=f'Resolution {r} ({len(leaf_by_res.get(r, []))} cells)', alpha=0.85)
        for r in sorted(resolution_colors.keys()) if r in leaf_by_res
    ]
    ax1.legend(handles=legend_patches, loc='upper right', fontsize=8,
              facecolor='white', edgecolor='#ccc', labelcolor='black', framealpha=0.9)

    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Longitude (deg)', color='#333', fontsize=10)
    ax1.set_ylabel('Latitude (deg)', color='#333', fontsize=10)
    ax1.tick_params(colors='#333')

    stat_text = f"CD-MCAR Adaptive Grid\nLeaf Cells: {len(leaf_set)}\n"
    stat_text += f"Tree Nodes: {len(tree_structure)}\n"
    stat_text += f"Targets: {len(detections_gdf)}"
    ax1.text(0.02, 0.98, stat_text, transform=ax1.transAxes, fontsize=9,
            color='#C41E3A', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#ccc'))

    fig1.suptitle(f'CD-MCAR Adaptive Grid — {prefix.rstrip("_")}',
                  fontsize=13, fontweight='bold', color='black', y=1.01)
    plt.tight_layout()
    fig1.savefig(f'{prefix}adaptive_grid_result.png', dpi=300, bbox_inches='tight',
                 facecolor=fig1.get_facecolor())
    plt.close()

    # 图2: 叶子节点分辨率分布直方图
    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 5))
    fig2.patch.set_facecolor('white')
    ax2.set_facecolor('white')

    res_vals = list(stats['resolution_distribution'].keys())
    leaf_cnts = [stats['resolution_distribution'][r] for r in res_vals]
    bar_colors = [resolution_colors.get(r, ('#888', '#aaa'))[0] for r in res_vals]

    bars = ax2.bar(res_vals, leaf_cnts, color=bar_colors, edgecolor='black', alpha=0.9)
    for bar, cnt in zip(bars, leaf_cnts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(cnt), ha='center', va='bottom', fontsize=10, color='black')

    ax2.set_xlabel('H3 Resolution', color='#333', fontsize=11)
    ax2.set_ylabel('Leaf Cell Count', color='#333', fontsize=11)
    ax2.set_xticks(res_vals)
    ax2.tick_params(colors='#333')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['bottom'].set_edgecolor('#999')
    ax2.spines['left'].set_edgecolor('#999')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig2.suptitle(f'Leaf Node Resolution Distribution — {prefix.rstrip("_")}',
                  fontsize=13, fontweight='bold', color='black')
    plt.tight_layout()
    fig2.savefig(f'{prefix}resolution_distribution.png', dpi=300, bbox_inches='tight',
                 facecolor=fig2.get_facecolor())
    plt.close()

    print(f"  输出: {prefix}adaptive_grid_result.png, {prefix}resolution_distribution.png")

    return {'prefix': prefix}

# ============================================================================
# 跨场景汇总
# ============================================================================

def generate_cross_scene_summary(all_results):
    """生成跨场景汇总分析"""
    print(f"\n{'='*60}")
    print(f"  跨场景汇总分析")
    print(f"{'='*60}")

    # 汇总表格
    summary_data = []
    for scene_id, result in all_results.items():
        metrics = result['step5']['metrics']
        step2 = result['step2']

        summary_data.append({
            'Scene': scene_id,
            'Scene Name': SCENES[scene_id]['name'],
            'Area (km2)': round(step2['A0_km2'], 2),
            'Targets': step2['detections_gdf'].shape[0],
            'Leaf Cells': metrics['leaf_count'],
            'Tree Nodes': metrics['tree_nodes'],
            'Res10 Baseline': metrics['ref_counts'].get(10, 0),
            'Reduction (%)': round(metrics['reduction_vs_res10'], 2),
            'Storage (MB)': round(metrics['tree_storage_mb'], 4),
            'Storage Savings (%)': round(metrics['storage_savings'], 2),
            'Query Time (ms)': round(metrics['query_time_ms'], 4),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('cross_scene_summary.csv', index=False, encoding='utf-8-sig')
    print(f"\n  跨场景汇总表已保存: cross_scene_summary.csv")

    # 生成汇总图
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor('white')
    fig.suptitle('CD-MCAR Six-Scene Cross-Scene Validation',
                 fontsize=14, fontweight='bold', color='black', y=1.02)

    scene_labels = [f"{row['Scene']}\n{row['Scene Name']}" for _, row in summary_df.iterrows()]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    # 图1: 格网减少率对比
    ax1 = axes[0, 0]
    ax1.set_facecolor('white')
    reductions = summary_df['Reduction (%)'].values
    bars1 = ax1.bar(scene_labels, reductions, color=colors, edgecolor='black', alpha=0.85)
    for bar, val in zip(bars1, reductions):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='black')
    ax1.set_ylabel('Grid Reduction (%)', color='#333', fontsize=10)
    ax1.set_title('Grid Count Reduction vs Res-10 Baseline', color='black', fontsize=11)
    ax1.tick_params(colors='#333', labelcolor='black')
    ax1.set_ylim(0, max(reductions) * 1.15)
    ax1.grid(True, alpha=0.3, axis='y')

    # 图2: 存储节省率对比
    ax2 = axes[0, 1]
    ax2.set_facecolor('white')
    savings = summary_df['Storage Savings (%)'].values
    bars2 = ax2.bar(scene_labels, savings, color=colors, edgecolor='black', alpha=0.85)
    for bar, val in zip(bars2, savings):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='black')
    ax2.set_ylabel('Storage Savings (%)', color='#333', fontsize=10)
    ax2.set_title('Storage Savings vs Res-10 Baseline', color='black', fontsize=11)
    ax2.tick_params(colors='#333', labelcolor='black')
    ax2.set_ylim(0, max(savings) * 1.15 if max(savings) > 0 else 100)
    ax2.grid(True, alpha=0.3, axis='y')

    # 图3: 叶子节点数对比
    ax3 = axes[1, 0]
    ax3.set_facecolor('white')
    leaf_counts = summary_df['Leaf Cells'].values
    res10_counts = summary_df['Res10 Baseline'].values
    x = np.arange(len(scene_labels))
    width = 0.35
    ax3.bar(x - width/2, res10_counts, width, label='Res-10 Baseline', color='#4a9eda', alpha=0.8)
    ax3.bar(x + width/2, leaf_counts, width, label='CD-MCAR Leaves', color='#FF6B6B', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(scene_labels, rotation=0, fontsize=8, color='#333')
    ax3.set_ylabel('Grid Count', color='#333', fontsize=10)
    ax3.set_title('Grid Count: Baseline vs CD-MCAR', color='black', fontsize=11)
    ax3.tick_params(colors='#333', labelcolor='black')
    ax3.legend(facecolor='white', edgecolor='#ccc', labelcolor='black')
    ax3.grid(True, alpha=0.3, axis='y')

    # 图4: 场景信息
    ax4 = axes[1, 1]
    ax4.set_facecolor('white')
    ax4.axis('off')

    info_text = "CD-MCAR Six-Scene Experiment Summary\n" + "="*40 + "\n\n"
    for _, row in summary_df.iterrows():
        info_text += f"{row['Scene']} ({row['Scene Name']})\n"
        info_text += f"  Area: {row['Area (km2)']:.2f} km²\n"
        info_text += f"  Targets: {row['Targets']}\n"
        info_text += f"  Leaf Cells: {row['Leaf Cells']}\n"
        info_text += f"  Reduction: {row['Reduction (%)']:.1f}%\n"
        info_text += f"  Storage Savings: {row['Storage Savings (%)']:.1f}%\n\n"

    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=8,
            color='black', va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', alpha=0.9, edgecolor='#ccc'))

    plt.tight_layout()
    fig.savefig('cross_scene_comparison.png', dpi=300, bbox_inches='tight',
               facecolor=fig.get_facecolor())
    plt.close()

    print(f"  跨场景对比图已保存: cross_scene_comparison.png")

    return summary_df

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*70)
    print("  CD-MCAR 六场景批量实验")
    print("  Six-Scene Batch Experiment for CD-MCAR Adaptive Grid Generation")
    print("="*70)
    print(f"  影像分辨率: Sentinel-2, 10m")
    print(f"  检测模型: YOLO11n-OBB")
    print(f"  对比基线: H3 Resolution 10 均匀覆盖")
    print(f"  分裂阈值: τ_n={TAU_N}, τ_d={TAU_D}, τ_s={TAU_S}")
    print(f"  最大分辨率: r_max={R_MAX}")
    print()

    all_results = {}
    total_start = time.time()

    # 逐景处理
    for scene_id, scene_config in SCENES.items():
        print(f"\n{'#'*70}")
        print(f"# 场景 {scene_id}: {scene_config['name']} ({scene_config['name_cn']})")
        print(f"# 数据文件: {scene_config['file']}")
        print(f"{'#'*70}")

        scene_start = time.time()

        # 检查文件是否存在
        if not os.path.exists(scene_config['file']):
            print(f"  [ERROR] 文件不存在: {scene_config['file']}")
            continue

        # 读取检测数据
        print(f"\n[数据加载]")
        detections_gdf = gpd.read_file(scene_config['file'])
        
        # CRS自动检测与修正
        src_crs = scene_config['crs']
        
        # 检测坐标值是否与声称的CRS不匹配
        if len(detections_gdf) > 0:
            bounds = detections_gdf.total_bounds
            max_coord = max(abs(bounds[0]), abs(bounds[1]), abs(bounds[2]), abs(bounds[3]))
            
            # 如果最大坐标值 > 180，说明不可能是WGS84经纬度，应该是米制坐标
            if max_coord > 180 and (detections_gdf.crs is None or detections_gdf.crs.to_epsg() == 4326):
                print(f"  [WARNING] 坐标值过大({max_coord:.1f}m)，自动修正CRS为: {src_crs}")
                detections_gdf = detections_gdf.set_crs(src_crs, allow_override=True)
            elif detections_gdf.crs is None:
                print(f"  GeoJSON无CRS元数据，应用配置CRS: {src_crs}")
                detections_gdf = detections_gdf.set_crs(src_crs, allow_override=True)
            elif detections_gdf.crs != src_crs:
                print(f"  GeoJSON CRS: {detections_gdf.crs}，转换到: {src_crs}")
                detections_gdf = detections_gdf.to_crs(src_crs)
        
        print(f"  成功加载 {len(detections_gdf)} 个检测目标")
        print(f"  坐标系: {detections_gdf.crs}")

        if len(detections_gdf) == 0:
            print(f"  [WARNING] 无检测目标，跳过此场景")
            continue

        # 步骤2: H0格网生成
        step2_result = step2_generate_h0_grid(scene_id, scene_config, detections_gdf)

        # 步骤3: CD-MCAR自适应分裂树
        step3_result = step3_build_adaptive_tree(step2_result)

        # 步骤4: 边界编码
        step4_result = step4_boundary_encoding(step2_result, step3_result)

        # 步骤5: 定量分析
        step5_result = step5_quantitative_analysis(step2_result, step3_result, step4_result)

        # 步骤6: 可视化
        step6_result = step6_visualization(step2_result, step3_result)

        scene_time = time.time() - scene_start
        print(f"\n  场景 {scene_id} 完成! 耗时: {scene_time:.2f} 秒")

        all_results[scene_id] = {
            'step2': step2_result,
            'step3': step3_result,
            'step4': step4_result,
            'step5': step5_result,
            'step6': step6_result,
        }

    # 跨场景汇总
    if all_results:
        summary_df = generate_cross_scene_summary(all_results)

        # 保存所有结果索引
        with open('all_scene_results_index.json', 'w', encoding='utf-8') as f:
            index_data = {}
            for scene_id, result in all_results.items():
                index_data[scene_id] = {
                    'name': SCENES[scene_id]['name'],
                    'area_km2': result['step2']['A0_km2'],
                    'targets': len(result['step2']['detections_gdf']),
                    'leaf_cells': result['step3']['stats']['leaf_nodes'],
                    'reduction_pct': round(result['step5']['metrics']['reduction_vs_res10'], 2),
                    'storage_savings_pct': round(result['step5']['metrics']['storage_savings'], 2),
                    'query_time_ms': round(result['step5']['metrics']['query_time_ms'], 4),
                }
            json.dump(index_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*70}")
        print(f"  CD-MCAR 六场景实验完成!")
        print(f"  总耗时: {time.time() - total_start:.2f} 秒")
        print(f"{'='*70}")

        # 打印最终汇总
        print(f"\n  === 最终汇总 ===")
        for scene_id, result in all_results.items():
            m = result['step5']['metrics']
            print(f"  {scene_id} ({SCENES[scene_id]['name']:25s}): "
                  f"Leaf={m['leaf_count']:5d}, "
                  f"Reduction={m['reduction_vs_res10']:.1f}%, "
                  f"Storage={m['storage_savings']:.1f}%")

    return all_results


if __name__ == '__main__':
    results = main()
