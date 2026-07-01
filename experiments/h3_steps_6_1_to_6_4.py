"""
步骤6.1~6.4: 结果评估与分析
- 步骤6.1: 可视化对比（定性分析）
- 步骤6.2: 定量分析 - 效率指标（格网数量、存储空间）
- 步骤6.3: 定量分析 - 应用效能验证（范围查询）
- 步骤6.4: 保存定量分析结果表

修正参考代码中的问题：
- h3 v4 API 修正: h3.h3_to_geo_boundary() -> h3.cell_to_boundary()
- h3 v4 API 修正: h3.polyfill() -> h3.h3shape_to_cells() / h3.polygon_to_cells()
- 使用 h3.average_hexagon_area() 代替 h3.get_hexagon_area_km2()
- 参考代码中部分变量未定义，已从文件重新加载
- 图B改为真实的H3初始格网（Resolution 7），而非"四叉树格网"
"""

import json
import time
import h3
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MPLPolygon, Patch
from matplotlib.colors import to_rgba
from shapely.geometry import shape, Polygon, mapping
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 数据加载
# ============================================================

def load_all_data():
    """加载所有前序步骤输出的数据"""
    print("=" * 60)
    print("加载数据")
    print("=" * 60)

    # 加载YOLO检测结果（detections.geojson）
    print("\n[1] 加载目标检测结果...")
    yolo_results = gpd.read_file('detections.geojson')
    print(f"  ✓ 加载了 {len(yolo_results)} 个检测目标")
    print(f"  ✓ 目标类别: {sorted(yolo_results['class'].unique().tolist())}")

    # 加载自适应树结构
    print("\n[2] 加载自适应树结构...")
    with open('adaptive_tree.json', 'r', encoding='utf-8') as f:
        tree_structure = json.load(f)
    print(f"  ✓ 加载了 {len(tree_structure)} 个H3格网节点")

    # 加载叶子节点
    print("\n[3] 加载叶子节点...")
    leaves_df = pd.read_csv('adaptive_tree_leaves.csv')
    leaf_set = set(leaves_df['h3_cell'].values)
    print(f"  ✓ 加载了 {len(leaf_set)} 个叶子节点")

    # 加载H0初始格网
    print("\n[4] 加载H0初始格网...")
    h0_df = pd.read_csv('H0_grid.csv')
    H0_initial = set(h0_df['h3_index'].values)
    print(f"  ✓ 加载了 {len(H0_initial)} 个H0格网（Resolution 7）")

    # 加载目标-格网关系
    print("\n[5] 加载目标-格网关系...")
    target_relations = pd.read_csv('target_cell_relations.csv')
    print(f"  ✓ 加载了 {len(target_relations)} 条关系")

    # 加载步骤5统计报告
    print("\n[6] 加载步骤5统计报告...")
    with open('step5_statistics.json', 'r', encoding='utf-8') as f:
        stats = json.load(f)
    print(f"  ✓ 加载统计报告完成")

    # 计算研究区域的边界框与总面积
    bbox = yolo_results.total_bounds  # [minx, miny, maxx, maxy]
    
    # 使用影像面积（来自area_result.txt）
    A0 = 233602077.63  # m²（来自步骤2.1的栅格面积计算）

    print(f"\n  研究区域边界框: [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")
    print(f"  研究区域总面积: {A0:.2e} m²")

    return yolo_results, tree_structure, leaf_set, H0_initial, target_relations, stats, bbox, A0


# ============================================================
# 步骤6.1：可视化对比（定性分析）
# ============================================================

def step_6_1_qualitative_visualization(yolo_results, tree_structure, leaf_set, H0_initial, bbox):
    """
    步骤6.1：绘制三图对比：
    (A) 原始影像与YOLO检测结果
    (B) 传统固定分辨率格网（H3 Resolution 7初始覆盖）
    (C) CD-MCAR Adaptive H3 Grid (Multi-Resolution Leaf Nodes)
    """
    print("\n" + "=" * 60)
    print("步骤6.1：可视化对比（定性分析）")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('white')

    # ── 扩展边界框，留出边距 ──────────────────────────────────
    pad_x = (bbox[2] - bbox[0]) * 0.02
    pad_y = (bbox[3] - bbox[1]) * 0.02
    xlim = (bbox[0] - pad_x, bbox[2] + pad_x)
    ylim = (bbox[1] - pad_y, bbox[3] + pad_y)

    # ── 目标类别颜色映射 ─────────────────────────────────────
    class_colors = {
        'soccer ball field':  '#FF6B6B',
        'storage tank':       '#FFD700',
        'plane':              '#00FF7F',
        'ground track field': '#FF8C00',
        'basketball court':   '#DA70D6',
        'bridge':             '#00BFFF',
        'roundabout':         '#FF1493',
        'tennis court':       '#7FFF00',
        'ship':               '#FF4500',
        'harbor':             '#20B2AA',
        'large vehicle':      '#9370DB',
        'small vehicle':      '#F0E68C',
    }
    
    # ──────────────────────────────────────────────────────────
    # 图A：原始影像与检测结果
    # ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('white')
    ax.set_title('(A) Original Image & YOLO Detections', fontsize=12, fontweight='bold',
                 color='black', pad=10)

    # 按类别绘制检测多边形
    plotted_classes = {}
    for idx, target in yolo_results.iterrows():
        geom = target['geometry']
        cls = target.get('class', 'unknown')
        color = class_colors.get(cls, '#AAAAAA')
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            poly = MPLPolygon(list(zip(x, y)),
                              fill=True, facecolor=color,
                              edgecolor='white', linewidth=0.5, alpha=0.7)
            ax.add_patch(poly)
            if cls not in plotted_classes:
                plotted_classes[cls] = color

    # 图例
    legend_elems = [Patch(facecolor=c, edgecolor='white', label=n, alpha=0.8)
                    for n, c in sorted(plotted_classes.items())]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=6,
              facecolor='white', edgecolor='#ccc', labelcolor='black',
              framealpha=0.9, ncol=1)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.tick_params(colors='#333', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#999')
    ax.set_xlabel('Longitude (°)', color='#333', fontsize=8)
    ax.set_ylabel('Latitude (°)', color='#333', fontsize=8)

    # 添加统计注释
    ax.text(0.02, 0.04,
            f"Targets: {len(yolo_results)}\nClasses: {len(plotted_classes)}",
            transform=ax.transAxes, fontsize=7,
            color='#C41E3A', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='#ccc'))

    # ──────────────────────────────────────────────────────────
    # 图B：传统固定分辨率格网（H3 Resolution 7）
    # ──────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor('white')
    ax.set_title('(B) Fixed-Resolution Grid\n(H3 Resolution 7 Uniform Coverage)',
                 fontsize=12, fontweight='bold', color='black', pad=10)

    # 绘制所有H0格网（Resolution 7）
    h0_list = list(H0_initial)
    for h in h0_list:
        try:
            boundary = h3.cell_to_boundary(h)  # [(lat, lon), ...]
            # boundary → (lon, lat) 给matplotlib
            xy = [(lon, lat) for lat, lon in boundary]
            poly = MPLPolygon(xy, fill=True,
                              facecolor='#1e3a5f', edgecolor='#4a9eda',
                              linewidth=0.8, alpha=0.6)
            ax.add_patch(poly)
        except Exception:
            pass

    # 叠加目标点（只显示中心点，简洁）
    for idx, target in yolo_results.iterrows():
        cx = target.get('cx_lon', target['geometry'].centroid.x)
        cy = target.get('cx_lat', target['geometry'].centroid.y)
        ax.plot(cx, cy, 'o', color='#FF6B6B', markersize=2, alpha=0.7, zorder=5)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.tick_params(colors='#333', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#999')
    ax.set_xlabel('Longitude (°)', color='#333', fontsize=8)

    ax.text(0.02, 0.04,
            f"Cells: {len(h0_list)}\nRes: 7 (uniform)",
            transform=ax.transAxes, fontsize=7,
            color='#2E86AB', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='#ccc'))

    # ──────────────────────────────────────────────────────────
    # Panel C: CD-MCAR Adaptive H3 Grid
    # ──────────────────────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor('white')
    ax.set_title('(C) CD-MCAR Adaptive H3 Grid\n(Multi-Resolution Leaf Nodes)',
                 fontsize=12, fontweight='bold', color='black', pad=10)

    # 分辨率 → 颜色映射（深色主题）
    resolution_colors = {
        7:  ('#2E86AB', '#82D8FF'),   # (face, edge)
        8:  ('#A23B72', '#FF82D8'),
        9:  ('#F18F01', '#FFD080'),
        10: ('#C73E1D', '#FF8060'),
    }
    res_face  = {r: c[0] for r, c in resolution_colors.items()}
    res_edge  = {r: c[1] for r, c in resolution_colors.items()}

    leaf_list = list(leaf_set)
    
    # 先画低分辨率（大格网）再画高分辨率（避免遮挡关系异常）
    leaf_by_res = {}
    for h_leaf in leaf_list:
        res = tree_structure.get(h_leaf, {}).get('resolution', 10)
        leaf_by_res.setdefault(res, []).append(h_leaf)

    for res in sorted(leaf_by_res.keys()):
        fc = res_face.get(res, '#888888')
        ec = res_edge.get(res, '#AAAAAA')
        alpha = 0.55 if res <= 8 else 0.45
        for h_leaf in leaf_by_res[res]:
            try:
                boundary = h3.cell_to_boundary(h_leaf)
                xy = [(lon, lat) for lat, lon in boundary]
                poly = MPLPolygon(xy, fill=True,
                                  facecolor=fc, edgecolor=ec,
                                  linewidth=0.4, alpha=alpha)
                ax.add_patch(poly)
            except Exception:
                pass

    # 图例
    legend_elems_c = [
        Patch(facecolor=res_face[r], edgecolor=res_edge[r],
              label=f'Resolution {r}  ({len(leaf_by_res.get(r, []))} cells)',
              alpha=0.85)
        for r in sorted(resolution_colors.keys())
    ]
    ax.legend(handles=legend_elems_c, loc='upper right', fontsize=6.5,
              facecolor='white', edgecolor='#ccc', labelcolor='black',
              framealpha=0.9)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.tick_params(colors='#333', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#999')
    ax.set_xlabel('Longitude (°)', color='#333', fontsize=8)

    leaf_res_counts = {r: len(v) for r, v in leaf_by_res.items()}
    stat_text = '\n'.join([f"Res {r}: {c}" for r, c in sorted(leaf_res_counts.items())])
    stat_text = f"Leaf Nodes: {len(leaf_set)}\n" + stat_text
    ax.text(0.02, 0.04, stat_text, transform=ax.transAxes, fontsize=6.5,
            color='#C41E3A', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='#ccc'))

    # ──────────────────────────────────────────────────────────
    # 总标题 & 保存
    # ──────────────────────────────────────────────────────────
    fig.suptitle('CD-MCAR Algorithm — Qualitative Comparison of DGGS Grids',
                 fontsize=14, fontweight='bold', color='black', y=1.02)

    plt.tight_layout(pad=1.5)
    output_path = 'qualitative_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"  ✓ 定性对比图已保存: {output_path}")
    return output_path


# ============================================================
# 步骤6.2：定量分析 - 效率指标
# ============================================================

def step_6_2_efficiency_metrics(yolo_results, tree_structure, leaf_set, A0):
    """
    步骤6.2：计算并对比效率指标
    - 格网数量（与固定分辨率全覆盖对比）
    - 存储空间估算
    """
    print("\n" + "=" * 60)
    print("步骤6.2：定量分析 - 效率指标")
    print("=" * 60)

    # ── 构建研究区域多边形（使用目标点凸包扩展）──────────────
    # 用H0边界框近似研究区域
    bbox = yolo_results.total_bounds  # [minx, miny, maxx, maxy]
    
    # 使用H3的 polygon_to_cells / h3shape_to_cells 计算不同分辨率的格网数
    # 构建研究区域LatLngPoly（h3 v4 API）
    # 注意：h3 v4 使用 (lat, lon) 顺序
    region_poly = h3.LatLngPoly([
        (bbox[3], bbox[0]),  # 左上 (lat, lon)
        (bbox[3], bbox[2]),  # 右上
        (bbox[1], bbox[2]),  # 右下
        (bbox[1], bbox[0]),  # 左下
    ])

    print(f"\n[6.2.1] 研究区域: 厦门海沧湾")
    print(f"  边界框: [{bbox[0]:.4f}°, {bbox[1]:.4f}°] ~ [{bbox[2]:.4f}°, {bbox[3]:.4f}°]")
    print(f"  总面积: {A0:.2e} m²  ({A0/1e6:.2f} km²)")

    # ── 计算不同分辨率的全覆盖格网数 ─────────────────────────
    print("\n[6.2.2] 计算各分辨率全覆盖格网数...")
    reference_counts = {}
    for res in [7, 8, 9, 10]:
        try:
            cells = h3.h3shape_to_cells(region_poly, res)
            reference_counts[res] = len(cells)
            print(f"  Resolution {res}: {len(cells):,} 个格网")
        except Exception as e:
            # 降级到 polygon_to_cells
            try:
                geom_dict = {
                    "type": "Polygon",
                    "coordinates": [[
                        [bbox[0], bbox[3]], [bbox[2], bbox[3]],
                        [bbox[2], bbox[1]], [bbox[0], bbox[1]],
                        [bbox[0], bbox[3]]
                    ]]
                }
                cells = h3.geo_to_cells(h3.geo_to_h3shape(geom_dict), res)
                reference_counts[res] = len(cells)
                print(f"  Resolution {res}: {len(cells):,} 个格网")
            except Exception as e2:
                print(f"  Resolution {res}: 计算失败 ({e2})")
                # 用面积估算
                avg_area_km2 = h3.average_hexagon_area(res, 'km2')
                est = int(A0 / 1e6 / avg_area_km2)
                reference_counts[res] = est
                print(f"  Resolution {res}: ~{est:,} 个格网（估算）")

    # The Hand方法的叶子格网数（实际结果）
    hand_count = len(leaf_set)
    # 使用Resolution 10作为主要对比基准（最细粒度）
    level_10_count = reference_counts.get(10, 0)
    level_9_count  = reference_counts.get(9, 0)
    level_7_count  = reference_counts.get(7, 0)

    print(f"\n[6.2.3] 格网数量对比:")
    print(f"  Resolution 7  全覆盖: {level_7_count:,} 个")
    print(f"  Resolution 9  全覆盖: {level_9_count:,} 个")
    print(f"  Resolution 10 全覆盖: {level_10_count:,} 个")
    print(f"  The Hand 自适应:       {hand_count:,} 个")
    if level_10_count > 0:
        reduction_vs_10 = (1 - hand_count / level_10_count) * 100
        print(f"  相比Resolution 10减少: {reduction_vs_10:.1f}%")
    if level_9_count > 0:
        reduction_vs_9 = (1 - hand_count / level_9_count) * 100
        print(f"  相比Resolution 9 减少: {reduction_vs_9:.1f}%")

    # ── 存储空间估算 ─────────────────────────────────────────
    # 传统方法：固定分辨率全覆盖（Resolution 9，常见选择），仅索引，约32字节/条
    # The Hand方法：多分辨率 + 编码元数据，约64字节/条
    print("\n[6.2.4] 存储空间估算:")

    bytes_per_traditional = 32  # 简单索引字节数
    bytes_per_hand = 64         # 包含编码信息的记录字节数

    # 对比Resolution 9（常见"中等精度"基准）
    traditional_count   = level_9_count if level_9_count > 0 else level_7_count * 7
    traditional_storage = traditional_count * bytes_per_traditional
    hand_storage        = len(tree_structure) * bytes_per_hand   # 包含所有树节点

    print(f"  传统方法 (Res 9, ~{bytes_per_traditional}B/条): {traditional_storage/1024/1024:.2f} MB  ({traditional_count:,} 条)")
    print(f"  The Hand (树节点, ~{bytes_per_hand}B/条):       {hand_storage/1024/1024:.2f} MB  ({len(tree_structure):,} 条)")
    if traditional_storage > 0:
        storage_savings = (1 - hand_storage / traditional_storage) * 100
        print(f"  节省存储空间: {storage_savings:.1f}%")
    else:
        storage_savings = 0.0

    # ── 精度指标：平均格网面积 ────────────────────────────────
    print("\n[6.2.5] 自适应格网精度指标:")
    
    # 从CSV中读取叶子节点分辨率分布
    leaves_df = pd.read_csv('adaptive_tree_leaves.csv')
    res_dist = leaves_df['resolution'].value_counts().to_dict()
    
    total_area_km2 = 0.0
    for res, count in res_dist.items():
        avg_area = h3.average_hexagon_area(res, 'km^2')
        total_area_km2 += avg_area * count
        print(f"  Resolution {res}: {count} 个格网 × {avg_area:.4f} km² = {avg_area*count:.2f} km²")
    
    avg_leaf_area = total_area_km2 / hand_count if hand_count > 0 else 0
    print(f"\n  叶子节点覆盖总面积: {total_area_km2:.2f} km²")
    print(f"  叶子节点平均面积:   {avg_leaf_area:.4f} km²")

    # ── 收集结果 ─────────────────────────────────────────────
    metrics = {
        'reference_counts': reference_counts,
        'hand_count':        hand_count,
        'tree_nodes':        len(tree_structure),
        'traditional_count': traditional_count,
        'traditional_storage_mb': traditional_storage / 1024 / 1024,
        'hand_storage_mb':   hand_storage / 1024 / 1024,
        'storage_savings_pct': storage_savings if traditional_storage > 0 else 0.0,
        'reduction_vs_9_pct':  reduction_vs_9 if level_9_count > 0 else 0.0,
        'reduction_vs_10_pct': reduction_vs_10 if level_10_count > 0 else 0.0,
        'avg_leaf_area_km2':  avg_leaf_area,
        'A0': A0,
        'res_distribution': res_dist,
    }

    return metrics


# ============================================================
# 步骤6.3：定量分析 - 应用效能验证（范围查询）
# ============================================================

def get_leaf_descendants(h3_cell, tree_structure, leaf_set):
    """
    递归获取一个格网节点的所有叶子后代。
    
    由于 target_cell_relations.csv 中记录的是 H0 层（Resolution 7）的格网，
    而叶子节点可能是 Resolution 8~10 的子格网，
    需要沿树向下展开，找到真正覆盖该区域的所有叶子节点。
    
    Parameters:
    - h3_cell: 起始格网（可能是内部节点或叶子节点）
    - tree_structure: 自适应树结构
    - leaf_set: 叶子节点集合
    
    Returns:
    - set: 所有叶子后代格网
    """
    if h3_cell in leaf_set:
        return {h3_cell}
    node = tree_structure.get(h3_cell)
    if not node:
        return set()
    result = set()
    for child in node.get('children', []):
        result |= get_leaf_descendants(child, tree_structure, leaf_set)
    return result


def step_6_3_query_performance(yolo_results, tree_structure, leaf_set, target_relations):
    """
    步骤6.3：基于H3索引的范围查询效能验证
    
    注意：target_cell_relations.csv 记录的 h3_cell 是 H0 层（Resolution 7）格网。
    对于已被分裂的H0格网，查询时需要向下展开至其叶子后代，以获得真实的自适应格网查询结果。
    这体现了The Hand多分辨率索引结构的层次查询能力。
    
    场景1：查询"所有包含船舶(ship)的格网"
    场景2：查询"所有包含飞机(plane)的格网"
    场景3：查询"高置信度(>0.9)目标覆盖的格网"
    场景4：查询"所有含目标的格网"
    """
    print("\n" + "=" * 60)
    print("步骤6.3：定量分析 - 应用效能验证（范围查询）")
    print("=" * 60)

    query_results_all = {}

    # ── 构建H3索引映射 ────────────────────────────────────────
    print("\n[6.3.1] 构建H3索引查找表（含层次展开）...")
    print("  说明：target_cell_relations.csv 的 h3_cell 为 H0层(Res=7)格网")
    print("        查询时向下展开至叶子后代，体现多分辨率层次查询")

    # 按目标类别分组 H0 格网
    class_to_h0cells = {}
    for _, row in target_relations.iterrows():
        cls  = row['target_class']
        cell = row['h3_cell']
        if cls not in class_to_h0cells:
            class_to_h0cells[cls] = set()
        class_to_h0cells[cls].add(cell)

    # 按置信度分组
    high_conf_h0_cells = set(
        target_relations[target_relations['confidence'] > 0.9]['h3_cell'].values
    )
    all_h0_cells = set(target_relations['h3_cell'].values)

    print(f"  ✓ 共 {len(all_h0_cells)} 个H0格网含目标, {len(leaf_set)} 个叶子节点")

    # ── 查询场景1：船舶目标 ───────────────────────────────────
    print("\n[6.3.2] 查询场景1: 包含'ship'的叶子格网")

    start_time = time.perf_counter()

    ship_h0_cells = class_to_h0cells.get('ship', set())
    ship_leaf_cells = set()
    for c in ship_h0_cells:
        ship_leaf_cells |= get_leaf_descendants(c, tree_structure, leaf_set)

    query_time_ship = (time.perf_counter() - start_time) * 1000

    print(f"  H0格网(Res=7)查到: {len(ship_h0_cells)} 个")
    print(f"  展开到叶子格网:    {len(ship_leaf_cells)} 个")
    print(f"  查询时间: {query_time_ship:.4f} ms")
    print(f"  查询效率: {len(ship_leaf_cells)/max(query_time_ship/1000, 1e-9):.0f} cells/s")

    query_results_all['ship'] = {
        'query_class':    'ship（船舶）',
        'h0_cells':       len(ship_h0_cells),
        'result_cells':   len(ship_leaf_cells),
        'query_time_ms':  query_time_ship,
        'throughput_cps': len(ship_leaf_cells) / max(query_time_ship / 1000, 1e-9),
    }

    # ── 查询场景2：飞机目标 ───────────────────────────────────
    print("\n[6.3.3] 查询场景2: 包含'plane'的叶子格网")

    start_time = time.perf_counter()

    plane_h0_cells = class_to_h0cells.get('plane', set())
    plane_leaf_cells = set()
    for c in plane_h0_cells:
        plane_leaf_cells |= get_leaf_descendants(c, tree_structure, leaf_set)

    query_time_plane = (time.perf_counter() - start_time) * 1000

    print(f"  H0格网(Res=7)查到: {len(plane_h0_cells)} 个")
    print(f"  展开到叶子格网:    {len(plane_leaf_cells)} 个")
    print(f"  查询时间: {query_time_plane:.4f} ms")
    print(f"  查询效率: {len(plane_leaf_cells)/max(query_time_plane/1000, 1e-9):.0f} cells/s")

    query_results_all['plane'] = {
        'query_class':    'plane（飞机）',
        'h0_cells':       len(plane_h0_cells),
        'result_cells':   len(plane_leaf_cells),
        'query_time_ms':  query_time_plane,
        'throughput_cps': len(plane_leaf_cells) / max(query_time_plane / 1000, 1e-9),
    }

    # ── 查询场景3：高置信度目标 ──────────────────────────────
    print("\n[6.3.4] 查询场景3: 高置信度(>0.9)目标覆盖的叶子格网")

    start_time = time.perf_counter()

    high_conf_leaf_cells = set()
    for c in high_conf_h0_cells:
        high_conf_leaf_cells |= get_leaf_descendants(c, tree_structure, leaf_set)

    query_time_hconf = (time.perf_counter() - start_time) * 1000

    print(f"  H0格网(Res=7)查到: {len(high_conf_h0_cells)} 个")
    print(f"  展开到叶子格网:    {len(high_conf_leaf_cells)} 个")
    print(f"  查询时间: {query_time_hconf:.4f} ms")
    print(f"  查询效率: {len(high_conf_leaf_cells)/max(query_time_hconf/1000, 1e-9):.0f} cells/s")

    query_results_all['high_confidence'] = {
        'query_class':    'confidence > 0.9',
        'h0_cells':       len(high_conf_h0_cells),
        'result_cells':   len(high_conf_leaf_cells),
        'query_time_ms':  query_time_hconf,
        'throughput_cps': len(high_conf_leaf_cells) / max(query_time_hconf / 1000, 1e-9),
    }

    # ── 查询场景4：全类别汇总查询 ────────────────────────────
    print("\n[6.3.5] 查询场景4: 全类别目标覆盖的叶子格网")

    start_time = time.perf_counter()

    all_target_leaf_cells = set()
    for c in all_h0_cells:
        all_target_leaf_cells |= get_leaf_descendants(c, tree_structure, leaf_set)

    query_time_all = (time.perf_counter() - start_time) * 1000

    print(f"  H0格网(Res=7)查到: {len(all_h0_cells)} 个")
    print(f"  展开到叶子格网:    {len(all_target_leaf_cells)} 个")
    print(f"  查询时间: {query_time_all:.4f} ms")
    print(f"  查询效率: {len(all_target_leaf_cells)/max(query_time_all/1000, 1e-9):.0f} cells/s")

    query_results_all['all_targets'] = {
        'query_class':    'all targets（全目标）',
        'h0_cells':       len(all_h0_cells),
        'result_cells':   len(all_target_leaf_cells),
        'query_time_ms':  query_time_all,
        'throughput_cps': len(all_target_leaf_cells) / max(query_time_all / 1000, 1e-9),
    }

    # ── 主查询结果（用于步骤6.4统计表）────────────────────────
    primary_query = query_results_all['ship']

    print(f"\n  ✓ 范围查询效能验证完成")
    print(f"  主查询(ship): {primary_query['result_cells']} cells / {primary_query['query_time_ms']:.4f} ms")

    return query_results_all, primary_query


# ============================================================
# 步骤6.4：保存定量分析结果表
# ============================================================

def step_6_4_save_quantitative(metrics, query_results_all, primary_query, stats):
    """
    步骤6.4：保存完整的定量分析结果表（CSV）及可视化（PNG）
    """
    print("\n" + "=" * 60)
    print("步骤6.4：保存定量分析结果表")
    print("=" * 60)

    ref_counts = metrics['reference_counts']
    level_9_count  = ref_counts.get(9, 0)
    level_10_count = ref_counts.get(10, 0)
    hand_count     = metrics['hand_count']
    trad_count     = metrics['traditional_count']

    # ── 存储对比说明：以Resolution 10作为最细粒度均匀覆盖基准 ──────
    # The Hand自适应格网最细到Res10，因此以Res10全覆盖为存储对比基准更有说服力
    level_10_count  = ref_counts.get(10, 0)
    bytes_per_traditional = 32
    bytes_per_hand = 64
    trad_storage_r10 = level_10_count * bytes_per_traditional if level_10_count > 0 else 0

    # ── 主结果表 ─────────────────────────────────────────────
    results_df = pd.DataFrame({
        'Metric': [
            'Grid Count (Resolution 7, uniform)',
            'Grid Count (Resolution 8, uniform)',
            'Grid Count (Resolution 9, uniform)',
            'Grid Count (Resolution 10, uniform)',
            'Grid Count (CD-MCAR adaptive)',
            'Grid Reduction vs Res-9 (%)',
            'Grid Reduction vs Res-10 (%)',
            'Storage (Traditional Res-10, ~32B/cell) MB',
            'Storage (CD-MCAR, ~64B/node) MB',
            'Storage Savings vs Res-10 (%)',
            'Avg Leaf Cell Area (km²)',
            'Query Cells - ship (leaf descendants)',
            'Query Time - ship (ms)',
            'Query Throughput - ship (cells/s)',
            'Query Cells - plane (leaf descendants)',
            'Query Time - plane (ms)',
            'Query Cells - confidence>0.9 (leaf descendants)',
            'Query Time - high conf (ms)',
            'Query Cells - all targets (leaf descendants)',
            'Query Time - all targets (ms)',
            'Tree Nodes (total)',
            'Leaf Nodes',
            'Split Nodes',
            'H0 Cells with Targets',
            'Total Target Records',
        ],
        'Value': [
            ref_counts.get(7, 'N/A'),
            ref_counts.get(8, 'N/A'),
            ref_counts.get(9, 'N/A'),
            ref_counts.get(10, 'N/A'),
            hand_count,
            f"{metrics['reduction_vs_9_pct']:.1f}%",
            f"{metrics['reduction_vs_10_pct']:.1f}%",
            f"{trad_storage_r10/1024/1024:.2f}" if trad_storage_r10 > 0 else 'N/A',
            f"{metrics['hand_storage_mb']:.2f}",
            f"{(1 - metrics['hand_storage_mb'] / (trad_storage_r10/1024/1024))*100:.1f}%" if trad_storage_r10 > 0 else 'N/A',
            f"{metrics['avg_leaf_area_km2']:.6f}",
            query_results_all['ship']['result_cells'],
            f"{query_results_all['ship']['query_time_ms']:.4f}",
            f"{query_results_all['ship']['throughput_cps']:.0f}",
            query_results_all['plane']['result_cells'],
            f"{query_results_all['plane']['query_time_ms']:.4f}",
            query_results_all['high_confidence']['result_cells'],
            f"{query_results_all['high_confidence']['query_time_ms']:.4f}",
            query_results_all['all_targets']['result_cells'],
            f"{query_results_all['all_targets']['query_time_ms']:.4f}",
            metrics['tree_nodes'],
            stats['leaf_cells'],
            stats.get('non_leaf_cells', 0),
            stats['cells_with_targets'],
            stats['total_targets_covered'],
        ],
        'Unit': [
            'cells', 'cells', 'cells', 'cells', 'cells',
            '%', '%',
            'MB', 'MB', '%',
            'km²',
            'cells', 'ms', 'cells/s',
            'cells', 'ms',
            'cells', 'ms',
            'cells', 'ms',
            'nodes', 'nodes', 'nodes',
            'cells', 'count',
        ],
        'Description': [
            'H3 Resolution 7 uniform full-cover grid count (H0 initial layer)',
            'H3 Resolution 8 uniform full-cover grid count',
            'H3 Resolution 9 uniform full-cover grid count',
            'H3 Resolution 10 uniform full-cover grid count (finest, baseline)',
            'CD-MCAR adaptive algorithm leaf node count',
            'Grid count change ratio vs Resolution 9 uniform',
            'Grid reduction ratio vs Resolution 10 uniform',
            'Traditional method (Res-10 full cover x 32B) estimated storage',
            'CD-MCAR method (all tree nodes x 64B) estimated storage',
            'Storage savings ratio vs Res-10 full cover',
            'Weighted average area of leaf nodes (km2)',
            'Leaf grid cells covering ship targets (H0->leaf expansion)',
            'Query time for ship target grid cells via H3 index (ms)',
            'Ship query throughput via H3 index (cells/s)',
            'Leaf grid cells covering plane targets (H0->leaf expansion)',
            'Query time for plane target grid cells via H3 index (ms)',
            'Leaf grid cells covering high-confidence (>0.9) targets',
            'Query time for high-confidence target grid cells via H3 index (ms)',
            'Leaf grid cells covering all-category targets',
            'Query time for all-target grid cells via H3 index (ms)',
            'Total adaptive tree nodes (including internal nodes)',
            'Leaf node count (final grid unit count)',
            'Internal nodes that underwent splitting',
            'H0 grid cells containing targets',
            'Total grid-target relation pairs',
        ]
    })

    csv_path = 'quantitative_analysis.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ 定量分析结果已保存: {csv_path}")
    print(f"  ✓ 共 {len(results_df)} 条指标记录")

    # ── 可视化：效率对比图 ────────────────────────────────────
    _plot_efficiency_chart(metrics, query_results_all, stats)

    return results_df


def _plot_efficiency_chart(metrics, query_results_all, stats):
    """绘制定量分析效率对比图（4子图）— 左列(柱/饼)紧凑化"""

    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor('white')
    fig.suptitle('CD-MCAR Algorithm — Quantitative Efficiency Analysis',
                 fontsize=14, fontweight='bold', color='black', y=1.02)

    # GridSpec: 左列(柱+饼)占1份宽度，右列(存储+查询)占2份宽度
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], hspace=0.30, wspace=0.25)

    # ── 子图1：格网数量对比（柱状图）─────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('white')

    ref_counts = metrics['reference_counts']
    labels = [f'Res-{r}\nUniform' for r in sorted(ref_counts.keys())] + ['CD-MCAR\n(Adaptive)']
    values = [ref_counts[r] for r in sorted(ref_counts.keys())] + [metrics['hand_count']]
    colors = ['#4a9eda', '#4a9eda', '#4a9eda', '#4a9eda', '#FF6B6B']

    bars = ax1.bar(labels, values, color=colors, edgecolor='#333', linewidth=0.8, alpha=0.85)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f'{val:,}', ha='center', va='bottom',
                 fontsize=8, color='black', fontweight='bold')

    ax1.set_title('Grid Count Comparison', fontsize=11, fontweight='bold', color='black')
    ax1.set_ylabel('Grid Count', color='#333', fontsize=9)
    ax1.tick_params(colors='#333', labelsize=8)
    ax1.set_facecolor('white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#999')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # ── 子图2：存储空间对比（饼图 + 柱图）────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('white')

    storage_labels = ['Traditional\n(Res-9 Uniform)', 'CD-MCAR\n(Adaptive)']
    storage_values = [metrics['traditional_storage_mb'], metrics['hand_storage_mb']]
    s_colors = ['#E74C3C', '#2ECC71']

    bars2 = ax2.bar(storage_labels, storage_values, color=s_colors,
                    edgecolor='#333', linewidth=0.8, alpha=0.85, width=0.5)
    for bar, val in zip(bars2, storage_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f'{val:.2f} MB', ha='center', va='bottom',
                 fontsize=10, color='black', fontweight='bold')

    if metrics['storage_savings_pct'] > 0:
        ax2.annotate(
            f"Save {metrics['storage_savings_pct']:.1f}%",
            xy=(1, storage_values[1]), xytext=(0.5, max(storage_values) * 0.6),
            fontsize=11, color='#C41E3A', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#C41E3A', lw=1.5),
            ha='center'
        )

    ax2.set_title('Storage Comparison', fontsize=11, fontweight='bold', color='black')
    ax2.set_ylabel('Storage (MB)', color='#333', fontsize=9)
    ax2.tick_params(colors='#333', labelsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#999')

    # ── 子图3：叶子节点分辨率分布（堆叠饼图）────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('white')

    res_dist = metrics['res_distribution']
    res_labels = [f'Resolution {r}' for r in sorted(res_dist.keys())]
    res_values = [res_dist[r] for r in sorted(res_dist.keys())]
    pie_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(res_labels)]

    wedges, texts, autotexts = ax3.pie(
        res_values, labels=res_labels, autopct='%1.1f%%',
        colors=pie_colors, startangle=90,
        pctdistance=0.75, labeldistance=1.1,
        textprops={'color': 'black', 'fontsize': 9}
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color('white')
        at.set_fontweight('bold')

    ax3.set_title('Leaf Node Resolution Distribution', fontsize=11, fontweight='bold', color='black')

    # 中心标注
    ax3.text(0, 0, f'Total\n{sum(res_values):,}',
             ha='center', va='center', fontsize=10,
             color='black', fontweight='bold')

    # ── 子图4：查询效能对比（横向柱图）──────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('white')

    q_labels = ['ship', 'plane', 'conf>0.9', 'all targets']
    q_times  = [
        query_results_all['ship']['query_time_ms'],
        query_results_all['plane']['query_time_ms'],
        query_results_all['high_confidence']['query_time_ms'],
        query_results_all['all_targets']['query_time_ms'],
    ]
    q_results = [
        query_results_all['ship']['result_cells'],
        query_results_all['plane']['result_cells'],
        query_results_all['high_confidence']['result_cells'],
        query_results_all['all_targets']['result_cells'],
    ]
    q_colors = ['#FF6B6B', '#FFD700', '#00FF7F', '#00BFFF']

    bars4 = ax4.barh(q_labels, q_times, color=q_colors, edgecolor='#333', linewidth=0.8, alpha=0.85)
    max_t = max(q_times) if max(q_times) > 0 else 1
    for bar, qt, qr in zip(bars4, q_times, q_results):
        ax4.text(bar.get_width() + max_t * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{qt:.3f} ms  ({qr} cells)',
                 va='center', ha='left', fontsize=8, color='black')

    ax4.set_title('Range Query Performance (Leaf Expansion)',
                  fontsize=11, fontweight='bold', color='black')
    ax4.set_xlabel('Query Time (ms)', color='#333', fontsize=9)
    ax4.tick_params(colors='#333', labelsize=8)
    for spine in ax4.spines.values():
        spine.set_edgecolor('#999')
    ax4.set_xlim(0, max_t * 3.5)

    plt.tight_layout(pad=2.0)
    output_path = 'quantitative_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ 定量分析可视化已保存: {output_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("步骤6：结果评估与分析")
    print("=" * 60)

    # 加载数据
    (yolo_results, tree_structure, leaf_set, H0_initial,
     target_relations, stats, bbox, A0) = load_all_data()

    # 步骤6.1：定性可视化对比
    print("\n>>> 步骤6.1：可视化对比...")
    step_6_1_qualitative_visualization(
        yolo_results, tree_structure, leaf_set, H0_initial, bbox)

    # 步骤6.2：效率指标计算
    print("\n>>> 步骤6.2：效率指标...")
    metrics = step_6_2_efficiency_metrics(yolo_results, tree_structure, leaf_set, A0)

    # 步骤6.3：范围查询效能验证
    print("\n>>> 步骤6.3：查询效能验证...")
    query_results_all, primary_query = step_6_3_query_performance(
        yolo_results, tree_structure, leaf_set, target_relations)

    # 步骤6.4：保存定量分析结果表
    print("\n>>> 步骤6.4：保存定量结果...")
    results_df = step_6_4_save_quantitative(
        metrics, query_results_all, primary_query, stats)

    print("\n" + "=" * 60)
    print("步骤6 完成！")
    print("=" * 60)
    print("\n输出文件:")
    print("  1. qualitative_comparison.png  — 定性可视化对比图（步骤6.1）")
    print("  2. quantitative_analysis.csv   — 定量分析指标表（步骤6.4）")
    print("  3. quantitative_analysis.png   — 定量分析效率图（步骤6.4）")

    print("\n=== 核心指标汇总 ===")
    ref9  = metrics['reference_counts'].get(9, 0)
    ref10 = metrics['reference_counts'].get(10, 0)
    print(f"  The Hand 叶子节点: {metrics['hand_count']:,} 个")
    print(f"  Resolution 9  均匀: {ref9:,} 个  →  相比减少 {metrics['reduction_vs_9_pct']:.1f}%")
    print(f"  Resolution 10 均匀: {ref10:,} 个  →  相比减少 {metrics['reduction_vs_10_pct']:.1f}%")
    print(f"  树节点总数:         {metrics['tree_nodes']:,}")
    print(f"  ship查询叶子格网:  {query_results_all['ship']['result_cells']} cells / {query_results_all['ship']['query_time_ms']:.3f} ms")
    print(f"  all查询叶子格网:   {query_results_all['all_targets']['result_cells']} cells / {query_results_all['all_targets']['query_time_ms']:.3f} ms")

    return results_df


if __name__ == '__main__':
    main()
