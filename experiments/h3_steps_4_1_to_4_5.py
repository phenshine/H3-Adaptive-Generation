"""
步骤4.1~4.5: 边界效应处理与编码
基于Algorithm B.2.2 (H3-LSP), B.2.3 (H3-CBFE), A.4.5 (h3_compact)

修正参考代码中的h3 v4 API错误：
- h3.k_ring() -> h3.grid_disk()
- h3.h3_to_children() -> h3.cell_to_children()
- h3.h3_get_resolution() -> h3.get_resolution()
- h3.h3_to_parent() -> h3.cell_to_parent()
- h3.compact() -> h3.compact_cells()
"""

import json
import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Polygon, Point
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 步骤4.1：边界层级不连续检测与平滑 (Algorithm B.2.2: H3-LSP)
# ============================================================

def detect_resolution_discontinuity(tree_structure, leaf_cells, delta_r_smooth=1):
    """
    检测层级间的分辨率不连续
    
    RDM = 1/|ε| * Σ max(0, |r_j - r_k| - Δr_smooth)²
    
    Parameters:
    - tree_structure: 字典，H3自适应树结构 {h3_cell: {resolution, targets, split, children}}
    - leaf_cells: 叶子节点集合
    - delta_r_smooth: 允许的最大分辨率差，默认1
    
    Returns:
    - discontinuities: 不连续列表
    - rdm_score: 分辨率不连续性度量
    """
    discontinuities = []
    
    # 构建叶子节点到分辨率的映射
    leaf_resolution = {}
    for h_cell in leaf_cells:
        if h_cell in tree_structure:
            leaf_resolution[h_cell] = tree_structure[h_cell]['resolution']
    
    # 对每个叶子节点，检查其邻域
    epsilon = len(leaf_cells)  # |ε| = 叶子节点数量
    
    total_penalty = 0
    
    for h_leaf in leaf_cells:
        if h_leaf not in leaf_resolution:
            continue
            
        r_leaf = leaf_resolution[h_leaf]
        
        # 获取同层邻域格网 (k_ring)
        try:
            neighbors = h3.grid_disk(h_leaf, 1)  # h3 v4 API
        except Exception as e:
            print(f"  Warning: grid_disk failed for {h_leaf}: {e}")
            continue
        
        for neighbor in neighbors:
            # 跳过自身
            if neighbor == h_leaf:
                continue
            
            # 只检查邻域中的叶子节点
            if neighbor in leaf_resolution:
                r_neighbor = leaf_resolution[neighbor]
                
                # 检查不连续性：分辨率差 > Δr_smooth
                r_diff = abs(r_leaf - r_neighbor)
                if r_diff > delta_r_smooth:
                    penalty = (r_diff - delta_r_smooth) ** 2
                    total_penalty += penalty
                    
                    discontinuities.append({
                        'cell1': h_leaf,
                        'cell2': neighbor,
                        'r1': r_leaf,
                        'r2': r_neighbor,
                        'r_diff': r_diff,
                        'penalty': penalty
                    })
    
    # 计算RDM
    rdm_score = total_penalty / max(epsilon, 1)
    
    return discontinuities, rdm_score


def smooth_discontinuities(tree_structure, discontinuities, leaf_cells, r_max=10):
    """
    平滑不连续：对粗分辨率叶子节点进行细分
    
    Parameters:
    - tree_structure: H3自适应树结构
    - discontinuities: 检测到的不连续列表
    - leaf_cells: 当前叶子节点集合
    - r_max: 最大分辨率
    
    Returns:
    - smoothed_leaves: 平滑后的叶子节点集合
    - refinement_log: 细分记录
    """
    refinement_log = []
    cells_to_refine = set()
    
    # 找出需要细分的格网（粗分辨率的那个）
    for disc in discontinuities:
        cell1, cell2 = disc['cell1'], disc['cell2']
        r1, r2 = disc['r1'], disc['r2']
        
        # 对分辨率较粗的格网进行细分
        if r1 < r2:
            cells_to_refine.add(cell1)
        elif r2 < r1:
            cells_to_refine.add(cell2)
    
    # 执行渐进式细分
    smoothed_leaves = set(leaf_cells)
    
    for coarse_cell in cells_to_refine:
        if coarse_cell not in tree_structure:
            continue
            
        current_res = tree_structure[coarse_cell]['resolution']
        target_res = min(current_res + 2, r_max)  # 最多增加2级
        
        if current_res >= target_res:
            continue
        
        # 渐进式细分
        refined = gradual_refinement_single(coarse_cell, current_res, target_res)
        
        # 从叶子集合中移除粗格网，添加细分后的格网
        if coarse_cell in smoothed_leaves:
            smoothed_leaves.remove(coarse_cell)
            smoothed_leaves.update(refined)
            
            refinement_log.append({
                'coarse_cell': coarse_cell,
                'original_res': current_res,
                'target_res': target_res,
                'refined_cells': list(refined)
            })
    
    return smoothed_leaves, refinement_log


# ============================================================
# 步骤4.2：逐步细分填充（Gradual Refinement）
# ============================================================

def gradual_refinement_single(h_cell, r_current, r_target):
    """
    对单个格网执行渐进式细分
    
    Gradual_Refine(h_j, r_j, r_j') = h3_uncompact({h_j}, r_j')
    
    Parameters:
    - h_cell: H3格网索引
    - r_current: 当前分辨率
    - r_target: 目标分辨率
    
    Returns:
    - refined_cells: 细分后的格网集合
    """
    current_cells = {h_cell}
    
    # 逐分辨率细分
    while r_current < r_target:
        next_cells = set()
        for h in current_cells:
            try:
                # h3 v4 API: cell_to_children()
                children = h3.cell_to_children(h, r_current + 1)
                next_cells.update(children)
            except Exception as e:
                print(f"  Warning: cell_to_children failed for {h}: {e}")
                next_cells.add(h)  # 保持原样
        
        current_cells = next_cells
        r_current += 1
    
    return current_cells


def gradual_refinement(coarse_cells, r_target, tree_structure):
    """
    执行渐进性细分填充（批量版本）
    
    Parameters:
    - coarse_cells: 粗分辨率格网列表
    - r_target: 目标分辨率
    - tree_structure: 树结构
    
    Returns:
    - refined_cells: {coarse_cell: {refined_children}} 字典
    """
    refined_cells = {}
    
    for h_coarse in coarse_cells:
        if h_coarse in tree_structure:
            current_res = tree_structure[h_coarse]['resolution']
            refined = gradual_refinement_single(h_coarse, current_res, r_target)
            refined_cells[h_coarse] = refined
    
    return refined_cells


# ============================================================
# 步骤4.3：H3 Compact操作 (Algorithm A.4.5)
# ============================================================

def compact_h3_cells(cell_set):
    """
    执行H3压缩操作
    
    Algorithm A.4.5: h3_compact
    将多个子分辨率格网合并为父格网（如果所有子都包含）
    
    Parameters:
    - cell_set: H3格网索引集合
    
    Returns:
    - compacted: 压缩后的格网集合
    """
    # 确保是字符串格式
    cell_list = [str(c) for c in cell_set]
    
    try:
        # h3 v4 API: compact_cells()
        compacted = h3.compact_cells(cell_list)
        return set(compacted)
    except Exception as e:
        print(f"  Warning: compact_cells failed: {e}")
        return cell_set


def expand_h3_cells(compacted_set, r_target):
    """
    执行H3展开操作（Compact的逆操作）
    
    Parameters:
    - compacted_set: 压缩后的格网集合
    - r_target: 目标分辨率
    
    Returns:
    - expanded: 展开后的格网集合
    """
    expanded = set()
    
    for h_cell in compacted_set:
        current_res = h3.get_resolution(h_cell)  # h3 v4 API
        
        if current_res < r_target:
            # 需要展开
            children = h3.cell_to_children(h_cell, r_target)
            expanded.update(children)
        else:
            # 已经是目标分辨率
            expanded.add(h_cell)
    
    return expanded


# ============================================================
# 步骤4.4：跨界目标融合编码 (Algorithm B.2.3: H3-CBFE)
# ============================================================

def find_nca(h3_cells):
    """
    找到H3格网列表的最近公共祖先 (Nearest Common Ancestor, NCA)
    
    Parameters:
    - h3_cells: H3格网索引列表
    
    Returns:
    - nca: 最近公共祖先格网
    - nca_resolution: NCA的分辨率
    """
    # 确保是字符串列表
    h3_cells = [str(h) for h in h3_cells if h]
    
    if not h3_cells:
        return None, -1
    
    if len(h3_cells) == 1:
        return h3_cells[0], h3.get_resolution(h3_cells[0])
    
    # 从最高分辨率开始，逐层上溯
    min_res = min(h3.get_resolution(h) for h in h3_cells)
    
    # 将所有格网上溯到同一分辨率层级
    ancestors_at_min_res = []
    for h in h3_cells:
        r = h3.get_resolution(h)
        if r > min_res:
            # 上溯到min_res
            h_ancestor = h3.cell_to_parent(h, min_res)  # h3 v4 API
            ancestors_at_min_res.append(h_ancestor)
        else:
            ancestors_at_min_res.append(h)
    
    # 如果已经在同一个格网，返回它
    if len(set(ancestors_at_min_res)) == 1:
        return ancestors_at_min_res[0], min_res
    
    # 否则继续上溯
    current_res = min_res
    current_ancestors = ancestors_at_min_res
    
    while current_res > 0:
        current_res -= 1
        parents = [h3.cell_to_parent(h, current_res) for h in current_ancestors]
        
        if len(set(parents)) == 1:
            return parents[0], current_res
        
        current_ancestors = parents
    
    # 如果到了分辨率0，返回原格网
    return h3_cells[0], h3.get_resolution(h3_cells[0])


def encode_boundary_targets(boundary_targets, tree_structure, targets_gdf):
    """
    对跨越H3边界的目标进行融合编码
    
    Algorithm B.2.3: H3-CBFE
    
    三种策略：
    (a) H3-Ascend: 上溯编码（适用于小目标）
    (b) H3-Primary-Secondary: 主从编码（适用于中等目标）
    (c) H3-Multi-Code: 多码联合编码（适用于大目标）
    
    Parameters:
    - boundary_targets: 边界目标列表 [{target_id, h3_coverage, confidence, area}, ...]
    - tree_structure: H3自适应树结构
    - targets_gdf: 目标GeoDataFrame
    
    Returns:
    - encoded_targets: 编码结果 {target_id: {strategy, code, h3_cells, confidence}}
    """
    encoded_targets = {}
    
    for target_info in boundary_targets:
        target_id = target_info['target_id']
        h3_cells_raw = target_info['h3_cells']
        
        # 确保h3_cells是字符串列表
        if isinstance(h3_cells_raw, list):
            h3_cells = [str(h) for h in h3_cells_raw if h]
        else:
            h3_cells = [str(h3_cells_raw)]
        
        confidence = target_info['confidence']
        area = target_info['area']
        
        # 计算NCA（最近公共祖先）
        nca_cell, nca_resolution = find_nca(h3_cells)
        
        # 根据目标特性选择编码策略
        # 策略选择规则（基于Algorithm B.2.3）：
        # 1. 小目标（覆盖格网数少或面积小）-> H3-Ascend
        # 2. 中等目标（置信度高）-> H3-Primary-Secondary
        # 3. 大目标 -> H3-Multi-Code
        
        n_cells = len(h3_cells)
        
        # 策略选择
        if nca_resolution >= 9 or n_cells <= 2:  # 小目标：分辨率高或覆盖格网少
            strategy = 'H3-Ascend'
            # 使用NCA作为编码
            code = nca_cell
        elif confidence >= 0.85 and n_cells <= 5:  # 中等目标：置信度高
            strategy = 'H3-Primary-Secondary'
            # 主格网 + 从属格网列表
            primary = h3_cells[0]
            secondary = h3_cells[1:] if len(h3_cells) > 1 else []
            code = {
                'primary': primary,
                'secondary': secondary,
                'nca': nca_cell
            }
        else:  # 大目标：默认多码联合
            strategy = 'H3-Multi-Code'
            # 所有覆盖格网的联合编码
            code = {
                'cells': h3_cells,
                'nca': nca_cell,
                'n_children': n_cells
            }
        
        encoded_targets[target_id] = {
            'h3_cells': h3_cells,
            'nca_cell': nca_cell,
            'nca_resolution': nca_resolution,
            'strategy': strategy,
            'code': code,
            'confidence': confidence,
            'area': area
        }
    
    return encoded_targets


# ============================================================
# 步骤4.5：执行边界处理（主函数）
# ============================================================

def load_tree_structure(json_path='adaptive_tree.json'):
    """加载H3自适应树结构"""
    with open(json_path, 'r') as f:
        tree_structure = json.load(f)
    return tree_structure


def load_leaf_cells(csv_path='adaptive_tree_leaves.csv'):
    """从CSV加载叶子节点"""
    df = pd.read_csv(csv_path)
    return set(df['h3_cell'].values)


def load_boundary_targets(geojson_path='targets_classified.geojson'):
    """
    加载边界目标（EDGE_CROSSING 或 MULTI_NEIGHBOR类别）
    
    Note: 根据步骤2.6的结果，所有目标都是CONTAINED，
    但我们可以模拟边界目标的处理逻辑
    """
    gdf = gpd.read_file(geojson_path)
    
    # 筛选边界目标（如果存在）
    if 'target_class' in gdf.columns:
        boundary_gdf = gdf[gdf['target_class'].isin(['EDGE_CROSSING', 'MULTI_NEIGHBOR'])]
    else:
        boundary_gdf = gdf.iloc[0:0]  # 空DataFrame
    
    # 如果没有边界目标，使用所有目标进行演示
    if len(boundary_gdf) == 0:
        print("  注意：未检测到EDGE_CROSSING或MULTI_NEIGHBOR目标")
        print("  使用所有目标进行编码演示...")
        boundary_gdf = gdf
    
    # 转换为编码函数所需的格式
    boundary_targets = []
    for idx, row in boundary_gdf.iterrows():
        # 解析h3_coverage - 可能是numpy数组、列表或字符串
        h3_coverage = row.get('h3_coverage_json', [])
        
        if isinstance(h3_coverage, np.ndarray):
            # numpy数组，直接转换为字符串列表
            h3_cells = [str(h) for h in h3_coverage.tolist()]
        elif isinstance(h3_coverage, str):
            try:
                import ast
                parsed = ast.literal_eval(h3_coverage)
                if isinstance(parsed, (list, np.ndarray)):
                    h3_cells = [str(h) for h in parsed]
                else:
                    h3_cells = [str(parsed)]
            except:
                h3_cells = [h3_coverage]
        elif isinstance(h3_coverage, (list, tuple)):
            h3_cells = [str(h) for h in h3_coverage]
        else:
            h3_cells = [str(h3_coverage)]
        
        if not h3_cells:
            h3_cells = [str(row.get('h3_center', ''))]
        
        boundary_targets.append({
            'target_id': int(idx) if isinstance(idx, (np.integer, np.int64)) else idx,
            'h3_cells': h3_cells,
            'confidence': float(row.get('confidence', 0.5)),
            'area': float(row.get('width_m', 0) * row.get('height_m', 1))
        })
    
    return boundary_targets, gdf


def visualize_boundary_process(discontinuities, refinement_log, encoded_targets, 
                              tree_structure, leaf_cells, output_path='h3_steps_4_result.png'):
    """
    可视化边界处理过程
    
    Parameters:
    - discontinuities: 检测到的不连续
    - refinement_log: 细分记录
    - encoded_targets: 编码结果
    - tree_structure: 树结构
    - leaf_cells: 叶子节点集合
    - output_path: 输出图片路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Step 4: Boundary Effect Processing and Encoding', fontsize=16, fontweight='bold')
    
    # 子图1：分辨率分布（处理前 vs 处理后）
    ax1 = axes[0, 0]
    
    # 处理前分辨率分布
    res_before = [tree_structure[cell]['resolution'] for cell in leaf_cells 
                  if cell in tree_structure]
    ax1.hist(res_before, bins=range(7, 12), alpha=0.5, label='Before Smoothing', 
             color='lightblue', edgecolor='black')
    
    # 处理后分辨率分布（模拟：细分后r8, r9增加）
    res_after = res_before.copy()
    for log in refinement_log:
        # 模拟细分效果
        for cell in log['refined_cells']:
            res_after.append(log['target_res'])
    
    if refinement_log:
        ax1.hist(res_after, bins=range(7, 12), alpha=0.5, label='After Smoothing', 
                 color='lightcoral', edgecolor='black')
    
    ax1.set_xlabel('Resolution', fontsize=10)
    ax1.set_ylabel('Count', fontsize=10)
    ax1.set_title('(a) Leaf Resolution Distribution', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2：不连续检测结果
    ax2 = axes[0, 1]
    
    if discontinuities:
        r_diffs = [d['r_diff'] for d in discontinuities]
        ax2.hist(r_diffs, bins=range(0, 5), alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(x=1, color='red', linestyle='--', label='Δr_smooth=1')
        ax2.set_xlabel('Resolution Difference |r_j - r_k|', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title(f'(b) Discontinuity Detection ({len(discontinuities)} found)', fontsize=11)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No Discontinuities\nDetected', ha='center', va='center', 
                 fontsize=12, transform=ax2.transAxes)
        ax2.set_title('(b) Discontinuity Detection', fontsize=11)
    
    ax2.grid(True, alpha=0.3)
    
    # 子图3：编码策略分布
    ax3 = axes[1, 0]
    
    if encoded_targets:
        strategies = [v['strategy'] for v in encoded_targets.values()]
        strategy_counts = pd.Series(strategies).value_counts()
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = ax3.bar(range(len(strategy_counts)), strategy_counts.values, 
                        color=colors[:len(strategy_counts)], alpha=0.7)
        ax3.set_xticks(range(len(strategy_counts)))
        ax3.set_xticklabels(strategy_counts.index, rotation=15, ha='right', fontsize=8)
        ax3.set_ylabel('Count', fontsize=10)
        ax3.set_title(f'(c) Encoding Strategy Distribution (n={len(encoded_targets)})', fontsize=11)
        
        # 添加数值标签
        for bar, count in zip(bars, strategy_counts.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                     str(count), ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No Encoded Targets', ha='center', va='center', 
                 fontsize=12, transform=ax3.transAxes)
        ax3.set_title('(c) Encoding Strategy Distribution', fontsize=11)
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 子图4：RDM分数和处理统计
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = "Boundary Processing Statistics\n" + "="*35 + "\n\n"
    stats_text += f"Discontinuities Detected: {len(discontinuities)}\n"
    stats_text += f"Cells Refined: {len(refinement_log)}\n"
    stats_text += f"Targets Encoded: {len(encoded_targets)}\n\n"
    
    if discontinuities:
        rdm = sum(d['penalty'] for d in discontinuities) / max(len(leaf_cells), 1)
        stats_text += f"RDM Score: {rdm:.4f}\n"
        stats_text += f"Max Resolution Diff: {max(d['r_diff'] for d in discontinuities)}\n"
        stats_text += f"Avg Resolution Diff: {np.mean([d['r_diff'] for d in discontinuities]):.2f}\n\n"
    
    if encoded_targets:
        strategy_dist = pd.Series([v['strategy'] for v in encoded_targets.values()]).value_counts()
        stats_text += "Encoding Strategy Distribution:\n"
        for strategy, count in strategy_dist.items():
            stats_text += f"  - {strategy}: {count} ({count/len(encoded_targets)*100:.1f}%)\n"
    
    ax4.text(0.1, 0.9, stats_text, fontsize=9, family='monospace', 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  可视化已保存到 {output_path}")
    
    return output_path


def save_encoding_results(encoded_targets, output_prefix='boundary_encoding'):
    """
    保存编码结果
    
    Parameters:
    - encoded_targets: 编码结果字典
    - output_prefix: 输出文件前缀
    
    Returns:
    - output_files: 输出文件列表
    """
    output_files = []
    
    # 1. 保存编码结果JSON
    json_path = f'{output_prefix}.json'
    with open(json_path, 'w') as f:
        json.dump(encoded_targets, f, indent=2, default=str)
    output_files.append(json_path)
    print(f"  ✓ 编码结果已保存到 {json_path}")
    
    # 2. 保存编码结果CSV（摘要）
    csv_data = []
    for target_id, info in encoded_targets.items():
        csv_data.append({
            'target_id': target_id,
            'strategy': info['strategy'],
            'nca_cell': info.get('nca_cell', ''),
            'nca_resolution': info.get('nca_resolution', ''),
            'n_h3_cells': len(info['h3_cells']),
            'confidence': info['confidence'],
            'area': info.get('area', 0)
        })
    
    if csv_data:
        csv_df = pd.DataFrame(csv_data)
        csv_path = f'{output_prefix}_summary.csv'
        csv_df.to_csv(csv_path, index=False)
        output_files.append(csv_path)
        print(f"  ✓ 编码摘要已保存到 {csv_path}")
    
    # 3. 保存不连续检测结果
    if 'discontinuities' in globals():
        pass  # 由主函数处理
    
    return output_files


def main():
    """主函数：执行步骤4.1~4.5"""
    print("=" * 60)
    print("步骤4: 边界效应处理与编码")
    print("=" * 60)
    
    # 加载数据
    print("\n[4.0] 加载数据...")
    tree_structure = load_tree_structure('adaptive_tree.json')
    leaf_cells = load_leaf_cells('adaptive_tree_leaves.csv')
    boundary_targets, targets_gdf = load_boundary_targets('targets_classified.geojson')
    
    print(f"  ✓ 树结构节点数: {len(tree_structure)}")
    print(f"  ✓ 叶子节点数: {len(leaf_cells)}")
    print(f"  ✓ 边界目标数: {len(boundary_targets)}")
    
    # ============================================================
    # 步骤4.1：边界层级不连续检测与平滑
    # ============================================================
    print("\n[4.1] 边界层级不连续检测与平滑 (Algorithm B.2.2: H3-LSP)...")
    
    discontinuities, rdm_score = detect_resolution_discontinuity(
        tree_structure, 
        leaf_cells, 
        delta_r_smooth=1
    )
    
    print(f"  ✓ 检测到 {len(discontinuities)} 处分辨率不连续")
    print(f"  ✓ RDM分数: {rdm_score:.4f}")
    
    # 保存不连续检测结果
    if discontinuities:
        disc_df = pd.DataFrame(discontinuities)
        disc_df.to_csv('boundary_discontinuities.csv', index=False)
        print(f"  ✓ 不连续检测结果已保存到 boundary_discontinuities.csv")
    
    # 平滑不连续
    print("\n  执行不连续平滑...")
    smoothed_leaves, refinement_log = smooth_discontinuities(
        tree_structure, 
        discontinuities, 
        leaf_cells, 
        r_max=10
    )
    
    print(f"  ✓ 平滑后叶子节点数: {len(smoothed_leaves)}")
    print(f"  ✓ 细分记录数: {len(refinement_log)}")
    
    if refinement_log:
        ref_df = pd.DataFrame(refinement_log)
        ref_df.to_csv('boundary_refinement_log.csv', index=False)
        print(f"  ✓ 细分记录已保存到 boundary_refinement_log.csv")
    
    # ============================================================
    # 步骤4.2：逐步细分填充（已在4.1中执行）
    # ============================================================
    print("\n[4.2] 逐步细分填充 (Gradual Refinement)...")
    print(f"  ✓ 渐进式细分已完成（见4.1结果）")
    
    # ============================================================
    # 步骤4.3：H3 Compact操作
    # ============================================================
    print("\n[4.3] H3 Compact操作 (Algorithm A.4.5)...")
    
    # 对叶子节点执行compact
    compacted = compact_h3_cells(leaf_cells)
    print(f"  ✓ Compact前格网数: {len(leaf_cells)}")
    print(f"  ✓ Compact后格网数: {len(compacted)}")
    print(f"  ✓ 压缩率: {(1 - len(compacted)/len(leaf_cells))*100:.1f}%")
    
    # 额外演示：对同一父格网的子格网进行compact
    print("\n  执行Compact演示（同一父格网的7个子格网）...")
    # 获取一个分裂节点的7个子格网
    demo_parent = None
    for cell, info in list(tree_structure.items())[:500]:
        if info.get('split') and info['resolution'] == 7:
            children = info.get('children', [])
            if len(children) == 7:
                demo_parent = cell
                demo_children = children
                break
    
    if demo_parent:
        print(f"    父格网: {demo_parent}")
        print(f"    7个子格网: {demo_children[:3]}... (共{len(demo_children)}个)")
        compacted_demo = h3.compact_cells(demo_children)
        print(f"    Compact后: {compacted_demo}")
        
        # 执行uncompact演示（指定目标分辨率）
        target_res = h3.get_resolution(demo_children[0]) + 1
        uncompacted_demo = h3.uncompact_cells(compacted_demo, target_res)
        print(f"    Uncompact (r={target_res})后格网数: {len(uncompacted_demo)}")
    
    # 保存compact结果
    compact_data = {
        'original_count': len(leaf_cells),
        'compacted_count': len(compacted),
        'compression_ratio': 1 - len(compacted)/len(leaf_cells),
        'compacted_cells': list(compacted),
        'demo': {
            'parent': demo_parent,
            'original_children': demo_children if demo_parent else [],
            'compacted_result': [str(c) for c in compacted_demo] if demo_parent else []
        }
    }
    with open('h3_compact_result.json', 'w') as f:
        json.dump(compact_data, f, indent=2)
    print(f"  ✓ Compact结果已保存到 h3_compact_result.json")
    
    # ============================================================
    # 步骤4.4：跨界目标融合编码
    # ============================================================
    print("\n[4.4] 跨界目标融合编码 (Algorithm B.2.3: H3-CBFE)...")
    
    encoded_results = encode_boundary_targets(
        boundary_targets, 
        tree_structure, 
        targets_gdf
    )
    
    print(f"  ✓ 编码完成：{len(encoded_results)} 个目标已编码")
    
    # 统计编码策略分布
    if encoded_results:
        strategies = [v['strategy'] for v in encoded_results.values()]
        strategy_counts = pd.Series(strategies).value_counts()
        print(f"\n  编码策略分布:")
        for strategy, count in strategy_counts.items():
            print(f"    - {strategy}: {count} ({count/len(encoded_results)*100:.1f}%)")
    
    # 保存编码结果
    output_files = save_encoding_results(encoded_results, 'boundary_encoding')
    
    # ============================================================
    # 步骤4.5：可视化与汇总
    # ============================================================
    print("\n[4.5] 生成可视化与汇总...")
    
    # 可视化
    viz_path = visualize_boundary_process(
        discontinuities, 
        refinement_log, 
        encoded_results,
        tree_structure, 
        leaf_cells,
        output_path='h3_steps_4_result.png'
    )
    
    # 保存处理统计
    stats = {
        'step_4_1': {
            'discontinuities_detected': len(discontinuities),
            'rdm_score': rdm_score,
            'cells_smoothed': len(refinement_log)
        },
        'step_4_2': {
            'gradual_refinement_applied': len(refinement_log) > 0
        },
        'step_4_3': {
            'cells_before_compact': len(leaf_cells),
            'cells_after_compact': len(compacted),
            'compression_ratio': 1 - len(compacted)/len(leaf_cells)
        },
        'step_4_4': {
            'targets_encoded': len(encoded_results),
            'encoding_strategies': {k: int(v) for k, v in strategy_counts.items()} if encoded_results else {}
        }
    }
    
    with open('step4_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ 处理统计已保存到 step4_statistics.json")
    
    # 最终汇总
    print("\n" + "=" * 60)
    print("步骤4 完成汇总")
    print("=" * 60)
    print(f"  ✓ 不连续检测: {len(discontinuities)} 处")
    print(f"  ✓ 渐进式细分: {len(refinement_log)} 个格网")
    print(f"  ✓ H3 Compact: {len(leaf_cells)} -> {len(compacted)} 格网")
    print(f"  ✓ 目标编码: {len(encoded_results)} 个")
    print(f"  ✓ 可视化: {viz_path}")
    print("=" * 60)
    
    return {
        'discontinuities': discontinuities,
        'refinement_log': refinement_log,
        'compacted': compacted,
        'encoded_targets': encoded_results,
        'stats': stats
    }


if __name__ == '__main__':
    results = main()
