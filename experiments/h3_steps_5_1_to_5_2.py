"""
步骤5.1~5.2: H3索引表创建与数据插入
- 步骤5.1: 创建H3索引表 (SQL DDL)
- 步骤5.2: 数据库连接与数据插入

修正参考代码中的问题：
- h3 v4 API 修正: h3.h3_to_geo_boundary() -> h3.cell_to_boundary()
- h3 v4 API 修正: h3.h3_get_hexagon_area_km2() -> h3.get_hexagon_area_km2()
- h3 v4 API 修正: h3.h3_to_parent() -> h3.cell_to_parent()
- 处理大数据集的批量插入
- 生成完整的SQL文件供PostgreSQL 17执行
"""

import json
import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Polygon, Point
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 步骤5.1：生成H3索引表创建SQL
# ============================================================

def generate_create_table_sql():
    """
    生成创建h3_index_adaptive表的SQL语句
    适用于PostgreSQL 17 + PostGIS
    
    Returns:
    - sql: CREATE TABLE SQL语句
    """
    
    sql = """
-- 步骤5.1：创建H3索引表
-- 适用于 PostgreSQL 17 + PostGIS
-- 执行前请确保已安装PostGIS扩展

-- 创建PostGIS扩展（如未安装）
CREATE EXTENSION IF NOT EXISTS postgis;

-- 删除已存在的表
DROP TABLE IF EXISTS h3_index_adaptive CASCADE;

-- 创建H3索引表
CREATE TABLE h3_index_adaptive (
    -- H3原生索引
    h3_index BIGINT PRIMARY KEY,
    resolution INT,
    is_leaf BOOLEAN,
    
    -- 目标信息
    target_count INT DEFAULT 0,
    target_ids TEXT[],
    
    -- 编码信息
    encoding_strategy VARCHAR(20),
    extended_code BYTEA,
    
    -- 元数据
    dominant_class INT,
    max_confidence FLOAT DEFAULT 0,
    area_km2 FLOAT,
    
    -- 索引加速
    parent_h3 BIGINT,
    geom GEOMETRY(Polygon, 4326),
    
    -- 边界标志
    is_boundary BOOLEAN DEFAULT FALSE,
    cross_cells TEXT[],
    
    FOREIGN KEY (parent_h3) REFERENCES h3_index_adaptive(h3_index)
);

-- 创建空间和属性索引
CREATE INDEX idx_h3_geom ON h3_index_adaptive USING GIST(geom);
CREATE INDEX idx_h3_res ON h3_index_adaptive(resolution);
CREATE INDEX idx_h3_parent ON h3_index_adaptive(parent_h3);
CREATE INDEX idx_h3_class ON h3_index_adaptive(dominant_class);
CREATE INDEX idx_h3_leaf ON h3_index_adaptive(is_leaf);

-- 创建H3索引的哈希索引（PostgreSQL 17特性）
CREATE INDEX idx_h3_index_hash ON h3_index_adaptive USING HASH(h3_index);

COMMENT ON TABLE h3_index_adaptive IS 'H3自适应索引表 - DGGS自适应生成算法输出';
COMMENT ON COLUMN h3_index_adaptive.h3_index IS 'H3格网索引（十六进制转大整数）';
COMMENT ON COLUMN h3_index_adaptive.resolution IS 'H3分辨率（0-15）';
COMMENT ON COLUMN h3_index_adaptive.is_leaf IS '是否为叶子节点';
COMMENT ON COLUMN h3_index_adaptive.target_count IS '包含的目标数量';
COMMENT ON COLUMN h3_index_adaptive.target_ids IS '目标ID数组';
COMMENT ON COLUMN h3_index_adaptive.encoding_strategy IS '编码策略：H3-Ascend/H3-Primary-Secondary/H3-Multi-Code';
COMMENT ON COLUMN h3_index_adaptive.dominant_class IS '主导目标类别';
COMMENT ON COLUMN h3_index_adaptive.max_confidence IS '最大置信度';
COMMENT ON COLUMN h3_index_adaptive.area_km2 IS '格网面积（平方公里）';
COMMENT ON COLUMN h3_index_adaptive.parent_h3 IS '父格网索引';
COMMENT ON COLUMN h3_index_adaptive.geom IS '格网几何（PostGIS Polygon, SRID: 4326）';
COMMENT ON COLUMN h3_index_adaptive.is_boundary IS '是否边界格网';
COMMENT ON COLUMN h3_index_adaptive.cross_cells IS '跨界目标关联格网';
"""
    
    return sql


# ============================================================
# 步骤5.2：准备插入数据
# ============================================================

def load_data():
    """
    加载之前步骤生成的数据文件
    
    Returns:
    - tree_structure: 自适应树结构
    - leaf_cells: 叶子节点列表
    - target_relations: 目标-格网关系
    - boundary_encoding: 边界编码结果
    - discontinuities: 不连续检测结果
    """
    
    print("=" * 60)
    print("步骤5.2：加载数据")
    print("=" * 60)
    
    # 加载自适应树结构
    print("\n[5.2.1] 加载自适应树结构...")
    with open('adaptive_tree.json', 'r', encoding='utf-8') as f:
        tree_structure = json.load(f)
    print(f"  ✓ 加载了 {len(tree_structure)} 个H3格网节点")
    
    # 加载叶子节点
    print("\n[5.2.2] 加载叶子节点...")
    leaves_df = pd.read_csv('adaptive_tree_leaves.csv')
    leaf_cells = set(leaves_df['h3_cell'].values)
    print(f"  ✓ 加载了 {len(leaf_cells)} 个叶子节点")
    
    # 加载目标-格网关系
    print("\n[5.2.3] 加载目标-格网关系...")
    target_relations = pd.read_csv('target_cell_relations.csv')
    print(f"  ✓ 加载了 {len(target_relations)} 条目标-格网关系")
    
    # 加载边界编码结果
    print("\n[5.2.4] 加载边界编码结果...")
    with open('boundary_encoding.json', 'r', encoding='utf-8') as f:
        boundary_encoding = json.load(f)
    print(f"  ✓ 加载了 {len(boundary_encoding)} 个目标的编码结果")
    
    # 加载不连续检测结果（如果存在）
    discontinuities = []
    try:
        disc_df = pd.read_csv('boundary_discontinuities.csv')
        discontinuities = disc_df.to_dict('records')
        print(f"  ✓ 加载了 {len(discontinuities)} 条不连续记录")
    except FileNotFoundError:
        print("  ⚠ 未找到不连续检测文件，使用空列表")
        discontinuities = []
    
    return tree_structure, leaf_cells, target_relations, boundary_encoding, discontinuities


def h3_to_int(h3_cell_str):
    """
    将H3索引（十六进制字符串）转换为大整数
    
    Parameters:
    - h3_cell_str: H3格网索引字符串（如 "8741a54e2ffffff"）
    
    Returns:
    - int: 大整数
    """
    # 移除可能的 "0x" 前缀
    h3_cell_str = h3_cell_str.lower().replace('0x', '')
    return int(h3_cell_str, 16)


def get_h3_polygon_wkt(h3_cell_str):
    """
    获取H3格网的多边形WKT表示
    
    Parameters:
    - h3_cell_str: H3格网索引字符串
    
    Returns:
    - wkt: WKT多边形字符串
    """
    # h3 v4 API: cell_to_boundary returns [(lat, lon), ...]
    boundary = h3.cell_to_boundary(h3_cell_str)
    
    # 构造WKT多边形: POLYGON((lon lat, lon lat, ...))
    # 注意：WKT使用 (lon lat) 顺序，而H3返回 (lat, lon)
    coords = []
    for lat, lon in boundary:
        coords.append(f"{lon} {lat}")
    
    # 闭合多边形（重复第一个点）
    coords.append(coords[0])
    
    wkt = f"POLYGON(({', '.join(coords)}))"
    return wkt


def prepare_insert_data(tree_structure, leaf_cells, target_relations, boundary_encoding, discontinuities):
    """
    准备插入数据
    
    Parameters:
    - tree_structure: 自适应树结构
    - leaf_cells: 叶子节点集合
    - target_relations: 目标-格网关系DataFrame
    - boundary_encoding: 边界编码结果
    - discontinuities: 不连续检测列表
    
    Returns:
    - insert_data: 插入数据列表
    """
    
    print("\n" + "=" * 60)
    print("步骤5.2：准备插入数据")
    print("=" * 60)
    
    insert_data = []
    
    # 构建不连续检测的查找表
    discontinuity_cells = set()
    for d in discontinuities:
        discontinuity_cells.add(d.get('cell1', ''))
    
    # 构建目标类别到整数的映射
    all_classes = target_relations['target_class'].unique().tolist()
    class_to_id = {cls: idx for idx, cls in enumerate(all_classes)}
    print(f"  目标类别映射: {class_to_id}")
    
    # 遍历树结构中的每个格网
    total_cells = len(tree_structure)
    processed = 0
    
    for h_cell, node_info in tree_structure.items():
        processed += 1
        
        if processed % 500 == 0:
            print(f"  处理进度: {processed}/{total_cells}")
        
        # 获取分辨率
        resolution = node_info.get('resolution', 7)
        
        # 判断是否为叶子节点
        is_leaf = (h_cell in leaf_cells)
        
        # 获取该格网中的目标
        cell_targets = target_relations[target_relations['h3_cell'] == h_cell]
        target_count = len(cell_targets)
        target_ids = cell_targets['target_id'].tolist()
        
        # 获取主导类别和最大置信度
        if target_count > 0:
            # 使用最多的目标类别作为主导类别（返回类别ID）
            mode_classes = cell_targets['target_class'].mode()
            if len(mode_classes) > 0:
                dominant_class_name = mode_classes.iloc[0]
                dominant_class = class_to_id.get(dominant_class_name, -1)
            else:
                dominant_class = -1
            max_confidence = cell_targets['confidence'].max()
        else:
            dominant_class = -1  # 无目标
            max_confidence = 0.0
        
        # 获取编码策略（从boundary_encoding中查找）
        encoding_strategy = None
        for target_id, encoding in boundary_encoding.items():
            if h_cell in encoding.get('h3_cells', []):
                encoding_strategy = encoding.get('strategy', 'H3-Ascend')
                break
        if encoding_strategy is None:
            encoding_strategy = 'H3-Ascend' if resolution >= 7 else 'H3-Multi-Code'
        
        # 获取格网面积
        try:
            area_km2 = h3.get_hexagon_area_km2(resolution)
        except Exception:
            area_km2 = 0.0
        
        # 获取父格网
        parent_h3 = None
        if resolution > 0:
            try:
                parent_str = h3.cell_to_parent(h_cell, resolution - 1)
                parent_h3 = h3_to_int(parent_str)
            except Exception:
                parent_h3 = None
        
        # 获取几何WKT
        try:
            geom_wkt = get_h3_polygon_wkt(h_cell)
        except Exception as e:
            print(f"  ⚠ 无法获取 {h_cell} 的几何: {e}")
            geom_wkt = None
        
        # 判断是否边界格网
        is_boundary = h_cell in discontinuity_cells
        
        # 获取跨界目标关联格网
        cross_cells = []
        for d in discontinuities:
            if d.get('cell1') == h_cell:
                cross_cells.append(d.get('cell2', ''))
        
        # 添加到插入数据
        insert_data.append({
            'h3_index': h3_to_int(h_cell),
            'resolution': resolution,
            'is_leaf': is_leaf,
            'target_count': target_count,
            'target_ids': target_ids,
            'encoding_strategy': encoding_strategy,
            'extended_code': None,
            'dominant_class': dominant_class,
            'max_confidence': max_confidence,
            'area_km2': area_km2,
            'parent_h3': parent_h3,
            'geom_wkt': geom_wkt,
            'is_boundary': is_boundary,
            'cross_cells': cross_cells
        })
    
    print(f"\n  ✓ 准备了 {len(insert_data)} 条插入数据")
    
    return insert_data


def generate_insert_sql(insert_data):
    """
    生成INSERT SQL语句
    
    Parameters:
    - insert_data: 插入数据列表
    
    Returns:
    - sql: INSERT SQL语句
    """
    
    print("\n" + "=" * 60)
    print("生成INSERT SQL语句")
    print("=" * 60)
    
    sql_lines = []
    sql_lines.append("-- 步骤5.2：插入H3索引数据")
    sql_lines.append("")
    sql_lines.append("-- 开始事务")
    sql_lines.append("BEGIN;")
    sql_lines.append("")
    
    # 分批插入
    batch_size = 100
    total = len(insert_data)
    
    for i in range(0, total, batch_size):
        batch = insert_data[i:i+batch_size]
        
        batch_num = i // batch_size + 1
        start_idx = i + 1
        end_idx = min(i + batch_size, total)
        sql_lines.append("-- 批次 {}: 记录 {} 到 {}".format(batch_num, start_idx, end_idx))
        
        for row in batch:
            # 处理target_ids数组
            if row['target_ids']:
                target_ids_sql = "'{" + ",".join(map(str, row['target_ids'])) + "}'::TEXT[]"
            else:
                target_ids_sql = "NULL"
            
            # 处理cross_cells数组
            if row['cross_cells']:
                cross_cells_sql = "'{" + ",".join(['"' + str(c) + '"' for c in row['cross_cells']]) + "}'::TEXT[]"
            else:
                cross_cells_sql = "NULL"
            
            # 处理geom
            if row['geom_wkt']:
                geom_sql = "ST_GeomFromText('{}', 4326)".format(row['geom_wkt'].replace("'", "''"))
            else:
                geom_sql = "NULL"
            
            # 处理parent_h3
            parent_sql = str(row['parent_h3']) if row['parent_h3'] is not None else "NULL"
            
            # 处理extended_code
            extended_code_sql = "NULL"  # BYTEA类型，暂时为NULL
            
            insert_sql = "INSERT INTO h3_index_adaptive (h3_index, resolution, is_leaf, target_count, target_ids, encoding_strategy, extended_code, dominant_class, max_confidence, area_km2, parent_h3, geom, is_boundary, cross_cells) VALUES ({}, {}, {}, {}, {}, '{}', {}, {}, {}, {}, {}, {}, {}, {});".format(
                row['h3_index'],
                row['resolution'],
                'TRUE' if row['is_leaf'] else 'FALSE',
                row['target_count'],
                target_ids_sql,
                row['encoding_strategy'],
                extended_code_sql,
                row['dominant_class'],  # Integer value
                row['max_confidence'],
                row['area_km2'],
                parent_sql,
                geom_sql,
                'TRUE' if row['is_boundary'] else 'FALSE',
                cross_cells_sql
            )
            
            sql_lines.append(insert_sql)
        
        sql_lines.append("")
    
    sql_lines.append("-- 提交事务")
    sql_lines.append("COMMIT;")
    sql_lines.append("")
    sql_lines.append(f"-- 总计插入 {total} 条记录")
    
    sql = "\n".join(sql_lines)
    
    print(f"  ✓ 生成了 {total} 条INSERT语句")
    
    return sql


def save_sql_file(create_table_sql, insert_sql):
    """
    保存SQL文件
    
    Parameters:
    - create_table_sql: 创建表SQL
    - insert_sql: 插入数据SQL
    """
    
    print("\n" + "=" * 60)
    print("保存SQL文件")
    print("=" * 60)
    
    # 保存完整的SQL文件
    output_file = 'h3_index_adaptive_full.sql'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(create_table_sql)
        f.write("\n\n")
        f.write(insert_sql)
    
    print(f"  ✓ 保存完整SQL文件: {output_file}")
    
    # 仅保存创建表SQL
    output_file = 'h3_index_adaptive_create.sql'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(create_table_sql)
    
    print(f"  ✓ 保存创建表SQL: {output_file}")
    
    # 仅保存插入数据SQL
    output_file = 'h3_index_adaptive_insert.sql'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(insert_sql)
    
    print(f"  ✓ 保存插入数据SQL: {output_file}")


def try_database_insert(insert_data):
    """
    尝试连接数据库并插入数据
    
    Parameters:
    - insert_data: 插入数据列表
    """
    
    print("\n" + "=" * 60)
    print("尝试数据库连接与插入")
    print("=" * 60)
    
    try:
        import psycopg2
        from psycopg2.extras import execute_batch
        
        # 数据库连接配置（可根据实际情况修改）
        db_config = {
            'dbname': 'maritime_data',
            'user': 'postgres',
            'password': 'your_password',
            'host': 'localhost',
            'port': 5432
        }
        
        print("\n[5.2.5] 连接数据库...")
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        print("  ✓ 数据库连接成功")
        
        # 准备批量插入数据
        print("\n[5.2.6] 准备批量插入...")
        
        insert_sql = """
            INSERT INTO h3_index_adaptive 
                (h3_index, resolution, is_leaf, target_count, target_ids,
                 encoding_strategy, extended_code, dominant_class, max_confidence,
                 area_km2, parent_h3, geom, is_boundary, cross_cells)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326), %s, %s)
        """
        
        data_for_insert = []
        for row in insert_data:
            # 处理target_ids
            target_ids_array = row['target_ids'] if row['target_ids'] else None
            
            # 处理cross_cells
            cross_cells_array = row['cross_cells'] if row['cross_cells'] else None
            
            data_for_insert.append((
                row['h3_index'],
                row['resolution'],
                row['is_leaf'],
                row['target_count'],
                target_ids_array,
                row['encoding_strategy'],
                row['extended_code'],
                row['dominant_class'],
                row['max_confidence'],
                row['area_km2'],
                row['parent_h3'],
                row['geom_wkt'],
                row['is_boundary'],
                cross_cells_array
            ))
        
        print(f"  ✓ 准备了 {len(data_for_insert)} 条数据")
        
        # 执行批量插入
        print("\n[5.2.7] 执行批量插入...")
        execute_batch(cursor, insert_sql, data_for_insert, page_size=1000)
        conn.commit()
        
        print(f"  ✓ 成功插入 {len(data_for_insert)} 条记录")
        
        cursor.close()
        conn.close()
        
    except ImportError:
        print("\n  ⚠ psycopg2未安装，跳过数据库插入")
        print("  ⚠ 请使用生成的SQL文件手动执行")
    except Exception as e:
        print(f"\n  ⚠ 数据库插入失败: {e}")
        print("  ⚠ 请使用生成的SQL文件手动执行")


def generate_statistics_report(tree_structure, insert_data):
    """
    生成统计报告
    
    Parameters:
    - tree_structure: 自适应树结构
    - insert_data: 插入数据列表
    
    Returns:
    - stats: 统计信息字典
    """
    
    print("\n" + "=" * 60)
    print("生成统计报告")
    print("=" * 60)
    
    stats = {
        'total_cells': len(insert_data),
        'leaf_cells': sum(1 for d in insert_data if d['is_leaf']),
        'non_leaf_cells': sum(1 for d in insert_data if not d['is_leaf']),
        'resolution_distribution': {},
        'encoding_strategy_distribution': {},
        'total_targets_covered': sum(d['target_count'] for d in insert_data),
        'cells_with_targets': sum(1 for d in insert_data if d['target_count'] > 0),
        'boundary_cells': sum(1 for d in insert_data if d['is_boundary']),
        'avg_targets_per_cell': 0,
        'avg_area_km2': 0
    }
    
    # 分辨率分布
    for d in insert_data:
        res = d['resolution']
        stats['resolution_distribution'][res] = stats['resolution_distribution'].get(res, 0) + 1
    
    # 编码策略分布
    for d in insert_data:
        strategy = d['encoding_strategy']
        stats['encoding_strategy_distribution'][strategy] = stats['encoding_strategy_distribution'].get(strategy, 0) + 1
    
    # 平均每个格网的目标数
    if stats['total_cells'] > 0:
        stats['avg_targets_per_cell'] = stats['total_targets_covered'] / stats['total_cells']
    
    # 平均面积
    areas = [d['area_km2'] for d in insert_data if d['area_km2'] > 0]
    if areas:
        stats['avg_area_km2'] = sum(areas) / len(areas)
    
    # 输出统计报告
    print(f"\n总格网数: {stats['total_cells']}")
    print(f"  叶子节点: {stats['leaf_cells']}")
    print(f"  非叶子节点: {stats['non_leaf_cells']}")
    print(f"\n分辨率分布:")
    for res, count in sorted(stats['resolution_distribution'].items()):
        print(f"  Resolution {res}: {count} 个格网")
    print(f"\n编码策略分布:")
    for strategy, count in stats['encoding_strategy_distribution'].items():
        print(f"  {strategy}: {count} 个格网")
    print(f"\n目标覆盖:")
    print(f"  总目标数: {stats['total_targets_covered']}")
    print(f"  有目标的格网: {stats['cells_with_targets']}")
    print(f"  平均每个格网目标数: {stats['avg_targets_per_cell']:.2f}")
    print(f"\n边界信息:")
    print(f"  边界格网数: {stats['boundary_cells']}")
    print(f"\n平均面积: {stats['avg_area_km2']:.2f} km²")
    
    # 保存统计报告
    output_file = 'step5_statistics.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n  ✓ 统计报告已保存: {output_file}")
    
    return stats


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主函数：执行步骤5.1~5.2
    """
    
    print("=" * 60)
    print("步骤5：H3索引表创建与数据插入")
    print("=" * 60)
    
    # 步骤5.1：生成创建表SQL
    print("\n>>> 步骤5.1：生成H3索引表创建SQL...")
    create_table_sql = generate_create_table_sql()
    print("  ✓ 创建表SQL已生成")
    
    # 步骤5.2：加载数据
    tree_structure, leaf_cells, target_relations, boundary_encoding, discontinuities = load_data()
    
    # 准备插入数据
    insert_data = prepare_insert_data(tree_structure, leaf_cells, target_relations, boundary_encoding, discontinuities)
    
    # 生成INSERT SQL
    insert_sql = generate_insert_sql(insert_data)
    
    # 保存SQL文件
    save_sql_file(create_table_sql, insert_sql)
    
    # 尝试数据库插入
    try_database_insert(insert_data)
    
    # 生成统计报告
    stats = generate_statistics_report(tree_structure, insert_data)
    
    print("\n" + "=" * 60)
    print("步骤5 完成！")
    print("=" * 60)
    print("\n输出文件:")
    print("  1. h3_index_adaptive_create.sql - 创建表SQL")
    print("  2. h3_index_adaptive_insert.sql - 插入数据SQL")
    print("  3. h3_index_adaptive_full.sql - 完整SQL（创建+插入）")
    print("  4. step5_statistics.json - 统计报告")
    print("\n请在PostgreSQL 17中执行SQL文件以完成数据库导入。")


if __name__ == '__main__':
    main()
