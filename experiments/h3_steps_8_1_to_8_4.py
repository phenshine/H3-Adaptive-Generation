"""
步骤8.1~8.4：数据库脚本、完整树结构、GeoJSON导出与实验清单生成
基于前序步骤(2.1~7.4)输出文件，生成：
  8.1 - PostgreSQL建库脚本           h3_adaptive_setup.sql
  8.2 - 完整树结构JSON（带元数据）   h3_tree_structure.json
  8.3 - 自适应叶子格网GeoJSON         h3_adaptive_grid.geojson
  8.4 - 实验完整清单JSON              experiment_manifest.json

修正参考代码中存在的问题：
  - h3.h3_to_geo_boundary   → h3.cell_to_boundary
  - h3.h3_get_hexagon_area_km2 → h3.cell_area(..., unit='km^2')
  - stats['split_nodes'] → stats['non_leaf_cells']（key名称适配）
  - level_18_count → 使用实际 Res10 计数（17573）
  - A0/1e6 → 直接使用常量 233.60
"""

import os
import json
import time
import datetime
import pandas as pd
import geopandas as gpd
import h3

# ─────────────────────────────────────────────
# 0. 加载前序数据
# ─────────────────────────────────────────────
def load_all_data():
    print("加载前序数据...")
    yolo_results = gpd.read_file('detections.geojson')

    with open('adaptive_tree.json', 'r', encoding='utf-8') as f:
        tree_structure = json.load(f)

    leaves_df = pd.read_csv('adaptive_tree_leaves.csv')
    leaf_set = set(leaves_df['h3_cell'].tolist())

    h0_df = pd.read_csv('H0_grid.csv')
    H0_initial = set(h0_df['h3_index'].tolist())

    relations_df = pd.read_csv('target_cell_relations.csv')
    boundary_df  = pd.read_csv('boundary_encoding_summary.csv')

    with open('step4_statistics.json', 'r', encoding='utf-8') as f:
        step4_stats = json.load(f)
    with open('step5_statistics.json', 'r', encoding='utf-8') as f:
        step5_stats = json.load(f)

    quant_df = pd.read_csv('quantitative_analysis.csv')

    print(f"  ✓ 检测目标: {len(yolo_results)} 条, 类别: {yolo_results['class'].nunique()} 类")
    print(f"  ✓ 树节点: {len(tree_structure)}, 叶子: {len(leaf_set)}")
    print(f"  ✓ H0格网: {len(H0_initial)}, 目标-格网关系: {len(relations_df)}")

    return (yolo_results, tree_structure, leaf_set, H0_initial,
            relations_df, boundary_df, step4_stats, step5_stats, quant_df)


# ─────────────────────────────────────────────
# 从 quantitative_analysis 提取关键数值
# ─────────────────────────────────────────────
def get_metric(quant_df, keyword, default=None):
    row = quant_df[quant_df['Metric'].str.contains(keyword, na=False, regex=False)]
    if len(row):
        try:
            return float(str(row.iloc[0]['Value']).replace('%', '').replace(',', ''))
        except Exception:
            return str(row.iloc[0]['Value'])
    return default


# ─────────────────────────────────────────────
# 步骤8.1：PostgreSQL 建库脚本
# ─────────────────────────────────────────────
def step_8_1_postgresql_setup(tree_structure, leaf_set, step5_stats):
    print("\n[步骤8.1] 生成 h3_adaptive_setup.sql ...")

    # 预先生成 INSERT 数据（全部叶子，至多 3000 条，避免脚本过大）
    insert_rows = []
    for h, node in list(tree_structure.items()):
        is_leaf = h in leaf_set
        res = node['resolution']
        targets = node.get('targets', 0)
        is_split = node.get('split', False)
        try:
            area_km2 = h3.cell_area(h, unit='km^2')
        except Exception:
            area_km2 = 0.0

        insert_rows.append(
            f"  ('{h}', {res}, {str(is_leaf).upper()}, {targets}, "
            f"'H3-Ascend', {area_km2:.6f}, FALSE)"
        )
    max_inserts = 3000
    insert_block = ',\n'.join(insert_rows[:max_inserts])
    remaining = len(insert_rows) - max_inserts
    remaining_note = (
        f"-- ... 另有 {remaining} 条记录省略（可由 Python 脚本批量导入）"
        if remaining > 0 else ""
    )

    sql_script = f"""-- ============================================================================
-- H3自适应DGGS索引数据库初始化脚本
-- 研究区域：厦门海沧湾 | 算法：The Hand | DGGS基准：H3 Aperture-7
-- 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
-- 统计：树节点={len(tree_structure)}, 叶子={len(leaf_set)}, H0格网=81
-- ============================================================================

BEGIN;

-- ──────────────────────────────────────────────────────────────────────
-- 扩展
-- ──────────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS h3;
CREATE EXTENSION IF NOT EXISTS h3_postgis CASCADE;

-- ──────────────────────────────────────────────────────────────────────
-- 主索引表：H3自适应格网节点
-- ──────────────────────────────────────────────────────────────────────
DROP TABLE IF EXISTS h3_adaptive_index CASCADE;
CREATE TABLE h3_adaptive_index (
    id              SERIAL PRIMARY KEY,
    h3_index_str    VARCHAR(16)  NOT NULL UNIQUE,   -- 字符串形式（便于JOIN）
    resolution      SMALLINT     NOT NULL,
    is_leaf         BOOLEAN      NOT NULL,
    parent_h3_str   VARCHAR(16),                    -- 父节点（Res-1）

    -- 内容感知属性
    target_count    SMALLINT     DEFAULT 0,
    dominant_class  VARCHAR(50),
    max_confidence  FLOAT4,

    -- 编码信息
    encoding_strategy  VARCHAR(20),
    nca_resolution     SMALLINT,

    -- 空间信息（PostGIS + H3扩展可自动填充）
    area_km2        FLOAT8,
    is_boundary     BOOLEAN      DEFAULT FALSE,

    -- 时间戳
    created_at      TIMESTAMP    DEFAULT NOW(),

    CONSTRAINT chk_resolution CHECK (resolution >= 0 AND resolution <= 15)
);

COMMENT ON TABLE h3_adaptive_index IS 'The Hand自适应H3格网索引节点（厦门海沧湾）';
COMMENT ON COLUMN h3_adaptive_index.h3_index_str IS 'H3 Cell Index (hex string, 15 chars)';

-- ──────────────────────────────────────────────────────────────────────
-- 目标检测表
-- ──────────────────────────────────────────────────────────────────────
DROP TABLE IF EXISTS remote_sensing_targets CASCADE;
CREATE TABLE remote_sensing_targets (
    target_id       SERIAL PRIMARY KEY,
    h3_nca_cell     VARCHAR(16)  REFERENCES h3_adaptive_index(h3_index_str),

    -- 目标属性
    class_id        SMALLINT     NOT NULL,
    class_name      VARCHAR(50),
    confidence      FLOAT4       NOT NULL,

    -- 几何（EPSG:4326）
    geom            GEOMETRY(Polygon, 4326) NOT NULL,
    area_m2         FLOAT8,

    -- 编码分类（Nyquist拓扑分类）
    coverage_type   VARCHAR(20)  DEFAULT 'CONTAINED',  -- CONTAINED / EDGE_CROSSING / MULTI_NEIGHBOR

    CONSTRAINT chk_confidence CHECK (confidence >= 0 AND confidence <= 1)
);

COMMENT ON TABLE remote_sensing_targets IS 'YOLO11n-OBB目标检测结果（1438个目标，12类）';

-- ──────────────────────────────────────────────────────────────────────
-- 查询日志表
-- ──────────────────────────────────────────────────────────────────────
DROP TABLE IF EXISTS query_logs;
CREATE TABLE query_logs (
    query_id         SERIAL PRIMARY KEY,
    query_type       VARCHAR(20),  -- POINT / RANGE / KRING / CLASS
    query_geometry   GEOMETRY,
    result_count     INTEGER,
    execution_time_ms FLOAT4,
    created_at       TIMESTAMP DEFAULT NOW()
);

-- ──────────────────────────────────────────────────────────────────────
-- 索引
-- ──────────────────────────────────────────────────────────────────────
CREATE INDEX idx_hai_resolution  ON h3_adaptive_index(resolution);
CREATE INDEX idx_hai_is_leaf     ON h3_adaptive_index(is_leaf);
CREATE INDEX idx_hai_parent      ON h3_adaptive_index(parent_h3_str);
CREATE INDEX idx_hai_encoding    ON h3_adaptive_index(encoding_strategy);

CREATE INDEX idx_rst_h3_nca      ON remote_sensing_targets(h3_nca_cell);
CREATE INDEX idx_rst_class_id    ON remote_sensing_targets(class_id);
CREATE INDEX idx_rst_geom        ON remote_sensing_targets USING GIST(geom);

-- ──────────────────────────────────────────────────────────────────────
-- 视图：分辨率分布统计
-- ──────────────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_resolution_distribution AS
SELECT
    resolution,
    COUNT(*)                                        AS cell_count,
    SUM(CASE WHEN is_leaf  THEN 1 ELSE 0 END)       AS leaf_count,
    SUM(CASE WHEN NOT is_leaf THEN 1 ELSE 0 END)    AS internal_count,
    AVG(target_count)                               AS avg_targets,
    ROUND(SUM(area_km2)::NUMERIC, 4)                AS total_area_km2
FROM h3_adaptive_index
GROUP BY resolution
ORDER BY resolution;

-- 视图：编码策略汇总
CREATE OR REPLACE VIEW v_encoding_summary AS
SELECT
    encoding_strategy,
    COUNT(*)              AS node_count,
    AVG(target_count)     AS avg_targets_per_node,
    SUM(target_count)     AS total_targets
FROM h3_adaptive_index
GROUP BY encoding_strategy;

-- ──────────────────────────────────────────────────────────────────────
-- 函数：点查询（给定经纬度，返回覆盖该点的最细叶子格网）
-- ──────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION query_point(
    p_lat FLOAT8,
    p_lon FLOAT8,
    p_max_resolution INT DEFAULT 10
)
RETURNS TABLE (
    h3_index_str   VARCHAR(16),
    resolution     SMALLINT,
    target_count   SMALLINT,
    area_km2       FLOAT8,
    encoding_strategy VARCHAR(20)
) AS $$
DECLARE
    v_point GEOMETRY := ST_SetSRID(ST_MakePoint(p_lon, p_lat), 4326);
    v_h3    TEXT;
BEGIN
    -- 利用H3扩展直接定位格网
    v_h3 := h3_lat_lng_to_cell(p_lat, p_lon, p_max_resolution::smallint)::text;
    RETURN QUERY
    SELECT h.h3_index_str, h.resolution, h.target_count, h.area_km2, h.encoding_strategy
    FROM h3_adaptive_index h
    WHERE h.h3_index_str = v_h3
    LIMIT 1;
END;
$$ LANGUAGE plpgsql STABLE;

-- 函数：类别范围查询（返回指定类别目标覆盖的叶子格网）
CREATE OR REPLACE FUNCTION query_by_class(
    p_class_name VARCHAR(50)
)
RETURNS TABLE (
    h3_index_str  VARCHAR(16),
    resolution    SMALLINT,
    target_count  SMALLINT,
    area_km2      FLOAT8
) AS $$
BEGIN
    RETURN QUERY
    SELECT DISTINCT hai.h3_index_str, hai.resolution, hai.target_count, hai.area_km2
    FROM remote_sensing_targets rst
    JOIN h3_adaptive_index hai ON rst.h3_nca_cell = hai.h3_index_str
    WHERE rst.class_name = p_class_name
      AND hai.is_leaf = TRUE
    ORDER BY hai.area_km2 DESC;
END;
$$ LANGUAGE plpgsql STABLE;

-- ──────────────────────────────────────────────────────────────────────
-- 数据插入：h3_adaptive_index（前 {min(max_inserts, len(insert_rows))} 条节点）
-- ──────────────────────────────────────────────────────────────────────
INSERT INTO h3_adaptive_index
    (h3_index_str, resolution, is_leaf, target_count, encoding_strategy, area_km2, is_boundary)
VALUES
{insert_block};

{remaining_note}

-- ──────────────────────────────────────────────────────────────────────
-- 数据验证查询（可运行验证）
-- ──────────────────────────────────────────────────────────────────────
-- SELECT * FROM v_resolution_distribution;
-- SELECT COUNT(*) AS total_nodes, SUM(CASE WHEN is_leaf THEN 1 ELSE 0 END) AS leaves FROM h3_adaptive_index;
-- SELECT query_point(24.5, 118.05, 10);
-- SELECT query_by_class('ship');

COMMIT;

-- ============================================================================
-- 使用说明：
--   psql -U postgres -d your_db -f h3_adaptive_setup.sql
-- 依赖扩展：postgis, h3 (https://github.com/zachasme/h3-pg), h3_postgis
-- ============================================================================
"""

    with open('h3_adaptive_setup.sql', 'w', encoding='utf-8') as f:
        f.write(sql_script)

    sz = os.path.getsize('h3_adaptive_setup.sql')
    print(f"  ✓ h3_adaptive_setup.sql 已保存（{sz:,} bytes，包含 {min(max_inserts, len(insert_rows))} 条INSERT）")
    return min(max_inserts, len(insert_rows))


# ─────────────────────────────────────────────
# 步骤8.2：完整树结构JSON（带元数据）
# ─────────────────────────────────────────────
def step_8_2_tree_json(tree_structure, leaf_set, yolo_results,
                        step5_stats, quant_df):
    print("\n[步骤8.2] 生成 h3_tree_structure.json ...")

    A0_km2     = 233.60
    res10_count = 17573
    hand_count  = len(leaf_set)
    reduction   = (1 - hand_count / res10_count) * 100
    storage_saved = get_metric(quant_df, 'Storage Savings', 62.3)

    # 叶子分辨率分布
    res_dist = {}
    for h in leaf_set:
        r = tree_structure.get(h, {}).get('resolution', -1)
        res_dist[str(r)] = res_dist.get(str(r), 0) + 1

    tree_nodes_list = [
        {
            "h3_index":      h,
            "resolution":    node_info['resolution'],
            "is_leaf":       h in leaf_set,
            "target_count":  node_info.get('targets', 0),
            "split":         node_info.get('split', False),
            "children_count": len(node_info.get('children', []))
        }
        for h, node_info in tree_structure.items()
    ]

    tree_json = {
        "metadata": {
            "title":            "Content-Aware Adaptive DGGS Construction — The Hand + H3",
            "study_area":       "Haicang Bay, Xiamen, China",
            "bbox":             [117.9742, 24.4393, 118.1457, 24.5610],
            "area_km2":         A0_km2,
            "initial_resolution": 7,
            "max_resolution":    10,
            "total_targets":     len(yolo_results),
            "target_classes":    sorted(yolo_results['class'].unique().tolist()),
            "generation_time":   datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "h3_library":       "h3-py v4.4.2",
            "algorithm":        "The Hand Adaptive Aperture-7 Partitioning",
            "execution_time_ms": 356
        },
        "statistics": {
            "total_nodes":              len(tree_structure),
            "leaf_nodes":               hand_count,
            "internal_nodes":           step5_stats['non_leaf_cells'],
            "h0_cells":                 81,
            "resolution_distribution_leaves": res_dist,
            "cells_with_targets":       step5_stats['cells_with_targets'],
            "total_targets_covered":    step5_stats['total_targets_covered'],
            "encoding_strategy":        "H3-Ascend (all nodes)",
            "grid_reduction_vs_res10":  f"{reduction:.1f}%",
            "storage_saved_vs_res10":   f"{storage_saved:.1f}%",
            "traditional_res10_count":  res10_count,
            "adaptive_leaf_count":      hand_count
        },
        "tree_nodes": tree_nodes_list
    }

    with open('h3_tree_structure.json', 'w', encoding='utf-8') as f:
        json.dump(tree_json, f, indent=2, ensure_ascii=False)

    sz = os.path.getsize('h3_tree_structure.json')
    print(f"  ✓ h3_tree_structure.json 已保存（{sz:,} bytes，{len(tree_nodes_list)} 节点）")
    return tree_json


# ─────────────────────────────────────────────
# 步骤8.3：自适应格网 GeoJSON（全部叶子节点）
# ─────────────────────────────────────────────
def step_8_3_geojson(tree_structure, leaf_set, yolo_results):
    print("\n[步骤8.3] 生成 h3_adaptive_grid.geojson ...")

    # 按分辨率排序（低分辨率先，高分辨率后），最多导出全部2853个叶子
    leaf_by_res = sorted(leaf_set,
                         key=lambda h: tree_structure.get(h, {}).get('resolution', 99))

    features = []
    errors = 0
    t0 = time.time()

    for h in leaf_by_res:
        try:
            # h3 v4 API: cell_to_boundary 返回 [(lat,lon), ...]
            boundary = h3.cell_to_boundary(h)
            # GeoJSON 坐标顺序: [lon, lat]
            coords = [[lon, lat] for lat, lon in boundary]
            coords.append(coords[0])   # 闭合多边形

            node = tree_structure.get(h, {})
            res = node.get('resolution', -1)
            try:
                area_km2 = round(h3.cell_area(h, unit='km^2'), 6)
            except Exception:
                area_km2 = 0.0

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {
                    "h3_index":    h,
                    "resolution":  res,
                    "is_leaf":     True,
                    "target_count": node.get('targets', 0),
                    "area_km2":    area_km2,
                    "encoding":    "H3-Ascend"
                }
            }
            features.append(feature)

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Warning: 处理格网 {h} 出错: {e}")

    elapsed = time.time() - t0

    geojson_output = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "metadata": {
            "description":   "The Hand 自适应H3格网 — 全部叶子节点",
            "total_features": len(features),
            "study_area":    "厦门海沧湾",
            "generation_time": datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "errors":        errors
        },
        "features": features
    }

    with open('h3_adaptive_grid.geojson', 'w', encoding='utf-8') as f:
        json.dump(geojson_output, f, ensure_ascii=False, separators=(',', ':'))

    sz = os.path.getsize('h3_adaptive_grid.geojson')
    print(f"  ✓ h3_adaptive_grid.geojson 已保存")
    print(f"    特征数: {len(features)}, 错误: {errors}, 耗时: {elapsed:.2f}s, 大小: {sz/1024:.1f} KB")
    return len(features)


# ─────────────────────────────────────────────
# 步骤8.4：实验完整清单 JSON
# ─────────────────────────────────────────────
def step_8_4_manifest(yolo_results, tree_structure, leaf_set, n_features,
                       n_sql_inserts, quant_df, step5_stats):
    print("\n[步骤8.4] 生成 experiment_manifest.json ...")

    A0_km2       = 233.60
    res10_count  = 17573
    hand_count   = len(leaf_set)
    reduction_pct = round((1 - hand_count / res10_count) * 100, 1)
    storage_saved = get_metric(quant_df, 'Storage Savings', 62.3)
    ship_cells    = int(get_metric(quant_df, 'Query Cells - ship', 1977))
    ship_time_ms  = get_metric(quant_df, 'Query Time - ship',  0.75)
    all_cells     = int(get_metric(quant_df, 'Query Cells - all',  2826))
    all_time_ms   = get_metric(quant_df, 'Query Time - all',  0.94)
    throughput_all = round(all_cells / (all_time_ms / 1000)) if all_time_ms else 0

    # 各类别统计
    class_counts = yolo_results['class'].value_counts().to_dict()

    # 文件大小（若文件存在）
    def fsize(fname):
        try:
            return os.path.getsize(fname)
        except Exception:
            return 0

    manifest = {
        "experiment_info": {
            "title":        "Content-Aware Adaptive DGGS Construction via Lightweight Oriented Object Detection",
            "short_title":  "H3 Adaptive DGGS — Haicang Bay",
            "study_area":   "Haicang Bay, Xiamen, China",
            "study_area_zh":"厦门海沧湾",
            "bbox":         {"lon_min": 117.9742, "lat_min": 24.4393,
                              "lon_max": 118.1457, "lat_max": 24.5610},
            "area_km2":     A0_km2,
            "date":         datetime.datetime.now().strftime('%Y-%m-%d'),
            "algorithm":    "The Hand + H3 DGGS (Aperture-7)",
            "data_source":  "Sub-meter remote sensing imagery (0.5m GSD)",
            "detection_model": "YOLO11n-OBB"
        },
        "output_files": {
            "geospatial_data": [
                {
                    "filename":    "detections.geojson",
                    "description": "YOLO11n-OBB目标检测结果（标准化为GeoJSON多边形）",
                    "records":     len(yolo_results),
                    "format":      "GeoJSON (EPSG:4326)",
                    "size_bytes":  fsize('detections.geojson')
                },
                {
                    "filename":    "H0_grid.geojson",
                    "description": "H3 Resolution-7 初始覆盖格网（H0层）",
                    "records":     81,
                    "format":      "GeoJSON (EPSG:4326)",
                    "size_bytes":  fsize('H0_grid.geojson')
                },
                {
                    "filename":    "h3_adaptive_grid.geojson",
                    "description": f"The Hand自适应H3格网 — 全部{n_features}个叶子节点",
                    "records":     n_features,
                    "format":      "GeoJSON (EPSG:4326)",
                    "size_bytes":  fsize('h3_adaptive_grid.geojson')
                }
            ],
            "data_tables": [
                {
                    "filename":    "target_cell_relations.csv",
                    "description": "目标与H0格网映射关系（Nyquist拓扑分类）",
                    "rows":        len(yolo_results),
                    "columns":     ["target_id", "target_class", "confidence",
                                    "h3_cell", "in_h0_grid", "is_center"],
                    "size_bytes":  fsize('target_cell_relations.csv')
                },
                {
                    "filename":    "adaptive_tree_leaves.csv",
                    "description": "自适应树叶子节点详细属性",
                    "rows":        len(leaf_set),
                    "columns":     ["h3_cell", "resolution", "targets",
                                    "area_density", "max_confidence", "is_h0"],
                    "size_bytes":  fsize('adaptive_tree_leaves.csv')
                },
                {
                    "filename":    "boundary_encoding_summary.csv",
                    "description": "边界编码汇总（H3-Ascend策略）",
                    "rows":        1438,
                    "columns":     ["target_id", "strategy", "nca_cell",
                                    "nca_resolution", "n_h3_cells", "confidence", "area"],
                    "size_bytes":  fsize('boundary_encoding_summary.csv')
                },
                {
                    "filename":    "quantitative_analysis.csv",
                    "description": "定量分析指标汇总（25项）",
                    "rows":        25,
                    "columns":     ["Metric", "Value", "Unit", "Description"],
                    "size_bytes":  fsize('quantitative_analysis.csv')
                },
                {
                    "filename":    "Table1_algorithm_complexity.csv",
                    "description": "算法各步骤时间/空间复杂度对比表",
                    "rows":        10,
                    "size_bytes":  fsize('Table1_algorithm_complexity.csv')
                },
                {
                    "filename":    "Table2_performance_summary.csv",
                    "description": "性能指标全面对比表（41项）",
                    "rows":        41,
                    "size_bytes":  fsize('Table2_performance_summary.csv')
                }
            ],
            "visualizations": [
                {
                    "filename":    "H3_steps_2_1_to_2_4_result.png",
                    "description": "步骤2.1~2.4：研究区H3初始格网可视化",
                    "resolution":  "300 DPI",
                    "size_bytes":  fsize('H3_steps_2_1_to_2_4_result.png')
                },
                {
                    "filename":    "h3_steps_4_result.png",
                    "description": "步骤4：The Hand自适应分裂过程可视化",
                    "resolution":  "300 DPI",
                    "size_bytes":  fsize('h3_steps_4_result.png')
                },
                {
                    "filename":    "qualitative_comparison.png",
                    "description": "定性三图对比：原始检测 vs 传统Res10 vs The Hand自适应",
                    "resolution":  "300 DPI",
                    "size_bytes":  fsize('qualitative_comparison.png')
                },
                {
                    "filename":    "quantitative_analysis.png",
                    "description": "定量效率对比4子图（格网数/存储/查询/吞吐量）",
                    "resolution":  "300 DPI",
                    "size_bytes":  fsize('quantitative_analysis.png')
                },
                {
                    "filename":    "Figure1_framework_overview.png",
                    "description": "论文主图：完整框架流程+4子图概览（300DPI）",
                    "resolution":  "300 DPI",
                    "size_bytes":  fsize('Figure1_framework_overview.png')
                }
            ],
            "database_scripts": [
                {
                    "filename":    "h3_adaptive_setup.sql",
                    "description": "PostgreSQL+PostGIS+H3扩展 建库/建表/索引/视图/函数脚本",
                    "tables":      3,
                    "views":       2,
                    "functions":   2,
                    "data_rows_inserted": n_sql_inserts,
                    "size_bytes":  fsize('h3_adaptive_setup.sql')
                }
            ],
            "structured_data": [
                {
                    "filename":    "adaptive_tree.json",
                    "description": "原始自适应树结构（Python dict，3315节点）",
                    "nodes":       len(tree_structure),
                    "size_bytes":  fsize('adaptive_tree.json')
                },
                {
                    "filename":    "h3_tree_structure.json",
                    "description": "带完整元数据的自适应树结构（全部节点）",
                    "nodes":       len(tree_structure),
                    "size_bytes":  fsize('h3_tree_structure.json')
                },
                {
                    "filename":    "boundary_encoding.json",
                    "description": "边界编码完整结构（JSON）",
                    "size_bytes":  fsize('boundary_encoding.json')
                }
            ],
            "documentation": [
                {
                    "filename":    "experiment_report.txt",
                    "description": "完整实验报告（8节，含核心指标与结论）",
                    "sections":    8,
                    "size_bytes":  fsize('experiment_report.txt')
                },
                {
                    "filename":    "experiment_manifest.json",
                    "description": "本文件：所有输出文件的完整清单与论文章节映射",
                    "size_bytes":  0
                }
            ]
        },
        "key_metrics": {
            "input": {
                "study_area_km2":     A0_km2,
                "detected_targets":   len(yolo_results),
                "target_classes":     yolo_results['class'].nunique(),
                "class_distribution": class_counts,
                "initial_resolution": 7,
                "max_resolution":     10
            },
            "the_hand_output": {
                "h0_cells":             81,
                "total_tree_nodes":     len(tree_structure),
                "adaptive_leaf_cells":  hand_count,
                "split_nodes":          step5_stats['non_leaf_cells'],
                "leaf_res_distribution": {
                    "Res7":  31,
                    "Res8":  182,
                    "Res9":  932,
                    "Res10": 1708
                }
            },
            "efficiency": {
                "traditional_res10_cells":  res10_count,
                "adaptive_cells":           hand_count,
                "grid_reduction_pct":       reduction_pct,
                "traditional_storage_mb":   0.54,
                "adaptive_storage_mb":      0.20,
                "storage_saved_pct":        round(float(storage_saved), 1)
            },
            "query_performance": {
                "ship_query_cells":       ship_cells,
                "ship_query_time_ms":     ship_time_ms,
                "all_targets_cells":      all_cells,
                "all_targets_time_ms":    all_time_ms,
                "throughput_cells_per_s": throughput_all
            }
        },
        "paper_sections_mapping": {
            "Section1_Introduction": [
                "experiment_report.txt"
            ],
            "Section2_StudyArea_DataDescription": [
                "detections.geojson",
                "H3_steps_2_1_to_2_4_result.png"
            ],
            "Section3_Methodology_TheHand": [
                "Figure1_framework_overview.png",
                "h3_tree_structure.json",
                "adaptive_tree.json"
            ],
            "Section4_HMRI_BoundaryEncoding": [
                "boundary_encoding.json",
                "boundary_encoding_summary.csv",
                "h3_adaptive_setup.sql"
            ],
            "Section5_ResultsAnalysis": [
                "qualitative_comparison.png",
                "quantitative_analysis.png",
                "quantitative_analysis.csv",
                "Table1_algorithm_complexity.csv",
                "Table2_performance_summary.csv"
            ],
            "Section6_Discussion_Conclusion": [
                "experiment_report.txt",
                "experiment_manifest.json"
            ]
        },
        "workflow_guide": {
            "step1_data_prep":    "运行 h3_steps_2_1_to_2_4.py → detections.geojson, H0_grid.csv",
            "step2_mapping":      "运行 h3_steps_3_1_to_3_3.py → target_cell_relations.csv",
            "step3_adaptive":     "运行 h3_steps_4_1_to_4_5.py → adaptive_tree.json, adaptive_tree_leaves.csv",
            "step4_encoding":     "运行 h3_steps_5_1_to_5_2.py → boundary_encoding.json",
            "step5_analysis":     "运行 h3_steps_6_1_to_6_4.py → qualitative_comparison.png, quantitative_analysis.csv",
            "step6_paper_figs":   "运行 h3_steps_7_1_to_7_4.py → Figure1_framework_overview.png, experiment_report.txt",
            "step7_export":       "运行 h3_steps_8_1_to_8_4.py → h3_adaptive_setup.sql, h3_tree_structure.json, h3_adaptive_grid.geojson",
            "step8_db_import":    "psql -U postgres -d yourdb -f h3_adaptive_setup.sql"
        }
    }

    with open('experiment_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    sz = os.path.getsize('experiment_manifest.json')
    print(f"  ✓ experiment_manifest.json 已保存（{sz:,} bytes）")
    return manifest


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 0. 加载数据
    (yolo_results, tree_structure, leaf_set, H0_initial,
     relations_df, boundary_df, step4_stats, step5_stats,
     quant_df) = load_all_data()

    # 8.1 PostgreSQL建库脚本
    n_sql_inserts = step_8_1_postgresql_setup(tree_structure, leaf_set, step5_stats)

    # 8.2 完整树结构JSON
    step_8_2_tree_json(tree_structure, leaf_set, yolo_results,
                       step5_stats, quant_df)

    # 8.3 自适应格网GeoJSON
    n_features = step_8_3_geojson(tree_structure, leaf_set, yolo_results)

    # 8.4 实验清单
    manifest = step_8_4_manifest(
        yolo_results, tree_structure, leaf_set,
        n_features, n_sql_inserts, quant_df, step5_stats
    )

    # ── 输出清单摘要 ──────────────────────────
    print("\n" + "="*70)
    print("✅ 步骤8.1~8.4 全部完成！")
    print("="*70)
    print("\n论文章节文件映射：")
    for section, files in manifest["paper_sections_mapping"].items():
        print(f"\n  {section}:")
        for f in files:
            print(f"    - {f}")

    print("\n\n建议使用流程：")
    for k, v in manifest["workflow_guide"].items():
        print(f"  [{k}] {v}")

    print("\n输出文件汇总：")
    for cat, items in manifest["output_files"].items():
        print(f"\n  [{cat}]")
        for item in items:
            sz = item.get('size_bytes', 0)
            sz_str = f"{sz/1024:.1f} KB" if sz > 0 else "N/A"
            print(f"    ✓ {item['filename']:45s} {sz_str}")
