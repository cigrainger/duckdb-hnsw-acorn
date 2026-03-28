-- full_benchmark.sql
-- Comprehensive benchmark: filtered search, RaBitQ, metadata joins, grouped search
--
-- Usage: ./build/release/duckdb -no-init < benchmarks/full_benchmark.sql
--
-- Generates synthetic data at 100K rows / 128 dims. All query vectors are
-- literals so the HNSW optimizer fires. Single-threaded for reproducibility.

.timer on
SET threads = 1;
SELECT setseed(0.42);
SET lambda_syntax = 'ENABLE_SINGLE_ARROW';
CREATE OR REPLACE MACRO rv(d) AS (list_transform(range(d), x -> random()::FLOAT));

-- =====================================================================
-- DATA SETUP: 100K rows, 128 dims, 10 categories, separate metadata table
-- =====================================================================
SELECT '>>> Generating 100K rows, 128 dims...' AS status;

CREATE OR REPLACE TABLE items AS
SELECT i AS id,
    rv(128)::FLOAT[128] AS vec,
    (i % 10) AS category,
    (i % 100) AS subcategory
FROM range(100000) t(i);

CREATE OR REPLACE TABLE metadata AS
SELECT i AS id,
    (i % 10) AS category,
    (i % 100) AS subcategory,
    'item_' || i AS name,
    (i % 3) AS tag
FROM range(100000) t(i);

-- Query vectors (constant literals, required for optimizer to fire)
-- q_center: [0.5, 0.5, ..., 0.5] — center of the random distribution
-- q_edge:   [0.0, 0.0, ..., 0.0] — edge of the distribution
CREATE OR REPLACE MACRO q_center() AS list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128];
CREATE OR REPLACE MACRO q_edge() AS list_resize([0.0::FLOAT], 128, 0.0::FLOAT)::FLOAT[128];

-- Results table
CREATE OR REPLACE TABLE bench_results (
    benchmark VARCHAR,
    variant VARCHAR,
    metric VARCHAR,
    value DOUBLE
);

-- =====================================================================
-- 1. FILTERED SEARCH: recall and result count at varying selectivities
-- =====================================================================
SELECT '>>> Benchmark 1: Filtered search' AS status;

CREATE INDEX idx ON items USING HNSW (vec);

-- Helper: brute force ground truth for each selectivity
-- 100% selectivity (no filter)
CREATE OR REPLACE TABLE gt_all AS
SELECT id FROM items ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;

-- 10% selectivity (category = 1)
CREATE OR REPLACE TABLE gt_10pct AS
SELECT id FROM items WHERE category = 1 ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;

-- 1% selectivity (subcategory = 1)
CREATE OR REPLACE TABLE gt_1pct AS
SELECT id FROM items WHERE subcategory = 1 ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;

-- 0.1% selectivity (subcategory = 1 AND tag = 0 from metadata, ~33 rows)
-- Can't use metadata join for ground truth — just brute force
CREATE OR REPLACE TABLE gt_01pct AS
SELECT id FROM items WHERE subcategory = 1 AND category = 1
ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;

-- --- No filter (100%) ---
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT id FROM items ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO bench_results VALUES ('filtered', '100%', 'latency_ms', epoch_ms(current_timestamp) - (SELECT t0 FROM _t));
INSERT INTO bench_results VALUES ('filtered', '100%', 'recall@10',
    (SELECT COUNT(*)::DOUBLE / 10 FROM res WHERE id IN (SELECT id FROM gt_all)));
INSERT INTO bench_results VALUES ('filtered', '100%', 'result_count', (SELECT COUNT(*) FROM res));

-- --- 10% selectivity ---
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT id FROM items WHERE category = 1
ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO bench_results VALUES ('filtered', '10%', 'latency_ms', epoch_ms(current_timestamp) - (SELECT t0 FROM _t));
INSERT INTO bench_results VALUES ('filtered', '10%', 'recall@10',
    (SELECT COUNT(*)::DOUBLE / 10 FROM res WHERE id IN (SELECT id FROM gt_10pct)));
INSERT INTO bench_results VALUES ('filtered', '10%', 'result_count', (SELECT COUNT(*) FROM res));

-- --- 1% selectivity ---
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT id FROM items WHERE subcategory = 1
ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO bench_results VALUES ('filtered', '1%', 'latency_ms', epoch_ms(current_timestamp) - (SELECT t0 FROM _t));
INSERT INTO bench_results VALUES ('filtered', '1%', 'recall@10',
    (SELECT COUNT(*)::DOUBLE / 10 FROM res WHERE id IN (SELECT id FROM gt_1pct)));
INSERT INTO bench_results VALUES ('filtered', '1%', 'result_count', (SELECT COUNT(*) FROM res));

-- --- 0.1% selectivity ---
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT id FROM items WHERE subcategory = 1 AND category = 1
ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO bench_results VALUES ('filtered', '0.1%', 'latency_ms', epoch_ms(current_timestamp) - (SELECT t0 FROM _t));
INSERT INTO bench_results VALUES ('filtered', '0.1%', 'recall@10',
    (SELECT COUNT(*)::DOUBLE / 10 FROM res WHERE id IN (SELECT id FROM gt_01pct)));
INSERT INTO bench_results VALUES ('filtered', '0.1%', 'result_count', (SELECT COUNT(*) FROM res));

DROP INDEX idx;

-- =====================================================================
-- 2. RABITQ: memory and recall at different oversample factors
-- =====================================================================
SELECT '>>> Benchmark 2: RaBitQ quantization' AS status;

-- Plain HNSW baseline
CREATE INDEX idx ON items USING HNSW (vec);
INSERT INTO bench_results VALUES ('rabitq', 'hnsw', 'vec_memory_kb',
    (SELECT vector_memory_usage FROM pragma_hnsw_index_info() WHERE index_name = 'idx') / 1024.0);

CREATE OR REPLACE TABLE res AS
SELECT id FROM items ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO bench_results VALUES ('rabitq', 'hnsw', 'recall@10',
    (SELECT COUNT(*)::DOUBLE / 10 FROM res WHERE id IN (SELECT id FROM gt_all)));
DROP INDEX idx;

-- RaBitQ at different oversample factors
SET hnsw_rabitq_oversample = 3;
CREATE INDEX idx ON items USING HNSW (vec) WITH (quantization = 'rabitq');
INSERT INTO bench_results VALUES ('rabitq', 'rabitq_3x', 'vec_memory_kb',
    (SELECT vector_memory_usage FROM pragma_hnsw_index_info() WHERE index_name = 'idx') / 1024.0);
CREATE OR REPLACE TABLE res AS
SELECT id FROM items ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO bench_results VALUES ('rabitq', 'rabitq_3x', 'recall@10',
    (SELECT COUNT(*)::DOUBLE / 10 FROM res WHERE id IN (SELECT id FROM gt_all)));
DROP INDEX idx;
RESET hnsw_rabitq_oversample;

SET hnsw_rabitq_oversample = 10;
CREATE INDEX idx ON items USING HNSW (vec) WITH (quantization = 'rabitq');
INSERT INTO bench_results VALUES ('rabitq', 'rabitq_10x', 'vec_memory_kb',
    (SELECT vector_memory_usage FROM pragma_hnsw_index_info() WHERE index_name = 'idx') / 1024.0);
CREATE OR REPLACE TABLE res AS
SELECT id FROM items ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO bench_results VALUES ('rabitq', 'rabitq_10x', 'recall@10',
    (SELECT COUNT(*)::DOUBLE / 10 FROM res WHERE id IN (SELECT id FROM gt_all)));
DROP INDEX idx;
RESET hnsw_rabitq_oversample;

SET hnsw_rabitq_oversample = 30;
CREATE INDEX idx ON items USING HNSW (vec) WITH (quantization = 'rabitq');
INSERT INTO bench_results VALUES ('rabitq', 'rabitq_30x', 'vec_memory_kb',
    (SELECT vector_memory_usage FROM pragma_hnsw_index_info() WHERE index_name = 'idx') / 1024.0);
CREATE OR REPLACE TABLE res AS
SELECT id FROM items ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO bench_results VALUES ('rabitq', 'rabitq_30x', 'recall@10',
    (SELECT COUNT(*)::DOUBLE / 10 FROM res WHERE id IN (SELECT id FROM gt_all)));
DROP INDEX idx;
RESET hnsw_rabitq_oversample;

-- =====================================================================
-- 3. METADATA JOIN: index-optimized vs brute force
-- =====================================================================
SELECT '>>> Benchmark 3: Metadata join' AS status;

CREATE INDEX idx ON items USING HNSW (vec);

-- 10% selectivity (category = 1, ~10K matching metadata rows)
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT m.name
FROM items e JOIN metadata m ON e.id = m.id
WHERE m.category = 1
ORDER BY array_distance(e.vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128])
LIMIT 10;
INSERT INTO bench_results VALUES ('metadata_join', '10%_with_index', 'latency_ms',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t));
INSERT INTO bench_results VALUES ('metadata_join', '10%_with_index', 'result_count',
    (SELECT COUNT(*) FROM res));

-- 1% selectivity (subcategory = 1, ~1K matching rows)
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT m.name
FROM items e JOIN metadata m ON e.id = m.id
WHERE m.subcategory = 1
ORDER BY array_distance(e.vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128])
LIMIT 10;
INSERT INTO bench_results VALUES ('metadata_join', '1%_with_index', 'latency_ms',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t));
INSERT INTO bench_results VALUES ('metadata_join', '1%_with_index', 'result_count',
    (SELECT COUNT(*) FROM res));

-- Compound filter (category = 1 AND tag = 0, ~3.3%)
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT m.name
FROM items e JOIN metadata m ON e.id = m.id
WHERE m.category = 1 AND m.tag = 0
ORDER BY array_distance(e.vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128])
LIMIT 10;
INSERT INTO bench_results VALUES ('metadata_join', '3.3%_compound_with_index', 'latency_ms',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t));
INSERT INTO bench_results VALUES ('metadata_join', '3.3%_compound_with_index', 'result_count',
    (SELECT COUNT(*) FROM res));

DROP INDEX idx;

-- Same queries without index (brute force)
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT m.name
FROM items e JOIN metadata m ON e.id = m.id
WHERE m.category = 1
ORDER BY array_distance(e.vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128])
LIMIT 10;
INSERT INTO bench_results VALUES ('metadata_join', '10%_brute_force', 'latency_ms',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t));

CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT m.name
FROM items e JOIN metadata m ON e.id = m.id
WHERE m.subcategory = 1
ORDER BY array_distance(e.vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128])
LIMIT 10;
INSERT INTO bench_results VALUES ('metadata_join', '1%_brute_force', 'latency_ms',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t));

CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT m.name
FROM items e JOIN metadata m ON e.id = m.id
WHERE m.category = 1 AND m.tag = 0
ORDER BY array_distance(e.vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128])
LIMIT 10;
INSERT INTO bench_results VALUES ('metadata_join', '3.3%_compound_brute_force', 'latency_ms',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t));

-- =====================================================================
-- 4. GROUPED NEAREST NEIGHBORS: index-optimized vs brute force
-- =====================================================================
SELECT '>>> Benchmark 4: Grouped nearest neighbors' AS status;

CREATE INDEX idx ON items USING HNSW (vec);

-- 10 groups (category), k=5
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT category, min_by(id, array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]), 5) AS closest
FROM items GROUP BY category;
INSERT INTO bench_results VALUES ('grouped', '10_groups_k5_with_index', 'latency_ms',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t));
INSERT INTO bench_results VALUES ('grouped', '10_groups_k5_with_index', 'group_count',
    (SELECT COUNT(*) FROM res));

-- 100 groups (subcategory), k=5
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT subcategory, min_by(id, array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]), 5) AS closest
FROM items GROUP BY subcategory;
INSERT INTO bench_results VALUES ('grouped', '100_groups_k5_with_index', 'latency_ms',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t));
INSERT INTO bench_results VALUES ('grouped', '100_groups_k5_with_index', 'group_count',
    (SELECT COUNT(*) FROM res));

-- 10 groups, k=20
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT category, min_by(id, array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]), 20) AS closest
FROM items GROUP BY category;
INSERT INTO bench_results VALUES ('grouped', '10_groups_k20_with_index', 'latency_ms',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t));

DROP INDEX idx;

-- Same queries without index (brute force)
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT category, min_by(id, array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]), 5) AS closest
FROM items GROUP BY category;
INSERT INTO bench_results VALUES ('grouped', '10_groups_k5_brute_force', 'latency_ms',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t));

CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT subcategory, min_by(id, array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]), 5) AS closest
FROM items GROUP BY subcategory;
INSERT INTO bench_results VALUES ('grouped', '100_groups_k5_brute_force', 'latency_ms',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t));

CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE OR REPLACE TABLE res AS
SELECT category, min_by(id, array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]), 20) AS closest
FROM items GROUP BY category;
INSERT INTO bench_results VALUES ('grouped', '10_groups_k20_brute_force', 'latency_ms',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t));

-- =====================================================================
-- RESULTS
-- =====================================================================
.timer off

SELECT '
============================================================
  FILTERED SEARCH (100K rows, 128 dims, LIMIT 10)
============================================================' AS '';

SELECT
    printf('%-8s', variant) AS selectivity,
    printf('%7.0f ms', MAX(value) FILTER (WHERE metric = 'latency_ms')) AS latency,
    printf('%5.0f%%', MAX(value) FILTER (WHERE metric = 'recall@10') * 100) AS "recall@10",
    printf('%d/10', MAX(value) FILTER (WHERE metric = 'result_count')::INT) AS results
FROM bench_results WHERE benchmark = 'filtered'
GROUP BY variant ORDER BY variant;

SELECT '
============================================================
  RABITQ QUANTIZATION (100K rows, 128 dims)
============================================================' AS '';

SELECT
    printf('%-12s', variant) AS method,
    printf('%5.0f%%', MAX(value) FILTER (WHERE metric = 'recall@10') * 100) AS "recall@10",
    printf('%8.0f KB', MAX(value) FILTER (WHERE metric = 'vec_memory_kb')) AS vec_memory,
    CASE WHEN variant != 'hnsw' THEN
        printf('%5.1fx',
            (SELECT MAX(value) FROM bench_results WHERE benchmark = 'rabitq' AND variant = 'hnsw' AND metric = 'vec_memory_kb')
            / NULLIF(MAX(value) FILTER (WHERE metric = 'vec_memory_kb'), 0))
    ELSE '' END AS compression
FROM bench_results WHERE benchmark = 'rabitq'
GROUP BY variant ORDER BY variant;

SELECT '
============================================================
  METADATA JOIN (100K rows, 128 dims, LIMIT 10)
============================================================' AS '';

SELECT
    printf('%-35s', variant) AS variant,
    printf('%7.0f ms', MAX(value) FILTER (WHERE metric = 'latency_ms')) AS latency,
    COALESCE(printf('%d/10', MAX(value) FILTER (WHERE metric = 'result_count')::INT), '') AS results
FROM bench_results WHERE benchmark = 'metadata_join'
GROUP BY variant ORDER BY variant;

SELECT '
============================================================
  GROUPED NEAREST NEIGHBORS (100K rows, 128 dims)
============================================================' AS '';

SELECT
    printf('%-35s', variant) AS variant,
    printf('%7.0f ms', MAX(value) FILTER (WHERE metric = 'latency_ms')) AS latency,
    COALESCE(printf('%d groups', MAX(value) FILTER (WHERE metric = 'group_count')::INT), '') AS groups
FROM bench_results WHERE benchmark = 'grouped'
GROUP BY variant ORDER BY variant;
