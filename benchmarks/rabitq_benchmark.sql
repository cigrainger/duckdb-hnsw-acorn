-- rabitq_benchmark.sql
-- Benchmark: RaBitQ vs plain HNSW vs brute force
--
-- Usage: ./build/release/duckdb < benchmarks/rabitq_benchmark.sql
--
-- Uses literal query vectors to ensure the HNSW index is actually used
-- (the optimizer requires constant arrays, not subqueries).

.timer on
SET threads = 1;

CREATE TABLE results (
    rows INT, dims INT, method VARCHAR,
    build_ms DOUBLE, recall DOUBLE, vec_kb DOUBLE
);

-- =====================================================================
-- 1000 rows, 32 dims
-- =====================================================================
SELECT '=== 1000 rows, 32 dims ===' AS config;
SELECT setseed(0.42);
SET lambda_syntax = 'ENABLE_SINGLE_ARROW';
CREATE OR REPLACE MACRO rv(d) AS (list_transform(range(d), x -> random()::FLOAT));

CREATE OR REPLACE TABLE data32 AS SELECT i AS id, rv(32)::FLOAT[32] AS vec FROM range(1000) t(i);

-- Ground truth (no index = brute force)
CREATE OR REPLACE TABLE gt32 AS
SELECT 'q1' AS q, id FROM data32 ORDER BY array_distance(vec, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]::FLOAT[32]) LIMIT 10;
INSERT INTO gt32
SELECT 'q2', id FROM data32 ORDER BY array_distance(vec, [0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9]::FLOAT[32]) LIMIT 10;
INSERT INTO gt32
SELECT 'q3', id FROM data32 ORDER BY array_distance(vec, [0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0]::FLOAT[32]) LIMIT 10;

-- Plain HNSW
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE INDEX idx32 ON data32 USING HNSW (vec);

CREATE OR REPLACE TABLE res32 AS
SELECT 'q1' AS q, id FROM data32 ORDER BY array_distance(vec, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]::FLOAT[32]) LIMIT 10;
INSERT INTO res32
SELECT 'q2', id FROM data32 ORDER BY array_distance(vec, [0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9]::FLOAT[32]) LIMIT 10;
INSERT INTO res32
SELECT 'q3', id FROM data32 ORDER BY array_distance(vec, [0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0]::FLOAT[32]) LIMIT 10;

INSERT INTO results SELECT 1000, 32, 'hnsw',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t),
    (SELECT AVG(r) FROM (SELECT q, COUNT(*) FILTER (WHERE id IN (SELECT id FROM gt32 g WHERE g.q = res32.q))::DOUBLE/10 AS r FROM res32 GROUP BY q)),
    (SELECT vector_memory_usage FROM pragma_hnsw_index_info() WHERE index_name = 'idx32') / 1024.0;
DROP INDEX idx32;

-- RaBitQ 3x
SET hnsw_rabitq_oversample = 3;
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE INDEX idx32 ON data32 USING HNSW (vec) WITH (quantization = 'rabitq');

CREATE OR REPLACE TABLE res32 AS
SELECT 'q1' AS q, id FROM data32 ORDER BY array_distance(vec, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]::FLOAT[32]) LIMIT 10;
INSERT INTO res32
SELECT 'q2', id FROM data32 ORDER BY array_distance(vec, [0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9]::FLOAT[32]) LIMIT 10;
INSERT INTO res32
SELECT 'q3', id FROM data32 ORDER BY array_distance(vec, [0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0]::FLOAT[32]) LIMIT 10;

INSERT INTO results SELECT 1000, 32, 'rabitq_3x',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t),
    (SELECT AVG(r) FROM (SELECT q, COUNT(*) FILTER (WHERE id IN (SELECT id FROM gt32 g WHERE g.q = res32.q))::DOUBLE/10 AS r FROM res32 GROUP BY q)),
    (SELECT vector_memory_usage FROM pragma_hnsw_index_info() WHERE index_name = 'idx32') / 1024.0;
DROP INDEX idx32;
RESET hnsw_rabitq_oversample;

-- RaBitQ 10x
SET hnsw_rabitq_oversample = 10;
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE INDEX idx32 ON data32 USING HNSW (vec) WITH (quantization = 'rabitq');

CREATE OR REPLACE TABLE res32 AS
SELECT 'q1' AS q, id FROM data32 ORDER BY array_distance(vec, [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]::FLOAT[32]) LIMIT 10;
INSERT INTO res32
SELECT 'q2', id FROM data32 ORDER BY array_distance(vec, [0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9]::FLOAT[32]) LIMIT 10;
INSERT INTO res32
SELECT 'q3', id FROM data32 ORDER BY array_distance(vec, [0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0]::FLOAT[32]) LIMIT 10;

INSERT INTO results SELECT 1000, 32, 'rabitq_10x',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t),
    (SELECT AVG(r) FROM (SELECT q, COUNT(*) FILTER (WHERE id IN (SELECT id FROM gt32 g WHERE g.q = res32.q))::DOUBLE/10 AS r FROM res32 GROUP BY q)),
    (SELECT vector_memory_usage FROM pragma_hnsw_index_info() WHERE index_name = 'idx32') / 1024.0;
DROP INDEX idx32;
RESET hnsw_rabitq_oversample;

-- =====================================================================
-- 10000 rows, 128 dims (same 3 query patterns, zero-padded to 128)
-- =====================================================================
SELECT '=== 10000 rows, 128 dims ===' AS config;
SELECT setseed(0.42);

CREATE OR REPLACE TABLE data128 AS SELECT i AS id, rv(128)::FLOAT[128] AS vec FROM range(10000) t(i);

-- Ground truth
CREATE OR REPLACE TABLE gt128 AS
SELECT 'q1' AS q, id FROM data128 ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO gt128 SELECT 'q2', id FROM data128 ORDER BY array_distance(vec, list_resize([0.0::FLOAT], 128, 0.0::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO gt128 SELECT 'q3', id FROM data128 ORDER BY array_distance(vec, list_resize([1.0::FLOAT], 128, 1.0::FLOAT)::FLOAT[128]) LIMIT 10;

-- Plain HNSW
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE INDEX idx128 ON data128 USING HNSW (vec);

CREATE OR REPLACE TABLE res128 AS
SELECT 'q1' AS q, id FROM data128 ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO res128 SELECT 'q2', id FROM data128 ORDER BY array_distance(vec, list_resize([0.0::FLOAT], 128, 0.0::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO res128 SELECT 'q3', id FROM data128 ORDER BY array_distance(vec, list_resize([1.0::FLOAT], 128, 1.0::FLOAT)::FLOAT[128]) LIMIT 10;

INSERT INTO results SELECT 10000, 128, 'hnsw',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t),
    (SELECT AVG(r) FROM (SELECT q, COUNT(*) FILTER (WHERE id IN (SELECT id FROM gt128 g WHERE g.q = res128.q))::DOUBLE/10 AS r FROM res128 GROUP BY q)),
    (SELECT vector_memory_usage FROM pragma_hnsw_index_info() WHERE index_name = 'idx128') / 1024.0;
DROP INDEX idx128;

-- RaBitQ 3x
SET hnsw_rabitq_oversample = 3;
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE INDEX idx128 ON data128 USING HNSW (vec) WITH (quantization = 'rabitq');

CREATE OR REPLACE TABLE res128 AS
SELECT 'q1' AS q, id FROM data128 ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO res128 SELECT 'q2', id FROM data128 ORDER BY array_distance(vec, list_resize([0.0::FLOAT], 128, 0.0::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO res128 SELECT 'q3', id FROM data128 ORDER BY array_distance(vec, list_resize([1.0::FLOAT], 128, 1.0::FLOAT)::FLOAT[128]) LIMIT 10;

INSERT INTO results SELECT 10000, 128, 'rabitq_3x',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t),
    (SELECT AVG(r) FROM (SELECT q, COUNT(*) FILTER (WHERE id IN (SELECT id FROM gt128 g WHERE g.q = res128.q))::DOUBLE/10 AS r FROM res128 GROUP BY q)),
    (SELECT vector_memory_usage FROM pragma_hnsw_index_info() WHERE index_name = 'idx128') / 1024.0;
DROP INDEX idx128;
RESET hnsw_rabitq_oversample;

-- RaBitQ 10x
SET hnsw_rabitq_oversample = 10;
CREATE OR REPLACE TABLE _t AS SELECT epoch_ms(current_timestamp) AS t0;
CREATE INDEX idx128 ON data128 USING HNSW (vec) WITH (quantization = 'rabitq');

CREATE OR REPLACE TABLE res128 AS
SELECT 'q1' AS q, id FROM data128 ORDER BY array_distance(vec, list_resize([0.5::FLOAT], 128, 0.5::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO res128 SELECT 'q2', id FROM data128 ORDER BY array_distance(vec, list_resize([0.0::FLOAT], 128, 0.0::FLOAT)::FLOAT[128]) LIMIT 10;
INSERT INTO res128 SELECT 'q3', id FROM data128 ORDER BY array_distance(vec, list_resize([1.0::FLOAT], 128, 1.0::FLOAT)::FLOAT[128]) LIMIT 10;

INSERT INTO results SELECT 10000, 128, 'rabitq_10x',
    epoch_ms(current_timestamp) - (SELECT t0 FROM _t),
    (SELECT AVG(r) FROM (SELECT q, COUNT(*) FILTER (WHERE id IN (SELECT id FROM gt128 g WHERE g.q = res128.q))::DOUBLE/10 AS r FROM res128 GROUP BY q)),
    (SELECT vector_memory_usage FROM pragma_hnsw_index_info() WHERE index_name = 'idx128') / 1024.0;
DROP INDEX idx128;
RESET hnsw_rabitq_oversample;

-- =====================================================================
-- RESULTS
-- =====================================================================
.timer off

SELECT
    printf('%5d', rows) AS rows,
    printf('%3d', dims) AS dims,
    printf('%-10s', method) AS method,
    printf('%7.0f', build_ms) AS "build_ms",
    printf('%5.1f%%', recall * 100) AS "recall@10",
    printf('%8.1f KB', vec_kb) AS "vec_mem",
    CASE WHEN method LIKE 'rabitq%' THEN
        printf('%5.1fx', (SELECT h.vec_kb FROM results h WHERE h.method = 'hnsw' AND h.rows = results.rows AND h.dims = results.dims) / NULLIF(vec_kb, 0))
    ELSE '' END AS "compression"
FROM results
ORDER BY rows, dims, method;
