-- Filtered HNSW Search Benchmark
-- Run with: ./build/release/duckdb < test/benchmark/filtered_search_bench.sql
--
-- Establishes baseline for filtered search behavior.
-- Measures: result count, recall@k, and query plan at different selectivities.
-- Ground truth computed via brute-force (no index).

.timer on

LOAD 'build/release/extension/vss/vss.duckdb_extension';

-- ============================================================
-- Setup: generate synthetic dataset
-- ============================================================

CREATE TABLE bench AS
SELECT
    i AS id,
    array_value(
        sin(i * 0.1)::FLOAT, cos(i * 0.2)::FLOAT, sin(i * 0.3)::FLOAT, cos(i * 0.4)::FLOAT,
        sin(i * 0.5)::FLOAT, cos(i * 0.6)::FLOAT, sin(i * 0.7)::FLOAT, cos(i * 0.8)::FLOAT
    ) AS vec,
    (i % 2) AS cat2,       -- 50% selectivity per value
    (i % 5) AS cat5,       -- 20% selectivity
    (i % 10) AS cat10,     -- 10% selectivity
    (i % 20) AS cat20,     -- 5% selectivity
    (i % 100) AS cat100,   -- 1% selectivity
FROM range(50000) t(i);

.print 'Dataset: 50000 rows, 8-dim vectors'

CREATE INDEX bench_idx ON bench USING HNSW (vec) WITH (metric = 'l2sq');

.print 'HNSW index built'

-- ============================================================
-- Check plan: does HNSW index get used with WHERE clause?
-- ============================================================

.print ''
.print '============================================================'
.print 'PLAN CHECK: Unfiltered (should use HNSW_INDEX_SCAN)'
.print '============================================================'

EXPLAIN SELECT id FROM bench ORDER BY array_distance(vec, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]::FLOAT[8]) LIMIT 20;

.print ''
.print '============================================================'
.print 'PLAN CHECK: Filtered (check if HNSW is used or falls back)'
.print '============================================================'

EXPLAIN SELECT id FROM bench WHERE cat10 = 0 ORDER BY array_distance(vec, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]::FLOAT[8]) LIMIT 20;

-- ============================================================
-- Baseline: unfiltered top-20
-- ============================================================

.print ''
.print '============================================================'
.print 'BASELINE: Unfiltered top-20'
.print '============================================================'

SELECT id, array_distance(vec, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]::FLOAT[8]) AS dist
FROM bench
ORDER BY dist
LIMIT 20;

-- ============================================================
-- Result count test at each selectivity
-- Shows how many of the requested 20 results we actually get
-- ============================================================

.print ''
.print '============================================================'
.print 'RESULT COUNT: How many of 20 requested rows are returned?'
.print '(Fewer than 20 = post-filter truncation problem)'
.print '============================================================'

SELECT '50% (cat2=0)' AS filter,
    (SELECT count(*) FROM (SELECT id FROM bench WHERE cat2 = 0 ORDER BY array_distance(vec, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]::FLOAT[8]) LIMIT 20)) AS got,
    20 AS wanted;

SELECT '20% (cat5=0)' AS filter,
    (SELECT count(*) FROM (SELECT id FROM bench WHERE cat5 = 0 ORDER BY array_distance(vec, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]::FLOAT[8]) LIMIT 20)) AS got,
    20 AS wanted;

SELECT '10% (cat10=0)' AS filter,
    (SELECT count(*) FROM (SELECT id FROM bench WHERE cat10 = 0 ORDER BY array_distance(vec, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]::FLOAT[8]) LIMIT 20)) AS got,
    20 AS wanted;

SELECT '5% (cat20=0)' AS filter,
    (SELECT count(*) FROM (SELECT id FROM bench WHERE cat20 = 0 ORDER BY array_distance(vec, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]::FLOAT[8]) LIMIT 20)) AS got,
    20 AS wanted;

SELECT '1% (cat100=0)' AS filter,
    (SELECT count(*) FROM (SELECT id FROM bench WHERE cat100 = 0 ORDER BY array_distance(vec, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]::FLOAT[8]) LIMIT 20)) AS got,
    20 AS wanted;

-- ============================================================
-- Recall test: compare HNSW filtered results vs brute force
-- ============================================================

.print ''
.print '============================================================'
.print 'RECALL: HNSW results vs brute-force ground truth'
.print '============================================================'

-- Ground truth at 10% selectivity (brute force, no index)
CREATE OR REPLACE TABLE gt AS
SELECT id FROM bench WHERE cat10 = 0
ORDER BY array_distance(vec, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]::FLOAT[8])
LIMIT 20;

-- HNSW result at 10% selectivity
CREATE OR REPLACE TABLE hnsw_result AS
SELECT id FROM bench WHERE cat10 = 0
ORDER BY array_distance(vec, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]::FLOAT[8])
LIMIT 20;

SELECT format('Recall@20 at 10%% selectivity: {}/{}',
    (SELECT count(*) FROM hnsw_result WHERE id IN (SELECT id FROM gt)),
    (SELECT count(*) FROM hnsw_result)
) AS recall;

.print ''
.print 'Done. After Brief 1, all "got" values should equal "wanted".'
