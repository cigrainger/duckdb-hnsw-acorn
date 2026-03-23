-- Movies Filtered HNSW Benchmark (local parquet)
--
-- Prerequisites: download the dataset first (see wiki_local_benchmark.sql header for pattern)
-- Run with: ./build/release/duckdb < test/benchmark/movies_benchmark.sql

.timer on
LOAD vss;

.print 'Loading movies dataset...'

CREATE TABLE movies AS
SELECT
    row_number() OVER () AS rid,
    id,
    title,
    original_language,
    genres,
    vote_average,
    status,
    -- Parse stringified embedding list into FLOAT array
    cast(embedding AS FLOAT[768]) AS vec
FROM '/tmp/movies_1m.parquet'
WHERE embedding IS NOT NULL AND len(embedding) > 10;

SELECT format('Loaded {} rows', count(*)) AS info FROM movies;
SELECT original_language, count(*) AS cnt FROM movies GROUP BY original_language ORDER BY cnt DESC LIMIT 5;

-- ============================================================
-- Build HNSW index
-- ============================================================
.print ''
.print 'Building HNSW index on 768-dim movie embeddings...'

CREATE INDEX movies_idx ON movies USING HNSW (vec) WITH (metric = 'cosine');

.print 'Index built.'

-- ============================================================
-- Query: find movies similar to a known movie
-- ============================================================
CREATE TABLE qvec AS SELECT vec AS q FROM movies WHERE title LIKE '%Matrix%' LIMIT 1;
SELECT format('Query movie: {}', (SELECT title FROM movies WHERE title LIKE '%Matrix%' LIMIT 1)) AS info;

-- ============================================================
-- Unfiltered baseline
-- ============================================================
.print ''
.print '=== UNFILTERED TOP-10 ==='

SELECT title, original_language, array_cosine_distance(vec, (SELECT q FROM qvec)) AS dist
FROM movies ORDER BY dist LIMIT 10;

-- ============================================================
-- Filtered by language (real categorical filter!)
-- ============================================================
.print ''
.print '=== FILTERED: English only ==='

SELECT title, array_cosine_distance(vec, (SELECT q FROM qvec)) AS dist
FROM movies WHERE original_language = 'en' ORDER BY dist LIMIT 10;

.print ''
.print '=== FILTERED: Japanese only (~1-2%) ==='

SELECT title, array_cosine_distance(vec, (SELECT q FROM qvec)) AS dist
FROM movies WHERE original_language = 'ja' ORDER BY dist LIMIT 10;

.print ''
.print '=== FILTERED: Korean only (~0.5-1%) ==='

SELECT title, array_cosine_distance(vec, (SELECT q FROM qvec)) AS dist
FROM movies WHERE original_language = 'ko' ORDER BY dist LIMIT 10;

-- ============================================================
-- Result count check
-- ============================================================
.print ''
.print '=== RESULT COUNTS (all should be 10) ==='

SELECT 'en (~70%)' AS filter, (SELECT count(*) FROM (SELECT rid FROM movies WHERE original_language = 'en' ORDER BY array_cosine_distance(vec, (SELECT q FROM qvec)) LIMIT 10)) AS got
UNION ALL SELECT 'ja (~2%)', (SELECT count(*) FROM (SELECT rid FROM movies WHERE original_language = 'ja' ORDER BY array_cosine_distance(vec, (SELECT q FROM qvec)) LIMIT 10))
UNION ALL SELECT 'ko (~1%)', (SELECT count(*) FROM (SELECT rid FROM movies WHERE original_language = 'ko' ORDER BY array_cosine_distance(vec, (SELECT q FROM qvec)) LIMIT 10))
UNION ALL SELECT 'zh (~0.5%)', (SELECT count(*) FROM (SELECT rid FROM movies WHERE original_language = 'zh' ORDER BY array_cosine_distance(vec, (SELECT q FROM qvec)) LIMIT 10));

.print ''
.print 'Done.'
