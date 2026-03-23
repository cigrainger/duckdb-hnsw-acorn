-- Real-World Movies Benchmark: 228k rows, 768-dim embeddings, real metadata filters
-- Run with: ./build/release/duckdb < test/benchmark/movies_real_benchmark.sql

LOAD vss;
.timer on

.print 'Loading 228k movies with 768-dim embeddings...'

CREATE TABLE movies AS
SELECT row_number() OVER () AS id, title, original_language, genres, vote_average,
    cast(embedding AS FLOAT[768]) AS vec
FROM '/tmp/movies_shard0.parquet'
WHERE embedding IS NOT NULL AND length(embedding) > 100;

SELECT format('Loaded {} rows', count(*)) AS info FROM movies;
SELECT original_language, count(*) AS cnt FROM movies GROUP BY original_language ORDER BY cnt DESC LIMIT 8;

.print ''
.print 'Building HNSW index on 768-dim vectors...'

CREATE INDEX idx ON movies USING HNSW (vec) WITH (metric = 'cosine');

.print 'Index built.'

-- Use first movie as query
CREATE TABLE qvec AS SELECT vec AS q FROM movies WHERE title LIKE '%Matrix%' LIMIT 1;
SELECT format('Query: {}', (SELECT title FROM movies WHERE title LIKE '%Matrix%' LIMIT 1)) AS info;

.print ''
.print '=== UNFILTERED TOP-10 ==='
SELECT title, original_language, array_cosine_distance(vec, (SELECT q FROM qvec)) AS dist
FROM movies ORDER BY dist LIMIT 10;

.print ''
.print '=== FILTERED: English only ==='
SELECT title, array_cosine_distance(vec, (SELECT q FROM qvec)) AS dist
FROM movies WHERE original_language = 'en' ORDER BY dist LIMIT 10;

.print ''
.print '=== FILTERED: Japanese only ==='
SELECT title, array_cosine_distance(vec, (SELECT q FROM qvec)) AS dist
FROM movies WHERE original_language = 'ja' ORDER BY dist LIMIT 10;

.print ''
.print '=== FILTERED: Korean only ==='
SELECT title, array_cosine_distance(vec, (SELECT q FROM qvec)) AS dist
FROM movies WHERE original_language = 'ko' ORDER BY dist LIMIT 10;

.print ''
.print '=== FILTERED: High rated (>= 8.0) ==='
SELECT title, original_language, vote_average, array_cosine_distance(vec, (SELECT q FROM qvec)) AS dist
FROM movies WHERE vote_average >= 8.0 ORDER BY dist LIMIT 10;

.print ''
.print '=== RESULT COUNTS (all should be 10) ==='
SELECT 'en (~60%)' AS filter, (SELECT count(*) FROM (SELECT id FROM movies WHERE original_language = 'en' ORDER BY array_cosine_distance(vec, (SELECT q FROM qvec)) LIMIT 10)) AS got
UNION ALL SELECT 'ja (~2%)', (SELECT count(*) FROM (SELECT id FROM movies WHERE original_language = 'ja' ORDER BY array_cosine_distance(vec, (SELECT q FROM qvec)) LIMIT 10))
UNION ALL SELECT 'ko (~1%)', (SELECT count(*) FROM (SELECT id FROM movies WHERE original_language = 'ko' ORDER BY array_cosine_distance(vec, (SELECT q FROM qvec)) LIMIT 10))
UNION ALL SELECT 'hi (~1%)', (SELECT count(*) FROM (SELECT id FROM movies WHERE original_language = 'hi' ORDER BY array_cosine_distance(vec, (SELECT q FROM qvec)) LIMIT 10))
UNION ALL SELECT 'rating>=8 (~5%)', (SELECT count(*) FROM (SELECT id FROM movies WHERE vote_average >= 8.0 ORDER BY array_cosine_distance(vec, (SELECT q FROM qvec)) LIMIT 10));

.print ''
.print 'Done. 228k movies, 768-dim, real metadata filters.'
