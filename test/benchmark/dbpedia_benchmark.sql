-- DBpedia 1M Entity Filtered HNSW Benchmark
-- Run with: ./build/release/duckdb < test/benchmark/dbpedia_benchmark.sql
--
-- Uses DBpedia 1M entities with OpenAI 1536-dim embeddings.
-- WARNING: Downloads ~11 GB, index build uses several GB RAM.

.timer on

INSTALL httpfs;
LOAD httpfs;

-- ============================================================
-- Step 1: Load dataset
-- ============================================================
.print 'Loading DBpedia 1M entity embeddings...'
.print '(1M rows, 1536-dim OpenAI embeddings, ~11 GB download)'
.print 'This will take a while...'

CREATE TABLE dbpedia AS
SELECT
    row_number() OVER () AS id,
    title,
    text,
    _categories,
    list_transform(openai, x -> x::FLOAT)::FLOAT[1536] AS embedding
FROM 'hf://datasets/KShivendu/dbpedia-entities-openai-1M/data/train-*.parquet';

SELECT format('Loaded {} rows', count(*)) AS status FROM dbpedia;

-- Add a categorical column derived from _categories
-- Many entities have multiple categories; pick the first one
.print 'Adding filter columns...'
ALTER TABLE dbpedia ADD COLUMN primary_category VARCHAR;
UPDATE dbpedia SET primary_category = _categories[1];

-- Check category distribution
SELECT primary_category, count(*) AS cnt
FROM dbpedia
WHERE primary_category IS NOT NULL
GROUP BY primary_category
ORDER BY cnt DESC
LIMIT 15;

-- ============================================================
-- Step 2: Build HNSW index
-- ============================================================
.print ''
.print 'Building HNSW index on 1M 1536-dim vectors...'
.print 'This will take several minutes and use several GB of RAM.'

CREATE INDEX dbpedia_idx ON dbpedia USING HNSW (embedding) WITH (metric = 'cosine');

.print 'Index built.'

-- ============================================================
-- Step 3: Query vector
-- ============================================================
CREATE TABLE query_store AS
SELECT embedding AS q FROM dbpedia WHERE title LIKE '%Einstein%' LIMIT 1;

SELECT format('Query: {}', (SELECT title FROM dbpedia WHERE title LIKE '%Einstein%' LIMIT 1)) AS info;

-- ============================================================
-- Step 4: Unfiltered baseline
-- ============================================================
.print ''
.print '============================================================'
.print 'BASELINE: Unfiltered top-10'
.print '============================================================'

SELECT title, array_cosine_distance(embedding, (SELECT q FROM query_store)) AS dist
FROM dbpedia
ORDER BY dist
LIMIT 10;

-- ============================================================
-- Step 5: Filtered by category
-- ============================================================
.print ''
.print '============================================================'
.print 'FILTERED: By most common category'
.print '============================================================'

-- Pick the most common category
CREATE TABLE top_cat AS
SELECT primary_category FROM dbpedia
WHERE primary_category IS NOT NULL
GROUP BY primary_category ORDER BY count(*) DESC LIMIT 1;

SELECT format('Filtering by category: {} ({} rows)',
    (SELECT primary_category FROM top_cat),
    (SELECT count(*) FROM dbpedia WHERE primary_category = (SELECT primary_category FROM top_cat))
) AS info;

SELECT title, array_cosine_distance(embedding, (SELECT q FROM query_store)) AS dist
FROM dbpedia
WHERE primary_category = (SELECT primary_category FROM top_cat)
ORDER BY dist
LIMIT 10;

-- ============================================================
-- Step 6: Result counts at various selectivities
-- ============================================================
.print ''
.print '============================================================'
.print 'RESULT COUNT across selectivities (should all be 10)'
.print '============================================================'

-- Use hash-based buckets for controlled selectivity
ALTER TABLE dbpedia ADD COLUMN bucket_100 INTEGER;
UPDATE dbpedia SET bucket_100 = abs(hash(title) % 100);

SELECT '50%' AS selectivity,
    (SELECT count(*) FROM (SELECT id FROM dbpedia WHERE bucket_100 < 50 ORDER BY array_cosine_distance(embedding, (SELECT q FROM query_store)) LIMIT 10)) AS got
UNION ALL SELECT '10%',
    (SELECT count(*) FROM (SELECT id FROM dbpedia WHERE bucket_100 < 10 ORDER BY array_cosine_distance(embedding, (SELECT q FROM query_store)) LIMIT 10))
UNION ALL SELECT '1%',
    (SELECT count(*) FROM (SELECT id FROM dbpedia WHERE bucket_100 = 0 ORDER BY array_cosine_distance(embedding, (SELECT q FROM query_store)) LIMIT 10));

.print ''
.print 'Done. 1M row benchmark complete.'
