-- Wikipedia Filtered HNSW Benchmark
-- Run with: ./build/release/duckdb < test/benchmark/wiki_benchmark.sql
--
-- Uses Cohere Simple English Wikipedia embeddings (~485k rows, 768-dim)
-- Tests filtered search at various selectivities with real data.

.timer on

INSTALL httpfs;
LOAD httpfs;

-- ============================================================
-- Step 1: Load dataset
-- ============================================================
.print 'Loading Cohere Simple English Wikipedia embeddings...'
.print '(~485k rows, 768-dim, ~3 GB download)'

CREATE TABLE wiki AS
SELECT
    row_number() OVER () AS id,
    title,
    text,
    wiki_id,
    paragraph_id,
    list_transform(emb, x -> x::FLOAT)::FLOAT[768] AS embedding
FROM 'hf://datasets/Cohere/wikipedia-22-12-simple-embeddings/data/train-*.parquet';

SELECT format('Loaded {} rows', count(*)) AS status FROM wiki;

-- Add filterable columns with known selectivities
.print 'Adding filter columns...'
ALTER TABLE wiki ADD COLUMN first_letter VARCHAR;
UPDATE wiki SET first_letter = upper(left(title, 1));

-- Check distribution
SELECT first_letter, count(*) AS cnt
FROM wiki
GROUP BY first_letter
ORDER BY cnt DESC
LIMIT 10;

-- ============================================================
-- Step 2: Build HNSW index
-- ============================================================
.print ''
.print 'Building HNSW index (this may take a few minutes)...'

CREATE INDEX wiki_idx ON wiki USING HNSW (embedding) WITH (metric = 'cosine');

.print 'Index built.'

-- ============================================================
-- Step 3: Pick a query vector (pre-compute as constant)
-- We'll use the embedding for 'Albert Einstein' as the query
-- ============================================================
.print ''
.print 'Selecting query vector...'

-- Store the query vector in a macro for reuse as a constant
CREATE TABLE query_store AS
SELECT embedding AS q FROM wiki WHERE title = 'Albert Einstein' LIMIT 1;

-- Verify we got one
SELECT format('Query vector dimension: {}', len(q)) AS status FROM query_store;

-- ============================================================
-- Step 4: Unfiltered baseline
-- ============================================================
.print ''
.print '============================================================'
.print 'BASELINE: Unfiltered top-10 nearest to "Albert Einstein"'
.print '============================================================'

-- Use a literal-ish approach: join with query_store
-- NOTE: This won't hit the HNSW optimizer (subquery).
-- For HNSW to activate, we need ORDER BY distance(vec, CONSTANT) LIMIT k
-- So let's just test timing for now.

SELECT title, array_cosine_distance(embedding, (SELECT q FROM query_store)) AS dist
FROM wiki
ORDER BY dist
LIMIT 10;

-- ============================================================
-- Step 5: Filtered search at various selectivities
-- ============================================================

.print ''
.print '============================================================'
.print 'FILTERED: Titles starting with S (~8% of rows)'
.print '============================================================'

SELECT title, array_cosine_distance(embedding, (SELECT q FROM query_store)) AS dist
FROM wiki
WHERE first_letter = 'S'
ORDER BY dist
LIMIT 10;

.print ''
.print '============================================================'
.print 'FILTERED: Titles starting with Z (~0.5% of rows)'
.print '============================================================'

SELECT title, array_cosine_distance(embedding, (SELECT q FROM query_store)) AS dist
FROM wiki
WHERE first_letter = 'Z'
ORDER BY dist
LIMIT 10;

-- ============================================================
-- Step 6: Result count verification
-- ============================================================
.print ''
.print '============================================================'
.print 'RESULT COUNT CHECK (should all be 10)'
.print '============================================================'

SELECT 'S (~8%)' AS filter,
    (SELECT count(*) FROM (SELECT id FROM wiki WHERE first_letter = 'S' ORDER BY array_cosine_distance(embedding, (SELECT q FROM query_store)) LIMIT 10)) AS got
UNION ALL SELECT 'M (~6%)',
    (SELECT count(*) FROM (SELECT id FROM wiki WHERE first_letter = 'M' ORDER BY array_cosine_distance(embedding, (SELECT q FROM query_store)) LIMIT 10))
UNION ALL SELECT 'Z (~0.5%)',
    (SELECT count(*) FROM (SELECT id FROM wiki WHERE first_letter = 'Z' ORDER BY array_cosine_distance(embedding, (SELECT q FROM query_store)) LIMIT 10))
UNION ALL SELECT 'X (~0.1%)',
    (SELECT count(*) FROM (SELECT id FROM wiki WHERE first_letter = 'X' ORDER BY array_cosine_distance(embedding, (SELECT q FROM query_store)) LIMIT 10));

.print ''
.print 'Done.'
