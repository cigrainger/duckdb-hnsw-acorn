-- Wikipedia Filtered HNSW Benchmark (local parquet)
--
-- Prerequisites:
--   1. Download with system duckdb:
--      duckdb -c "INSTALL httpfs; LOAD httpfs; COPY (SELECT * FROM 'hf://datasets/Cohere/wikipedia-22-12-simple-embeddings/data/train-*.parquet') TO '/tmp/wiki_embeddings.parquet';"
--
--   2. Run with our built duckdb:
--      ./build/release/duckdb < test/benchmark/wiki_local_benchmark.sql

.timer on
LOAD hnsw_acorn;

-- ============================================================
-- Load from local parquet
-- ============================================================
.print 'Loading local wiki embeddings...'

CREATE TABLE wiki AS
SELECT
    row_number() OVER () AS id,
    title,
    text,
    wiki_id,
    paragraph_id,
    emb::FLOAT[768] AS embedding
FROM '/tmp/wiki_embeddings.parquet';

SELECT format('Loaded {} rows', count(*)) AS status FROM wiki;

-- Filter column
ALTER TABLE wiki ADD COLUMN first_letter VARCHAR;
UPDATE wiki SET first_letter = upper(left(title, 1));

-- Distribution
SELECT first_letter, count(*) AS cnt FROM wiki GROUP BY first_letter ORDER BY cnt DESC LIMIT 5;

-- ============================================================
-- Build HNSW index
-- ============================================================
.print ''
.print 'Building HNSW index on 768-dim vectors...'

CREATE INDEX wiki_idx ON wiki USING HNSW (embedding) WITH (metric = 'cosine');

.print 'Index built.'

-- ============================================================
-- Pick query vector (Einstein)
-- ============================================================
CREATE TABLE qvec AS SELECT embedding AS q FROM wiki WHERE title = 'Albert Einstein' LIMIT 1;
SELECT format('Query: Albert Einstein (dim={})', len(q)) AS info FROM qvec;

-- ============================================================
-- Unfiltered baseline
-- ============================================================
.print ''
.print '=== UNFILTERED TOP-10 ==='

SELECT title, array_cosine_distance(embedding, (SELECT q FROM qvec)) AS dist
FROM wiki ORDER BY dist LIMIT 10;

-- ============================================================
-- Filtered searches
-- ============================================================
.print ''
.print '=== FILTERED: first_letter = S (~8%) ==='

SELECT title, array_cosine_distance(embedding, (SELECT q FROM qvec)) AS dist
FROM wiki WHERE first_letter = 'S' ORDER BY dist LIMIT 10;

.print ''
.print '=== FILTERED: first_letter = Z (~0.5%) ==='

SELECT title, array_cosine_distance(embedding, (SELECT q FROM qvec)) AS dist
FROM wiki WHERE first_letter = 'Z' ORDER BY dist LIMIT 10;

-- ============================================================
-- Result count check
-- ============================================================
.print ''
.print '=== RESULT COUNTS (all should be 10) ==='

SELECT 'No filter' AS filter, (SELECT count(*) FROM (SELECT id FROM wiki ORDER BY array_cosine_distance(embedding, (SELECT q FROM qvec)) LIMIT 10)) AS got
UNION ALL SELECT 'S (~8%)', (SELECT count(*) FROM (SELECT id FROM wiki WHERE first_letter = 'S' ORDER BY array_cosine_distance(embedding, (SELECT q FROM qvec)) LIMIT 10))
UNION ALL SELECT 'M (~6%)', (SELECT count(*) FROM (SELECT id FROM wiki WHERE first_letter = 'M' ORDER BY array_cosine_distance(embedding, (SELECT q FROM qvec)) LIMIT 10))
UNION ALL SELECT 'Z (~0.5%)', (SELECT count(*) FROM (SELECT id FROM wiki WHERE first_letter = 'Z' ORDER BY array_cosine_distance(embedding, (SELECT q FROM qvec)) LIMIT 10))
UNION ALL SELECT 'X (~0.1%)', (SELECT count(*) FROM (SELECT id FROM wiki WHERE first_letter = 'X' ORDER BY array_cosine_distance(embedding, (SELECT q FROM qvec)) LIMIT 10));

.print ''
.print 'Done.'
