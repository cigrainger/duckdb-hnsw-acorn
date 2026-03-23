-- Wiki Word Embeddings Benchmark (496k rows, 300-dim)
-- Run with: ./build/release/duckdb < test/benchmark/words_benchmark.sql

.timer on
LOAD hnsw_acorn;

.print 'Loading 496k word embeddings (300-dim)...'

CREATE TABLE words AS
SELECT
    row_number() OVER () AS id,
    words AS word,
    upper(left(words, 1)) AS first_letter,
    length(words) AS word_len,
    embedding::FLOAT[300] AS vec
FROM '/tmp/wiki_words_300d.parquet';

SELECT format('Loaded {} rows', count(*)) AS info FROM words;
SELECT first_letter, count(*) AS cnt FROM words GROUP BY first_letter ORDER BY cnt DESC LIMIT 5;

.print ''
.print 'Building HNSW index on 300-dim vectors...'

CREATE INDEX words_idx ON words USING HNSW (vec);

.print 'Index built.'

-- Query: words similar to "science"
CREATE TABLE qvec AS SELECT vec AS q FROM words WHERE word = 'science' LIMIT 1;

.print ''
.print '=== UNFILTERED: top-10 nearest to "science" ==='
SELECT word, array_distance(vec, (SELECT q FROM qvec)) AS dist
FROM words ORDER BY dist LIMIT 10;

.print ''
.print '=== FILTERED: words starting with S ==='
SELECT word, array_distance(vec, (SELECT q FROM qvec)) AS dist
FROM words WHERE first_letter = 'S' ORDER BY dist LIMIT 10;

.print ''
.print '=== FILTERED: words starting with Z ==='
SELECT word, array_distance(vec, (SELECT q FROM qvec)) AS dist
FROM words WHERE first_letter = 'Z' ORDER BY dist LIMIT 10;

.print ''
.print '=== FILTERED: short words only (len <= 4) ==='
SELECT word, array_distance(vec, (SELECT q FROM qvec)) AS dist
FROM words WHERE word_len <= 4 ORDER BY dist LIMIT 10;

.print ''
.print '=== RESULT COUNTS (all should be 10) ==='

SELECT 'S (~8%)' AS filter, (SELECT count(*) FROM (SELECT id FROM words WHERE first_letter = 'S' ORDER BY array_distance(vec, (SELECT q FROM qvec)) LIMIT 10)) AS got
UNION ALL SELECT 'B (~5%)', (SELECT count(*) FROM (SELECT id FROM words WHERE first_letter = 'B' ORDER BY array_distance(vec, (SELECT q FROM qvec)) LIMIT 10))
UNION ALL SELECT 'Z (~0.3%)', (SELECT count(*) FROM (SELECT id FROM words WHERE first_letter = 'Z' ORDER BY array_distance(vec, (SELECT q FROM qvec)) LIMIT 10))
UNION ALL SELECT 'X (~0.1%)', (SELECT count(*) FROM (SELECT id FROM words WHERE first_letter = 'X' ORDER BY array_distance(vec, (SELECT q FROM qvec)) LIMIT 10))
UNION ALL SELECT 'len<=4 (~20%)', (SELECT count(*) FROM (SELECT id FROM words WHERE word_len <= 4 ORDER BY array_distance(vec, (SELECT q FROM qvec)) LIMIT 10));

.print ''
.print 'Done.'
