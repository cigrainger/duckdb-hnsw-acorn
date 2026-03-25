# DuckDB-HNSW-ACORN

A DuckDB extension for vector similarity search with **filtered HNSW** (ACORN-1) and **RaBitQ binary quantization**.

Fork of [duckdb/duckdb-vss](https://github.com/duckdb/duckdb-vss).

## Why this fork?

The upstream `duckdb-vss` extension has two limitations:

1. **Filtered search is broken.** `WHERE` clauses are applied *after* the HNSW index returns results, so `SELECT ... WHERE category = 'X' ORDER BY distance LIMIT 10` often returns fewer than 10 rows.
2. **No vector compression.** Every vector is stored as full F32, so the index memory scales linearly with dimensions.

This fork fixes both:

- **ACORN-1 filtered search** pushes filter predicates *into* HNSW graph traversal, ensuring filtered queries return the correct number of results.
- **RaBitQ quantization** compresses vectors to 1 bit per dimension (~21x memory reduction at 128 dims) with a rescore phase that preserves result quality.

## Quick start

```sql
-- Create a table with vectors
CREATE TABLE items AS
SELECT i AS id,
    array_value(random(), random(), random())::FLOAT[3] AS vec,
    (i % 5) AS category
FROM range(10000) t(i);

-- Standard HNSW index
CREATE INDEX idx ON items USING HNSW (vec);

-- Nearest neighbor search
SELECT * FROM items ORDER BY array_distance(vec, [0.5, 0.5, 0.5]::FLOAT[3]) LIMIT 10;

-- Filtered search (returns exactly 10 results matching the filter)
SELECT * FROM items
WHERE category = 1
ORDER BY array_distance(vec, [0.5, 0.5, 0.5]::FLOAT[3])
LIMIT 10;
```

No special syntax — the optimizer detects `WHERE` + `ORDER BY distance` + `LIMIT` and uses filtered HNSW search automatically.

## RaBitQ quantization

For large-dimension vectors (128+), RaBitQ dramatically reduces index memory:

```sql
CREATE INDEX idx ON items USING HNSW (vec) WITH (quantization = 'rabitq');
```

Everything else works identically — queries, filters, persistence. The index stores binary-quantized vectors and rescores candidates against the original F32 vectors for exact ranking.

| Metric | 128 dims | 256 dims | 768 dims |
|--------|----------|----------|----------|
| F32 bytes/vec | 512 | 1024 | 3072 |
| RaBitQ bytes/vec | 24 | 40 | 104 |
| **Compression** | **21x** | **26x** | **30x** |

### Configuration

```sql
-- Oversample factor: search N*oversample candidates, rescore to top-N (default 3)
SET hnsw_rabitq_oversample = 10;
```

Higher oversample = better recall, slightly slower queries.

### Benchmark (10K rows, 128 dims, L2sq)

| Method | Recall@10 | Vec Memory | Compression |
|--------|-----------|------------|-------------|
| HNSW | 66.7% | 5000 KB | — |
| RaBitQ 3x | 66.7% | 234 KB | 21.3x |
| RaBitQ 10x | 83.3% | 234 KB | 21.3x |

RaBitQ 10x achieves higher recall than plain HNSW because the rescore phase reranks with exact distances.

Run the benchmark yourself: `./build/release/duckdb < benchmarks/rabitq_benchmark.sql`

## Filtered search (ACORN-1)

Filter predicates are evaluated during HNSW graph traversal using the [ACORN-1 algorithm](https://arxiv.org/abs/2403.04871):

```sql
SELECT * FROM items
WHERE category = 'X'
ORDER BY array_distance(vec, [1,2,3]::FLOAT[3])
LIMIT 10;
```

Prepared statements work for parameterized queries:

```sql
PREPARE search AS SELECT * FROM items
WHERE category = $2
ORDER BY array_distance(vec, $1::FLOAT[3])
LIMIT 10;

EXECUTE search([1,2,3], 'X');
```

### Selectivity-based strategy

| Selectivity | Strategy |
|-------------|----------|
| >60% | Standard HNSW (post-filter) |
| 1–60% | ACORN-1 (two-hop expansion) |
| <1% | Brute-force exact scan |

Thresholds are configurable:

```sql
SET hnsw_acorn_threshold = 0.6;        -- default
SET hnsw_bruteforce_threshold = 0.01;  -- default
```

### Filtered search benchmark (228K movies, 768-dim)

| Filter | Selectivity | Upstream | ACORN-1 |
|--------|-------------|----------|---------|
| English only | ~60% | ~10/10 | **10/10** |
| Japanese only | ~3% | 0–1/10 | **10/10** |
| Korean only | ~1% | 0/10 | **10/10** |
| Rating >= 8.0 | ~5% | 0/10 | **10/10** |

## Distance metrics

| Metric | Option | Function | Operator |
|--------|--------|----------|----------|
| Euclidean (L2sq) | `l2sq` (default) | `array_distance` | `<->` |
| Cosine | `cosine` | `array_cosine_distance` | `<=>` |
| Inner product | `ip` | `array_negative_inner_product` | — |

```sql
CREATE INDEX my_idx ON items USING HNSW (vec) WITH (metric = 'cosine');
CREATE INDEX my_idx ON items USING HNSW (vec) WITH (metric = 'cosine', quantization = 'rabitq');
```

## Index options

```sql
CREATE INDEX idx ON items USING HNSW (vec) WITH (
    metric = 'l2sq',           -- distance metric (l2sq, cosine, ip)
    quantization = 'rabitq',   -- vector quantization (rabitq, none)
    ef_construction = 200,     -- graph construction search width
    ef_search = 200,           -- query-time search width
    M = 16,                    -- max connections per node
    M0 = 32                    -- max connections at layer 0
);
```

Runtime settings:

```sql
SET hnsw_ef_search = 100;              -- override ef_search at query time
SET hnsw_rabitq_oversample = 10;       -- RaBitQ rescore oversample factor
SET hnsw_acorn_threshold = 0.6;        -- ACORN-1 selectivity threshold
SET hnsw_bruteforce_threshold = 0.01;  -- brute-force selectivity threshold
```

## Index inspection

```sql
SELECT * FROM pragma_hnsw_index_info();
```

Returns: index name, table, metric, dimensions, count, capacity, memory usage, quantization type, bytes per vector, vector memory usage, levels, and per-level statistics.

## Inserts, updates, deletes

The index supports mutations after creation. For best performance, create the index after bulk loading data.

Deletes are lazily marked — run `PRAGMA hnsw_compact_index('idx')` to reclaim space.

## Persistence

```sql
SET hnsw_enable_experimental_persistence = true;
```

Indexes persist across restarts when using a disk-backed database. The full index is serialized on checkpoint and deserialized on first access.

## Limitations

- Only `FLOAT` array types are supported.
- The index must fit in RAM (not buffer-managed).
- Subquery query vectors fall back to sequential scan — use literals or prepared statements.
- RaBitQ with inner product metric has lower recall than L2sq or cosine (the L2sq graph proxy doesn't align well with IP ordering).

## Building

```sh
make            # release build
make debug      # debug build
make test       # run all tests
```

Binaries:

```
./build/release/duckdb                                    # shell with extension loaded
./build/release/test/unittest                             # test runner
./build/release/extension/hnsw_acorn/hnsw_acorn.duckdb_extension  # loadable extension
```
