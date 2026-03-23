# DuckDB-HNSW-ACORN

> **This is a fork of [duckdb/duckdb-vss](https://github.com/duckdb/duckdb-vss) that adds ACORN-1 filtered HNSW search.**
>
> The upstream extension has a critical limitation: `WHERE` clauses are applied *after* the HNSW index returns results, so `SELECT ... WHERE category = 'X' ORDER BY distance LIMIT 10` often returns fewer than 10 rows. This fork pushes filter predicates *into* the HNSW graph traversal using the [ACORN-1 algorithm](https://arxiv.org/abs/2403.04871), ensuring filtered queries return the correct number of results with high recall.
>
> **What changed:**
> - Filter predicates are evaluated during HNSW graph traversal, not after
> - ACORN-1 two-hop expansion through failed neighbors recovers graph connectivity under selective filtering
> - Selectivity-based strategy switching: >60% selectivity uses post-filter, 1-60% uses ACORN-1, <1% uses brute-force exact scan
> - Per-node expansion threshold (Lucene's 90% rule) skips two-hop when the neighborhood is already well-connected
> - Configurable thresholds: `SET hnsw_acorn_threshold = 0.6` and `SET hnsw_bruteforce_threshold = 0.01`
>
> **Benchmark (496k word embeddings, 300-dim):**
> | Filter selectivity | Upstream (post-filter) | This fork (ACORN-1) |
> |---|---|---|
> | ~8% (letter S) | 0-2 of 10 | **10 of 10** |
> | ~0.3% (letter Z) | 0 of 10 | **10 of 10** |
> | ~0.1% (letter X) | 0 of 10 | **10 of 10** |

---

*Original README follows.*

---

# DuckDB-VSS

Vector Similarity Search for DuckDB

This is an experimental extension for DuckDB that adds indexing support to accelerate Vector Similarity Search using DuckDB's new fixed-size `ARRAY` type added in version v0.10.0.
This extension is based on the [usearch](https://github.com/unum-cloud/usearch) library and serves as a proof of concept for providing a custom index type, in this case a HNSW index, from within an extension and exposing it to DuckDB.

## Filtered Search (ACORN-1)

This fork adds support for filtered vector search. Queries with `WHERE` clauses now push filter predicates into the HNSW index traversal:

```sql
-- This now returns exactly 10 results with category = 'X',
-- ordered by distance. Upstream would return 0-2 results.
SELECT * FROM my_table
WHERE category = 'X'
ORDER BY array_distance(vec, [1,2,3]::FLOAT[3])
LIMIT 10;
```

No special syntax required — the optimizer automatically detects `WHERE` + `ORDER BY distance` + `LIMIT` patterns and uses filtered HNSW search.

### Configuration

```sql
-- Selectivity above which ACORN-1 is skipped (default 0.6 = 60%)
SET hnsw_acorn_threshold = 0.6;

-- Selectivity below which brute-force exact scan is used (default 0.01 = 1%)
SET hnsw_bruteforce_threshold = 0.01;
```

## Usage

To create a new HNSW index on a table with an `ARRAY` column, use the `CREATE INDEX` statement with the `USING HNSW` clause. For example:
```sql
CREATE TABLE my_vector_table (vec FLOAT[3]);
INSERT INTO my_vector_table SELECT array_value(a,b,c) FROM range(1,10) ra(a), range(1,10) rb(b), range(1,10) rc(c);
CREATE INDEX my_hnsw_index ON my_vector_table USING HNSW (vec);
```

The index will then be used to accelerate queries that use a `ORDER BY` clause evaluating one of the supported distance metric functions against the indexed columns and a constant vector, followed by a `LIMIT` clause. For example:
```sql
SELECT * FROM my_vector_table ORDER BY array_distance(vec, [1,2,3]::FLOAT[3]) LIMIT 3;

# We can verify that the index is being used by checking the EXPLAIN output 
# and looking for the HNSW_INDEX_SCAN node in the plan

EXPLAIN SELECT * FROM my_vector_table ORDER BY array_distance(vec, [1,2,3]::FLOAT[3]) LIMIT 3;

┌───────────────────────────┐
│         PROJECTION        │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│             #0            │
└─────────────┬─────────────┘                             
┌─────────────┴─────────────┐
│         PROJECTION        │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│            vec            │
│array_distance(vec, [1.0, 2│
│         .0, 3.0])         │
└─────────────┬─────────────┘                             
┌─────────────┴─────────────┐
│      HNSW_INDEX_SCAN      │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│   t1 (HNSW INDEX SCAN :   │
│           my_idx)         │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│            vec            │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│           EC: 3           │
└───────────────────────────┘               
```

By default the HNSW index will be created using the euclidean distance `l2sq` (L2-norm squared) metric, matching DuckDBs `array_distance` function, but other distance metrics can be used by specifying the `metric` option during index creation. For example:
```sql
CREATE INDEX my_hnsw_cosine_index ON my_vector_table USING HNSW (vec) WITH (metric = 'cosine');
```

The following table shows the supported distance metrics and their corresponding DuckDB functions

| Description | Metric | Function                       |
| --- | --- |--------------------------------|
| Euclidean distance | `l2sq` | `array_distance`               |
| Cosine similarity | `cosine` | `array_cosine_distance`        |
| Inner product | `ip` | `array_negative_inner_product` |

## Inserts, Updates,  Deletes and Re-Compaction

The HNSW index does support inserting, updating and deleting rows from the table after index creation. However, there are two things to keep in mind:  
- Its faster to create the index after the table has been populated with data as the initial bulk load can make better use of parallelism on large tables.
- Deletes are not immediately reflected in the index, but are instead "marked" as deleted, which can cause the index to grow stale over time and negatively impact query quality and performance.

To address this, you can call the `PRAGMA hnsw_compact_index('<index name>')` pragma function to trigger a re-compaction of the index pruning deleted items, or re-create the index after a significant number of updates.

## Limitations 

- Only vectors consisting of `FLOAT`s are supported at the moment.
- The index itself is not buffer managed and must be able to fit into RAM memory. 

With that said, the index will be persisted into the database if you run DuckDB with a disk-backed database file. But there is no incremental updates, so every time DuckDB performs a checkpoint the entire index will be serialized to disk and overwrite its previous blocks. Similarly, the index will be deserialized back into main memory in its entirety after a restart of the database, although this will be deferred until you first access the table associated with the index. Depending on how large the index is, the deserialization process may take some time, but it should be faster than simply dropping and re-creating the index. 

---

## Building the extension

### Build steps
To build the extension, run:
```sh
make
```
The main binaries that will be built are:
```sh
./build/release/duckdb
./build/release/test/unittest
./build/release/extension/vss/vss.duckdb_extension
```
- `duckdb` is the binary for the duckdb shell with the extension code automatically loaded.
- `unittest` is the test runner of duckdb. Again, the extension is already linked into the binary.
- `vss.duckdb_extension` is the loadable binary as it would be distributed.

## Running the extension
To run the extension code, simply start the shell with `./build/release/duckdb`.

## Running the tests
Thes SQL tests can be run using:
```sh
make test
```
