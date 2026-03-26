#pragma once

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/storage_index.hpp"

namespace duckdb {

class Index;

// This is created by the optimizer rule
struct HNSWIndexScanBindData final : public TableScanBindData {
	explicit HNSWIndexScanBindData(TableCatalogEntry &table, Index &index, idx_t limit,
	                               unsafe_unique_array<float> query)
	    : TableScanBindData(table), index(index), limit(limit), query(std::move(query)) {
	}

	//! The index to use
	Index &index;

	//! The limit of the scan
	idx_t limit;

	//! The query vector
	unsafe_unique_array<float> query;

	//! Optional: table filters to push into the index scan (for filtered search).
	//! Mutable because the DuckDB scan API takes non-const optional_ptr<TableFilterSet>.
	mutable TableFilterSet table_filters;

	//! Column IDs for the filter scan. Includes all columns needed by filters
	//! (in positions matching the filter keys) plus ROW_ID at the end.
	vector<StorageIndex> filter_scan_column_ids;

	//! Types for the filter scan columns (matching filter_scan_column_ids).
	vector<LogicalType> filter_scan_types;

	//! --- Metadata join fields (optional) ---
	//! When set, the scan pre-executes a metadata table query to derive the
	//! filter bitset for ACORN-1 filtered search. This implements cross-table
	//! filtered kNN: JOIN metadata ON id WHERE metadata.col = value.

	//! The metadata table to scan for matching join keys
	optional_ptr<TableCatalogEntry> metadata_table;

	//! The join key column index in the indexed (embeddings) table
	idx_t indexed_join_column = DConstants::INVALID_INDEX;

	//! The join key column index in the metadata table
	idx_t metadata_join_column = DConstants::INVALID_INDEX;

	//! Filters to apply when scanning the metadata table
	mutable TableFilterSet metadata_filters;

	//! Column IDs for the metadata filter scan
	vector<StorageIndex> metadata_scan_column_ids;

	//! Types for the metadata filter scan columns
	vector<LogicalType> metadata_scan_types;

	bool HasMetadataJoin() const {
		return metadata_table.get() != nullptr;
	}

	//! --- Grouped TopK fields (optional) ---
	//! When set, the scan runs per-group ACORN-1 filtered search.
	//! For each distinct value of the group column, builds a filter bitset
	//! and runs a separate HNSW filtered search.
	idx_t group_column = DConstants::INVALID_INDEX;

	bool HasGroupedSearch() const {
		return group_column != DConstants::INVALID_INDEX;
	}

public:
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<HNSWIndexScanBindData>();
		return &other.table == &table;
	}

	bool HasFilters() const {
		return !table_filters.filters.empty();
	}
};

struct HNSWIndexScanFunction {
	static TableFunction GetFunction();
};

//! Extract table filters from a LogicalGet into HNSWIndexScanBindData.
//! Remaps filter keys from table column indices to storage column OIDs
//! and clears the GET's table_filters.
void ExtractFiltersIntoBind(DuckTableEntry &duck_table, LogicalGet &get, HNSWIndexScanBindData &bind_data);

} // namespace duckdb
