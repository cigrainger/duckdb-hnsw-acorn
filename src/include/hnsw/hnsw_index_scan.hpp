#pragma once

#include "duckdb/common/helper.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/table/table_scan.hpp"
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

	//! Optional: table filters to push into the index scan (for filtered search)
	//! The filter keys match positions in filter_scan_column_ids.
	TableFilterSet table_filters;

	//! Column IDs for the filter scan. Includes all columns needed by filters
	//! (in positions matching the filter keys) plus ROW_ID at the end.
	vector<StorageIndex> filter_scan_column_ids;

	//! Types for the filter scan columns (matching filter_scan_column_ids).
	vector<LogicalType> filter_scan_types;

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

} // namespace duckdb
