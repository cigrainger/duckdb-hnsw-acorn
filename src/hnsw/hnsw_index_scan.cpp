#include "hnsw/hnsw_index_scan.hpp"

#include <unordered_set>

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/dependency_list.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/storage_index.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"

namespace duckdb {

void ExtractFiltersIntoBind(DuckTableEntry &duck_table, LogicalGet &get, HNSWIndexScanBindData &bind_data) {
	if (get.table_filters.filters.empty()) {
		return;
	}
	idx_t scan_pos = 0;
	unordered_map<idx_t, idx_t> key_remap;
	for (const auto &entry : get.table_filters.filters) {
		auto table_col_idx = entry.first;
		auto &col = duck_table.GetColumn(LogicalIndex(table_col_idx));
		bind_data.filter_scan_column_ids.emplace_back(StorageIndex(col.StorageOid()));
		bind_data.filter_scan_types.push_back(col.GetType());
		key_remap[table_col_idx] = scan_pos++;
	}
	for (auto &entry : get.table_filters.filters) {
		bind_data.table_filters.filters[key_remap[entry.first]] = entry.second->Copy();
	}
	get.table_filters.filters.clear();
}

BindInfo HNSWIndexScanBindInfo(const optional_ptr<FunctionData> bind_data_p) {
	auto &bind_data = bind_data_p->Cast<HNSWIndexScanBindData>();
	return BindInfo(bind_data.table);
}

//-------------------------------------------------------------------------
// Global State
//-------------------------------------------------------------------------
struct HNSWIndexScanGlobalState : public GlobalTableFunctionState {
	ColumnFetchState fetch_state;
	TableScanState local_storage_state;
	vector<StorageIndex> column_ids;

	// Index scan state
	unique_ptr<IndexScanState> index_state;
	Vector row_ids = Vector(LogicalType::ROW_TYPE);

	DataChunk all_columns;
	vector<idx_t> projection_ids;
};

static unique_ptr<GlobalTableFunctionState> HNSWIndexScanInitGlobal(ClientContext &context,
                                                                    TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<HNSWIndexScanBindData>();

	auto result = make_uniq<HNSWIndexScanGlobalState>();

	// Setup the scan state for the local storage
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	result->column_ids.reserve(input.column_ids.size());

	// Figure out the storage column ids
	for (auto &id : input.column_ids) {
		storage_t col_id = id;
		if (id != DConstants::INVALID_INDEX) {
			col_id = bind_data.table.GetColumn(LogicalIndex(id)).StorageOid();
		}
		result->column_ids.emplace_back(col_id);
	}

	// Initialize the storage scan state
	result->local_storage_state.Initialize(result->column_ids, context, input.filters);
	local_storage.InitializeScan(bind_data.table.GetStorage(), result->local_storage_state.local_state, input.filters);

	// Initialize the scan state for the index
	auto &hnsw_index = bind_data.index.Cast<HNSWIndex>();

	if (bind_data.HasMetadataJoin()) {
		// Metadata join: scan the metadata table with its filters to get matching
		// join keys, then look up which row_ids in the indexed table have those
		// keys, and build a filter bitset for ACORN-1 filtered search.
		auto &data_table = bind_data.table.GetStorage();
		auto &transaction = DuckTransaction::Get(context, bind_data.table.catalog);
		auto &meta_table_entry = const_cast<DuckTableEntry &>(bind_data.metadata_table->Cast<DuckTableEntry>());
		auto &meta_data_table = meta_table_entry.GetStorage();

		// Step 1: Scan metadata table with filters → collect matching join key values
		auto meta_scan_col_ids = bind_data.metadata_scan_column_ids;
		auto meta_scan_types = bind_data.metadata_scan_types;

		TableScanState meta_scan_state;
		meta_scan_state.Initialize(meta_scan_col_ids);
		meta_data_table.InitializeScan(context, transaction, meta_scan_state, meta_scan_col_ids,
		                               &bind_data.metadata_filters);

		unordered_set<int64_t> matching_keys;
		DataChunk meta_chunk;
		meta_chunk.Initialize(context, meta_scan_types);
		while (true) {
			meta_chunk.Reset();
			meta_data_table.Scan(transaction, meta_chunk, meta_scan_state);
			if (meta_chunk.size() == 0) {
				break;
			}
			meta_chunk.Flatten();
			// Join key is at position 0 in our scan columns
			auto key_data = FlatVector::GetData<int64_t>(meta_chunk.data[0]);
			for (idx_t i = 0; i < meta_chunk.size(); i++) {
				matching_keys.insert(key_data[i]);
			}
		}

		// Step 2: Scan the indexed table's join column + ROW_ID to find matching row_ids
		auto &idx_duck_table = bind_data.table.Cast<DuckTableEntry>();
		auto join_col_storage_id = idx_duck_table.GetColumn(
		    LogicalIndex(bind_data.indexed_join_column)).StorageOid();
		vector<StorageIndex> idx_scan_col_ids = {StorageIndex(join_col_storage_id)};
		vector<LogicalType> idx_scan_types = {
		    idx_duck_table.GetColumn(LogicalIndex(bind_data.indexed_join_column)).GetType()};
		idx_scan_col_ids.emplace_back(StorageIndex(COLUMN_IDENTIFIER_ROW_ID));
		idx_scan_types.push_back(LogicalType::ROW_TYPE);

		TableScanState idx_scan_state;
		idx_scan_state.Initialize(idx_scan_col_ids);
		data_table.InitializeScan(context, transaction, idx_scan_state, idx_scan_col_ids);

		auto total_rows = data_table.GetTotalRows();
		vector<uint64_t> filter_bitset((total_rows / 64) + 1, 0);

		DataChunk idx_chunk;
		idx_chunk.Initialize(context, idx_scan_types);
		while (true) {
			idx_chunk.Reset();
			data_table.Scan(transaction, idx_chunk, idx_scan_state);
			if (idx_chunk.size() == 0) {
				break;
			}
			idx_chunk.Flatten();
			auto key_data = FlatVector::GetData<int64_t>(idx_chunk.data[0]);
			auto rid_data = FlatVector::GetData<row_t>(idx_chunk.data[1]);
			for (idx_t i = 0; i < idx_chunk.size(); i++) {
				if (matching_keys.count(key_data[i])) {
					auto rid = static_cast<idx_t>(rid_data[i]);
					auto word = rid / 64;
					if (word >= filter_bitset.size()) {
						filter_bitset.resize(word + 1, 0);
					}
					filter_bitset[word] |= (1ULL << (rid % 64));
				}
			}
		}

		result->index_state = hnsw_index.InitializeFilteredScan(bind_data.query.get(), bind_data.limit,
		                                                        std::move(filter_bitset), context);
	} else if (bind_data.HasFilters()) {
		// Build a filter bitset by scanning the table with the filter predicates
		auto &data_table = bind_data.table.GetStorage();
		auto &transaction = DuckTransaction::Get(context, bind_data.table.catalog);

		// Build scan column IDs: filter columns + ROW_ID (last).
		auto filter_scan_col_ids = bind_data.filter_scan_column_ids;
		auto filter_scan_types = bind_data.filter_scan_types;
		filter_scan_col_ids.emplace_back(StorageIndex(COLUMN_IDENTIFIER_ROW_ID));
		filter_scan_types.push_back(LogicalType::ROW_TYPE);

		// Pass table filters to InitializeScan so DuckDB uses zone maps to skip
		// non-matching row groups and evaluates filters per-row natively.
		// This avoids a full table scan — only matching rows are returned.
		TableScanState scan_state;
		scan_state.Initialize(filter_scan_col_ids);
		data_table.InitializeScan(context, transaction, scan_state, filter_scan_col_ids, &bind_data.table_filters);

		auto total_rows = data_table.GetTotalRows();
		vector<uint64_t> filter_bitset((total_rows / 64) + 1, 0);

		DataChunk scan_chunk;
		scan_chunk.Initialize(context, filter_scan_types);

		// ROW_ID is the last column. All rows returned by Scan already
		// pass the filter — just collect their row_ids into the bitset.
		auto row_id_col_idx = filter_scan_types.size() - 1;
		while (true) {
			scan_chunk.Reset();
			data_table.Scan(transaction, scan_chunk, scan_state);
			if (scan_chunk.size() == 0) {
				break;
			}
			auto row_id_data = FlatVector::GetData<row_t>(scan_chunk.data[row_id_col_idx]);
			for (idx_t i = 0; i < scan_chunk.size(); i++) {
				auto rid = static_cast<idx_t>(row_id_data[i]);
				auto word = rid / 64;
				if (word >= filter_bitset.size()) {
					filter_bitset.resize(word + 1, 0);
				}
				filter_bitset[word] |= (1ULL << (rid % 64));
			}
		}

		result->index_state = hnsw_index.InitializeFilteredScan(bind_data.query.get(), bind_data.limit,
		                                                        std::move(filter_bitset), context);
	} else {
		result->index_state = hnsw_index.InitializeScan(bind_data.query.get(), bind_data.limit, context);
	}

	// For RaBitQ indexes, rescore oversampled candidates against original F32 vectors
	if (hnsw_index.IsRaBitQ() && result->index_state) {
		HNSWIndex::RescoreRaBitQCandidates(*result->index_state, hnsw_index,
		                                   bind_data.table.Cast<DuckTableEntry>(), bind_data.limit, context);
	}

	if (!input.CanRemoveFilterColumns()) {
		return std::move(result);
	}

	// We need this to project out what we scan from the underlying table.
	result->projection_ids = input.projection_ids;

	auto &duck_table = bind_data.table.Cast<DuckTableEntry>();
	const auto &columns = duck_table.GetColumns();
	vector<LogicalType> scanned_types;
	for (const auto &col_idx : input.column_indexes) {
		if (col_idx.IsRowIdColumn()) {
			scanned_types.emplace_back(LogicalType::ROW_TYPE);
		} else {
			scanned_types.push_back(columns.GetColumn(col_idx.ToLogical()).Type());
		}
	}
	result->all_columns.Initialize(context, scanned_types);

	return std::move(result);
}

//-------------------------------------------------------------------------
// Execute
//-------------------------------------------------------------------------
static void HNSWIndexScanExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {

	auto &bind_data = data_p.bind_data->Cast<HNSWIndexScanBindData>();
	auto &state = data_p.global_state->Cast<HNSWIndexScanGlobalState>();
	auto &transaction = DuckTransaction::Get(context, bind_data.table.catalog);

	// Scan the index for row id's
	auto row_count = bind_data.index.Cast<HNSWIndex>().Scan(*state.index_state, state.row_ids);
	if (row_count == 0) {
		// Short-circuit if the index had no more rows
		output.SetCardinality(0);
		return;
	}

	// Fetch the data from the local storage given the row ids
	if (state.projection_ids.empty()) {
		bind_data.table.GetStorage().Fetch(transaction, output, state.column_ids, state.row_ids, row_count,
		                                   state.fetch_state);
		return;
	}

	// Otherwise, we need to first fetch into our scan chunk, and then project out the result
	state.all_columns.Reset();
	bind_data.table.GetStorage().Fetch(transaction, state.all_columns, state.column_ids, state.row_ids, row_count,
	                                   state.fetch_state);
	output.ReferenceColumns(state.all_columns, state.projection_ids);
}

//-------------------------------------------------------------------------
// Statistics
//-------------------------------------------------------------------------
static unique_ptr<BaseStatistics> HNSWIndexScanStatistics(ClientContext &context, const FunctionData *bind_data_p,
                                                          column_t column_id) {
	auto &bind_data = bind_data_p->Cast<HNSWIndexScanBindData>();
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	if (local_storage.Find(bind_data.table.GetStorage())) {
		// we don't emit any statistics for tables that have outstanding transaction-local data
		return nullptr;
	}
	return bind_data.table.GetStatistics(context, column_id);
}

//-------------------------------------------------------------------------
// Dependency
//-------------------------------------------------------------------------
void HNSWIndexScanDependency(LogicalDependencyList &entries, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<HNSWIndexScanBindData>();
	entries.AddDependency(bind_data.table);

	// TODO: Add dependency to index here?
}

//-------------------------------------------------------------------------
// Cardinality
//-------------------------------------------------------------------------
unique_ptr<NodeStatistics> HNSWIndexScanCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<HNSWIndexScanBindData>();
	return make_uniq<NodeStatistics>(bind_data.limit, bind_data.limit);
}

//-------------------------------------------------------------------------
// ToString
//-------------------------------------------------------------------------
static InsertionOrderPreservingMap<string> HNSWIndexScanToString(TableFunctionToStringInput &input) {
	D_ASSERT(input.bind_data);
	InsertionOrderPreservingMap<string> result;
	auto &bind_data = input.bind_data->Cast<HNSWIndexScanBindData>();
	result["Table"] = bind_data.table.name;
	result["HNSW Index"] = bind_data.index.GetIndexName();
	return result;
}

//-------------------------------------------------------------------------
// Get Function
//-------------------------------------------------------------------------
TableFunction HNSWIndexScanFunction::GetFunction() {
	TableFunction func("hnsw_index_scan", {}, HNSWIndexScanExecute);
	func.init_local = nullptr;
	func.init_global = HNSWIndexScanInitGlobal;
	func.statistics = HNSWIndexScanStatistics;
	func.dependency = HNSWIndexScanDependency;
	func.cardinality = HNSWIndexScanCardinality;
	func.pushdown_complex_filter = nullptr;
	func.to_string = HNSWIndexScanToString;
	func.table_scan_progress = nullptr;
	func.projection_pushdown = true;
	func.filter_pushdown = false;
	func.get_bind_info = HNSWIndexScanBindInfo;

	return func;
}

//-------------------------------------------------------------------------
// Register
//-------------------------------------------------------------------------
void HNSWModule::RegisterIndexScan(ExtensionLoader &loader) {
	loader.RegisterFunction(HNSWIndexScanFunction::GetFunction());
}

} // namespace duckdb
