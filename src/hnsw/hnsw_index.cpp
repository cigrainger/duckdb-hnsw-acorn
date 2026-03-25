#include "hnsw/hnsw_index.hpp"

#include <algorithm>

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/common/column_index.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/enums/index_constraint_type.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/validity_mask.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_size.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/execution/index/index_type.hpp"
#include "duckdb/execution/index/index_type_set.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/setting_info.hpp"
#include "duckdb/optimizer/matcher/expression_type_matcher.hpp"
#include "duckdb/optimizer/matcher/function_matcher.hpp"
#include "duckdb/optimizer/matcher/set_matcher.hpp"
#include "duckdb/optimizer/matcher/type_matcher.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/partial_block_manager.hpp"
#include "duckdb/storage/storage_info.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "hnsw/hnsw.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "usearch/duckdb_usearch.hpp"

namespace duckdb {
//------------------------------------------------------------------------------
// Linked Blocks
//------------------------------------------------------------------------------

class LinkedBlock {
public:
	static constexpr const idx_t BLOCK_SIZE = Storage::DEFAULT_BLOCK_SIZE - sizeof(validity_t);
	static constexpr const idx_t BLOCK_DATA_SIZE = BLOCK_SIZE - sizeof(IndexPointer);
	static_assert(BLOCK_SIZE > sizeof(IndexPointer), "Block size must be larger than the size of an IndexPointer");

	IndexPointer next_block;
	char data[BLOCK_DATA_SIZE] = {0};
};

constexpr idx_t LinkedBlock::BLOCK_DATA_SIZE;
constexpr idx_t LinkedBlock::BLOCK_SIZE;

class LinkedBlockReader {
private:
	FixedSizeAllocator &allocator;

	IndexPointer root_pointer;
	IndexPointer current_pointer;
	idx_t position_in_block;

public:
	LinkedBlockReader(FixedSizeAllocator &allocator, IndexPointer root_pointer)
	    : allocator(allocator), root_pointer(root_pointer), current_pointer(root_pointer), position_in_block(0) {
	}

	void Reset() {
		current_pointer = root_pointer;
		position_in_block = 0;
	}

	idx_t ReadData(data_ptr_t buffer, idx_t length) {
		idx_t bytes_read = 0;
		while (bytes_read < length) {

			// TODO: Check if current pointer is valid

			auto block = allocator.Get<const LinkedBlock>(current_pointer, false);
			auto block_data = block->data;
			auto data_to_read = std::min(length - bytes_read, LinkedBlock::BLOCK_DATA_SIZE - position_in_block);
			std::memcpy(buffer + bytes_read, block_data + position_in_block, data_to_read);

			bytes_read += data_to_read;
			position_in_block += data_to_read;

			if (position_in_block == LinkedBlock::BLOCK_DATA_SIZE) {
				position_in_block = 0;
				current_pointer = block->next_block;
			}
		}

		return bytes_read;
	}
};

class LinkedBlockWriter {
private:
	FixedSizeAllocator &allocator;

	IndexPointer root_pointer;
	IndexPointer current_pointer;
	idx_t position_in_block;

public:
	LinkedBlockWriter(FixedSizeAllocator &allocator, IndexPointer root_pointer)
	    : allocator(allocator), root_pointer(root_pointer), current_pointer(root_pointer), position_in_block(0) {
	}

	void ClearCurrentBlock() {
		auto block = allocator.Get<LinkedBlock>(current_pointer, true);
		block->next_block.Clear();
		memset(block->data, 0, LinkedBlock::BLOCK_DATA_SIZE);
	}

	void Reset() {
		current_pointer = root_pointer;
		position_in_block = 0;
		ClearCurrentBlock();
	}

	void WriteData(const_data_ptr_t buffer, idx_t length) {
		idx_t bytes_written = 0;
		while (bytes_written < length) {
			auto block = allocator.Get<LinkedBlock>(current_pointer, true);
			auto block_data = block->data;
			auto data_to_write = std::min(length - bytes_written, LinkedBlock::BLOCK_DATA_SIZE - position_in_block);
			std::memcpy(block_data + position_in_block, buffer + bytes_written, data_to_write);

			bytes_written += data_to_write;
			position_in_block += data_to_write;

			if (position_in_block == LinkedBlock::BLOCK_DATA_SIZE) {
				position_in_block = 0;
				block->next_block = allocator.New();
				current_pointer = block->next_block;
				ClearCurrentBlock();
			}
		}
	}
};

//------------------------------------------------------------------------------
// HNSWIndex Methods
//------------------------------------------------------------------------------

// Constructor
HNSWIndex::HNSWIndex(const string &name, IndexConstraintType index_constraint_type, const vector<column_t> &column_ids,
                     TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
                     AttachedDatabase &db, const case_insensitive_map_t<Value> &options, const IndexStorageInfo &info,
                     idx_t estimated_cardinality)
    : BoundIndex(name, TYPE_NAME, index_constraint_type, column_ids, table_io_manager, unbound_expressions, db) {

	if (index_constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("HNSW indexes do not support unique or primary key constraints");
	}

	// Create a allocator for the linked blocks
	auto &block_manager = table_io_manager.GetIndexBlockManager();
	linked_block_allocator = make_uniq<FixedSizeAllocator>(sizeof(LinkedBlock), block_manager);

	// We only support one ARRAY column
	D_ASSERT(logical_types.size() == 1);
	auto &vector_type = logical_types[0];
	D_ASSERT(vector_type.id() == LogicalTypeId::ARRAY);

	// Get the size of the vector
	auto vector_size = ArrayType::GetSize(vector_type);
	auto vector_child_type = ArrayType::GetChildType(vector_type);

	// Get the scalar kind from the array child type. This parameter should be verified during binding.
	auto scalar_kind = unum::usearch::scalar_kind_t::f32_k;
	auto scalar_kind_val = SCALAR_KIND_MAP.find(static_cast<uint8_t>(vector_child_type.id()));
	if (scalar_kind_val != SCALAR_KIND_MAP.end()) {
		scalar_kind = scalar_kind_val->second;
	}

	// Try to get the vector metric from the options, this parameter should be verified during binding.
	auto metric_kind = unum::usearch::metric_kind_t::l2sq_k;
	auto metric_kind_opt = options.find("metric");
	if (metric_kind_opt != options.end()) {
		auto metric_kind_val = METRIC_KIND_MAP.find(metric_kind_opt->second.GetValue<string>());
		if (metric_kind_val != METRIC_KIND_MAP.end()) {
			metric_kind = metric_kind_val->second;
		}
	}

	// Check for RaBitQ quantization option
	auto quantization_opt = options.find("quantization");
	if (quantization_opt != options.end()) {
		auto q = quantization_opt->second.GetValue<string>();
		if (StringUtil::CIEquals(q, "rabitq")) {
			is_rabitq_ = true;
		}
	}

	// Create the usearch metric — either standard F32 or RaBitQ
	unum::usearch::metric_punned_t metric;

	if (is_rabitq_) {
		// Initialize RaBitQ distance context (lives as long as the index)
		rabitq_distance_ctx_ = make_uniq<RaBitQDistanceContext>();
		rabitq_distance_ctx_->dimensions = vector_size;
		rabitq_distance_ctx_->binary_bytes = (vector_size + 7) / 8;

		// RaBitQ always uses L2sq for graph construction/traversal.
		// The rescore phase computes the exact distance for the original metric.
		// L2sq on centroid-subtracted residuals is a good proxy for all metrics
		// because it captures neighborhood structure regardless of final metric.
		auto distance_fn = reinterpret_cast<std::uintptr_t>(&RaBitQDistanceL2sq);

		metric = unum::usearch::metric_punned_t::custom(
		    vector_size, RaBitQBytesPerVector(vector_size), distance_fn,
		    reinterpret_cast<std::uintptr_t>(rabitq_distance_ctx_.get()), metric_kind);

		// Initialize RaBitQ state (centroid will be set during construction or loaded from disk)
		rabitq_state_ = make_uniq<RaBitQState>();
		rabitq_state_->dimensions = vector_size;
		rabitq_state_->original_metric = metric_kind;
	} else {
		metric = unum::usearch::metric_punned_t(vector_size, metric_kind, scalar_kind);
	}
	unum::usearch::index_dense_config_t config = {};

	// We dont need to do key lookups (id -> vector) in the index, DuckDB stores the vectors separately
	config.enable_key_lookups = false;

	auto ef_construction_opt = options.find("ef_construction");
	if (ef_construction_opt != options.end()) {
		config.expansion_add = ef_construction_opt->second.GetValue<int32_t>();
	}

	auto ef_search_opt = options.find("ef_search");
	if (ef_search_opt != options.end()) {
		config.expansion_search = ef_search_opt->second.GetValue<int32_t>();
	}

	auto m_opt = options.find("m");
	if (m_opt != options.end()) {
		config.connectivity = m_opt->second.GetValue<int32_t>();
		config.connectivity_base = config.connectivity * 2;
	}

	auto m0_opt = options.find("m0");
	if (m0_opt != options.end()) {
		config.connectivity_base = m0_opt->second.GetValue<int32_t>();
	}

	index = unum::usearch::index_dense_gt<row_t>::make(metric, config);

	auto lock = rwlock.GetExclusiveLock();
	// Is this a new index or an existing index?
	if (info.IsValid()) {
		// This is an old index that needs to be loaded

		// Set the root node
		root_block_ptr.Set(info.root);
		D_ASSERT(info.allocator_infos.size() == 1);
		linked_block_allocator->Init(info.allocator_infos[0]);

		// Is there anything to deserialize? We could have an empty index
		if (!info.allocator_infos[0].buffer_ids.empty()) {
			LinkedBlockReader reader(*linked_block_allocator, root_block_ptr);

			// Detect RaBitQ stream: RaBitQ indexes start with magic byte 'R' (0x52).
			// Legacy usearch streams start with uint32 matrix_rows (little-endian),
			// whose low byte is the row count mod 256 — never 'R' (0x52 = 82) for
			// any practical index.
			uint8_t first_byte = 0;
			reader.ReadData(reinterpret_cast<data_ptr_t>(&first_byte), sizeof(first_byte));

			bool stream_is_rabitq = (first_byte == 'R');

			if (stream_is_rabitq) {
				// Initialize RaBitQ state (options may not survive restart)
				if (!is_rabitq_) {
					is_rabitq_ = true;
					rabitq_distance_ctx_ = make_uniq<RaBitQDistanceContext>();
					rabitq_distance_ctx_->dimensions = vector_size;
					rabitq_distance_ctx_->binary_bytes = (vector_size + 7) / 8;

					auto distance_fn = reinterpret_cast<std::uintptr_t>(&RaBitQDistanceL2sq);

					auto rabitq_metric = unum::usearch::metric_punned_t::custom(
					    vector_size, RaBitQBytesPerVector(vector_size), distance_fn,
					    reinterpret_cast<std::uintptr_t>(rabitq_distance_ctx_.get()), metric_kind);

					rabitq_state_ = make_uniq<RaBitQState>();
					rabitq_state_->dimensions = vector_size;
					rabitq_state_->original_metric = metric_kind;

					// Rebuild the index with the RaBitQ metric
					index = unum::usearch::index_dense_gt<row_t>::make(rabitq_metric, config);
				}

				uint64_t centroid_dims = 0;
				reader.ReadData(reinterpret_cast<data_ptr_t>(&centroid_dims), sizeof(centroid_dims));
				rabitq_state_->centroid.resize(centroid_dims);
				reader.ReadData(reinterpret_cast<data_ptr_t>(rabitq_state_->centroid.data()),
				                centroid_dims * sizeof(float));
			}

			// Load the usearch index stream.
			// For legacy (non-RaBitQ) format, we consumed the first byte which is
			// part of the usearch stream — inject it back on the first read call.
			bool inject_first_byte = !stream_is_rabitq;
			USearchIndexType::serialization_config_t sconfig;
			sconfig.preserve_metric = is_rabitq_;

			index.load_from_stream(
			    [&](void *data, size_t size) -> bool {
				    if (inject_first_byte) {
					    inject_first_byte = false;
					    auto ptr = static_cast<data_ptr_t>(data);
					    ptr[0] = first_byte;
					    if (size == 1) {
						    return true;
					    }
					    return (reader.ReadData(ptr + 1, size - 1) + 1) == size;
				    }
				    return size == reader.ReadData(static_cast<data_ptr_t>(data), size);
			    },
			    sconfig);
		}
	} else {
		index.reserve(MinValue(static_cast<idx_t>(32), estimated_cardinality));
	}
	index_size = index.size();

	function_matcher = MakeFunctionMatcher();
}

idx_t HNSWIndex::GetVectorSize() const {
	return index.dimensions();
}

string HNSWIndex::GetMetric() const {
	switch (index.metric().metric_kind()) {
	case unum::usearch::metric_kind_t::l2sq_k:
		return "l2sq";
	case unum::usearch::metric_kind_t::cos_k:
		return "cosine";
	case unum::usearch::metric_kind_t::ip_k:
		return "ip";
	default:
		throw InternalException("Unknown metric kind");
	}
}

const case_insensitive_map_t<unum::usearch::metric_kind_t> HNSWIndex::METRIC_KIND_MAP = {
    {"l2sq", unum::usearch::metric_kind_t::l2sq_k},
    {"cosine", unum::usearch::metric_kind_t::cos_k},
    {"ip", unum::usearch::metric_kind_t::ip_k},
    /* TODO: Add the rest of these later
    {"divergence", unum::usearch::metric_kind_t::divergence_k},
    {"hamming", unum::usearch::metric_kind_t::hamming_k},
    {"jaccard", unum::usearch::metric_kind_t::jaccard_k},
    {"haversine", unum::usearch::metric_kind_t::haversine_k},
    {"pearson", unum::usearch::metric_kind_t::pearson_k},
    {"sorensen", unum::usearch::metric_kind_t::sorensen_k},
    {"tanimoto", unum::usearch::metric_kind_t::tanimoto_k}
     */
};

const unordered_map<uint8_t, unum::usearch::scalar_kind_t> HNSWIndex::SCALAR_KIND_MAP = {
    {static_cast<uint8_t>(LogicalTypeId::FLOAT), unum::usearch::scalar_kind_t::f32_k},
    /* TODO: Add the rest of these later
    {static_cast<uint8_t>(LogicalTypeId::DOUBLE), unum::usearch::scalar_kind_t::f64_k},
    {static_cast<uint8_t>(LogicalTypeId::TINYINT), unum::usearch::scalar_kind_t::i8_k},
    {static_cast<uint8_t>(LogicalTypeId::SMALLINT), unum::usearch::scalar_kind_t::i16_k},
    {static_cast<uint8_t>(LogicalTypeId::INTEGER), unum::usearch::scalar_kind_t::i32_k},
    {static_cast<uint8_t>(LogicalTypeId::BIGINT), unum::usearch::scalar_kind_t::i64_k},
    {static_cast<uint8_t>(LogicalTypeId::UTINYINT), unum::usearch::scalar_kind_t::u8_k},
    {static_cast<uint8_t>(LogicalTypeId::USMALLINT), unum::usearch::scalar_kind_t::u16_k},
    {static_cast<uint8_t>(LogicalTypeId::UINTEGER), unum::usearch::scalar_kind_t::u32_k},
    {static_cast<uint8_t>(LogicalTypeId::UBIGINT), unum::usearch::scalar_kind_t::u64_k}
    */
};

unique_ptr<HNSWIndexStats> HNSWIndex::GetStats() {
	auto lock = rwlock.GetExclusiveLock();
	auto result = make_uniq<HNSWIndexStats>();

	result->max_level = index.max_level();
	result->count = index.size();
	result->capacity = index.capacity();
	result->approx_size = index.memory_usage();

	for (idx_t i = 0; i < index.max_level(); i++) {
		result->level_stats.push_back(index.stats(i));
	}

	return result;
}

// Scan State
struct HNSWIndexScanState : public IndexScanState {
	idx_t current_row = 0;
	idx_t total_rows = 0;
	unique_array<row_t> row_ids = nullptr;

	// RaBitQ rescore fields
	bool needs_rescore = false;
	unique_array<float> query_vector;
	idx_t vector_dimensions = 0;
	unum::usearch::metric_kind_t metric_kind = unum::usearch::metric_kind_t::l2sq_k;
};

static idx_t GetEfSearch(const HNSWIndex &hnsw_index, ClientContext &context) {
	auto ef_search = hnsw_index.index.expansion_search();
	Value hnsw_ef_search_opt;
	if (context.TryGetCurrentSetting("hnsw_ef_search", hnsw_ef_search_opt)) {
		if (!hnsw_ef_search_opt.IsNull() && hnsw_ef_search_opt.type() == LogicalType::BIGINT) {
			auto val = hnsw_ef_search_opt.GetValue<int64_t>();
			if (val > 0) {
				ef_search = static_cast<idx_t>(val);
			}
		}
	}
	return ef_search;
}

static idx_t GetRaBitQOversample(ClientContext &context) {
	idx_t oversample = 3;
	Value oversample_opt;
	if (context.TryGetCurrentSetting("hnsw_rabitq_oversample", oversample_opt)) {
		if (!oversample_opt.IsNull()) {
			auto val = oversample_opt.GetValue<int64_t>();
			if (val > 0) {
				oversample = static_cast<idx_t>(val);
			}
		}
	}
	return oversample;
}

unique_ptr<IndexScanState> HNSWIndex::InitializeScan(float *query_vector, idx_t limit, ClientContext &context) {
	auto state = make_uniq<HNSWIndexScanState>();
	auto ef_search = GetEfSearch(*this, context);

	auto lock = rwlock.GetSharedLock();

	if (is_rabitq_) {
		// RaBitQ path: quantize query, oversample, mark for rescore
		auto oversample = GetRaBitQOversample(context);
		auto search_limit = limit * oversample;
		bool normalize = rabitq_state_->original_metric == unum::usearch::metric_kind_t::cos_k;

		// Quantize query vector
		auto bbq_size = RaBitQBytesPerVector(rabitq_state_->dimensions);
		auto bbq_query = make_uniq_array<uint8_t>(bbq_size);
		RaBitQQuantizeVector(query_vector, *rabitq_state_, bbq_query.get(), normalize);

		auto search_result =
		    index.ef_search(reinterpret_cast<const unum::usearch::b1x8_t *>(bbq_query.get()), search_limit, ef_search);

		state->current_row = 0;
		state->total_rows = search_result.size();
		state->row_ids = make_uniq_array<row_t>(search_result.size());
		search_result.dump_to(state->row_ids.get());

		// Store info for rescore
		state->needs_rescore = true;
		state->vector_dimensions = rabitq_state_->dimensions;
		state->metric_kind = rabitq_state_->original_metric;
		state->query_vector = make_uniq_array<float>(rabitq_state_->dimensions);
		memcpy(state->query_vector.get(), query_vector, rabitq_state_->dimensions * sizeof(float));
	} else {
		auto search_result = index.ef_search(query_vector, limit, ef_search);

		state->current_row = 0;
		state->total_rows = search_result.size();
		state->row_ids = make_uniq_array<row_t>(search_result.size());
		search_result.dump_to(state->row_ids.get());
	}

	return std::move(state);
}

unique_ptr<IndexScanState> HNSWIndex::InitializeFilteredScan(float *query_vector, idx_t limit,
                                                             vector<uint64_t> filter_bitset, ClientContext &context) {
	auto state = make_uniq<HNSWIndexScanState>();
	auto ef_search = GetEfSearch(*this, context);

	// For RaBitQ, oversample the limit for the search phase
	auto search_limit = limit;
	if (is_rabitq_) {
		auto oversample = GetRaBitQOversample(context);
		search_limit = limit * oversample;
	}

	// Compute selectivity from bitset
	idx_t popcount = 0;
	for (auto &w : filter_bitset) {
		// Portable popcount: works on GCC, Clang, and MSVC
#if defined(__GNUC__) || defined(__clang__)
		popcount += __builtin_popcountll(w);
#elif defined(_MSC_VER)
		popcount += __popcnt64(w);
#else
		auto v = w;
		while (v) {
			popcount++;
			v &= v - 1;
		}
#endif
	}
	// Use index size as denominator — matches the keys the predicate will be checked against.
	// Clamp to [0, 1] since popcount (from table scan) and index.size() can diverge after deletes.
	auto total = index.size();
	float selectivity = total > 0 ? static_cast<float>(popcount) / static_cast<float>(total) : 0.0f;
	if (selectivity > 1.0f) {
		selectivity = 1.0f;
	}

	// Get selectivity thresholds (configurable via SET)
	float acorn_threshold = 0.6f;
	float bruteforce_threshold = 0.01f;

	Value acorn_thresh_opt;
	if (context.TryGetCurrentSetting("hnsw_acorn_threshold", acorn_thresh_opt)) {
		if (!acorn_thresh_opt.IsNull()) {
			acorn_threshold = acorn_thresh_opt.GetValue<float>();
		}
	}
	Value bf_thresh_opt;
	if (context.TryGetCurrentSetting("hnsw_bruteforce_threshold", bf_thresh_opt)) {
		if (!bf_thresh_opt.IsNull()) {
			bruteforce_threshold = bf_thresh_opt.GetValue<float>();
		}
	}

	// Build predicate from filter bitset (owned by this function via move).
	// LIFETIME: captures filter_bitset by reference. Safe because
	// ef_acorn1_filtered_search executes synchronously — the predicate
	// does not outlive this function scope.
	auto predicate = [&filter_bitset](row_t key) -> bool {
		auto word = static_cast<idx_t>(key) / 64;
		auto bit = static_cast<idx_t>(key) % 64;
		return word < filter_bitset.size() && (filter_bitset[word] & (1ULL << bit)) != 0;
	};

	// Auto-tune ef_search for filtered search: at low selectivity, the search
	// needs to explore more candidates to find enough matching rows.
	// Scale inversely with selectivity, capped at index size.
	if (selectivity > 0.0f && selectivity < 1.0f) {
		auto needed = static_cast<idx_t>(static_cast<float>(search_limit * 2) / selectivity);
		if (needed > ef_search) {
			ef_search = (needed < index.size()) ? needed : static_cast<idx_t>(index.size());
		}
	}

	// For RaBitQ, quantize the query vector before searching
	unique_array<uint8_t> rabitq_query;
	if (is_rabitq_) {
		auto bbq_size = RaBitQBytesPerVector(rabitq_state_->dimensions);
		rabitq_query = make_uniq_array<uint8_t>(bbq_size);
		bool normalize = rabitq_state_->original_metric == unum::usearch::metric_kind_t::cos_k;
		RaBitQQuantizeVector(query_vector, *rabitq_state_, rabitq_query.get(), normalize);
	}

	auto lock = rwlock.GetSharedLock();

	USearchIndexType::search_result_t search_result;
	if (is_rabitq_) {
		auto bbq_ptr = reinterpret_cast<const unum::usearch::b1x8_t *>(rabitq_query.get());
		if (selectivity > acorn_threshold) {
			search_result = index.ef_search(bbq_ptr, search_limit, ef_search);
		} else if (popcount > 0 && selectivity < bruteforce_threshold) {
			search_result = index.ef_acorn1_filtered_search(bbq_ptr, search_limit, ef_search, predicate,
			                                                /*thread=*/0, /*exact=*/true);
		} else {
			search_result = index.ef_acorn1_filtered_search(bbq_ptr, search_limit, ef_search, predicate);
		}
	} else {
		if (selectivity > acorn_threshold) {
			search_result = index.ef_search(query_vector, search_limit, ef_search);
		} else if (popcount > 0 && selectivity < bruteforce_threshold) {
			search_result = index.ef_acorn1_filtered_search(query_vector, search_limit, ef_search, predicate,
			                                                /*thread=*/0, /*exact=*/true);
		} else {
			search_result = index.ef_acorn1_filtered_search(query_vector, search_limit, ef_search, predicate);
		}
	}

	state->current_row = 0;
	state->total_rows = search_result.size();
	state->row_ids = make_uniq_array<row_t>(search_result.size());
	search_result.dump_to(state->row_ids.get());

	// For RaBitQ, mark for rescore with original query
	if (is_rabitq_) {
		state->needs_rescore = true;
		state->vector_dimensions = rabitq_state_->dimensions;
		state->metric_kind = rabitq_state_->original_metric;
		state->query_vector = make_uniq_array<float>(rabitq_state_->dimensions);
		memcpy(state->query_vector.get(), query_vector, rabitq_state_->dimensions * sizeof(float));
	}

	return std::move(state);
}

idx_t HNSWIndex::Scan(IndexScanState &state, Vector &result, idx_t result_offset) {
	auto &scan_state = state.Cast<HNSWIndexScanState>();

	idx_t count = 0;
	auto row_ids = FlatVector::GetData<row_t>(result) + result_offset;

	// Push the row ids into the result vector, up to STANDARD_VECTOR_SIZE or the
	// end of the result set
	while (count < STANDARD_VECTOR_SIZE && scan_state.current_row < scan_state.total_rows) {
		row_ids[count++] = scan_state.row_ids[scan_state.current_row++];
	}

	return count;
}

struct MultiScanState final : IndexScanState {
	Vector vec;
	vector<row_t> row_ids;
	size_t ef_search;
	MultiScanState(size_t ef_search_p) : vec(LogicalType::ROW_TYPE, nullptr), ef_search(ef_search_p) {
	}
};

unique_ptr<IndexScanState> HNSWIndex::InitializeMultiScan(ClientContext &context) {
	// Try to get the ef_search parameter from the database or use the default value
	auto ef_search = index.expansion_search();

	Value hnsw_ef_search_opt;
	if (context.TryGetCurrentSetting("hnsw_ef_search", hnsw_ef_search_opt)) {
		if (!hnsw_ef_search_opt.IsNull() && hnsw_ef_search_opt.type() == LogicalType::BIGINT) {
			const auto val = hnsw_ef_search_opt.GetValue<int64_t>();
			if (val > 0) {
				ef_search = static_cast<idx_t>(val);
			}
		}
	}
	// Return the new state
	return make_uniq<MultiScanState>(ef_search);
}

idx_t HNSWIndex::ExecuteMultiScan(IndexScanState &state_p, float *query_vector, idx_t limit) {
	auto &state = state_p.Cast<MultiScanState>();

	USearchIndexType::search_result_t search_result;
	{
		auto lock = rwlock.GetSharedLock();
		if (is_rabitq_) {
			// Quantize query and search with b1x8 overload
			auto bbq_size = RaBitQBytesPerVector(rabitq_state_->dimensions);
			auto bbq_query = make_uniq_array<uint8_t>(bbq_size);
			bool normalize = rabitq_state_->original_metric == unum::usearch::metric_kind_t::cos_k;
			RaBitQQuantizeVector(query_vector, *rabitq_state_, bbq_query.get(), normalize);
			// No oversample/rescore in join path — the join operator handles batching
			// and doesn't support per-query rescore yet. Use higher ef_search instead.
			search_result = index.ef_search(
			    reinterpret_cast<const unum::usearch::b1x8_t *>(bbq_query.get()), limit, state.ef_search);
		} else {
			search_result = index.ef_search(query_vector, limit, state.ef_search);
		}
	}

	const auto offset = state.row_ids.size();
	state.row_ids.resize(state.row_ids.size() + search_result.size());
	search_result.dump_to(state.row_ids.data() + offset);

	return search_result.size();
}

const Vector &HNSWIndex::GetMultiScanResult(IndexScanState &state) {
	auto &scan_state = state.Cast<MultiScanState>();
	FlatVector::SetData(scan_state.vec, (data_ptr_t)scan_state.row_ids.data());
	return scan_state.vec;
}

void HNSWIndex::ResetMultiScan(IndexScanState &state) {
	auto &scan_state = state.Cast<MultiScanState>();
	scan_state.row_ids.clear();
}

void HNSWIndex::SetRaBitQCentroid(vector<float> centroid) {
	D_ASSERT(is_rabitq_);
	D_ASSERT(rabitq_state_);
	rabitq_state_->centroid = std::move(centroid);
}

void HNSWIndex::RescoreRaBitQCandidates(IndexScanState &index_state, const HNSWIndex &hnsw_index,
                                         DuckTableEntry &table, idx_t limit, ClientContext &context) {
	auto &scan_state = index_state.Cast<HNSWIndexScanState>();
	if (!scan_state.needs_rescore || scan_state.total_rows == 0) {
		return;
	}

	auto &data_table = table.GetStorage();
	auto &transaction = DuckTransaction::Get(context, table.catalog);

	auto dims = scan_state.vector_dimensions;
	auto candidate_count = scan_state.total_rows;
	auto final_limit = MinValue(limit, candidate_count);

	// Get the vector column's storage index
	auto vec_col_id = hnsw_index.GetColumnIds()[0];
	vector<StorageIndex> fetch_col_ids = {StorageIndex(vec_col_id)};

	// Build a row_id vector for fetching
	Vector candidate_rowids(LogicalType::ROW_TYPE, candidate_count);
	auto rid_data = FlatVector::GetData<row_t>(candidate_rowids);
	memcpy(rid_data, scan_state.row_ids.get(), candidate_count * sizeof(row_t));

	// Fetch original F32 vectors for all candidates
	DataChunk fetched;
	fetched.Initialize(Allocator::DefaultAllocator(), hnsw_index.logical_types);
	ColumnFetchState fetch_state;
	data_table.Fetch(transaction, fetched, fetch_col_ids, candidate_rowids, candidate_count, fetch_state);

	// Compute exact distances
	auto &vec_vec = fetched.data[0];
	auto &vec_child = ArrayVector::GetEntry(vec_vec);
	auto vec_data = FlatVector::GetData<float>(vec_child);
	auto query = scan_state.query_vector.get();

	// Choose exact distance function based on metric
	using dist_fn_t = float (*)(const float *, const float *, idx_t);
	dist_fn_t dist_fn;
	switch (scan_state.metric_kind) {
	case unum::usearch::metric_kind_t::cos_k:
		dist_fn = ExactDistanceCosine;
		break;
	case unum::usearch::metric_kind_t::ip_k:
		dist_fn = ExactDistanceIP;
		break;
	default:
		dist_fn = ExactDistanceL2sq;
		break;
	}

	// Compute distances and build (distance, index) pairs
	struct DistIdx {
		float distance;
		idx_t idx;
	};
	vector<DistIdx> scored(candidate_count);
	for (idx_t i = 0; i < candidate_count; i++) {
		scored[i].distance = dist_fn(query, vec_data + i * dims, dims);
		scored[i].idx = i;
	}

	// Partial sort to get top-K
	std::partial_sort(scored.begin(), scored.begin() + final_limit, scored.end(),
	                  [](const DistIdx &a, const DistIdx &b) { return a.distance < b.distance; });

	// Replace scan state with rescored top-K
	auto new_row_ids = make_uniq_array<row_t>(final_limit);
	for (idx_t i = 0; i < final_limit; i++) {
		new_row_ids[i] = scan_state.row_ids[scored[i].idx];
	}

	scan_state.row_ids = std::move(new_row_ids);
	scan_state.total_rows = final_limit;
	scan_state.current_row = 0;
	scan_state.needs_rescore = false;
}

void HNSWIndex::CommitDrop(IndexLock &index_lock) {
	// Acquire an exclusive lock to drop the index
	auto lock = rwlock.GetExclusiveLock();

	index.reset();
	index_size = 0;
	// TODO: Maybe we can drop these much earlier?
	linked_block_allocator->Reset();
	root_block_ptr.Clear();
}

void HNSWIndex::Construct(DataChunk &input, Vector &row_ids, idx_t thread_idx) {
	D_ASSERT(row_ids.GetType().InternalType() == ROW_TYPE);
	D_ASSERT(logical_types[0] == input.data[0].GetType());

	// Mark this index as dirty so we checkpoint it properly
	is_dirty = true;

	auto count = input.size();
	input.Flatten();

	auto &vec_vec = input.data[0];
	auto &vec_child_vec = ArrayVector::GetEntry(vec_vec);
	auto array_size = ArrayType::GetSize(vec_vec.GetType());

	auto vec_child_data = FlatVector::GetData<float>(vec_child_vec);
	auto rowid_data = FlatVector::GetData<row_t>(row_ids);

	auto to_add_count = FlatVector::Validity(vec_vec).CountValid(count);

	// Check if we need to resize the index
	// We keep the size of the index in a separate atomic to avoid
	// locking exclusively when checking
	bool needs_resize = false;
	{
		auto lock = rwlock.GetSharedLock();
		if (index_size.fetch_add(to_add_count) + to_add_count > index.capacity()) {
			needs_resize = true;
		}
	}

	// We need to "upgrade" the lock to exclusive to resize the index
	if (needs_resize) {
		auto lock = rwlock.GetExclusiveLock();
		// Do we still need to resize?
		// Another thread might have resized it already
		auto size = index_size.load();
		if (size > index.capacity()) {
			// Add some extra space so that we don't need to resize too often
			index.reserve(NextPowerOfTwo(size));
		}
	}

	{
		// Now we can be sure that we have enough space in the index
		auto lock = rwlock.GetSharedLock();
		for (idx_t out_idx = 0; out_idx < count; out_idx++) {
			if (FlatVector::IsNull(vec_vec, out_idx)) {
				// Dont add nulls
				continue;
			}

			auto rowid = rowid_data[out_idx];
			USearchIndexType::add_result_t result;
			if (is_rabitq_ && rabitq_state_) {
				// Quantize before inserting
				auto bbq_size = RaBitQBytesPerVector(rabitq_state_->dimensions);
				auto bbq_buf = make_uniq_array<uint8_t>(bbq_size);
				bool normalize = rabitq_state_->original_metric == unum::usearch::metric_kind_t::cos_k;
				RaBitQQuantizeVector(vec_child_data + (out_idx * array_size), *rabitq_state_, bbq_buf.get(),
				                     normalize);
				result = index.add(rowid, reinterpret_cast<const unum::usearch::b1x8_t *>(bbq_buf.get()),
				                   thread_idx);
			} else {
				result = index.add(rowid, vec_child_data + (out_idx * array_size), thread_idx);
			}
			if (!result) {
				throw InternalException("Failed to add to the HNSW index: %s", result.error.what());
			}
		}
	}
}

void HNSWIndex::Compact() {
	// Mark this index as dirty so we checkpoint it properly
	is_dirty = true;

	// Acquire an exclusive lock to compact the index
	auto lock = rwlock.GetExclusiveLock();
	// Re-compact the index
	auto result = index.compact();
	if (!result) {
		throw InternalException("Failed to compact the HNSW index: %s", result.error.what());
	}

	index_size = index.size();
}

void HNSWIndex::Delete(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
	// Mark this index as dirty so we checkpoint it properly
	is_dirty = true;

	auto count = input.size();
	rowid_vec.Flatten(count);
	auto row_id_data = FlatVector::GetData<row_t>(rowid_vec);

	// For deleting from the index, we need an exclusive lock
	auto _lock = rwlock.GetExclusiveLock();

	for (idx_t i = 0; i < input.size(); i++) {
		auto result = index.remove(row_id_data[i]);
	}

	index_size = index.size();
}

ErrorData HNSWIndex::Insert(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
	Construct(input, rowid_vec, unum::usearch::index_dense_t::any_thread());
	return ErrorData {};
}

ErrorData HNSWIndex::Append(IndexLock &lock, DataChunk &appended_data, Vector &row_identifiers) {
	DataChunk expression_result;
	expression_result.Initialize(Allocator::DefaultAllocator(), logical_types);

	// first resolve the expressions for the index
	ExecuteExpressions(appended_data, expression_result);

	// now insert into the index
	Construct(expression_result, row_identifiers, unum::usearch::index_dense_t::any_thread());

	return ErrorData {};
}

void HNSWIndex::PersistToDisk() {
	// Acquire an exclusive lock to persist the index
	auto lock = rwlock.GetExclusiveLock();

	// If there haven't been any changes, we don't need to rewrite the index again
	if (!is_dirty) {
		return;
	}

	// Write
	if (root_block_ptr.Get() == 0) {
		root_block_ptr = linked_block_allocator->New();
	}

	LinkedBlockWriter writer(*linked_block_allocator, root_block_ptr);
	writer.Reset();

	// For RaBitQ indexes, write a header with magic + centroid before usearch stream.
	// Legacy (non-RaBitQ) indexes write usearch stream directly — no header.
	// On load, we detect RaBitQ by checking if the first byte is 'R' (0x52),
	// which cannot be the first byte of a legacy usearch stream (which starts
	// with uint32 matrix_rows — the low byte of a row count is never 'R' for
	// any practical index size).
	if (is_rabitq_ && rabitq_state_) {
		uint8_t magic = 'R'; // 0x52 — identifies RaBitQ stream
		writer.WriteData(reinterpret_cast<const_data_ptr_t>(&magic), sizeof(magic));

		uint64_t centroid_dims = rabitq_state_->dimensions;
		writer.WriteData(reinterpret_cast<const_data_ptr_t>(&centroid_dims), sizeof(centroid_dims));
		writer.WriteData(reinterpret_cast<const_data_ptr_t>(rabitq_state_->centroid.data()),
		                 centroid_dims * sizeof(float));
	}

	index.save_to_stream([&](const void *data, size_t size) {
		writer.WriteData(static_cast<const_data_ptr_t>(data), size);
		return true;
	});

	is_dirty = false;
}

IndexStorageInfo HNSWIndex::SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) {

	PersistToDisk();

	IndexStorageInfo info;
	info.name = name;
	info.root = root_block_ptr.Get();

	// Use the partial block manager to serialize allocator data.
	auto &block_manager = table_io_manager.GetIndexBlockManager();
	PartialBlockManager partial_block_manager(context, block_manager, PartialBlockType::FULL_CHECKPOINT);
	linked_block_allocator->SerializeBuffers(partial_block_manager);
	partial_block_manager.FlushPartialBlocks();
	info.allocator_infos.push_back(linked_block_allocator->GetInfo());

	return info;
}

IndexStorageInfo HNSWIndex::SerializeToWAL(const case_insensitive_map_t<Value> &options) {

	PersistToDisk();

	IndexStorageInfo info;
	info.name = name;
	info.root = root_block_ptr.Get();
	info.buffers.push_back(linked_block_allocator->InitSerializationToWAL());
	info.allocator_infos.push_back(linked_block_allocator->GetInfo());

	return info;
}

idx_t HNSWIndex::GetInMemorySize(IndexLock &state) {
	// TODO: This is not correct: its a lower bound, but it's a start
	return index.memory_usage();
}

bool HNSWIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	throw NotImplementedException("HNSWIndex::MergeIndexes() not implemented");
}

void HNSWIndex::Vacuum(IndexLock &state) {
}

void HNSWIndex::Verify(IndexLock &l) {
	// No-op: HNSW index verification not implemented
}

string HNSWIndex::ToString(IndexLock &l, bool display_ascii) {
	return StringUtil::Format("HNSW Index [%s] (%llu entries)", GetIndexName(), index.size());
}

void HNSWIndex::VerifyAllocations(IndexLock &state) {
	throw NotImplementedException("HNSWIndex::VerifyAllocations() not implemented");
}

//------------------------------------------------------------------------------
// Can rewrite index expression?
//------------------------------------------------------------------------------
static void TryBindIndexExpressionInternal(Expression &expr, idx_t table_idx, const vector<column_t> &index_columns,
                                           const vector<ColumnIndex> &table_columns, bool &success, bool &found) {

	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		found = true;
		auto &ref = expr.Cast<BoundColumnRefExpression>();

		// Rewrite the column reference to fit in the current set of bound column ids
		ref.binding.table_index = table_idx;

		const auto referenced_column = index_columns[ref.binding.column_index];
		for (idx_t i = 0; i < table_columns.size(); i++) {
			if (table_columns[i].GetPrimaryIndex() == referenced_column) {
				ref.binding.column_index = i;
				return;
			}
		}
		success = false;
	}

	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) {
		TryBindIndexExpressionInternal(child, table_idx, index_columns, table_columns, success, found);
	});
}

bool HNSWIndex::TryBindIndexExpression(LogicalGet &get, unique_ptr<Expression> &result) const {
	auto expr_ptr = unbound_expressions.back()->Copy();

	auto &expr = *expr_ptr;
	auto &index_columns = GetColumnIds();
	auto &table_columns = get.GetColumnIds();

	auto success = true;
	auto found = false;

	TryBindIndexExpressionInternal(expr, get.table_index, index_columns, table_columns, success, found);

	if (success && found) {
		result = std::move(expr_ptr);
		return true;
	}
	return false;
}

bool HNSWIndex::TryMatchDistanceFunction(const unique_ptr<Expression> &expr,
                                         vector<reference<Expression>> &bindings) const {
	return function_matcher->Match(*expr, bindings);
}

unique_ptr<ExpressionMatcher> HNSWIndex::MakeFunctionMatcher() const {
	unordered_set<string> distance_functions;
	switch (index.metric().metric_kind()) {
	case unum::usearch::metric_kind_t::l2sq_k:
		distance_functions = {"array_distance", "<->"};
		break;
	case unum::usearch::metric_kind_t::cos_k:
		distance_functions = {"array_cosine_distance", "<=>"};
		break;
	case unum::usearch::metric_kind_t::ip_k:
		distance_functions = {"array_negative_inner_product", "<#>"};
		break;
	default:
		throw NotImplementedException("Unknown metric kind");
	}

	auto matcher = make_uniq<FunctionExpressionMatcher>();
	matcher->function = make_uniq<ManyFunctionMatcher>(distance_functions);
	matcher->expr_type = make_uniq<SpecificExpressionTypeMatcher>(ExpressionType::BOUND_FUNCTION);
	matcher->policy = SetMatcher::Policy::UNORDERED;

	auto lhs_matcher = make_uniq<ExpressionMatcher>();
	lhs_matcher->type = make_uniq<SpecificTypeMatcher>(LogicalType::ARRAY(LogicalType::FLOAT, GetVectorSize()));
	matcher->matchers.push_back(std::move(lhs_matcher));

	auto rhs_matcher = make_uniq<ExpressionMatcher>();
	rhs_matcher->type = make_uniq<SpecificTypeMatcher>(LogicalType::ARRAY(LogicalType::FLOAT, GetVectorSize()));
	matcher->matchers.push_back(std::move(rhs_matcher));

	return std::move(matcher);
}

void HNSWIndex::VerifyBuffers(IndexLock &lock) {
	// Verify the linked block allocator buffers
	linked_block_allocator->VerifyBuffers();
}

//------------------------------------------------------------------------------
// Register Index Type
//------------------------------------------------------------------------------
void HNSWModule::RegisterIndex(DatabaseInstance &db) {

	IndexType index_type;

	index_type.name = HNSWIndex::TYPE_NAME;
	index_type.create_instance = [](CreateIndexInput &input) -> unique_ptr<BoundIndex> {
		auto res = make_uniq<HNSWIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
		                                input.unbound_expressions, input.db, input.options, input.storage_info);
		return std::move(res);
	};
	index_type.create_plan = HNSWIndex::CreatePlan;

	// Register persistence option
	db.config.AddExtensionOption("hnsw_enable_experimental_persistence",
	                             "experimental: enable creating HNSW indexes in persistent databases",
	                             LogicalType::BOOLEAN, Value::BOOLEAN(false));

	// Register scan options
	db.config.AddExtensionOption("hnsw_ef_search",
	                             "experimental: override the ef_search parameter when scanning HNSW indexes",
	                             LogicalType::BIGINT);

	// ACORN-1 filtered search thresholds
	db.config.AddExtensionOption(
	    "hnsw_acorn_threshold", "selectivity above which ACORN-1 is skipped (standard HNSW + post-filter used instead)",
	    LogicalType::FLOAT, Value::FLOAT(0.6f));
	db.config.AddExtensionOption("hnsw_bruteforce_threshold",
	                             "selectivity below which brute-force exact scan is used instead of ACORN-1",
	                             LogicalType::FLOAT, Value::FLOAT(0.01f));

	// RaBitQ quantization settings
	db.config.AddExtensionOption("hnsw_rabitq_oversample",
	                             "rescore oversample factor for RaBitQ-quantized HNSW indexes (default 3)",
	                             LogicalType::BIGINT, Value::BIGINT(3));

	// Register the index type
	db.config.GetIndexTypes().RegisterIndexType(index_type);
}

} // namespace duckdb
