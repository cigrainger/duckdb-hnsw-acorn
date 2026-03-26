#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/function/scalar/struct_utils.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_unnest_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_unnest.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "duckdb/storage/storage_index.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"

namespace duckdb {

//------------------------------------------------------------------------------
// Physical Operator
//------------------------------------------------------------------------------

class PhysicalHNSWIndexJoin final : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

	PhysicalHNSWIndexJoin(PhysicalPlan &physical_plan, const vector<LogicalType> &types_p,
	                      const idx_t estimated_cardinality, DuckTableEntry &table_p, HNSWIndex &hnsw_index_p,
	                      const idx_t limit_p)
	    : PhysicalOperator(physical_plan, TYPE, types_p, estimated_cardinality), table(table_p),
	      hnsw_index(hnsw_index_p), limit(limit_p) {
	}

public:
	string GetName() const override;
	bool ParallelOperator() const override;
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;
	InsertionOrderPreservingMap<string> ParamsToString() const override;

public:
	DuckTableEntry &table;
	HNSWIndex &hnsw_index;
	idx_t limit;

	vector<column_t> inner_column_ids;
	vector<idx_t> inner_projection_ids;

	idx_t outer_vector_column;
	idx_t inner_vector_column;
};

string PhysicalHNSWIndexJoin::GetName() const {
	return "HNSW_INDEX_JOIN";
}

bool PhysicalHNSWIndexJoin::ParallelOperator() const {
	return false;
}

// TODO: Assert at most k = SVS
class HNSWIndexJoinState final : public OperatorState {
public:
	idx_t input_idx = 0;

	ColumnFetchState fetch_state;
	TableScanState local_storage_state;
	vector<StorageIndex> physical_column_ids;

	// Index scan state
	unique_ptr<IndexScanState> index_state;
	SelectionVector match_sel;
};

unique_ptr<OperatorState> PhysicalHNSWIndexJoin::GetOperatorState(ExecutionContext &context) const {
	auto result = make_uniq<HNSWIndexJoinState>();

	auto &local_storage = LocalStorage::Get(context.client, table.catalog);
	result->physical_column_ids.reserve(inner_column_ids.size());

	// Figure out the storage column ids from the projection expression
	for (auto &id : inner_column_ids) {
		storage_t col_id = id;
		if (id != DConstants::INVALID_INDEX) {
			col_id = table.GetColumn(LogicalIndex(id)).StorageOid();
		}
		result->physical_column_ids.emplace_back(col_id);
	}

	// Initialize selection vector
	result->match_sel.Initialize();

	// Initialize the storage scan state
	result->local_storage_state.Initialize(result->physical_column_ids, nullptr);
	local_storage.InitializeScan(table.GetStorage(), result->local_storage_state.local_state, nullptr);

	// Initialize the index scan state
	result->index_state = hnsw_index.InitializeMultiScan(context.client);

	return std::move(result);
}

OperatorResultType PhysicalHNSWIndexJoin::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                  GlobalOperatorState &gstate, OperatorState &ostate) const {
	auto &state = ostate.Cast<HNSWIndexJoinState>();
	auto &transcation = DuckTransaction::Get(context.client, table.catalog);

	input.Flatten();

	// The first 0..inner_column_ids.size() columns are the inner table columns
	const auto MATCH_COLUMN_OFFSET = inner_column_ids.size();
	// The next column is the row number
	const auto OUTER_COLUMN_OFFSET = MATCH_COLUMN_OFFSET + 1;
	// The rest of the columns are the outer table columns

	auto &rhs_vector_vector = input.data[outer_vector_column];
	auto &rhs_vector_child = ArrayVector::GetEntry(rhs_vector_vector);
	const auto rhs_vector_size = ArrayType::GetSize(rhs_vector_vector.GetType());
	const auto rhs_vector_ptr = FlatVector::GetData<float>(rhs_vector_child);

	// We mimic the window row_number() operator here and output the row number in each batch, basically.
	const auto row_number_vector = FlatVector::GetData<int64_t>(chunk.data[MATCH_COLUMN_OFFSET]);

	hnsw_index.ResetMultiScan(*state.index_state);

	// How many batches are we going to process?
	const auto batch_count = MinValue(input.size() - state.input_idx, STANDARD_VECTOR_SIZE / limit);
	idx_t output_idx = 0;
	for (idx_t batch_idx = 0; batch_idx < batch_count; batch_idx++, state.input_idx++) {

		// Get the next batch
		const auto rhs_vector_data = rhs_vector_ptr + state.input_idx * rhs_vector_size;

		// Scan the index for row ids
		const auto match_count = hnsw_index.ExecuteMultiScan(*state.index_state, rhs_vector_data, limit);
		for (idx_t i = 0; i < match_count; i++) {
			state.match_sel.set_index(output_idx, state.input_idx);
			row_number_vector[output_idx] = i + 1; // Note: 1-indexed!
			output_idx++;
		}
	}

	const auto &row_ids = hnsw_index.GetMultiScanResult(*state.index_state);

	// Execute one big fetch for the LHS
	table.GetStorage().Fetch(transcation, chunk, state.physical_column_ids, row_ids, output_idx, state.fetch_state);

	// Now slice the chunk so that we include the rhs too
	chunk.Slice(input, state.match_sel, output_idx, OUTER_COLUMN_OFFSET);

	// Set the cardinality
	chunk.SetCardinality(output_idx);

	if (state.input_idx == input.size()) {
		state.input_idx = 0;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	return OperatorResultType::HAVE_MORE_OUTPUT;
}

InsertionOrderPreservingMap<string> PhysicalHNSWIndexJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	auto table_name = table.name;
	auto index_name = hnsw_index.name;
	result.insert("table", table_name);
	result.insert("index", index_name);
	result.insert("limit", to_string(limit));
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

//------------------------------------------------------------------------------
// Logical Operator
//------------------------------------------------------------------------------

class LogicalHNSWIndexJoin final : public LogicalExtensionOperator {
public:
	explicit LogicalHNSWIndexJoin(const idx_t table_index_p, DuckTableEntry &table_p, HNSWIndex &hnsw_index_p,
	                              const idx_t limit_p)
	    : table_index(table_index_p), table(table_p), hnsw_index(hnsw_index_p), limit(limit_p) {
	}

public:
	string GetName() const override;
	void ResolveTypes() override;
	vector<ColumnBinding> GetColumnBindings() override;
	vector<ColumnBinding> GetLeftBindings();
	vector<ColumnBinding> GetRightBindings();
	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;
	idx_t EstimateCardinality(ClientContext &context) override;

public:
	idx_t table_index;

	DuckTableEntry &table;
	HNSWIndex &hnsw_index;
	idx_t limit;

	vector<column_t> inner_column_ids;
	vector<idx_t> inner_projection_ids;
	vector<LogicalType> inner_returned_types;

	idx_t outer_vector_column;
	idx_t inner_vector_column;
};

string LogicalHNSWIndexJoin::GetName() const {
	return "HNSW_INDEX_JOIN";
}

void LogicalHNSWIndexJoin::ResolveTypes() {
	if (inner_column_ids.empty()) {
		inner_column_ids.push_back(COLUMN_IDENTIFIER_ROW_ID);
	}
	types.clear();

	if (inner_projection_ids.empty()) {
		for (const auto &index : inner_column_ids) {
			if (index == COLUMN_IDENTIFIER_ROW_ID) {
				types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				types.push_back(inner_returned_types[index]);
			}
		}
	} else {
		for (const auto &proj_index : inner_projection_ids) {
			const auto &index = inner_column_ids[proj_index];
			if (index == COLUMN_IDENTIFIER_ROW_ID) {
				types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				types.push_back(inner_returned_types[index]);
			}
		}
	}

	// Always add the row_number after the inner columns
	types.emplace_back(LogicalType::BIGINT);

	// Also add the types of the right hand side
	auto &right_types = children[0]->types;
	types.insert(types.end(), right_types.begin(), right_types.end());
}

vector<ColumnBinding> LogicalHNSWIndexJoin::GetLeftBindings() {
	vector<ColumnBinding> result;
	if (inner_projection_ids.empty()) {
		for (idx_t col_idx = 0; col_idx < inner_column_ids.size(); col_idx++) {
			result.emplace_back(table_index, col_idx);
		}
	} else {
		for (auto proj_id : inner_projection_ids) {
			result.emplace_back(table_index, proj_id);
		}
	}

	// Always add the row number last
	result.emplace_back(table_index, inner_column_ids.size());

	return result;
}

vector<ColumnBinding> LogicalHNSWIndexJoin::GetRightBindings() {
	vector<ColumnBinding> result;
	for (auto &binding : children[0]->GetColumnBindings()) {
		result.push_back(binding);
	}
	return result;
}

vector<ColumnBinding> LogicalHNSWIndexJoin::GetColumnBindings() {
	vector<ColumnBinding> result;
	auto left_bindings = GetLeftBindings();
	auto right_bindings = GetRightBindings();
	result.insert(result.end(), left_bindings.begin(), left_bindings.end());
	result.insert(result.end(), right_bindings.begin(), right_bindings.end());
	return result;
}

PhysicalOperator &LogicalHNSWIndexJoin::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {

	auto &result = planner.Make<PhysicalHNSWIndexJoin>(types, estimated_cardinality, table, hnsw_index, limit);
	auto &cast_result = result.Cast<PhysicalHNSWIndexJoin>();
	cast_result.limit = limit;
	cast_result.inner_column_ids = inner_column_ids;
	cast_result.inner_projection_ids = inner_projection_ids;
	cast_result.outer_vector_column = outer_vector_column;
	cast_result.inner_vector_column = inner_vector_column;

	// Plan the	child
	auto &plan = planner.CreatePlan(*children[0]);
	result.children.push_back(plan);
	return result;
}

idx_t LogicalHNSWIndexJoin::EstimateCardinality(ClientContext &context) {
	// The cardinality of the HNSW index join is the cardinality of the outer table
	if (has_estimated_cardinality) {
		return estimated_cardinality;
	}

	const auto child_cardinality = children[0]->EstimateCardinality(context);
	estimated_cardinality = child_cardinality * limit;
	has_estimated_cardinality = true;

	return estimated_cardinality;
}

//------------------------------------------------------------------------------
// Optimizer
//------------------------------------------------------------------------------

class HNSWIndexJoinOptimizer : public OptimizerExtension {
public:
	HNSWIndexJoinOptimizer();
	static bool TryOptimize(Binder &binder, ClientContext &context, unique_ptr<LogicalOperator> &root,
	                        unique_ptr<LogicalOperator> &plan);
	static void OptimizeRecursive(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &root,
	                              unique_ptr<LogicalOperator> &plan, bool has_aggregate_above);
	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);
};

HNSWIndexJoinOptimizer::HNSWIndexJoinOptimizer() {
	optimize_function = Optimize;
}

class CardinalityResetter final : public LogicalOperatorVisitor {
public:
	ClientContext &context;

	explicit CardinalityResetter(ClientContext &context_p) : context(context_p) {
	}

	void VisitOperator(LogicalOperator &op) override {
		op.has_estimated_cardinality = false;
		VisitOperatorChildren(op);
		op.EstimateCardinality(context);
	}
};

bool HNSWIndexJoinOptimizer::TryOptimize(Binder &binder, ClientContext &context, unique_ptr<LogicalOperator> &root,
                                         unique_ptr<LogicalOperator> &plan) {

	//------------------------------------------------------------------------------
	// Match: PROJECTION → [FILTER →] DELIM_JOIN → [PROJECTION(s)] → UNNEST →
	//        AGGREGATE(arg_min_nulls_last) → PROJECTION → CROSS_PRODUCT
	//
	// We match at the PROJECTION level because the DELIM_JOIN's inner side
	// uses struct-packed UNNEST output that we replace entirely.
	//------------------------------------------------------------------------------
	if (plan->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return false;
	}
	auto &top_proj = plan->Cast<LogicalProjection>();
	if (top_proj.children.size() != 1) {
		return false;
	}

	// Navigate through an optional FILTER between the PROJECTION and DELIM_JOIN
	// (appears when the lateral join has a WHERE clause on joined columns)
	auto *delim_join_ptr = top_proj.children[0].get();
	if (delim_join_ptr->type == LogicalOperatorType::LOGICAL_FILTER && delim_join_ptr->children.size() == 1) {
		delim_join_ptr = delim_join_ptr->children[0].get();
	}

	if (delim_join_ptr->type != LogicalOperatorType::LOGICAL_DELIM_JOIN ||
	    delim_join_ptr->children.size() != 2) {
		return false;
	}
	auto &delim_join = delim_join_ptr->Cast<LogicalJoin>();

	// Outer table (queries)
	if (delim_join.children[1]->type != LogicalOperatorType::LOGICAL_GET ||
	    delim_join.children[1]->children.size() != 0) {
		return false;
	}
	auto outer_get_ptr = &delim_join.children[1];
	auto &outer_get = (*outer_get_ptr)->Cast<LogicalGet>();

	// Navigate through projection(s) to find UNNEST.
	LogicalOperator *cursor = delim_join.children[0].get();
	while (cursor->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		if (cursor->children.empty()) {
			return false;
		}
		cursor = cursor->children.back().get();
	}

	// Match UNNEST
	if (cursor->type != LogicalOperatorType::LOGICAL_UNNEST || cursor->children.size() != 1) {
		return false;
	}
	auto &unnest = cursor->Cast<LogicalUnnest>();

	// Match AGGREGATE (arg_min_nulls_last)
	auto &agg_child = unnest.children[0];
	if (agg_child->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return false;
	}
	auto &agg = agg_child->Cast<LogicalAggregate>();

	// Must have exactly one aggregate expression
	if (agg.expressions.size() != 1) {
		return false;
	}
	auto &agg_expr = agg.expressions[0];
	if (agg_expr->type != ExpressionType::BOUND_AGGREGATE) {
		return false;
	}
	auto &agg_func_expr = agg_expr->Cast<BoundAggregateExpression>();

	// Match arg_min_nulls_last or min_by with 3 children
	if (agg_func_expr.function.name != "arg_min_nulls_last" && agg_func_expr.function.name != "min_by") {
		return false;
	}
	if (agg_func_expr.children.size() != 3) {
		return false;
	}

	// Extract k value from children[2]
	if (agg_func_expr.children[2]->type != ExpressionType::VALUE_CONSTANT) {
		return false;
	}
	auto &limit_expr = agg_func_expr.children[2]->Cast<BoundConstantExpression>();
	auto k_value = limit_expr.value.GetValue<int64_t>();
	if (k_value < 0 || k_value >= STANDARD_VECTOR_SIZE) {
		return false;
	}

	// The distance expression is children[1] of the aggregate.
	// It may be a direct function call or a column ref into the inner projection.
	const unique_ptr<Expression> *agg_distance_expr = &agg_func_expr.children[1];

	// Navigate AGGREGATE → PROJECTION → CROSS_PRODUCT
	if (agg.children.size() != 1) {
		return false;
	}
	auto &agg_proj_child = agg.children[0];
	if (agg_proj_child->type != LogicalOperatorType::LOGICAL_PROJECTION || agg_proj_child->children.size() != 1) {
		return false;
	}
	auto &inner_proj = agg_proj_child->Cast<LogicalProjection>();

	if (inner_proj.children[0]->type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT ||
	    inner_proj.children[0]->children.size() != 2) {
		return false;
	}
	auto &cross_product = inner_proj.children[0]->Cast<LogicalCrossProduct>();

	// Extract DELIM_GET and inner SEQ_SCAN from CROSS_PRODUCT
	unique_ptr<LogicalOperator> *delim_get_ptr;
	unique_ptr<LogicalOperator> *inner_get_ptr;

	auto &cp_lhs = cross_product.children[0];
	auto &cp_rhs = cross_product.children[1];
	if (cp_lhs->type == LogicalOperatorType::LOGICAL_DELIM_GET && cp_rhs->type == LogicalOperatorType::LOGICAL_GET) {
		delim_get_ptr = &cp_lhs;
		inner_get_ptr = &cp_rhs;
	} else if (cp_rhs->type == LogicalOperatorType::LOGICAL_DELIM_GET &&
	           cp_lhs->type == LogicalOperatorType::LOGICAL_GET) {
		delim_get_ptr = &cp_rhs;
		inner_get_ptr = &cp_lhs;
	} else {
		return false;
	}

	auto &delim_get = (*delim_get_ptr)->Cast<LogicalDelimGet>();
	auto &inner_get = (*inner_get_ptr)->Cast<LogicalGet>();
	if (inner_get.function.name != "seq_scan") {
		return false;
	}

	// Resolve the distance expression: if it's a column ref into the inner
	// projection, follow it to the actual function expression.
	const unique_ptr<Expression> *distance_expr_ptr = agg_distance_expr;
	if ((*distance_expr_ptr)->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &ref = (*distance_expr_ptr)->Cast<BoundColumnRefExpression>();
		if (ref.binding.table_index == inner_proj.table_index &&
		    ref.binding.column_index < inner_proj.expressions.size()) {
			distance_expr_ptr = &inner_proj.expressions[ref.binding.column_index];
		}
	}

	//------------------------------------------------------------------------------
	// Match the index
	//------------------------------------------------------------------------------
	auto &table = *inner_get.GetTable();
	if (!table.IsDuckTable()) {
		// We can only replace the scan if the table is a duck table
		return false;
	}
	auto &duck_table = table.Cast<DuckTableEntry>();
	auto &table_info = *table.GetStorage().GetDataTableInfo();

	HNSWIndex *index_ptr = nullptr;
	vector<reference<Expression>> bindings;

	table_info.BindIndexes(context, HNSWIndex::TYPE_NAME);
	for(auto &index : table_info.GetIndexes().Indexes()) {
		if (!index.IsBound() || HNSWIndex::TYPE_NAME != index.GetIndexType()) {
			continue;
		}
		auto &cast_index = index.Cast<HNSWIndex>();

		// Reset the bindings
		bindings.clear();
		if (!cast_index.TryMatchDistanceFunction(*distance_expr_ptr, bindings)) {
			continue;
		}
		unique_ptr<Expression> bound_index_expr = nullptr;
		if (!cast_index.TryBindIndexExpression(inner_get, bound_index_expr)) {
			continue;
		}

		// We also have to replace the outer table index here with the delim_get table index
		ExpressionIterator::EnumerateExpression(bound_index_expr, [&](Expression &child) {
			if (child.type == ExpressionType::BOUND_COLUMN_REF) {
				auto &bound_colref_expr = child.Cast<BoundColumnRefExpression>();
				if (bound_colref_expr.binding.table_index == outer_get.table_index) {
					bound_colref_expr.binding.table_index = delim_get.table_index;
				}
			}
		});

		auto &lhs_dist_expr = bindings[1];
		auto &rhs_dist_expr = bindings[2];

		// Figure out which of the arguments to the distance function is the index expression (and move it to the rhs)
		// If the index expression is not part of this distance function, we can't optimize, return false.
		if (lhs_dist_expr.get().Equals(*bound_index_expr)) {
			if (!rhs_dist_expr.get().Equals(*bound_index_expr)) {
				std::swap(lhs_dist_expr, rhs_dist_expr);
			} else {
				continue;
			}
		}

		// Save the pointer to the index
		index_ptr = &cast_index;
		break;
	}
	if (!index_ptr) {
		return false;
	}

	// Fuck it, for now dont allow expressions on the index
	if (bindings[1].get().type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	if (bindings[2].get().type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	const auto &outer_ref_expr = bindings[1].get().Cast<BoundColumnRefExpression>();
	const auto &inner_ref_expr = bindings[2].get().Cast<BoundColumnRefExpression>();

	// Sanity check
	if (inner_ref_expr.binding.table_index != inner_get.table_index) {
		// Well, we have to reference the rhs
		return false;
	}

	//------------------------------------------------------------------------------
	// Now create the HNSWIndexJoin operator
	//------------------------------------------------------------------------------

	auto index_join = make_uniq<LogicalHNSWIndexJoin>(binder.GenerateTableIndex(), duck_table, *index_ptr, k_value);
	for (auto &column_id : inner_get.GetColumnIds()) {
		index_join->inner_column_ids.emplace_back(column_id.GetPrimaryIndex());
	}
	index_join->inner_projection_ids = inner_get.projection_ids;
	index_join->inner_returned_types = inner_get.returned_types;

	// TODO: this is kind of unsafe, column_index != physical index
	index_join->outer_vector_column = outer_ref_expr.binding.column_index;
	index_join->inner_vector_column = inner_ref_expr.binding.column_index;

	// -------------------------------------------------------------------------
	// Build replacement plan: replace PROJECTION + DELIM_JOIN with
	//   PROJECTION (same table_index) → HNSW_INDEX_JOIN → SEQ_SCAN (outer)
	//
	// Resolve each top_proj expression through the intermediate projection
	// chain to map struct-unpacked UNNEST refs to index join columns.
	// -------------------------------------------------------------------------
	auto ij_table = index_join->table_index;

	// Build mapping: for each column in the inner table, what's its index in the join output?
	// The inner_get columns map to index_join columns 0..N-1.
	// Outer table columns keep their original bindings (table_index stays the same).

	// Build a mapping from intermediate projection bindings to actual source
	// columns. The intermediate projection(s) between DELIM_JOIN and UNNEST
	// unpack the arg_min struct result. We trace each binding back to either
	// an inner table column or an outer table column.
	//
	// The intermediate projection contains expressions like:
	//   #0       → group key (outer qvec)
	//   #[2.0]   → struct field 0 of unnest result (inner col 0)
	//   #[2.1]   → struct field 1 of unnest result (inner col 1)
	//   etc.
	//
	// Rather than parse these, we use the known structure:
	// - The arg_min_nulls_last's children[0] is struct_pack(inner_col_0, inner_col_1, ...)
	//   The struct fields correspond to inner projection columns 0, 1, 2, ...
	// - The aggregate's group columns correspond to outer (delim_get) columns
	//
	// We walk the top_proj expressions, resolve each BOUND_COLUMN_REF through
	// the intermediate projection chain, and map to index join columns.

	// Build a lookup: for each expression in the intermediate projection
	// chain, trace to a source column. We navigate from the DELIM_JOIN's
	// left child (the outermost intermediate projection) inward.
	//
	// Actually, the simplest approach: resolve each top_proj expression by
	// recursively inlining column refs through the projections until we hit
	// a binding we recognize (inner_get, delim_get, or outer_get).

	auto inner_get_table = inner_get.table_index;
	auto delim_get_table = delim_get.table_index;
	auto outer_get_table = outer_get.table_index;

	// Collect all intermediate projections between DELIM_JOIN and UNNEST
	vector<LogicalProjection *> intermediate_projs;
	{
		LogicalOperator *c = delim_join.children[0].get();
		while (c->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			intermediate_projs.push_back(&c->Cast<LogicalProjection>());
			c = c->children.back().get();
		}
	}

	// Function to resolve a column ref through intermediate projections
	// Returns the resolved expression (may be a column ref to inner_get/delim_get/outer_get,
	// or a function expression like array_distance)
	std::function<unique_ptr<Expression>(const BoundColumnRefExpression &)> resolve_ref;
	resolve_ref = [&](const BoundColumnRefExpression &ref) -> unique_ptr<Expression> {
		// Check if it references a known table directly
		if (ref.binding.table_index == inner_get_table ||
		    ref.binding.table_index == outer_get_table) {
			return ref.Copy();
		}
		if (ref.binding.table_index == delim_get_table) {
			// DELIM_GET → outer table
			auto new_ref = ref.Copy();
			new_ref->Cast<BoundColumnRefExpression>().binding.table_index = outer_get_table;
			return new_ref;
		}

		// Check intermediate projections
		for (auto *proj : intermediate_projs) {
			if (ref.binding.table_index == proj->table_index &&
			    ref.binding.column_index < proj->expressions.size()) {
				auto &proj_expr = proj->expressions[ref.binding.column_index];
				if (proj_expr->type == ExpressionType::BOUND_COLUMN_REF) {
					return resolve_ref(proj_expr->Cast<BoundColumnRefExpression>());
				}
				// Struct extract: #[col.field] — the struct was built by struct_pack
				// in arg_min_nulls_last. Field N maps to inner projection column N.
				if (proj_expr->type == ExpressionType::STRUCT_EXTRACT) {
					// The struct extract has children[0] = the struct column ref,
					// and an index indicating which field. The field index maps
					// to inner projection columns (struct_pack packs them in order).
					auto &func = proj_expr->Cast<BoundFunctionExpression>();
					if (!func.children.empty() && func.children[0]->type == ExpressionType::BOUND_COLUMN_REF) {
						// The field index is in the bind_info
						auto field_idx = func.bind_info->Cast<StructExtractBindData>().index;
						if (field_idx < inner_proj.expressions.size()) {
							auto &inner_expr = inner_proj.expressions[field_idx];
							if (inner_expr->type == ExpressionType::BOUND_COLUMN_REF) {
								return resolve_ref(inner_expr->Cast<BoundColumnRefExpression>());
							}
							return inner_expr->Copy();
						}
					}
				}
				// Return as-is for other expression types
				return proj_expr->Copy();
			}
		}

		// Check the inner projection (below AGG)
		if (ref.binding.table_index == inner_proj.table_index &&
		    ref.binding.column_index < inner_proj.expressions.size()) {
			auto &inner_expr = inner_proj.expressions[ref.binding.column_index];
			if (inner_expr->type == ExpressionType::BOUND_COLUMN_REF) {
				return resolve_ref(inner_expr->Cast<BoundColumnRefExpression>());
			}
			return inner_expr->Copy();
		}

		// UNNEST table: the unnested struct fields map to inner projection columns.
		// The arg_min_nulls_last packs inner columns via struct_pack, and UNNEST
		// produces them back. This relies on struct_pack preserving the order of
		// inner projection columns (which DuckDB guarantees — struct_pack always
		// packs fields in the order they appear in the SELECT list).
		if (ref.binding.table_index == unnest.unnest_index) {
			if (ref.binding.column_index < inner_proj.expressions.size()) {
				auto &inner_expr = inner_proj.expressions[ref.binding.column_index];
				if (inner_expr->type == ExpressionType::BOUND_COLUMN_REF) {
					return resolve_ref(inner_expr->Cast<BoundColumnRefExpression>());
				}
				return inner_expr->Copy();
			}
		}

		// Aggregate group columns → these are the outer (delim_get) columns
		// Group bindings use agg.group_index, not agg.aggregate_index.
		if (ref.binding.table_index == agg.group_index) {
			if (ref.binding.column_index < agg.groups.size()) {
				auto &group_expr = agg.groups[ref.binding.column_index];
				if (group_expr->type == ExpressionType::BOUND_COLUMN_REF) {
					return resolve_ref(group_expr->Cast<BoundColumnRefExpression>());
				}
				return group_expr->Copy();
			}
		}

		// Unresolvable — signal failure
		return nullptr;
	};

	// Rebuild top projection expressions with index join references.
	// If any expression can't be resolved, bail out (don't optimize).
	vector<unique_ptr<Expression>> new_exprs;
	bool resolution_failed = false;
	for (auto &expr : top_proj.expressions) {
		if (resolution_failed) {
			break;
		}
		if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
			auto &ref = expr->Cast<BoundColumnRefExpression>();
			auto resolved = resolve_ref(ref);
			if (!resolved) {
				resolution_failed = true;
				break;
			}

			// Remap resolved refs to index join columns
			if (resolved->type == ExpressionType::BOUND_COLUMN_REF) {
				auto &rref = resolved->Cast<BoundColumnRefExpression>();
				if (rref.binding.table_index == inner_get_table) {
					rref.binding.table_index = ij_table;
				}
			} else {
				// Function expression (like distance) — replace inner/delim refs inside
				ExpressionIterator::EnumerateExpression(resolved, [&](Expression &child) {
					if (child.type == ExpressionType::BOUND_COLUMN_REF) {
						auto &cr = child.Cast<BoundColumnRefExpression>();
						if (cr.binding.table_index == inner_get_table) {
							cr.binding.table_index = ij_table;
						} else if (cr.binding.table_index == delim_get_table) {
							cr.binding.table_index = outer_get_table;
						}
					}
				});
			}
			new_exprs.push_back(std::move(resolved));
		} else {
			// Non-ref expression (e.g., arithmetic on lateral outputs).
			// Walk all column refs inside and remap them. If any ref points to
			// an internal table we can't resolve, bail out.
			auto new_expr = expr->Copy();
			bool expr_failed = false;
			ExpressionIterator::EnumerateExpression(new_expr, [&](Expression &child) {
				if (expr_failed || child.type != ExpressionType::BOUND_COLUMN_REF) {
					return;
				}
				auto &cr = child.Cast<BoundColumnRefExpression>();
				if (cr.binding.table_index == inner_get_table) {
					cr.binding.table_index = ij_table;
				} else if (cr.binding.table_index == delim_get_table) {
					cr.binding.table_index = outer_get_table;
				} else if (cr.binding.table_index != outer_get_table &&
				           cr.binding.table_index != ij_table) {
					// References an old internal table (UNNEST, AGG, intermediate proj)
					// that we can't remap in this context — bail out.
					expr_failed = true;
				}
			});
			if (expr_failed) {
				resolution_failed = true;
				break;
			}
			new_exprs.push_back(std::move(new_expr));
		}
	}

	if (resolution_failed) {
		return false;
	}

	// Create the new projection with the same table_index so parent
	// references to top_proj bindings remain valid.
	auto new_proj = make_uniq<LogicalProjection>(top_proj.table_index, std::move(new_exprs));

	// Wire up: outer_get → index_join → new_proj
	index_join->children.emplace_back(std::move(*outer_get_ptr));
	new_proj->children.emplace_back(std::move(index_join));
	new_proj->EstimateCardinality(context);

	// Replace the PROJECTION + DELIM_JOIN subtree.
	// Since the new projection uses the same table_index as the old one,
	// references from above (the root PROJECTION) remain valid.
	plan = std::move(new_proj);

	CardinalityResetter cardinality_resetter(context);
	cardinality_resetter.VisitOperator(*root);

	return true;
}

void HNSWIndexJoinOptimizer::OptimizeRecursive(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &root,
                                               unique_ptr<LogicalOperator> &plan, bool has_aggregate_above) {
	// Don't optimize lateral joins under aggregates — the GROUP BY's expressions
	// reference bindings from inside the DELIM_JOIN that we can't safely remap.
	if (!has_aggregate_above && !TryOptimize(input.optimizer.binder, input.context, root, plan)) {
		// Recursively optimize the children
		bool child_has_agg = has_aggregate_above ||
		    plan->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY;
		for (auto &child : plan->children) {
			OptimizeRecursive(input, root, child, child_has_agg);
		}
	}
}

void HNSWIndexJoinOptimizer::Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	OptimizeRecursive(input, plan, plan, false);
}

//------------------------------------------------------------------------------
// Register
//------------------------------------------------------------------------------

void HNSWModule::RegisterJoinOptimizer(DatabaseInstance &db) {
	// Register the JoinOptimizer
	OptimizerExtension::Register(db.config, HNSWIndexJoinOptimizer());
}

} // namespace duckdb
