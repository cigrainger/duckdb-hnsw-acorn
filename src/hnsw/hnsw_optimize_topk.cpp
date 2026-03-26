#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table/table_index_list.hpp"
#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/hnsw_index_scan.hpp"

namespace duckdb {

//------------------------------------------------------------------------------
// Optimizer Helpers
//------------------------------------------------------------------------------

static unique_ptr<Expression> CreateListOrderByExpr(ClientContext &context, unique_ptr<Expression> elem_expr,
                                                    unique_ptr<Expression> order_expr,
                                                    unique_ptr<Expression> filter_expr) {
	auto func_entry =
	    Catalog::GetEntry<AggregateFunctionCatalogEntry>(context, "", "", "list", OnEntryNotFound::RETURN_NULL);
	if (!func_entry) {
		return nullptr;
	}

	auto func = func_entry->functions.GetFunctionByOffset(0);
	vector<unique_ptr<Expression>> arguments;
	arguments.push_back(std::move(elem_expr));

	auto agg_bind_data = func.bind(context, func, arguments);
	auto new_agg_expr =
	    make_uniq<BoundAggregateExpression>(func, std::move(arguments), std::move(std::move(filter_expr)),
	                                        std::move(agg_bind_data), AggregateType::NON_DISTINCT);

	// We also need to order the list items by the distance
	BoundOrderByNode order_by_node(OrderType::ASCENDING, OrderByNullType::NULLS_LAST, std::move(order_expr));
	new_agg_expr->order_bys = make_uniq<BoundOrderModifier>();
	new_agg_expr->order_bys->orders.push_back(std::move(order_by_node));

	return std::move(new_agg_expr);
}

//------------------------------------------------------------------------------
// Main Optimizer
//------------------------------------------------------------------------------
// This optimizer rewrites
//
//	AGG(MIN_BY(t1.col1, distance_func(t1.col2, query_vector), k)) <- TABLE_SCAN(t1)
//  =>
//	AGG(LIST(col1 ORDER BY distance_func(col2, query_vector) ASC)) <- HNSW_INDEX_SCAN(t1, query_vector, k)
//

class HNSWTopKOptimizer : public OptimizerExtension {
public:
	HNSWTopKOptimizer() {
		optimize_function = Optimize;
	}

	static bool TryOptimize(Binder &binder, ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		// Look for a Aggregate operator
		if (plan->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
			return false;
		}
		// Look for a expression that is a distance expression
		auto &agg = plan->Cast<LogicalAggregate>();
		if (agg.expressions.size() != 1) {
			return false;
		}

		auto &agg_expr = agg.expressions[0];
		if (agg_expr->type != ExpressionType::BOUND_AGGREGATE) {
			return false;
		}
		auto &agg_func_expr = agg_expr->Cast<BoundAggregateExpression>();
		if (agg_func_expr.function.name != "min_by") {
			return false;
		}
		if (agg_func_expr.children.size() != 3) {
			return false;
		}
		if (agg_func_expr.children[2]->type != ExpressionType::VALUE_CONSTANT) {
			return false;
		}
		const auto &col_expr = agg_func_expr.children[0];
		const auto &dist_expr = agg_func_expr.children[1];
		const auto &limit_expr = agg_func_expr.children[2];

		// we need the aggregate to be on top of a projection
		if (agg.children.size() != 1) {
			return false;
		}

		// The child must be a table scan (possibly through a projection for compression)
		unique_ptr<LogicalOperator> *get_ptr_ref = &agg.children[0];
		if ((*get_ptr_ref)->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			if ((*get_ptr_ref)->children.size() != 1 ||
			    (*get_ptr_ref)->children[0]->type != LogicalOperatorType::LOGICAL_GET) {
				return false;
			}
			get_ptr_ref = &(*get_ptr_ref)->children[0];
		} else if ((*get_ptr_ref)->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		auto &get_ptr = *get_ptr_ref;
		auto &get = get_ptr->Cast<LogicalGet>();
		if (get.function.name != "seq_scan") {
			return false;
		}

		if (get.dynamic_filters && get.dynamic_filters->HasFilters()) {
			// Cant push down!
			return false;
		}

		// Get the table
		auto &table = *get.GetTable();
		if (!table.IsDuckTable()) {
			return false;
		}

		auto &duck_table = table.Cast<DuckTableEntry>();
		auto &table_info = *table.GetStorage().GetDataTableInfo();

		unique_ptr<HNSWIndexScanBindData> bind_data = nullptr;
		vector<reference<Expression>> bindings;

		table_info.BindIndexes(context, HNSWIndex::TYPE_NAME);
		for (auto &index : table_info.GetIndexes().Indexes()) {
			if (!index.IsBound() || HNSWIndex::TYPE_NAME != index.GetIndexType()) {
				continue;
			}
			auto &cast_index = index.Cast<HNSWIndex>();

			// Reset the bindings
			bindings.clear();

			// Check that the projection expression is a distance function that matches the index
			if (!cast_index.TryMatchDistanceFunction(dist_expr, bindings)) {
				continue;
			}
			// Check that the HNSW index actually indexes the expression
			unique_ptr<Expression> index_expr;
			if (!cast_index.TryBindIndexExpression(get, index_expr)) {
				continue;
			}

			// If there's a compression projection between AGGREGATE and GET,
			// the dist_expr's vec binding uses the projection's table_index,
			// not the GET's. Rebind the index_expr to match.
			if (agg.children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &compress_proj = agg.children[0]->Cast<LogicalProjection>();
				ExpressionIterator::EnumerateExpression(index_expr, [&](Expression &child) {
					if (child.type == ExpressionType::BOUND_COLUMN_REF) {
						auto &ref = child.Cast<BoundColumnRefExpression>();
						if (ref.binding.table_index == get.table_index) {
							// Find which projection column maps to this GET column
							auto get_col_ids = get.GetColumnIds();
							for (idx_t i = 0; i < compress_proj.expressions.size(); i++) {
								if (compress_proj.expressions[i]->type == ExpressionType::BOUND_COLUMN_REF) {
									auto &proj_ref = compress_proj.expressions[i]->Cast<BoundColumnRefExpression>();
									if (proj_ref.binding.table_index == get.table_index &&
									    proj_ref.binding.column_index == ref.binding.column_index) {
										ref.binding.table_index = compress_proj.table_index;
										ref.binding.column_index = i;
										break;
									}
								}
							}
						}
					}
				});
			}

			// Now, ensure that one of the bindings is a constant vector, and the other our index expression
			auto &const_expr_ref = bindings[1];
			auto &index_expr_ref = bindings[2];

			if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT || !index_expr->Equals(index_expr_ref)) {
				// Swap the bindings and try again
				std::swap(const_expr_ref, index_expr_ref);
				if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
				    !index_expr->Equals(index_expr_ref)) {
					// Nope, not a match, we can't optimize.
					continue;
				}
			}

			const auto vector_size = cast_index.GetVectorSize();
			const auto &matched_vector = const_expr_ref.get().Cast<BoundConstantExpression>().value;

			auto query_vector = make_unsafe_uniq_array<float>(vector_size);
			auto vector_elements = ArrayValue::GetChildren(matched_vector);
			for (idx_t i = 0; i < vector_size; i++) {
				query_vector[i] = vector_elements[i].GetValue<float>();
			}
			auto k_limit = limit_expr->Cast<BoundConstantExpression>().value.GetValue<int32_t>();
			if (k_limit <= 0 || k_limit >= STANDARD_VECTOR_SIZE) {
				continue;
			}
			bind_data = make_uniq<HNSWIndexScanBindData>(duck_table, cast_index, k_limit, std::move(query_vector));
			break;
		}

		if (!bind_data) {
			// No index found
			return false;
		}

		// Push table filters into the bind data for filtered HNSW search
		ExtractFiltersIntoBind(duck_table, get, *bind_data);

		// For grouped aggregation, set the group column for per-group ACORN-1 search.
		// The group expression must be a simple column ref on the same table.
		if (!agg.groups.empty()) {
			if (agg.groups.size() != 1 || agg.groups[0]->type != ExpressionType::BOUND_COLUMN_REF) {
				return false;
			}
			auto &group_ref = agg.groups[0]->Cast<BoundColumnRefExpression>();
			// Resolve through compression projection if present
			idx_t group_col_idx = group_ref.binding.column_index;
			if (agg.children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &proj = agg.children[0]->Cast<LogicalProjection>();
				if (group_ref.binding.table_index == proj.table_index &&
				    group_col_idx < proj.expressions.size()) {
					auto &proj_expr = proj.expressions[group_col_idx];
					if (proj_expr->type == ExpressionType::BOUND_COLUMN_REF) {
						group_col_idx = proj_expr->Cast<BoundColumnRefExpression>().binding.column_index;
					}
				}
			}
			// Map through GET's column_ids to table column index
			auto get_col_ids = get.GetColumnIds();
			if (group_col_idx < get_col_ids.size()) {
				bind_data->group_column = get_col_ids[group_col_idx].GetPrimaryIndex();
			} else {
				return false;
			}
		}

		// Replace the aggregate with a index scan + projection
		get.function = HNSWIndexScanFunction::GetFunction();
		const auto cardinality = get.function.cardinality(context, bind_data.get());
		get.has_estimated_cardinality = cardinality->has_estimated_cardinality;
		get.estimated_cardinality = cardinality->estimated_cardinality;
		get.bind_data = std::move(bind_data);

		// For ungrouped: replace min_by with LIST (index scan returns exactly K results)
		// For grouped: keep min_by (per-group ACORN-1 search handles the filtering,
		//              min_by selects top-K per group from the per-group results)
		if (agg.groups.empty()) {
			agg.expressions[0] = CreateListOrderByExpr(context, col_expr->Copy(), dist_expr->Copy(),
			                                           agg_func_expr.filter ? agg_func_expr.filter->Copy() : nullptr);
		}

		return true;
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		if (!TryOptimize(input.optimizer.binder, input.context, plan)) {
			// Recursively optimize the children
			for (auto &child : plan->children) {
				Optimize(input, child);
			}
		}
	}
};

void HNSWModule::RegisterTopKOptimizer(DatabaseInstance &db) {
	// Register the TopKOptimizer
	OptimizerExtension::Register(db.config, HNSWTopKOptimizer());
}

} // namespace duckdb
