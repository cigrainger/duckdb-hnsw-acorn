#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table/table_index_list.hpp"
#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/hnsw_index_scan.hpp"

namespace duckdb {

//-----------------------------------------------------------------------------
// Plan rewriter
//-----------------------------------------------------------------------------
class HNSWIndexScanOptimizer : public OptimizerExtension {
public:
	HNSWIndexScanOptimizer() {
		optimize_function = Optimize;
	}

	static bool TryOptimize(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		// Look for a TopN operator
		auto &op = *plan;

		if (op.type != LogicalOperatorType::LOGICAL_TOP_N) {
			return false;
		}

		auto &top_n = op.Cast<LogicalTopN>();

		if (top_n.orders.size() != 1) {
			// We can only optimize if there is a single order by expression right now
			return false;
		}

		const auto &order = top_n.orders[0];

		if (order.type != OrderType::ASCENDING) {
			// We can only optimize if the order by expression is ascending
			return false;
		}

		if (order.expression->type != ExpressionType::BOUND_COLUMN_REF) {
			// The expression has to reference the child operator (a projection with the distance function)
			return false;
		}
		const auto &bound_column_ref = order.expression->Cast<BoundColumnRefExpression>();

		// find the expression that is referenced
		if (top_n.children.size() != 1 || top_n.children.front()->type != LogicalOperatorType::LOGICAL_PROJECTION) {
			// The child has to be a projection
			return false;
		}

		auto &projection = top_n.children.front()->Cast<LogicalProjection>();

		// This the expression that is referenced by the order by expression
		const auto projection_index = bound_column_ref.binding.column_index;
		const auto &projection_expr = projection.expressions[projection_index];

		// The projection must sit on top of a get (or a filter on top of a get)
		if (projection.children.size() != 1) {
			return false;
		}

		unique_ptr<LogicalOperator> *get_ptr_ref = &projection.children.front();
		LogicalFilter *logical_filter = nullptr;
		LogicalComparisonJoin *comparison_join = nullptr;
		LogicalGet *metadata_get = nullptr;
		idx_t indexed_join_col = DConstants::INVALID_INDEX;
		idx_t metadata_join_col = DConstants::INVALID_INDEX;

		if ((*get_ptr_ref)->type == LogicalOperatorType::LOGICAL_FILTER) {
			logical_filter = &(*get_ptr_ref)->Cast<LogicalFilter>();
			if (logical_filter->children.size() != 1 ||
			    logical_filter->children.front()->type != LogicalOperatorType::LOGICAL_GET) {
				return false;
			}
			get_ptr_ref = &logical_filter->children.front();
		} else if ((*get_ptr_ref)->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			// Metadata join: PROJECTION → JOIN → [GET(indexed), GET(metadata)]
			comparison_join = &(*get_ptr_ref)->Cast<LogicalComparisonJoin>();
			if (comparison_join->join_type != JoinType::INNER || comparison_join->children.size() != 2) {
				return false;
			}
			// Must be a simple equi-join with one condition
			if (comparison_join->conditions.size() != 1 ||
			    comparison_join->conditions[0].comparison != ExpressionType::COMPARE_EQUAL) {
				return false;
			}
			auto &join_cond = comparison_join->conditions[0];

			// Both sides must be GET (seq_scan)
			auto &lhs = comparison_join->children[0];
			auto &rhs = comparison_join->children[1];
			if (lhs->type != LogicalOperatorType::LOGICAL_GET || rhs->type != LogicalOperatorType::LOGICAL_GET) {
				return false;
			}

			// Figure out which side has the HNSW index — we'll determine that below
			// during index matching. For now, try the left side first.
			// Extract join column indices from the join condition
			if (join_cond.left->type != ExpressionType::BOUND_COLUMN_REF ||
			    join_cond.right->type != ExpressionType::BOUND_COLUMN_REF) {
				return false;
			}
			auto &join_left_ref = join_cond.left->Cast<BoundColumnRefExpression>();
			auto &join_right_ref = join_cond.right->Cast<BoundColumnRefExpression>();

			auto &lhs_get = lhs->Cast<LogicalGet>();
			auto &rhs_get = rhs->Cast<LogicalGet>();

			// Determine which side is the indexed table and which is the metadata table.
			// We'll try both and see which has the HNSW index (below).
			if (join_left_ref.binding.table_index == lhs_get.table_index) {
				// Left side join column matches left GET → left is indexed, right is metadata
				// Map binding column_index through GET's column_ids to table column index
				auto lhs_col_ids = lhs_get.GetColumnIds();
				auto rhs_col_ids = rhs_get.GetColumnIds();
				if (join_left_ref.binding.column_index >= lhs_col_ids.size() ||
				    join_right_ref.binding.column_index >= rhs_col_ids.size()) {
					return false;
				}
				indexed_join_col = lhs_col_ids[join_left_ref.binding.column_index].GetPrimaryIndex();
				metadata_join_col = rhs_col_ids[join_right_ref.binding.column_index].GetPrimaryIndex();
				get_ptr_ref = &lhs;
				metadata_get = &rhs_get;
			} else {
				auto lhs_col_ids = lhs_get.GetColumnIds();
				auto rhs_col_ids = rhs_get.GetColumnIds();
				if (join_right_ref.binding.column_index >= rhs_col_ids.size() ||
				    join_left_ref.binding.column_index >= lhs_col_ids.size()) {
					return false;
				}
				indexed_join_col = rhs_col_ids[join_right_ref.binding.column_index].GetPrimaryIndex();
				metadata_join_col = lhs_col_ids[join_left_ref.binding.column_index].GetPrimaryIndex();
				get_ptr_ref = &rhs;
				metadata_get = &lhs_get;
			}
		} else if ((*get_ptr_ref)->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		auto &get_ptr = *get_ptr_ref;
		auto &get = get_ptr->Cast<LogicalGet>();
		// Check if the get is a table scan
		if (get.function.name != "seq_scan") {
			return false;
		}

		if (get.dynamic_filters && get.dynamic_filters->HasFilters()) {
			// Cant push down!
			return false;
		}

		// We have a top-n operator on top of a table scan
		// We can replace the function with a custom index scan (if the table has a custom index)

		// Get the table
		auto &table = *get.GetTable();
		if (!table.IsDuckTable()) {
			// We can only replace the scan if the table is a duck table
			return false;
		}

		auto &duck_table = table.Cast<DuckTableEntry>();
		auto &table_info = *table.GetStorage().GetDataTableInfo();

		// Find the index
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
			if (!cast_index.TryMatchDistanceFunction(projection_expr, bindings)) {
				continue;
			}
			// Check that the HNSW index actually indexes the expression
			unique_ptr<Expression> index_expr;
			if (!cast_index.TryBindIndexExpression(get, index_expr)) {
				continue;
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

			bind_data = make_uniq<HNSWIndexScanBindData>(duck_table, cast_index, top_n.limit, std::move(query_vector));
			break;
		}

		if (!bind_data) {
			// No index found
			return false;
		}

		// If this is a metadata join, populate the join info in bind_data
		if (comparison_join && metadata_get) {
			auto &meta_table = *metadata_get->GetTable();
			if (!meta_table.IsDuckTable()) {
				return false;
			}

			// Join key must be BIGINT (we read it as int64_t during scan init)
			auto idx_join_col_type = duck_table.GetColumn(LogicalIndex(indexed_join_col)).GetType();
			if (idx_join_col_type.id() != LogicalTypeId::BIGINT) {
				return false;
			}

			bind_data->metadata_table = &meta_table;
			bind_data->indexed_join_column = indexed_join_col;
			bind_data->metadata_join_column = metadata_join_col;

			// Set up metadata scan: reuse ExtractFiltersIntoBind pattern for filter columns,
			// then prepend the join key column.
			auto &meta_duck_table = meta_table.Cast<DuckTableEntry>();

			// Set up metadata scan columns: join key at position 0, then filter columns.
			// Filter keys must be remapped to positions in our scan column_ids vector
			// (DuckDB's TableFilterSet keys are indices into the column_ids vector).
			auto &meta_join_col = meta_duck_table.GetColumn(LogicalIndex(metadata_join_col));
			bind_data->metadata_scan_column_ids.emplace_back(StorageIndex(meta_join_col.StorageOid()));
			bind_data->metadata_scan_types.push_back(meta_join_col.GetType());

			idx_t meta_scan_pos = 1; // filter columns start after join key
			for (const auto &entry : metadata_get->table_filters.filters) {
				auto table_col_idx = entry.first;
				auto &col = meta_duck_table.GetColumn(LogicalIndex(table_col_idx));
				bind_data->metadata_scan_column_ids.emplace_back(StorageIndex(col.StorageOid()));
				bind_data->metadata_scan_types.push_back(col.GetType());
				bind_data->metadata_filters.filters[meta_scan_pos] = entry.second->Copy();
				meta_scan_pos++;
			}
		}

		// Push table filters into the bind data for filtered HNSW search
		ExtractFiltersIntoBind(duck_table, get, *bind_data);

		const auto cardinality = get.function.cardinality(context, bind_data.get());
		get.function = HNSWIndexScanFunction::GetFunction();
		get.has_estimated_cardinality = cardinality->has_estimated_cardinality;
		get.estimated_cardinality = cardinality->estimated_cardinality;
		get.bind_data = std::move(bind_data);

		if (comparison_join) {
			// Metadata join: keep the TOP_N above the JOIN to enforce LIMIT
			// after the JOIN re-attaches metadata columns. The HNSW_INDEX_SCAN
			// uses ACORN-1 with the metadata-derived bitset for filtered search,
			// but the JOIN can duplicate rows (non-1:1 keys), so TOP_N must stay.
			return true;
		}

		if (get.table_filters.filters.empty()) {
			// No filters — just remove TopN
			plan = std::move(top_n.children[0]);
			return true;
		}

		// Pull up the filters as a LogicalFilter node (keeps plan structure intact)
		get.projection_ids.clear();
		get.types.clear();

		auto new_filter = make_uniq<LogicalFilter>();
		auto &get_column_ids = get.GetColumnIds();
		for (const auto &entry : get.table_filters.filters) {
			idx_t column_id = entry.first;
			auto &type = get.returned_types[column_id];
			bool found = false;
			for (idx_t i = 0; i < get_column_ids.size(); i++) {
				if (get_column_ids[i].GetPrimaryIndex() == column_id) {
					column_id = i;
					found = true;
					break;
				}
			}
			if (!found) {
				throw InternalException("Could not find column id for filter");
			}
			auto column = make_uniq<BoundColumnRefExpression>(type, ColumnBinding(get.table_index, column_id));
			new_filter->expressions.push_back(entry.second->ToExpression(*column));
		}
		new_filter->children.push_back(std::move(get_ptr));
		new_filter->ResolveOperatorTypes();
		get_ptr = std::move(new_filter);

		// Remove the TopN operator
		plan = std::move(top_n.children[0]);
		return true;
	}

	static bool OptimizeChildren(ClientContext &context, unique_ptr<LogicalOperator> &plan) {

		auto ok = TryOptimize(context, plan);
		// Recursively optimize the children
		for (auto &child : plan->children) {
			ok |= OptimizeChildren(context, child);
		}
		return ok;
	}

	static void MergeProjections(unique_ptr<LogicalOperator> &plan) {
		if (plan->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			if (plan->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &child = plan->children[0];

				if (child->children[0]->type == LogicalOperatorType::LOGICAL_GET &&
				    child->children[0]->Cast<LogicalGet>().function.name == "hnsw_index_scan") {
					auto &parent_projection = plan->Cast<LogicalProjection>();
					auto &child_projection = child->Cast<LogicalProjection>();

					column_binding_set_t referenced_bindings;
					for (auto &expr : parent_projection.expressions) {
						ExpressionIterator::EnumerateExpression(expr, [&](Expression &expr_ref) {
							if (expr_ref.type == ExpressionType::BOUND_COLUMN_REF) {
								auto &bound_column_ref = expr_ref.Cast<BoundColumnRefExpression>();
								referenced_bindings.insert(bound_column_ref.binding);
							}
						});
					}

					auto child_bindings = child_projection.GetColumnBindings();
					for (idx_t i = 0; i < child_projection.expressions.size(); i++) {
						auto &expr = child_projection.expressions[i];
						auto &outgoing_binding = child_bindings[i];

						if (referenced_bindings.find(outgoing_binding) == referenced_bindings.end()) {
							// The binding is not referenced
							// We can remove this expression. But positionality matters so just replace with int.
							expr = make_uniq_base<Expression, BoundConstantExpression>(Value(LogicalType::TINYINT));
						}
					}
					return;
				}
			}
		}
		for (auto &child : plan->children) {
			MergeProjections(child);
		}
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		auto did_use_hnsw_scan = OptimizeChildren(input.context, plan);
		if (did_use_hnsw_scan) {
			MergeProjections(plan);
		}
	}
};

//-----------------------------------------------------------------------------
// Register
//-----------------------------------------------------------------------------
void HNSWModule::RegisterScanOptimizer(DatabaseInstance &db) {
	// Register the optimizer extension
	OptimizerExtension::Register(db.config, HNSWIndexScanOptimizer());
}

} // namespace duckdb
