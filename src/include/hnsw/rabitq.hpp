#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/vector.hpp"
#include "usearch/index_plugins.hpp"

namespace duckdb {

class ColumnDataCollection;

//------------------------------------------------------------------------------
// RaBitQ State
//------------------------------------------------------------------------------

struct RaBitQState {
	vector<float> centroid;
	idx_t dimensions;
	unum::usearch::metric_kind_t original_metric;
};

struct RaBitQDistanceContext {
	idx_t dimensions;
	idx_t binary_bytes; // ceil(dimensions / 8)
};

//------------------------------------------------------------------------------
// RaBitQ Functions
//------------------------------------------------------------------------------

//! Compute the number of bytes per quantized vector: ceil(D/8) for bits + 2 floats for corrections
idx_t RaBitQBytesPerVector(idx_t dimensions);

//! Compute centroid from all vectors in a collection. The collection must have
//! the vector column at index 0. If normalize_inputs is true, each vector is
//! L2-normalized before averaging (used for cosine metric).
vector<float> RaBitQComputeCentroid(ColumnDataCollection &collection, idx_t dimensions, bool normalize_inputs = false);

//! Quantize a single F32 vector into RaBitQ format.
//! Output layout: [D/8 bytes sign bits] [float norm] [float vdot]
//! If normalize_input is true, the input vector is normalized to unit length
//! before computing the residual (used for cosine metric).
void RaBitQQuantizeVector(const float *input, const RaBitQState &state, uint8_t *output, bool normalize_input = false);

//! RaBitQ approximate distance function (L2 squared estimator).
//! Used for all metrics during HNSW graph construction/traversal. For cosine,
//! vectors are pre-normalized so L2sq preserves cosine ordering.
//! Signature: (uptr_t a, uptr_t b, uptr_t state) -> float
float RaBitQDistanceL2sq(std::size_t a_ptr, std::size_t b_ptr, std::size_t ctx_ptr);

//! Compute exact L2 squared distance between two F32 vectors
float ExactDistanceL2sq(const float *a, const float *b, idx_t dimensions);

//! Compute exact cosine distance between two F32 vectors
float ExactDistanceCosine(const float *a, const float *b, idx_t dimensions);

//! Compute exact negative inner product between two F32 vectors
float ExactDistanceIP(const float *a, const float *b, idx_t dimensions);

} // namespace duckdb
