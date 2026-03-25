#include "hnsw/rabitq.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"

#include <cmath>
#include <cstring>

namespace duckdb {

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

idx_t RaBitQBytesPerVector(idx_t dimensions) {
	// Binary bits (ceil(D/8)) + norm (float) + vdot (float)
	return ((dimensions + 7) / 8) + sizeof(float) * 2;
}

static idx_t HammingDistance(const uint8_t *a, const uint8_t *b, idx_t binary_bytes) {
	idx_t hamming = 0;
	idx_t i = 0;

	// Process 8 bytes at a time using 64-bit popcount
	for (; i + 8 <= binary_bytes; i += 8) {
		uint64_t xa;
		uint64_t xb;
		std::memcpy(&xa, a + i, sizeof(uint64_t));
		std::memcpy(&xb, b + i, sizeof(uint64_t));
#if defined(__GNUC__) || defined(__clang__)
		hamming += static_cast<idx_t>(__builtin_popcountll(xa ^ xb));
#elif defined(_MSC_VER)
		hamming += static_cast<idx_t>(__popcnt64(xa ^ xb));
#else
		uint64_t v = xa ^ xb;
		while (v) {
			hamming++;
			v &= v - 1;
		}
#endif
	}

	// Handle remaining bytes
	for (; i < binary_bytes; i++) {
#if defined(__GNUC__) || defined(__clang__)
		hamming += static_cast<idx_t>(__builtin_popcount(a[i] ^ b[i]));
#else
		uint8_t v = a[i] ^ b[i];
		while (v) {
			hamming++;
			v &= v - 1;
		}
#endif
	}

	return hamming;
}

//------------------------------------------------------------------------------
// Centroid Computation
//------------------------------------------------------------------------------

vector<float> RaBitQComputeCentroid(ColumnDataCollection &collection, idx_t dimensions, bool normalize_inputs) {
	vector<float> centroid(dimensions, 0.0f);
	idx_t count = 0;

	DataChunk scan_chunk;
	collection.InitializeScanChunk(scan_chunk);
	ColumnDataScanState scan_state;
	collection.InitializeScan(scan_state);

	while (collection.Scan(scan_state, scan_chunk)) {
		auto &vec_vec = scan_chunk.data[0];
		auto vec_count = scan_chunk.size();

		UnifiedVectorFormat vec_format;
		vec_vec.ToUnifiedFormat(vec_count, vec_format);

		auto &data_vec = ArrayVector::GetEntry(vec_vec);
		UnifiedVectorFormat data_format;
		data_vec.ToUnifiedFormat(vec_count * dimensions, data_format);
		auto data_ptr = UnifiedVectorFormat::GetData<float>(data_format);

		for (idx_t i = 0; i < vec_count; i++) {
			auto vec_idx = vec_format.sel->get_index(i);
			if (!vec_format.validity.RowIsValid(vec_idx)) {
				continue;
			}

			const float *vec = data_ptr + vec_idx * dimensions;

			if (normalize_inputs) {
				// For cosine metric: normalize each vector to unit length before averaging
				float norm_sq = 0.0f;
				for (idx_t d = 0; d < dimensions; d++) {
					norm_sq += vec[d] * vec[d];
				}
				float inv_norm = (norm_sq > 0.0f) ? (1.0f / std::sqrt(norm_sq)) : 0.0f;
				for (idx_t d = 0; d < dimensions; d++) {
					centroid[d] += vec[d] * inv_norm;
				}
			} else {
				for (idx_t d = 0; d < dimensions; d++) {
					centroid[d] += vec[d];
				}
			}
			count++;
		}
	}

	if (count > 0) {
		float inv_count = 1.0f / static_cast<float>(count);
		for (idx_t d = 0; d < dimensions; d++) {
			centroid[d] *= inv_count;
		}
	}

	return centroid;
}

//------------------------------------------------------------------------------
// Vector Quantization
//------------------------------------------------------------------------------

void RaBitQQuantizeVector(const float *input, const RaBitQState &state, uint8_t *output, bool normalize_input) {
	auto dimensions = state.dimensions;
	auto binary_bytes = (dimensions + 7) / 8;

	// Heap-allocate residual buffer — safe for any dimension count.
	auto residual_buf = make_uniq_array<float>(dimensions);
	auto residual = residual_buf.get();

	// Step 1: Optionally normalize input (for cosine metric)
	if (normalize_input) {
		float input_norm = 0.0f;
		for (idx_t d = 0; d < dimensions; d++) {
			input_norm += input[d] * input[d];
		}
		input_norm = std::sqrt(input_norm);
		if (input_norm > 0.0f) {
			float inv_norm = 1.0f / input_norm;
			for (idx_t d = 0; d < dimensions; d++) {
				residual[d] = input[d] * inv_norm - state.centroid[d];
			}
		} else {
			for (idx_t d = 0; d < dimensions; d++) {
				residual[d] = -state.centroid[d];
			}
		}
	} else {
		// Step 2: Subtract centroid
		for (idx_t d = 0; d < dimensions; d++) {
			residual[d] = input[d] - state.centroid[d];
		}
	}

	// Step 3: Compute norm of residual
	float norm_sq = 0.0f;
	for (idx_t d = 0; d < dimensions; d++) {
		norm_sq += residual[d] * residual[d];
	}
	float norm = std::sqrt(norm_sq);

	// Step 4: Normalize to unit vector and sign-bit quantize
	// Also compute vdot = <unit, sign(unit)> = sum(|unit[d]|)
	float vdot = 0.0f;
	std::memset(output, 0, binary_bytes);

	if (norm > 0.0f) {
		float inv_norm = 1.0f / norm;
		for (idx_t d = 0; d < dimensions; d++) {
			float unit_d = residual[d] * inv_norm;
			// Set bit if >= 0 (sign bit quantization)
			if (unit_d >= 0.0f) {
				output[d / 8] |= (1u << (d % 8));
			}
			// vdot accumulates |unit_d|, since sign(unit_d) * unit_d = |unit_d|
			vdot += std::fabs(unit_d);
		}
		// Normalize vdot: in RaBitQ, the binary vector has entries ±1/sqrt(D),
		// so <unit, binary> = vdot / sqrt(D). We store vdot / sqrt(D).
		vdot /= std::sqrt(static_cast<float>(dimensions));
	} else {
		// Zero vector: all bits 1 (arbitrary), corrections are zero
		std::memset(output, 0xFF, binary_bytes);
	}

	// Step 5: Write correction factors after the binary bits
	std::memcpy(output + binary_bytes, &norm, sizeof(float));
	std::memcpy(output + binary_bytes + sizeof(float), &vdot, sizeof(float));
}

//------------------------------------------------------------------------------
// Distance Functions
//------------------------------------------------------------------------------

float RaBitQDistanceL2sq(std::size_t a_ptr, std::size_t b_ptr, std::size_t ctx_ptr) {
	auto &ctx = *reinterpret_cast<const RaBitQDistanceContext *>(ctx_ptr);
	auto a = reinterpret_cast<const uint8_t *>(a_ptr);
	auto b = reinterpret_cast<const uint8_t *>(b_ptr);
	auto bb = ctx.binary_bytes;
	auto D = ctx.dimensions;

	// Extract correction factors
	float norm_a, vdot_a, norm_b, vdot_b;
	std::memcpy(&norm_a, a + bb, sizeof(float));
	std::memcpy(&vdot_a, a + bb + sizeof(float), sizeof(float));
	std::memcpy(&norm_b, b + bb, sizeof(float));
	std::memcpy(&vdot_b, b + bb + sizeof(float), sizeof(float));

	// Hamming distance between sign-bit vectors
	auto hamming = HammingDistance(a, b, bb);

	// Convert hamming to dot product in {-1, +1} space:
	// <b_a, b_b> = D - 2*hamming
	float binary_dot = static_cast<float>(D) - 2.0f * static_cast<float>(hamming);

	// RaBitQ inner product estimator for unit vectors u_a, u_b:
	//   binary_dot = <sign_a, sign_b> in {-1,+1}^D = D - 2*hamming
	//   Reconstructed binary vectors are ±1/sqrt(D), so <recon_a, recon_b> = binary_dot / D
	//   vdot_i = <u_i, recon_i> captures correlation between unit vector and its binary rep.
	//   Estimator: <u_a, u_b> ≈ vdot_a * vdot_b * binary_dot / D
	float unit_dot_approx = vdot_a * vdot_b * binary_dot / static_cast<float>(D);

	// ||v_a - v_b||^2 = ||r_a - r_b||^2  (centroid cancels)
	//                  = norm_a^2 + norm_b^2 - 2 * norm_a * norm_b * <u_a, u_b>
	return norm_a * norm_a + norm_b * norm_b - 2.0f * norm_a * norm_b * unit_dot_approx;
}

//------------------------------------------------------------------------------
// Exact Distance Functions (for rescoring)
//------------------------------------------------------------------------------

float ExactDistanceL2sq(const float *a, const float *b, idx_t dimensions) {
	float dist = 0.0f;
	for (idx_t d = 0; d < dimensions; d++) {
		float diff = a[d] - b[d];
		dist += diff * diff;
	}
	return dist;
}

float ExactDistanceCosine(const float *a, const float *b, idx_t dimensions) {
	float dot = 0.0f;
	float norm_a = 0.0f;
	float norm_b = 0.0f;
	for (idx_t d = 0; d < dimensions; d++) {
		dot += a[d] * b[d];
		norm_a += a[d] * a[d];
		norm_b += b[d] * b[d];
	}
	float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
	if (denom == 0.0f) {
		return 1.0f;
	}
	return 1.0f - (dot / denom);
}

float ExactDistanceIP(const float *a, const float *b, idx_t dimensions) {
	float dot = 0.0f;
	for (idx_t d = 0; d < dimensions; d++) {
		dot += a[d] * b[d];
	}
	return -dot; // negative inner product (lower = more similar)
}

} // namespace duckdb
