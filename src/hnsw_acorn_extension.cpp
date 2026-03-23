#include "hnsw_acorn_extension.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "hnsw/hnsw.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	// Register the HNSW index module
	HNSWModule::Register(loader);
}

void HnswAcornExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string HnswAcornExtension::Name() {
	return "hnsw_acorn";
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(hnsw_acorn, loader) {
	duckdb::LoadInternal(loader);
}
}
