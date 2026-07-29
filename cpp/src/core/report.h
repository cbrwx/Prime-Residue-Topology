#pragma once
#include "core/pipeline.h"
#include <string>

namespace prt {

// Full text summary of a run (config + all headline measurements).
std::string make_summary(const Results& R);

// Create results/<run_...>/ under base_dir; returns the path ("" on failure).
std::string make_run_dir(const std::string& base_dir, const Results& R);

// Write summary.txt + CSVs of every aggregate the pipeline measured into dir.
// Returns false if any file failed to open.
bool export_data(const Results& R, const std::string& dir);

} // namespace prt
