#pragma once
// Full run capture: summary + CSVs + 3D snapshots from several angles.
#include <string>
#include "core/pipeline.h"

struct AppState;

// Writes results/<run_...>/ with all data files and (if the 3D scene is
// available) PNG snapshots of all four explorer views at three angles each,
// using the current explorer parameters. Returns the folder path ("" on error).
std::string export_run(AppState& st, const prt::Results& R);
