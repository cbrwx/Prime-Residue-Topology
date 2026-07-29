#pragma once
#include "core/pipeline.h"

namespace prt {

// Honest out-of-sample evaluation.
// Density predictor: classes mod L ranked by prime density measured on
// (0, N/2] only; precision evaluated on (N/2, N] with exact integer counts
// per class. Transition predictor: next-prime residue mod q from the
// training-half transition matrix, scored on the test half.
PredReport compute_prediction(uint64_t N, uint32_t L,
                              const std::vector<uint32_t>& jointL,
                              const std::vector<uint32_t>& jointL_train,
                              const TransStats& ts);

} // namespace prt
