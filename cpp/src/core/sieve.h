#pragma once
// Multithreaded segmented sieve of Eratosthenes.
// Primes are delivered strictly in increasing order, in batches, so downstream
// accumulators can rely on ordering (gaps, consecutive-prime transitions).
#include <cstdint>
#include <vector>
#include <functional>
#include <atomic>

namespace prt {

struct SieveProgress {
    std::atomic<uint64_t> done_upto{0};   // all primes <= this value have been delivered
    std::atomic<bool>*    cancel = nullptr;
};

// Simple sieve, returns all primes <= limit.
std::vector<uint32_t> simple_sieve(uint32_t limit);

// Calls batch(ptr, count) with strictly increasing primes covering [2, N].
// Segments are sieved in parallel; delivery is sequential and ordered.
// Returns false if cancelled via prog->cancel.
bool for_each_prime(uint64_t N,
                    const std::function<void(const uint64_t*, size_t)>& batch,
                    SieveProgress* prog = nullptr,
                    unsigned threads = 0);

} // namespace prt
