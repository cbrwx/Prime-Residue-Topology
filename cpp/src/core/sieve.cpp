#include "core/sieve.h"
#include <cmath>
#include <bit>
#include <algorithm>
#include <deque>
#include <future>
#include <thread>

namespace prt {

std::vector<uint32_t> simple_sieve(uint32_t limit) {
    std::vector<uint32_t> primes;
    if (limit < 2) return primes;
    std::vector<uint8_t> comp(limit + 1, 0);
    for (uint32_t i = 2; i <= limit; ++i) {
        if (!comp[i]) {
            primes.push_back(i);
            if ((uint64_t)i * i <= limit)
                for (uint64_t j = (uint64_t)i * i; j <= limit; j += i)
                    comp[j] = 1;
        }
    }
    return primes;
}

// Sieve odd numbers in [lo, hi), lo odd. Returns primes found, in order.
static std::vector<uint64_t> sieve_segment(uint64_t lo, uint64_t hi,
                                           const std::vector<uint32_t>* base) {
    const uint64_t nbits = (hi - lo + 1) / 2;      // odd numbers in [lo, hi)
    std::vector<uint64_t> bits((nbits + 63) / 64, 0);

    for (uint32_t p : *base) {
        if (p == 2) continue;
        const uint64_t p2 = (uint64_t)p * p;
        if (p2 >= hi) break;
        uint64_t start = ((lo + p - 1) / p) * p;    // first multiple >= lo
        if (start < p2) start = p2;                 // composites below p^2 already covered
        if ((start & 1) == 0) start += p;           // odd multiples only
        for (uint64_t j = start; j < hi; j += 2ull * p) {
            const uint64_t idx = (j - lo) >> 1;
            bits[idx >> 6] |= 1ull << (idx & 63);
        }
    }

    std::vector<uint64_t> out;
    out.reserve((size_t)(nbits / 8) + 16);
    for (size_t w = 0; w < bits.size(); ++w) {
        uint64_t x = ~bits[w];
        if (w == bits.size() - 1) {
            const uint64_t rem = nbits - (uint64_t)w * 64;
            if (rem < 64) x &= (1ull << rem) - 1;
        }
        while (x) {
            const int b = std::countr_zero(x);
            x &= x - 1;
            out.push_back(lo + 2ull * ((uint64_t)w * 64 + (uint64_t)b));
        }
    }
    return out;
}

bool for_each_prime(uint64_t N,
                    const std::function<void(const uint64_t*, size_t)>& batch,
                    SieveProgress* prog, unsigned threads) {
    if (N < 2) return true;
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());

    const uint64_t two = 2;
    batch(&two, 1);
    if (prog) prog->done_upto.store(2);
    if (N == 2) return true;

    uint32_t root = (uint32_t)std::sqrt((double)N) + 2;
    static std::vector<uint32_t> base;              // reused across calls (single pipeline at a time)
    base = simple_sieve(root);

    const uint64_t SEG = 1ull << 22;                // numbers per segment (~4.2M => 256 KB bitmap)
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (uint64_t s = 3; s <= N; s += SEG) {
        uint64_t a = s, b = std::min(N + 1, s + SEG);
        if ((a & 1) == 0) ++a;
        if (a < b) ranges.emplace_back(a, b);
    }

    std::deque<std::future<std::vector<uint64_t>>> inflight;
    size_t next = 0;
    const size_t window = (size_t)threads + 2;
    auto launch = [&]() {
        if (next < ranges.size()) {
            auto [a, b] = ranges[next++];
            inflight.push_back(std::async(std::launch::async, sieve_segment, a, b, &base));
        }
    };
    for (size_t i = 0; i < window; ++i) launch();

    size_t seg_idx = 0;
    while (!inflight.empty()) {
        std::vector<uint64_t> primes = inflight.front().get();
        inflight.pop_front();
        launch();
        if (prog && prog->cancel && prog->cancel->load()) {
            // drain remaining futures before returning
            while (!inflight.empty()) { inflight.front().wait(); inflight.pop_front(); }
            return false;
        }
        if (!primes.empty()) batch(primes.data(), primes.size());
        if (prog) prog->done_upto.store(ranges[seg_idx].second - 1);
        ++seg_idx;
    }
    return true;
}

} // namespace prt
