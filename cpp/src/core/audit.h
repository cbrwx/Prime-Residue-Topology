#pragma once
// Sequence auditor: run the honest correlation machinery on ANY integer
// sequence read from a file. Detects occupancy structure (wheel-like),
// residue memory between successive terms, structure beyond difference
// statistics, and order-2 memory - all out-of-sample (first half trains,
// second half tests).
#include "core/pipeline.h"
#include <string>
#include <vector>

namespace prt {

struct AuditConfig {
    uint32_t q = 10;                 // modulus for the order-2 memory test
};

struct AuditResult {
    bool valid = false;
    std::string error;
    std::string source;

    uint64_t count = 0;
    int64_t vmin = 0, vmax = 0;
    double monotone_frac = 0;        // fraction of strictly increasing steps

    std::vector<ModulusStats> mods;  // marginal occupancy vs UNIFORM null
    std::vector<PatternRow> rows;    // q = 3..100, general-class analysis

    uint32_t q2 = 0;
    double acc_order1 = 0, acc_order2 = 0, acc_uniform = 0;
    uint64_t test_triples = 0;

    std::vector<int32_t>  diff_x;    // observed differences (|d| <= 1023)
    std::vector<uint64_t> diff_count;
    uint64_t diff_overflow = 0;
};

// Analyze an in-memory sequence (order matters; need not be monotone).
AuditResult run_audit(const std::vector<int64_t>& seq, const AuditConfig& cfg);

// Parse integers from a text file (any non-numeric separators) and audit.
AuditResult audit_file(const std::string& path, const AuditConfig& cfg);

std::string audit_summary(const AuditResult& A);

// Create results/audit_<timestamp>_<name>/ and write summary + CSVs.
// Returns folder path ("" on failure).
std::string export_audit(const AuditResult& A);

} // namespace prt
