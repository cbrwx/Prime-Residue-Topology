#pragma once
// Shared state between the window shell and the UI panels.
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include "core/pipeline.h"
#include "core/audit.h"
#include "app/render3d.h"

struct AppState {
    // ---- pipeline ----
    prt::Config cfg;
    prt::Progress progress;
    std::thread worker;
    std::atomic<bool> running{false};
    std::shared_ptr<prt::Results> results;   // last completed run (may be null)
    std::mutex results_mx;
    std::atomic<uint64_t> generation{0};     // bumped when results change

    // ---- UI selections ----
    int    n_choice = 3;                     // index into N presets
    int    L_choice = 1;                     // 0: 210, 1: 2310
    int    mod_sel = 1;                      // selected statistics modulus
    bool   gaps_log = true;
    bool   races_logx = true;

    // ---- 3D explorer ----
    prt3d::Scene scene;
    bool scene_ok = false;
    prt3d::Camera cam;
    int   view_mode = 1;                     // 0 strands, 1 lattice, 2 torus, 3 spectral
    int   strands_m = 30;
    int   strands_points = 10'000'000;
    int   lat_m1 = 7, lat_m2 = 11, lat_m3 = 30;   // must divide L
    int   torus_m1 = 105, torus_m2 = 22;
    int   pair_gmax = 120;                        // gap axis extent in pair space
    float point_size = 22.0f;   // default matches the lattice start view
    bool  glow = false;
    // cache key for rebuilding the point cloud only when inputs change
    uint64_t cloud_key = ~0ull;

    // ---- run capture ----
    const void* last_export_ptr = nullptr;   // Results* already exported
    std::string last_export_path;

    // ---- sequence auditor ----
    std::thread audit_worker;
    std::atomic<bool> audit_running{false};
    std::shared_ptr<prt::AuditResult> audit;
    std::mutex audit_mx;
    char audit_path[512] = "";
    const void* audit_export_ptr = nullptr;
    std::string audit_export_path;

    void start_audit() {
        if (audit_running.load()) return;
        if (audit_worker.joinable()) audit_worker.join();
        audit_running.store(true);
        const std::string path = audit_path;
        prt::AuditConfig ac;
        ac.q = cfg.q;
        audit_worker = std::thread([this, path, ac]() {
            auto r = std::make_shared<prt::AuditResult>(prt::audit_file(path, ac));
            {
                std::lock_guard<std::mutex> lk(audit_mx);
                audit = std::move(r);
            }
            audit_running.store(false);
        });
    }

    std::shared_ptr<prt::AuditResult> get_audit() {
        std::lock_guard<std::mutex> lk(audit_mx);
        return audit;
    }

    void start_run() {
        if (running.load()) return;
        if (worker.joinable()) worker.join();
        progress.cancel.store(false);
        progress.frac.store(0);
        progress.stage.store(1);
        running.store(true);
        prt::Config c = cfg;
        worker = std::thread([this, c]() {
            auto r = std::make_shared<prt::Results>(prt::run_pipeline(c, &progress));
            {
                std::lock_guard<std::mutex> lk(results_mx);
                if (r->valid) results = std::move(r);
            }
            generation.fetch_add(1);
            running.store(false);
        });
    }

    void cancel_run() { progress.cancel.store(true); }

    std::shared_ptr<prt::Results> get_results() {
        std::lock_guard<std::mutex> lk(results_mx);
        return results;
    }

    ~AppState() {
        progress.cancel.store(true);
        if (worker.joinable()) worker.join();
        if (audit_worker.joinable()) audit_worker.join();
    }
};

// Draws the whole UI for one frame.
void draw_ui(AppState& st);
