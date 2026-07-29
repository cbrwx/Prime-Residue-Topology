#include "app/run_export.h"
#include "app/app_state.h"
#include "app/clouds.h"
#include "core/report.h"
#include <cstdio>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

std::string export_run(AppState& st, const prt::Results& R) {
    const std::string dir = prt::make_run_dir("results", R);
    if (dir.empty()) return dir;

    prt::export_data(R, dir);

    if (st.scene_ok) {
        struct ViewDef {
            const char* name;
            int mode;
            float psize;
            bool glow;
        };
        const ViewDef views[] = {
            {"strands", 0, 3.0f, true},
            {"lattice", 1, 22.0f, false},
            {"torus", 2, 18.0f, false},
            {"spectral", 3, 26.0f, false},
            {"pairspace", 4, 16.0f, false},
        };
        struct Angle { float yaw, pitch; };
        const Angle angles[] = {{0.7f, 0.35f}, {2.8f, 0.35f}, {4.9f, 1.15f}};

        const int W = 1280, H = 860;
        std::vector<float> pts;
        std::vector<unsigned char> rgba;
        stbi_flip_vertically_on_write(1);   // GL rows are bottom-up

        for (const ViewDef& v : views) {
            switch (v.mode) {
                case 0: prtclouds::build_strands(R, st.strands_m, st.strands_points, pts); break;
                case 1: prtclouds::build_lattice(R, st.lat_m1, st.lat_m2, st.lat_m3, pts); break;
                case 2: prtclouds::build_torus(R, st.torus_m1, st.torus_m2, pts); break;
                case 3: prtclouds::build_spectral(R, pts); break;
                case 4: prtclouds::build_pairspace(R, st.pair_gmax, pts); break;
            }
            if (pts.empty()) continue;
            st.scene.set_points(pts);
            for (int a = 0; a < 3; ++a) {
                prt3d::Camera cam;
                cam.yaw = angles[a].yaw;
                cam.pitch = angles[a].pitch;
                if (!st.scene.snapshot(cam, W, H, v.psize, v.glow, rgba)) continue;
                char path[512];
                std::snprintf(path, sizeof(path), "%s/view_%s_angle%d.png",
                              dir.c_str(), v.name, a + 1);
                stbi_write_png(path, W, H, 4, rgba.data(), W * 4);
            }
        }
        st.cloud_key = ~0ull;   // force the explorer to rebuild its own cloud
    }
    return dir;
}
