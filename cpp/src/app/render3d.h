#pragma once
// Offscreen 3D point-cloud renderer (FBO -> texture, shown via ImGui::Image).
#include <cstdint>
#include <vector>
#include "app/gl_loader.h"

namespace prt3d {

struct Camera {
    float yaw = 0.7f, pitch = 0.35f, dist = 3.4f;
    float pan_x = 0.0f, pan_y = 0.0f;
};

class Scene {
public:
    bool init();                    // build shader program; needs current GL context
    void shutdown();

    // Interleaved x,y,z,value (value in [0,1] -> colormap).
    void set_points(const std::vector<float>& xyzv);
    size_t point_count() const { return npoints_; }

    // Renders into an offscreen texture of the given size; returns texture id.
    // glow=true: additive blending, no depth (nebula look for dense clouds).
    unsigned render(const Camera& cam, int w, int h, float point_size, bool glow);

    // Renders and reads back RGBA8 pixels (bottom-up rows, as OpenGL delivers).
    bool snapshot(const Camera& cam, int w, int h, float point_size, bool glow,
                  std::vector<unsigned char>& rgba);

private:
    bool ensure_fbo(int w, int h);
    GLuint prog_ = 0, vao_ = 0, vbo_ = 0;
    GLuint fbo_ = 0, tex_ = 0, depth_ = 0;
    int fbw_ = 0, fbh_ = 0;
    GLint loc_mvp_ = -1, loc_psize_ = -1, loc_glow_ = -1;
    size_t npoints_ = 0;
};

} // namespace prt3d
