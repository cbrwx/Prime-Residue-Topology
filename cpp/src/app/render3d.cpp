#include "app/render3d.h"
#include "app/mat4.h"
#include <cstdio>

namespace prt3d {

static const char* kVert = R"GLSL(
#version 330 core
layout(location=0) in vec4 av;   // xyz + value
uniform mat4 mvp;
uniform float psize;
out float val;
void main() {
    gl_Position = mvp * vec4(av.xyz, 1.0);
    val = av.w;
    float w = max(gl_Position.w, 0.1);
    gl_PointSize = clamp(psize / w, 1.0, 64.0);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 330 core
in float val;
uniform int glow;
out vec4 frag;

vec3 viridis(float t) {
    const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}

void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    vec3 col = viridis(clamp(val, 0.0, 1.0));
    if (glow == 1) {
        float a = exp(-2.5 * r2);
        frag = vec4(col * a, a * 0.55);
    } else {
        float edge = smoothstep(1.0, 0.8, r2);
        frag = vec4(col, edge);
    }
}
)GLSL";

static GLuint compile(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
        std::fprintf(stderr, "shader error: %s\n", log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

bool Scene::init() {
    const GLuint vs = compile(GL_VERTEX_SHADER, kVert);
    const GLuint fs = compile(GL_FRAGMENT_SHADER, kFrag);
    if (!vs || !fs) return false;
    prog_ = glCreateProgram();
    glAttachShader(prog_, vs);
    glAttachShader(prog_, fs);
    glLinkProgram(prog_);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(prog_, GL_LINK_STATUS, &ok);
    if (!ok) return false;
    loc_mvp_ = glGetUniformLocation(prog_, "mvp");
    loc_psize_ = glGetUniformLocation(prog_, "psize");
    loc_glow_ = glGetUniformLocation(prog_, "glow");
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    return true;
}

void Scene::shutdown() {
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (prog_) glDeleteProgram(prog_);
    if (fbo_) glDeleteFramebuffers(1, &fbo_);
    if (depth_) glDeleteRenderbuffers(1, &depth_);
    if (tex_) glDeleteTextures(1, &tex_);
    prog_ = vao_ = vbo_ = fbo_ = tex_ = depth_ = 0;
}

void Scene::set_points(const std::vector<float>& xyzv) {
    npoints_ = xyzv.size() / 4;
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(xyzv.size() * sizeof(float)),
                 xyzv.empty() ? nullptr : xyzv.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glBindVertexArray(0);
}

bool Scene::ensure_fbo(int w, int h) {
    if (w == fbw_ && h == fbh_ && fbo_) return true;
    if (fbo_) glDeleteFramebuffers(1, &fbo_);
    if (depth_) glDeleteRenderbuffers(1, &depth_);
    if (tex_) glDeleteTextures(1, &tex_);

    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenRenderbuffers(1, &depth_);
    glBindRenderbuffer(GL_RENDERBUFFER, depth_);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);

    glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_);
    const bool ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (ok) { fbw_ = w; fbh_ = h; }
    return ok;
}

unsigned Scene::render(const Camera& cam, int w, int h, float point_size, bool glow) {
    if (w < 8 || h < 8 || !prog_) return 0;
    if (!ensure_fbo(w, h)) return 0;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, w, h);
    glClearColor(0.045f, 0.05f, 0.07f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const float cy = std::cos(cam.yaw), sy = std::sin(cam.yaw);
    const float cp = std::cos(cam.pitch), sp = std::sin(cam.pitch);
    Vec3 eye = {cam.dist * cp * sy, cam.dist * sp, cam.dist * cp * cy};
    Vec3 center = {cam.pan_x, cam.pan_y, 0.0f};
    eye.x += center.x; eye.y += center.y; eye.z += center.z;
    const Mat4 view = look_at(eye, center, {0, 1, 0});
    const Mat4 proj = perspective(0.9f, (float)w / (float)h, 0.05f, 100.0f);
    const Mat4 mvp = mul(proj, view);

    glUseProgram(prog_);
    glUniformMatrix4fv(loc_mvp_, 1, GL_FALSE, mvp.m);
    glUniform1f(loc_psize_, point_size);
    glUniform1i(loc_glow_, glow ? 1 : 0);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);      // compatibility profile: needed for gl_PointCoord
    if (glow) {
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    } else {
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, (GLsizei)npoints_);
    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return tex_;
}

bool Scene::snapshot(const Camera& cam, int w, int h, float point_size, bool glow,
                     std::vector<unsigned char>& rgba) {
    if (!render(cam, w, h, point_size, glow)) return false;
    rgba.assign((size_t)w * h * 4, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return true;
}

} // namespace prt3d
