#include "app/gl_loader.h"

#define PRT_DEFINE(name) PFN_##name name = nullptr;
PRT_GL_FUNCS(PRT_DEFINE)
#undef PRT_DEFINE

static void* get_proc(const char* name) {
    void* p = (void*)wglGetProcAddress(name);
    if (p == nullptr || p == (void*)1 || p == (void*)2 || p == (void*)3 || p == (void*)-1) {
        static HMODULE mod = GetModuleHandleA("opengl32.dll");
        p = mod ? (void*)GetProcAddress(mod, name) : nullptr;
    }
    return p;
}

bool prt_gl_load() {
    bool ok = true;
#define PRT_LOAD(name) name = (PFN_##name)get_proc(#name); if (!name) ok = false;
    PRT_GL_FUNCS(PRT_LOAD)
#undef PRT_LOAD
    return ok;
}
