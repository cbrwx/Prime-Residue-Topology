// Win32 + WGL + Dear ImGui shell for the Prime Residue Topology explorer.
#include "app/gl_loader.h"
#include "app/app_state.h"
#include "imgui.h"
#include "implot.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_opengl3.h"
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <thread>

// prefer the discrete GPU on hybrid systems
extern "C" {
__declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

static HGLRC g_glrc = nullptr;
static HDC g_hdc = nullptr;
static int g_width = 1600, g_height = 950;
static bool g_quit = false;

static LRESULT WINAPI WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wp, lp)) return 1;
    switch (msg) {
        case WM_SIZE:
            if (wp != SIZE_MINIMIZED) {
                g_width = LOWORD(lp);
                g_height = HIWORD(lp);
            }
            return 0;
        case WM_SYSCOMMAND:
            if ((wp & 0xfff0) == SC_KEYMENU) return 0;   // no ALT menu
            break;
        case WM_DESTROY:
            g_quit = true;
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static bool create_gl(HWND hwnd) {
    g_hdc = GetDC(hwnd);
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    const int pf = ChoosePixelFormat(g_hdc, &pfd);
    if (!pf || !SetPixelFormat(g_hdc, pf, &pfd)) return false;
    g_glrc = wglCreateContext(g_hdc);
    if (!g_glrc) return false;
    wglMakeCurrent(g_hdc, g_glrc);

    typedef BOOL(WINAPI * PFNSWAPINTERVAL)(int);
    if (auto swap = (PFNSWAPINTERVAL)wglGetProcAddress("wglSwapIntervalEXT"))
        swap(1);
    return true;
}

int APIENTRY WinMain(HINSTANCE hinst, HINSTANCE, LPSTR cmdline, int) {
    ImGui_ImplWin32_EnableDpiAwareness();
    const bool autorun = cmdline && strstr(cmdline, "--autorun") != nullptr;
    const char* narg = cmdline ? strstr(cmdline, "--n=") : nullptr;

    WNDCLASSEXW wc = {sizeof(wc), CS_OWNDC, WndProc, 0, 0, hinst,
                      nullptr, LoadCursor(nullptr, IDC_ARROW), nullptr, nullptr,
                      L"prt_window", nullptr};
    RegisterClassExW(&wc);
    HWND hwnd = CreateWindowW(wc.lpszClassName, L"Prime Residue Topology",
                              WS_OVERLAPPEDWINDOW, 60, 40, g_width, g_height,
                              nullptr, nullptr, hinst, nullptr);
    if (!create_gl(hwnd)) {
        MessageBoxA(nullptr, "Failed to create an OpenGL context.", "prime_topology", MB_ICONERROR);
        return 1;
    }

    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = "prime_topology_ui.ini";
    ImGui::StyleColorsDark();
    ImGui::GetStyle().WindowRounding = 0.0f;

    ImGui_ImplWin32_InitForOpenGL(hwnd);
    ImGui_ImplOpenGL3_Init("#version 130");

    AppState st;
    st.scene_ok = prt_gl_load() && st.scene.init();
    if (narg) {
        const double v = atof(narg + 4);
        if (v >= 100.0) st.cfg.N = (uint64_t)v;
    }
    if (autorun) st.start_run();

    ShowWindow(hwnd, SW_SHOWMAXIMIZED);
    UpdateWindow(hwnd);

    auto frame_start = std::chrono::steady_clock::now();
    while (!g_quit) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
            if (msg.message == WM_QUIT) g_quit = true;
        }
        if (g_quit) break;
        if (IsIconic(hwnd)) { Sleep(16); continue; }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        draw_ui(st);

        ImGui::Render();
        glViewport(0, 0, g_width, g_height);
        glClearColor(0.06f, 0.06f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SwapBuffers(g_hdc);
        // vsync stops blocking while the display is asleep, which lets this loop
        // spin uncapped at 100% gpu - enforce a ~60 fps floor ourselves
        const auto min_frame = std::chrono::microseconds(16667);
        const auto elapsed = std::chrono::steady_clock::now() - frame_start;
        if (elapsed < min_frame)
            std::this_thread::sleep_for(min_frame - elapsed);
        frame_start = std::chrono::steady_clock::now();
    }

    st.progress.cancel.store(true);
    if (st.worker.joinable()) st.worker.join();
    st.scene.shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(g_glrc);
    ReleaseDC(hwnd, g_hdc);
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, hinst);
    return 0;
}
