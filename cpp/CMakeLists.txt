cmake_minimum_required(VERSION 3.20)
project(prime_topology CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/W3 /EHsc /utf-8 /MP)
endif()

# ---------------- core number-crunching library ----------------
add_library(prt_core STATIC
  src/core/sieve.cpp
  src/core/pipeline.cpp
  src/core/stats.cpp
  src/core/topology.cpp
  src/core/predict.cpp
  src/core/report.cpp
  src/core/audit.cpp
)
target_include_directories(prt_core PUBLIC src)

# ---------------- CLI: self-tests + headless runs ----------------
add_executable(prt_cli src/cli/main_cli.cpp)
target_link_libraries(prt_cli PRIVATE prt_core)

# ---------------- vendored Dear ImGui + ImPlot ----------------
set(IMGUI_DIR  ${CMAKE_SOURCE_DIR}/external/imgui)
set(IMPLOT_DIR ${CMAKE_SOURCE_DIR}/external/implot)

add_library(imgui STATIC
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/backends/imgui_impl_win32.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
  ${IMPLOT_DIR}/implot.cpp
  ${IMPLOT_DIR}/implot_items.cpp
)
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends ${IMPLOT_DIR})

# ---------------- GUI application ----------------
if(EXISTS ${CMAKE_SOURCE_DIR}/src/app/main_win32.cpp)
add_executable(prime_topology WIN32
  src/app/main_win32.cpp
  src/app/gl_loader.cpp
  src/app/render3d.cpp
  src/app/ui_panels.cpp
  src/app/clouds.cpp
  src/app/run_export.cpp
)
target_include_directories(prime_topology PRIVATE ${CMAKE_SOURCE_DIR}/external/stb)
target_link_libraries(prime_topology PRIVATE prt_core imgui opengl32 imm32 dwmapi comdlg32)
endif()
