project(sdf_experiments)
cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
enable_language(CUDA)
find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

find_package(PythonInterp REQUIRED)
if (PYTHON_INCLUDE_DIRS)
  include_directories(${PYTHON_INCLUDE_DIRS})
  message("Using python version: " ${PYTHONLIBS_VERSION_STRING})
else ()
  message(FATAL_ERROR "python not found!")
endif ()

# override PYTHON_EXECUTABLE to use virtual environment
set(PYTHON_EXECUTABLE ../venv/bin/python)
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from distutils import sysconfig; print(sysconfig.get_python_lib())"
    OUTPUT_VARIABLE SITE_PACKAGES_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

pybind11_add_module(sdf_experiments_py pybind.cpp)
add_library(sdf_renderer sdf_renderer.cu sdf_shapes.cu)
set_target_properties(sdf_renderer PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON)
#target_compile_options(sdf_renderer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G -lineinfo>)

target_link_libraries(sdf_experiments_py PUBLIC sdf_renderer)

install(TARGETS sdf_experiments_py LIBRARY DESTINATION ${SITE_PACKAGES_DIR}/sdf_experiments)

# gui stuff
set(imgui_src imgui/imgui.cpp imgui/imgui_demo.cpp imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp imgui/imgui_widgets.cpp imgui/backends/imgui_impl_opengl3.cpp
    imgui/backends/imgui_impl_glfw.cpp)
add_library(gui gui.cpp ${imgui_src})
find_package(glfw3 REQUIRED)
find_package(OpenGL)
find_package(Eigen3)
target_include_directories(gui PUBLIC ${EIGEN3_INCLUDE_DIR})

target_link_libraries(gui glfw)
target_link_libraries(gui OpenGL::GL)
target_link_libraries(gui Eigen3::Eigen)
target_include_directories(gui PUBLIC imgui/backends)
target_include_directories(gui PUBLIC imgui)

target_link_libraries(sdf_experiments_py PUBLIC gui)