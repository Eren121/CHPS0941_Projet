cmake_minimum_required(VERSION 3.9)
if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
project(VSProject LANGUAGES CXX VERSION 0.0.0)

if (NOT WIN32)
# visual studio doesn't like these (not need them):
set (CMAKE_CXX_FLAGS "--std=c++11")
set (CUDA_PROPAGATE_HOST_FLAGS ON)
endif()

mark_as_advanced(CUDA_SDK_ROOT_DIR)

# ------------------------------------------------------------------
# build glfw
# ------------------------------------------------------------------
set(OpenGL_GL_PREFERENCE LEGACY)
if (WIN32)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE) #Desactive la génération de la doc
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE) #Desactive les tests
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) #Desactive la compilation des exemples
  find_package(OpenGL REQUIRED)
  
  set(glfw_dir ${PROJECT_SOURCE_DIR}/libs/glfw3/)
  include_directories(${glfw_dir}/include)
  add_subdirectory(${glfw_dir} EXCLUDE_FROM_ALL)
else()
  find_package(glfw3 REQUIRED)
endif()

# ------------------------------------------------------------------
# build glew
# ------------------------------------------------------------------
if (WIN32)
  set(glew_dir ${PROJECT_SOURCE_DIR}/libs/glew-2.1.0/)
  include_directories(./libs/glew-2.1.0/include)
  add_subdirectory(./libs/glew-2.1.0/build/cmake/) #chemin vers les sources de GLew 2.1.0
else()
  find_package(glew REQUIRED)
endif()



# ------------------------------------------------------------------
# build OptiX
# ------------------------------------------------------------------
set(gdt_dir ${PROJECT_SOURCE_DIR}/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")
include(${gdt_dir}/cmake/configure_build_type.cmake)
include(${gdt_dir}/cmake/configure_optix.cmake)

include_directories(${OptiX_INCLUDE})

add_executable(VSProject
  ${embedded_ptx_code}
  ./src/main.cpp
  ./src/screenDisplay.cpp
  ./src/screenDisplay.h 
  ./src/vec.h 
  ./src/camera.h
  ./src/Scene.hpp
  ./src/Scene.cpp
  ./src/trianglemesh.h
  ./src/trianglemesh.cpp
  ./src/optixRender.cpp
  ./src/optixRender.h
  ./src/volume.h
  ./src/volume.cpp
  ./src/mmatrix.h
  ./src/mmatrix.cpp
)

include_directories(${OptiX_INCLUDE})

add_definitions(
    -DTW_STATIC
    -DTW_NO_LIB_PRAGMA
    -DTW_NO_DIRECT3D
    -DGLEW_STATIC
)

target_link_libraries(VSProject
  OpenGL::GL
  imgui
  glfw
  ${OPENGL_LIBRARY}
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )


#compile shader
enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)

# CUDA PACKAGE
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER clang++)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()


add_custom_command(TARGET VSProject POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_BINARY_DIR}/raygen.dir/$<CONFIG>/raygen.ptx $<TARGET_FILE_DIR:VSProject>/raygen.ptx)


# COMPILE CU FILES
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./)
add_library(mesh OBJECT shader/mesh.cu)
set_property(TARGET mesh PROPERTY CUDA_ARCHITECTURES 50-virtual)
set_target_properties(mesh PROPERTIES
CUDA_PTX_COMPILATION ON)

add_library(raygen OBJECT shader/raygen.cu)
set_property(TARGET raygen PROPERTY CUDA_ARCHITECTURES 50-virtual)
set_target_properties(raygen PROPERTIES
CUDA_PTX_COMPILATION ON)

add_library(volume OBJECT shader/volume.cu)
set_property(TARGET volume PROPERTY CUDA_ARCHITECTURES 50-virtual)
set_target_properties(volume PROPERTIES CUDA_PTX_COMPILATION ON)

add_custom_command(TARGET VSProject POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_BINARY_DIR}/volume.dir/$<CONFIG>/volume.ptx $<TARGET_FILE_DIR:VSProject>/volume.ptx)

add_custom_command(TARGET VSProject POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_BINARY_DIR}/raygen.dir/$<CONFIG>/raygen.ptx $<TARGET_FILE_DIR:VSProject>/raygen.ptx)

add_custom_command(TARGET VSProject POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_BINARY_DIR}/mesh.dir/$<CONFIG>/mesh.ptx $<TARGET_FILE_DIR:VSProject>/mesh.ptx)

# ImGUI
set(IMGUI_DIR "./libs/imgui/")

add_library("imgui" "${IMGUI_DIR}/imgui.cpp"
                  "${IMGUI_DIR}/imgui_demo.cpp"
                  "${IMGUI_DIR}/imgui_draw.cpp"
                  "${IMGUI_DIR}/imgui_tables.cpp"
                  "${IMGUI_DIR}/imgui_widgets.cpp"
                  
                  "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
                  "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp")

target_include_directories("imgui" PRIVATE "${IMGUI_DIR}")
target_include_directories(VSProject PRIVATE "${IMGUI_DIR}")
target_include_directories("imgui" PRIVATE "${IMGUI_DIR}/backends/")
target_include_directories(VSProject PRIVATE "${IMGUI_DIR}/backends/")
target_link_libraries(VSProject "imgui" "${CMAKE_DL_LIBS}")

add_custom_command(TARGET VSProject POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_BINARY_DIR}/mesh.dir/$<CONFIG>/mesh.ptx $<TARGET_FILE_DIR:VSProject>/mesh.ptx)


