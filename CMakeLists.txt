
cmake_minimum_required(VERSION 3.0)
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
# and final build rules for the project
# ------------------------------------------------------------------
set(gdt_dir ${PROJECT_SOURCE_DIR}/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")
include(${gdt_dir}/cmake/configure_build_type.cmake)
include(${gdt_dir}/cmake/configure_optix.cmake)

add_executable(VSProject
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
  glfw
  ${OPENGL_LIBRARY}
  OpenGL::GL
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )


