cmake_minimum_required(VERSION 3.0)
if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(VSProject LANGUAGES CUDA VERSION 0.0.0)

add_library(myptx OBJECT mesh.cu raygen.cu devicePrograms.cu)
set_property(TARGET myptx PROPERTY CUDA_PTX_COMPILATION ON)

# ------------------------------------------------------------------
# and final build rules for the project
# ------------------------------------------------------------------
set(gdt_dir ../)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")
include(${gdt_dir}/cmake/configure_build_type.cmake)
include(${gdt_dir}/cmake/configure_optix.cmake)

include_directories(${OptiX_INCLUDE})