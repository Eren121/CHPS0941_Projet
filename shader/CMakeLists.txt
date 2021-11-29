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



# COMPILE CU FILES
add_library(mesh OBJECT mesh.cu)
set_target_properties(mesh PROPERTIES
CUDA_PTX_COMPILATION ON)

unable_language(CUDA)