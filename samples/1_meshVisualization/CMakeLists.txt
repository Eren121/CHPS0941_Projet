add_executable(1_meshVisualization
  ./main.cpp
  ${CMAKE_SOURCE_DIR}/common/screenDisplay.cpp
  ${CMAKE_SOURCE_DIR}/common/Scene.cpp
  ${CMAKE_SOURCE_DIR}/common/image.cpp
  ${CMAKE_SOURCE_DIR}/common/trianglemesh.cpp
  ./optixRender.h
  ./optixRender.cpp
  ${CMAKE_SOURCE_DIR}/common/meshloader.cpp
  ${CMAKE_SOURCE_DIR}/common/volume.cpp
  ${CMAKE_SOURCE_DIR}/common/mmatrix.cpp
)

target_link_libraries(1_meshVisualization
  CHPS0941_Additions
  OpenGL::GL
  imgui
  glfw
  assimp-vc142-mt
  ${OPENGL_LIBRARY}
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )

