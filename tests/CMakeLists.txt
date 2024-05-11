cmake_minimum_required(VERSION 3.20)

project(TestRasterizer LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "70;75;86")

# Include directories, adjust paths as needed if this CMakeLists is in tests/
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cuda_rasterizer)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/glm)

# Add executable for the tests
add_executable(TestRasterizer test_forward.cu)

# If the tests need to link against the main CudaRasterizer library or any other libraries:
target_link_libraries(TestRasterizer PRIVATE CudaRasterizer)
