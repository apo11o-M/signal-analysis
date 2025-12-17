cmake_minimum_required(VERSION 3.13)
project(signal-analysis LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

# add_subdirectory(core)
# add_subdirectory(utils)

add_executable(signal-analysis main.cpp)

target_include_directories(signal-analysis PUBLIC 
    core/include
    utils/include
)
