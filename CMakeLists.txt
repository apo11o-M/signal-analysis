cmake_minimum_required(VERSION 3.13)

project(signal-analysis LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA enabled..")
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    set(CMAKE_CUDA_STANDARD 17)
else()
    message(STATUS "CUDA not available..")
endif()

# add_subdirectory(core)
# add_subdirectory(utils)
# add_subdirectory(tx)

add_executable(signal-analysis main.cpp)

target_include_directories(signal-analysis PUBLIC 
    core/include
    utils/include
    tx/include
    rx/include
)
