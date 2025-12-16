cmake_minimum_required(VERSION 3.13)
# set(CMAKE_CUDA_ARCHITECTURES "75")
project(signal-analysis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)

add_executable(signal-analysis main.cpp)

