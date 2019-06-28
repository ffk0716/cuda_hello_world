cmake_minimum_required(VERSION 3.11)

project(demo)
enable_language(CUDA)

find_library(CUDNN_LIB cudnn HINTS $ENV{CUDNN_ROOT}/lib64)
find_path(CUDNN_INCLUDE cudnn.h HINTS $ENV{CUDNN_ROOT}/include)

include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CUDNN_INCLUDE})

set(CMAKE_CXX_STANDARD 11)
add_executable(demo main.cxx pow.cu)
set_property(TARGET demo PROPERTY CUDA_STANDARD 11)
