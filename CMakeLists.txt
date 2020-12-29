cmake_minimum_required(VERSION 3.11)

project(demo)
enable_language(CUDA)
enable_testing()

# setup CUDNN lib and include
# find_library(CUDNN_LIB cudnn HINTS $ENV{CUDNN_ROOT}/lib64)
# find_path(CUDNN_INCLUDE cudnn.h HINTS $ENV{CUDNN_ROOT}/include)

# compile flag
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# exe
add_executable(demo
    main.cxx
    div.cu
    pow.cu)
set_property(TARGET demo PROPERTY CUDA_STANDARD 11)

add_test(NAME demo COMMAND demo)
