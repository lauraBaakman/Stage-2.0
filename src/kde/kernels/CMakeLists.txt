cmake_minimum_required(VERSION 3.5)
project(kernels)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES standardGaussian.c.c standardGaussian.c.h)
add_executable(kernels ${SOURCE_FILES})