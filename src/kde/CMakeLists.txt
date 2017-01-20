cmake_minimum_required(VERSION 3.5)
project(kernels)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
        kde.c kde.h
        densityFunctions.c densityFunctions.h
        kernels.c kernels.h
        parzen.c parzen.h)
add_executable(kde ${SOURCE_FILES})

find_package(PythonLibs REQUIRED)
include_directories(
        SYSTEM
        /Users/laura/.virtualenvs/stage/lib/python3.5/site-packages/numpy/core/include
        /Users/laura/.virtualenvs/stage/include/python3.5m
        /Users/laura/.virtualenvs/stage/include/python3.5m/Python.h
        /Library/Frameworks/Python.framework/Headers
)