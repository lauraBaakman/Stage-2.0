cmake_minimum_required(VERSION 3.5)
project(kernels)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
        kdeModule.c kdeModule.h
        kernels/kernels.c kernels/kernels.h
        kernels/kernelsModule.c kernels/kernelsModule.h
        parzen.c parzen.h
        utils.c utils.h modifeidbreiman.c modifeidbreiman.h utils/utilsModule.c utils/utilsModule.h)
add_executable(kde ${SOURCE_FILES})

find_package(PythonLibs REQUIRED)
include_directories(
        SYSTEM
        /Users/laura/.virtualenvs/stage/lib/python3.5/site-packages/numpy/core/include
        /Users/laura/.virtualenvs/stage/include/python3.5m
        /Users/laura/.virtualenvs/stage/include/python3.5m/Python.h
        /Library/Frameworks/Python.framework/Headers
        /usr/include
        /usr/local/include
)