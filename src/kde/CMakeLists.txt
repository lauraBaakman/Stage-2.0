cmake_minimum_required(VERSION 3.5)
project(kernels)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
        kdeModule.c kdeModule.h
        kernels/kernels.c kernels/kernels.h
        kernels/kernelsModule.c kernels/kernelsModule.h
        parzen.c parzen.h parzen.ih
        sambe.c sambe.h sambe.ih
        utils.c utils.h utils.ih
        modifeidbreiman.c modifeidbreiman.h
        utils/utilsModule.c utils/utilsModule.h
        utils/distancematrix.c utils/distancematrix.h utils/distancematrix.ih
        utils/knn.c utils/knn.h utils/knn.ih
        utils/covariancematrix.c utils/covariancematrix.h utils/covariancematrix.ih
        utils/eigenvalues.c utils/eigenvalues.h
        utils/geometricmean.c utils/geometricmean.h)
add_executable(kde ${SOURCE_FILES})

find_package(gsl COMPONENTS system filesystem REQUIRED)

find_package(PythonLibs REQUIRED)
include_directories(
        SYSTEM
        ${gsl_INCLUDE_DIRS}
        /Users/laura/.virtualenvs/stage/lib/python3.5/site-packages/numpy/core/include
        /Users/laura/.virtualenvs/stage/include/python3.5m
        /Users/laura/.virtualenvs/stage/include/python3.5m/Python.h
        /Library/Frameworks/Python.framework/Headers
        /usr/include
        /usr/local/include
)