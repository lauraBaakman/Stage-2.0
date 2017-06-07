cmake_minimum_required(VERSION 3.5)
project(kernels)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
        ../benchmark_main.c
        kdeModule.c kdeModule.h
        kernels/kernels.c kernels/kernels.h kernels/kernels.ih
        kernels/kernelsModule.c kernels/kernelsModule.h
        kernels/gaussian.c kernels/gaussian.h kernels/gaussian.ih
        kernels/epanechnikov.c kernels/epanechnikov.h kernels/epanechnikov.ih
        kernels/testkernel.c kernels/testkernel.h kernels/testkernel.ih
        parzen.c parzen.h parzen.ih
        sambe.c sambe.h sambe.ih
        utils.c utils.h utils.ih
        modifeidbreiman.c modifeidbreiman.h
        ../test_main.c
        tests/test_sambe.c tests/test_sambe.h
        tests/test_parzen.c tests/test_parzen.h
        ../lib/CuTest.c ../lib/CuTest.h ../lib/CuTestUtils.c ../lib/CuTestUtils.h
        utils/utilsModule.c utils/utilsModule.h
        utils/distancematrix.c utils/distancematrix.h utils/distancematrix.ih
        utils/knn.c utils/knn.h utils/knn.ih
        utils/covariancematrix.c utils/covariancematrix.h utils/covariancematrix.ih
        utils/eigenvalues.c utils/eigenvalues.h
        utils/geometricmean.c utils/geometricmean.h
        utils/gsl_utils.c utils/gsl_utils.h)
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