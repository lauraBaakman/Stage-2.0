# C Stuff
CC=gcc

CFLAGS=-std=c99 -fopenmp

LDLIBS=-lgsl -lgslcblas -lm -lpthread
LDDIRS=-L/usr/local/lib/
LDFLAGS=-fopenmp

IDIRS=-I/usr/local/include/

OS := $(shell uname)
ifeq ($(OS),Darwin)
	# PYRUNFLAGS=arch -i386 this breaks stuff, as soonas openmp is added other stuff will break, for now lets run only on Zeus.
endif
ifeq ($(OS),LINUX)
endif

# PYTHON STUFF
PYTHON=python
PYBUILDFLAGS=build_ext --inplace

TEST_SOURCES=./test_utils.c ./lib/CuTest.c ./lib/CuTestUtils.c \
	./kde/tests/test_kde.c \
	./kde/tests/test_parzen.c \
	./kde/tests/test_mbe.c \
	./kde/tests/test_sambe.c \
	./kde/utils/tests/test_utils.c \
	./kde/utils/tests/test_knn.c \
	./kde/kernels/tests/test_kernels.c \
	./kde/kernels/tests/test_epanechnikov.c \
	./kde/kernels/tests/test_gaussian.c 
TEST_OBJECTS=$(TEST_SOURCES:.c=.o)
TEST_DEPS=./test_utils.h ./lib/CuTest.h ./lib/CuTestUtils.h\
	./kde/tests/test_kde.h \
	./kde/tests/test_parzen.h \
	./kde/tests/test_mbe.h \
	./kde/tests/test_sambe.h \
	./kde/utils/tests/test_utils.h \
	./kde/utils/tests/test_knn.h \
	./kde/kernels/tests/test_kernels.h \
	./kde/kernels/tests/test_epanechnikov.h \
	./kde/kernels/tests/test_gaussian.h 
SOURCES=./kde/sambe.c ./kde/parzen.c ./kde/mbe.c \
    ./kde/kernels/kernels.c \
    ./kde/kernels/gaussian.c ./kde/kernels/epanechnikov.c ./kde/kernels/testkernel.c \
    ./kde/utils/knn.c \
    ./kde/utils/covariancematrix.c \
    ./kde/utils/eigenvalues.c \
    ./kde/utils/gsl_utils.c \
    ./lib/kdtree/kdtree.c
OBJECTS=$(SOURCES:.c=.o)
DEPS=./kde/sambe.h ./kde/sambe.ih \
    ./kde/parzen.h ./kde/parzen.ih \
    ./kde/mbe.h ./kde/mbe.ih \
    ./kde/kernels/kernels.h ./kde/kernels/kernels.ih \
    ./kde/kernels/gaussian.h ./kde/kernels/gaussian.ih \
    ./kde/kernels/epanechnikov.h ./kde/kernels/epanechnikov.ih \
    ./kde/kernels/testkernel.h ./kde/kernels/testkernel.ih \
    ./kde/utils/knn.h ./kde/utils/knn.ih \
    ./kde/utils/covariancematrix.h ./kde/utils/covariancematrix.ih \
    ./kde/utils/eigenvalues.h \
    ./kde/utils/gsl_utils.h \
    ./lib/kdtree/kdtree.h

.PHONY: clean
.PHONY: run_tests
.PHONY: modules
.PHONY: run_kernel_tests
.PHONY: run_kde_tests
.PHONY: run_utils_tests

# http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
%.o: %.c $(DEPS) $(TEST_DEPS)
	@$(CC) $(IDIRS) -c -o $@ $< $(CFLAGS)
default: clean run_tests
test_main: test_main.o	
	@$(CC) $(IDIRS) -c -o $@ $< $(CFLAGS)
tests: $(OBJECTS) $(TEST_OBJECTS) test_main.o
	@$(CC) $(CFLAGS) $(LDFLAGS) -o tests.out $(OBJECTS) $(TEST_OBJECTS) test_main.o $(LDLIBS) $(LDDIRS) $(IDIRS)	
benchmark_main: benchmark_main.o	
	@$(CC) $(IDIRS) -c -o $@ $< $(CFLAGS)
benchmark: $(OBJECTS) benchmark_main.o
	@$(CC) $(CFLAGS) $(LDFLAGS) -o benchmark.out $(OBJECTS) benchmark_main.o $(LDLIBS) $(LDDIRS) $(IDIRS)	
_kde: $(OBJECTS)
	@$(PYTHON) kde/setup.py $(PYBUILDFLAGS)
	@cp _kde.so ./kde/_kde.so
_kernels: $(OBJECTS)
	$(PYTHON) kde/kernels/setup.py $(PYBUILDFLAGS)	
	@cp _kernels.so ./kde/kernels/_kernels.so
_utils: $(OBJECTS)
	$(PYTHON) kde/utils/setup.py $(PYBUILDFLAGS)		
	@cp _utils.so ./kde/utils/_utils.so
modules: _kde _kernels _utils
run_tests: modules tests
	@echo -e "\n\033[0;31m Python Tests \033[0m\n"
	@export PYTHONWARNINGS="ignore"; $(PYRUNFLAGS) $(PYTHON) -m unittest discover -s . -t .
	@echo -e "\n\033[0;31m C Tests \033[0m"
	@./tests.out	
run_kernel_tests: _kernels _utils tests
	@echo -e "\n\033[0;31m Python Tests \033[0m\n"
	@export PYTHONWARNINGS="ignore"; $(PYRUNFLAGS) $(PYTHON) -m unittest discover -s ./kde/kernels -t .
	@echo -e "\n\033[0;31m C Tests \033[0m"
	@./tests.out		
run_kde_tests: _kernels _utils _kde tests
	@echo -e "\n\033[0;31m Python Tests \033[0m\n"
	@export PYTHONWARNINGS="ignore"; $(PYRUNFLAGS) $(PYTHON) -m unittest discover -s ./kde/ -t .
	@echo -e "\n\033[0;31m C Tests \033[0m"
	@./tests.out			
run_utils_tests: _utils tests
	@echo -e "\n\033[0;31m Python Tests \033[0m\n"
	@export PYTHONWARNINGS="ignore"; $(PYRUNFLAGS) $(PYTHON) -m unittest discover -s ./kde/utils -t .
	@echo -e "\n\033[0;31m C Tests \033[0m"
	@./tests.out				
clean:
	@find . -name "*.o" -delete
	@find . -name "*.out" -delete
	@find . -name "*.so" -delete