CC=gcc
CFLAGS=-std=c99
LDLIBS=-lgsl -lgslcblas -lm
LDDIRS=-L/usr/local/lib/
IDIRS=-I/usr/local/include/

TEST_SOURCES=./lib/CuTest.c ./lib/CuTestUtils.c \
	./kde/tests/test_kde.c \
	./kde/tests/test_sambe.c \
	./kde/tests/test_parzen.c \
	./kde/tests/test_mbe.c \
	./kde/utils/tests/test_utils.c \
	./kde/utils/tests/test_knn.c 
TEST_OBJECTS=$(TEST_SOURCES:.c=.o)
TEST_DEPS=./test_constants.h ./lib/CuTest.h ./lib/CuTestUtils.h\
	./kde/tests/test_kde.h \
	./kde/tests/test_sambe.h \
	./kde/tests/test_parzen.h \
	./kde/tests/test_mbe.h \
	./kde/utils/tests/test_utils.h \
	./kde/utils/tests/test_knn.h	

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

# Working command OS X
# gcc test_main.c -I/usr/local/include/ -L/usr/local/lib/ -lgsl -lgslcblas

# Working command Linux
# gcc -Wall -I/usr/local/include -c temp.c
# gcc temp.o -lgsl -lgslcblas -lm

# http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
%.o: %.c $(DEPS) $(TEST_DEPS)
	$(CC) $(IDIRS) -c -o $@ $< $(CFLAGS)
test_main: test_main.o	
	$(CC) $(IDIRS) -c -o $@ $< $(CFLAGS)
tests: $(OBJECTS) $(TEST_OBJECTS) test_main.o
	$(CC) $(CFLAGS) -o tests.out $(OBJECTS) $(TEST_OBJECTS) test_main.o $(LDLIBS) $(LDDIRS) $(IDIRS)	
benchmark_main: benchmark_main.o	
	$(CC) $(IDIRS) -c -o $@ $< $(CFLAGS)
benchmark: $(OBJECTS) benchmark_main.o
	$(CC) $(CFLAGS) -o benchmark.out $(OBJECTS) benchmark_main.o $(LDLIBS) $(LDDIRS) $(IDIRS)	
clean:
	find . -name "*.o" -delete
	find . -name "*.out" -delete