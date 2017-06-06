CC=gcc
CFLAGS=-std=c99
LDLIBS=-lgsl -lgslcblas -lm
LDDIRS=-L/usr/local/lib/
IDIRS=-I/usr/local/include/

SOURCES=./test_main.c ./lib/CuTest.c ./lib/CuTestUtils.c\
    ./kde/tests/test_sambe.c ./kde/sambe.c \
    ./kde/kernels/kernels.c \
    ./kde/kernels/gaussian.c ./kde/kernels/epanechnikov.c ./kde/kernels/testkernel.c \
    ./kde/utils/distancematrix.c \
    ./kde/utils/knn.c \
    ./kde/utils/covariancematrix.c \
    ./kde/utils/eigenvalues.c \
    ./kde/utils/gsl_utils.c
OBJECTS=$(SOURCES:.c=.o)
DEPS=./test_constants.h ./lib/CuTest.h ./lib/CuTestUtils.h\
    ./kde/tests/test_sambe.h ./kde/sambe.h ./kde/sambe.ih \
    ./kde/kernels/kernels.h ./kde/kernels/kernels.ih \
    ./kde/kernels/gaussian.h ./kde/kernels/gaussian.ih \
    ./kde/kernels/epanechnikov.h ./kde/kernels/epanechnikov.ih \
    ./kde/kernels/testkernel.h ./kde/kernels/testkernel.ih \
    ./kde/utils/distancematrix.ih ./kde/utils/distancematrix.h \
    ./kde/utils/knn.h ./kde/utils/knn.ih \
    ./kde/utils/covariancematrix.h ./kde/utils/covariancematrix.ih \
    ./kde/utils/eigenvalues.h \
    ./kde/utils/gsl_utils.h

# Working command OS X
# gcc test_main.c -I/usr/local/include/ -L/usr/local/lib/ -lgsl -lgslcblas

# Working command Linux
# gcc -Wall -I/usr/local/include -c temp.c
# gcc temp.o -lgsl -lgslcblas -lm

RM=rm

# http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
%.o: %.c $(DEPS)
	$(CC) $(IDIRS) -c -o $@ $< $(CFLAGS)
test: $(OBJECTS)
	$(CC) $(CFLAGS) -o test_main.out $(OBJECTS) $(LDLIBS) $(LDDIRS) $(IDIRS)
clean:
	$(RM) *o *.out