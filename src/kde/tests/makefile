CC=gcc
CFLAGS=-std=c99
LDLIBS=-lgsl -lgslcblas -lm
LDDIRS=-L/usr/local/lib/
IDIRS=-I/usr/local/include/

SOURCES=test_main.c CuTest.c CuTestUtils.c\
    test_sambe.c ../sambe.c \
    ../kernels/kernels.c \
    ../kernels/gaussian.c ../kernels/epanechnikov.c ../kernels/testkernel.c \
    ../utils/distancematrix.c \
    ../utils/knn.c \
    ../utils/covariancematrix.c \
    ../utils/eigenvalues.c \
    ../utils/gsl_utils.c
OBJECTS=$(SOURCES:.c=.o)
DEPS=test_constants.h CuTest.h CuTestUtils.h\
    test_sambe.h ../sambe.h ../sambe.ih \
    ../kernels/kernels.h ../kernels/kernels.ih \
    ../kernels/gaussian.h ../kernels/gaussian.ih \
    ../kernels/epanechnikov.h ../kernels/epanechnikov.ih \
    ../kernels/testkernel.h ../kernels/testkernel.ih \
    ../utils/distancematrix.ih ../utils/distancematrix.h \
    ../utils/knn.h ../utils/knn.ih \
    ../utils/covariancematrix.h ../utils/covariancematrix.ih \
    ../utils/eigenvalues.h \
    ../utils/gsl_utils.h

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