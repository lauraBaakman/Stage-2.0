CC=gcc
CFLAGS=-std=c99
LDLIBS=-lgsl -lgslcblas
LDDIRS=-L/usr/local/lib/
IDIRS=-I/usr/local/include/

SOURCES=test_main.c CuTest.c CuTestUtils.c\
    test_sambe.c ../sambe.c \
    ../kernels/kernels.c \
    ../utils/distancematrix.c \
    ../utils/knn.c \
    ../utils/covariancematrix.c \
    ../utils/eigenvalues.c
OBJECTS=$(SOURCES:.c=.o)
DEPS=test_constants.h CuTest.h CuTestUtils.h\
    test_sambe.h ../sambe.h ../sambe.ih \
    ../kernels/kernels.h ../kernels/kernels.ih \
    ../utils/distancematrix.ih ../utils/distancematrix.h \
    ../utils/knn.h ../utils/knn.ih \
    ../utils/covariancematrix.h ../utils/covariancematrix.ih \
    ../utils/eigenvalues.h \

# Working command
# gcc test_main.c -I/usr/local/include/ -L/usr/local/lib/ -lgsl -lgslcblas

RM=rm

# http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
%.o: %.c $(DEPS)
	$(CC) $(IDIRS) -c -o $@ $< $(CFLAGS)
test: $(OBJECTS)
	$(CC) $(CFLAGS) -o test_main.out $(OBJECTS) $(LDLIBS) $(LDDIRS) $(IDIRS)
clean:
	$(RM) *o *.out