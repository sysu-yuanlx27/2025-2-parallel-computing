CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -std=c11
LDFLAGS ?=
LDLIBS ?= -pthread -lm

all: libparallel_for.so matmul_test heated_plate_pthreads heated_plate_openmp

parallel_for.o: parallel_for.c parallel_for.h
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

libparallel_for.so: parallel_for.o
	$(CC) -shared -o $@ $^ -pthread

matmul_test: matmul_test.c parallel_for.h libparallel_for.so
	$(CC) $(CFLAGS) $< -L. -lparallel_for -Wl,-rpath,'$$ORIGIN' $(LDFLAGS) $(LDLIBS) -o $@

heated_plate_pthreads: heated_plate_pthreads.c parallel_for.h libparallel_for.so
	$(CC) $(CFLAGS) $< -L. -lparallel_for -Wl,-rpath,'$$ORIGIN' $(LDFLAGS) $(LDLIBS) -o $@

heated_plate_openmp: ../heated_plate_openmp.c
	$(CC) $(CFLAGS) -fopenmp $< $(LDFLAGS) -lm -o $@

clean:
	rm -f *.o *.so matmul_test heated_plate_pthreads heated_plate_openmp

.PHONY: all clean
