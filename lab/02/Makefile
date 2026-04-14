CXX := mpicxx
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O3 -march=native
SRC := src/matmul.cpp
TARGET := build/matmul_mpi_v2

.PHONY: all clean run run-p2p run-collective bench

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@

run: run-collective

run-p2p: $(TARGET)
	mpirun --oversubscribe -np 4 $(TARGET) 512 512 512 --comm p2p --no-print

run-collective: $(TARGET)
	mpirun --oversubscribe -np 4 $(TARGET) 512 512 512 --comm collective --no-print

bench: $(TARGET)
	python3 benchmark.py --binary ./$(TARGET)

clean:
	rm -f $(TARGET)
