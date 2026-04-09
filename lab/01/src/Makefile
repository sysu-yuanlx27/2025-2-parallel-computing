CXX := mpicxx
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O3 -march=native
TARGET := matmul_mpi

.PHONY: all clean run bench

all: $(TARGET)

$(TARGET): matmul.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

run: $(TARGET)
	mpirun --oversubscribe -np 4 ./$(TARGET) 512 512 512 --no-print

bench: $(TARGET)
	python3 benchmark.py --binary ./$(TARGET)

clean:
	rm -f $(TARGET)
