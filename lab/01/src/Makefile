CXX ?= g++
CXXFLAGS_BASE = -std=c++17 -Wall -Wextra

TARGET_O0 = matmul_O0
TARGET_O3 = matmul_O3
TARGET_MKL = matmul_mkl

.PHONY: all clean run-o0 run-o3 run-py bench bench-mkl

all: $(TARGET_O0) $(TARGET_O3)

$(TARGET_O0): matmul.cpp
	$(CXX) $(CXXFLAGS_BASE) -O0 matmul.cpp -o $(TARGET_O0)

$(TARGET_O3): matmul.cpp
	$(CXX) $(CXXFLAGS_BASE) -O3 -march=native matmul.cpp -o $(TARGET_O3)

# Requires MKL installation and MKLROOT configured.
$(TARGET_MKL): matmul.cpp
	$(CXX) $(CXXFLAGS_BASE) -O3 -DUSE_MKL matmul.cpp -o $(TARGET_MKL) \
	-I$(MKLROOT)/include -L$(MKLROOT)/lib/intel64 \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

run-o0: $(TARGET_O0)
	./$(TARGET_O0) 512 512 512 naive

run-o3: $(TARGET_O3)
	./$(TARGET_O3) 512 512 512 reorder

run-py:
	python3 matmul.py 512 512 512

bench:
	python3 benchmark.py --m 512 --n 512 --k 512

bench-mkl:
	python3 benchmark.py --m 512 --n 512 --k 512 --include-mkl

clean:
	rm -f $(TARGET_O0) $(TARGET_O3) $(TARGET_MKL)
