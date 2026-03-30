Size: m=512, n=512, k=512
Peak GFLOPS: 80.000
| Version | Time (sec) | Relative Speedup | Absolute Speedup | GFLOPS | Peak Performance (%) |
|---|---:|---:|---:|---:|---:|
| 1 Python | 11.797337 | 1.000 | 1.000 | 0.023 | 0.029 |
| 2 C/C++ | 1.163168 | 10.142 | 10.142 | 0.231 | 0.289 |
| 3 Adjust Loop Order | 0.694146 | 1.676 | 16.995 | 0.387 | 0.484 |
| 4 Compiler Optimization | 0.026696 | 26.002 | 441.914 | 10.055 | 12.569 |
| 5 Loop Unrolling | 0.026168 | 1.020 | 450.831 | 10.258 | 12.822 |
