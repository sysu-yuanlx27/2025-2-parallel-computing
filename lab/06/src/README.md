# Lab 06

```bash
make
./matmul_test 256 4 static
./matmul_test 256 4 dynamic
./heated_plate_pthreads 4 static
./heated_plate_pthreads 4 dynamic
chmod +x benchmark.sh
./benchmark.sh
```

`libparallel_for.so` 提供实验要求的 `parallel_for(...)`，默认采用静态分块；
`parallel_for_schedule(...)` 额外支持 `static` / `dynamic` 调度和动态调度块大小。
