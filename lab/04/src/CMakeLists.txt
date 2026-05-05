cmake_minimum_required(VERSION 3.10)
project(lab04_openmp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(matmul_openmp matmul_openmp.cpp)
target_link_libraries(matmul_openmp PRIVATE OpenMP::OpenMP_CXX)

add_executable(array_sum_openmp array_sum_openmp.cpp)
target_link_libraries(array_sum_openmp PRIVATE OpenMP::OpenMP_CXX)
