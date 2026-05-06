cmake_minimum_required(VERSION 3.16)
project(lab05_parallel_matmul LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_executable(openmp_matmul src/openmp_matmul.cpp)
target_link_libraries(openmp_matmul PRIVATE OpenMP::OpenMP_CXX)

add_executable(pthread_parallel_for_matmul src/pthread_parallel_for_matmul.cpp)
target_link_libraries(pthread_parallel_for_matmul PRIVATE Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(openmp_matmul PRIVATE -Wall -Wextra -O3 -march=native)
    target_compile_options(pthread_parallel_for_matmul PRIVATE -Wall -Wextra -O3 -march=native)
endif()
