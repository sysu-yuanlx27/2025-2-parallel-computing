cmake_minimum_required(VERSION 3.16)
project(lab03_pthreads LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(matmul_pthreads src/matmul_pthreads.cpp)
add_executable(array_sum_pthreads src/array_sum_pthreads.cpp)

find_package(Threads REQUIRED)
target_link_libraries(matmul_pthreads PRIVATE Threads::Threads)
target_link_libraries(array_sum_pthreads PRIVATE Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(matmul_pthreads PRIVATE -Wall -Wextra -O3 -march=native)
    target_compile_options(array_sum_pthreads PRIVATE -Wall -Wextra -O3 -march=native)
endif()
