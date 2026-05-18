cmake_minimum_required(VERSION 3.16)
project(lab07_mpi_fft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(MPI REQUIRED)

add_executable(fft_mpi src/fft_mpi.cpp)
target_link_libraries(fft_mpi PRIVATE MPI::MPI_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(fft_mpi PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native)
endif()
