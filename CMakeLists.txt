cmake_minimum_required(VERSION 3.14)
project(fpcpp)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(benchmark GIT_REPOSITORY https://github.com/google/benchmark GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(benchmark)

FetchContent_Declare(GSL GIT_REPOSITORY https://github.com/Microsoft/GSL GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(GSL)

set(OPTIONAL_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(Optional GIT_REPOSITORY https://github.com/TartanLlama/optional GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(Optional)

#set(EXPECTED_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
#FetchContent_Declare(Expected GIT_REPOSITORY https://github.com/TartanLlama/expected GIT_SHALLOW TRUE)
#FetchContent_MakeAvailable(Expected)

add_executable(sum sum.cpp)
add_executable(wc wc.cpp)
add_executable(algs algs.cpp)
target_link_libraries(algs PRIVATE benchmark)
add_executable(words words.cpp)
target_Link_libraries(words optional)
add_executable(maze maze.cpp)
add_executable(lazyeval lazyeval.cpp)

add_executable(memoization memoization.cpp)
target_compile_options(memoization PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/W4>)

add_executable(log log.cpp)
target_compile_options(log PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/W4>)
