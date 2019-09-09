cmake_minimum_required(VERSION 3.11)
project(fpcpp)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
set(EP_UPDATE_DISCONNECTED TRUE)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(benchmark_ext
  GIT_REPOSITORY https://github.com/google/benchmark
  GIT_SHALLOW TRUE
  UPDATE_DISCONNECTED TRUE
)
#FetchContent_MakeAvailable(benchmark_ext) # CMake >= 3.14
FetchContent_GetProperties(benchmark_ext)
if(NOT googletest_POPULATED)
  FetchContent_Populate(benchmark_ext)
  add_subdirectory(${benchmark_ext_SOURCE_DIR} ${benchmark_ext_BINARY_DIR})
endif()

FetchContent_Declare(gsl_ext
  GIT_REPOSITORY https://github.com/Microsoft/GSL
  GIT_SHALLOW TRUE
  UPDATE_DISCONNECTED TRUE
)
#FetchContent_MakeAvailable(gsl_ext) # CMake >= 3.14
FetchContent_GetProperties(gsl_ext)
if(NOT googletest_POPULATED)
  FetchContent_Populate(gsl_ext)
  add_subdirectory(${gsl_ext_SOURCE_DIR} ${gsl_ext_BINARY_DIR})
endif()

set(opt_url "https://raw.githubusercontent.com/TartanLlama/optional/master/include/tl/optional.hpp")
file(DOWNLOAD ${opt_url} ${CMAKE_CURRENT_BINARY_DIR}/tl/optional.hpp)
add_library(optional INTERFACE)
target_include_directories(optional INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

set(exp_url "https://raw.githubusercontent.com/TartanLlama/expected/master/include/tl/expected.hpp")
file(DOWNLOAD ${exp_url} ${CMAKE_CURRENT_BINARY_DIR}/tl/expected.hpp)
add_library(expected INTERFACE)
target_include_directories(expected INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_executable(sum sum.cpp)

add_executable(wc wc.cpp)
target_link_libraries(wc PRIVATE $<$<CXX_COMPILER_ID:GNU>:-lstdc++fs>)

add_executable(algs algs.cpp)
target_link_libraries(algs PRIVATE benchmark)

add_executable(words words.cpp)
target_Link_libraries(words PRIVATE optional $<$<CXX_COMPILER_ID:GNU>:-lstdc++fs>)

add_executable(maze maze.cpp)

add_executable(lazyeval lazyeval.cpp)

add_executable(memoization memoization.cpp)
target_compile_options(memoization PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_executable(log log.cpp)
target_compile_options(log PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall -march=native>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /arch:AVX2>
)
