cmake_minimum_required(VERSION 3.11)
project(fpcpp)

set(CMAKE_CXX_STANDARD 17)
if(MSVC)
  string(REPLACE "/W3" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

include(FetchContent)
set(EP_UPDATE_DISCONNECTED TRUE)

FetchContent_Declare(googletest_ext
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_SHALLOW TRUE
  UPDATE_DISCONNECTED TRUE
)
#FetchContent_MakeAvailable(googletest_ext) # CMake >= 3.14
FetchContent_GetProperties(googletest_ext)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest_ext)
  add_subdirectory(${googletest_ext_SOURCE_DIR} ${googletest_ext_BINARY_DIR})
endif()

include(GoogleTest)
if(MSVC)
  string(REPLACE "/MDd" "/MTd" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
endif()

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
target_compile_options(expected INTERFACE
  $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus>
)

add_library(helpers INTERFACE)
target_include_directories(helpers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(helpers INTERFACE optional expected)

add_executable(sum sum.cpp)
target_compile_options(sum PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_executable(wc wc.cpp)
target_compile_options(wc PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
target_link_libraries(wc PRIVATE helpers $<$<CXX_COMPILER_ID:GNU>:-lstdc++fs>)

add_executable(algs algs.cpp)
target_compile_options(algs PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
target_link_libraries(algs PRIVATE helpers benchmark)

add_executable(words words.cpp)
target_compile_options(words PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
target_Link_libraries(words PRIVATE optional $<$<CXX_COMPILER_ID:GNU>:-lstdc++fs>)

add_executable(maze maze.cpp)
target_compile_options(maze PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_executable(lazyeval lazyeval.cpp)
target_compile_options(lazyeval PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

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

add_executable(ill ill.cpp)
target_compile_options(ill PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_executable(tennis tennis.cpp)
target_compile_options(tennis PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
target_Link_libraries(tennis PRIVATE helpers gtest gtest_main $<$<PLATFORM_ID:Linux>:pthread>)
gtest_discover_tests(tennis)

add_executable(mtry mtry.cpp)
target_compile_options(mtry PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
target_Link_libraries(mtry PRIVATE helpers)

add_executable(expected_tests expected_tests.cpp)
target_compile_options(expected_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /Zc:__cplusplus>
)
target_link_libraries(expected_tests gtest gtest_main $<$<PLATFORM_ID:Linux>:pthread>)

add_executable(tmp tmp.cpp)
target_compile_options(tmp PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_executable(curried curried.cpp)
target_compile_options(curried PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_executable(actors actors.cpp)
target_compile_options(actors PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_executable(list list.cpp)
target_compile_options(list PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
target_Link_libraries(list PRIVATE helpers gtest gtest_main $<$<PLATFORM_ID:Linux>:pthread>)
gtest_discover_tests(list)

