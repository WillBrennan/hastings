cmake_minimum_required(VERSION 3.22)
project(hastings)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
include(CTest)
include(GoogleTest)

enable_testing()

add_subdirectory(thirdparty)

find_package(GLog       REQUIRED)
find_package(OpenCV     REQUIRED)

file(GLOB_RECURSE       lib_hastings_srcs       hastings/*.cpp)
add_library(hastings    ${lib_hastings_srcs})
target_include_directories(hastings PUBLIC .)
target_link_libraries(hastings PUBLIC glog ${OpenCV_LIBS} Remotery)

add_subdirectory(examples)

file(GLOB_RECURSE   test_cxx_source_files           tests/*.cpp)
add_executable(test_hastings               ${test_cxx_source_files})
target_link_libraries(test_hastings        GTest::gtest_main hastings)

gtest_discover_tests(test_hastings)

file(GLOB_RECURSE   all_cxx_files   hastings/*.cpp hastings/*.h tests/*.cpp tests/*.h)
find_program(clang_format "clang-format")
add_custom_target(format COMMAND ${clang_format} -i -style=file ${all_cxx_files})