cmake_minimum_required(VERSION 3.22)
project(hastings LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
include(CTest)
include(GoogleTest)

enable_testing()

find_package(GLog           REQUIRED)
find_package(OpenCV         REQUIRED)
find_package(Boost          REQUIRED COMPONENTS filesystem system thread regex)
find_package(CUDA           REQUIRED)
find_package(CUDAToolkit    REQUIRED)
add_subdirectory(thirdparty)

file(GLOB_RECURSE       lib_hastings_srcs       hastings/*.cpp hastings/*.cu)
add_library(hastings    ${lib_hastings_srcs})
target_include_directories(hastings PUBLIC . ${TENSORRT_INCLUDE_DIRS})
target_link_libraries(hastings PUBLIC glog ${OpenCV_LIBS} ${Boost_LIBRARIES} Remotery nlohmann_json::nlohmann_json CUDA::cudart ${TENSORRT_LIBRARIES})

add_subdirectory(examples)

file(GLOB_RECURSE   test_cxx_source_files           tests/*.cpp)
add_executable(test_hastings               ${test_cxx_source_files})
target_link_libraries(test_hastings        GTest::gtest_main hastings GTest::gmock Remotery)

gtest_discover_tests(test_hastings)

file(GLOB_RECURSE   all_cxx_files   *.h *.cpp)
find_program(clang_format "clang-format")
add_custom_target(format COMMAND ${clang_format} -i -style=file ${all_cxx_files})