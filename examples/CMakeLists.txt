cmake_minimum_required(VERSION 3.22)
project(hastings)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV     REQUIRED)
find_package(GLog       REQUIRED)
find_package(GFlags     REQUIRED)


add_executable(example_webcam example_webcam.cpp)
target_link_libraries(example_webcam hastings ${OpenCV_LIBS} glog gflags)