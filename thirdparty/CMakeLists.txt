include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           v1.14.0
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

add_library(Remotery  ${CMAKE_CURRENT_SOURCE_DIR}/Remotery/lib/Remotery.c)
target_include_directories(Remotery PUBLIC Remotery/lib/)

set(NVINFER_INCLUDE_DIR /home/will/code/thirdparty/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4/TensorRT-10.0.1.6/include/)
set(NVINFER_LIBS_DIR /home/will/code/thirdparty/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4/TensorRT-10.0.1.6/lib)

find_library(NVINFER  NAMES nvinfer HINTS ${NVINFER_LIBS_DIR})
find_library(NVINFERPLUGIN NAMES nvinfer_plugin HINTS ${NVINFER_LIBS_DIR})
find_library(NVONNXPARSER NAMES nvonnxparser HINTS ${NVINFER_LIBS_DIR})

if(CUDA_FOUND AND NVINFER AND NVINFERPLUGIN AND NVONNXPARSER)
  set(TENSORRT_LIBRARIES    ${CUDA_LIBRARIES} ${NVINFER} ${NVINFERPLUGIN} ${NVONNXPARSER} PARENT_SCOPE)
  set(TENSORRT_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} ${NVINFER_INCLUDE_DIR} PARENT_SCOPE)
  set(TENSORRT_FOUND ON)
  message("-- Found - TensorRT")
else()
  message("TensorRT NOT Available")
  set(TENSORRT_FOUND OFF)
endif()