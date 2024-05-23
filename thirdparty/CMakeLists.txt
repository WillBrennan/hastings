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