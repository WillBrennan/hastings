include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           v1.14.0
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(googletest)


add_library(Remotery  Remotery/lib/Remotery.c)
target_include_directories(Remotery PUBLIC Remotery/lib/)