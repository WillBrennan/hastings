#include <gtest/gtest.h>

#include "hastings/helpers/profile_marker.h"

TEST(ProfilerConnectionTest, ConstructorDestructor) { hastings::ProfilerConnection profilerConnection; }

TEST(ProfilerFunctionMarkerTest, ConstructorDestructor) { hastings::ProfilerFunctionMarker profilerFunctionMarker("test_function"); }

TEST(ProfilerFrameMarkerTest, ConstructorDestructor) { hastings::ProfilerFrameMarker profilerFrameMarker("test_frame"); }
