#pragma once

#include <string>

namespace hastings {
class ProfilerConnection {
  public:
    explicit ProfilerConnection();
    ~ProfilerConnection();
};
class ProfilerFunctionMarker {
  public:
    explicit ProfilerFunctionMarker(const std::string& name);
    ~ProfilerFunctionMarker();
};

class ProfilerFrameMarker {
  public:
    explicit ProfilerFrameMarker(const std::string& name);
    ~ProfilerFrameMarker();
};
}  // namespace hastings