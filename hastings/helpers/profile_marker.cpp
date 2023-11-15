#include "hastings/helpers/profile_marker.h"

#include <Remotery.h>
#include <glog/logging.h>

namespace hastings {

Remotery* rmt_ = nullptr;

ProfilerConnection::ProfilerConnection() { _rmt_CreateGlobalInstance(&rmt_); }

ProfilerConnection::~ProfilerConnection() {
    auto rmt = _rmt_GetGlobalInstance();
    _rmt_DestroyGlobalInstance(rmt);
}

ProfilerFunctionMarker::ProfilerFunctionMarker(const std::string& name) { _rmt_BeginCPUSample(name.data(), 0, nullptr); }

ProfilerFunctionMarker::~ProfilerFunctionMarker() { _rmt_EndCPUSample(); }

ProfilerFrameMarker::ProfilerFrameMarker(const std::string& name) { rmt_MarkFrame(); }

ProfilerFrameMarker::~ProfilerFrameMarker() {}
}  // namespace hastings