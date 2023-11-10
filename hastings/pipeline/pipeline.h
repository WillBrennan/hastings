#pragma once

#include <memory>
#include <thread>

#include "hastings/pipeline/node.h"

namespace hastings {
class PipelineInterface {
  public:
    PipelineInterface() = default;
    virtual ~PipelineInterface() = default;

    virtual void add(NodeInterface::Ptr&& node) = 0;

    template <class T, class... Args>
    void add(Args&&... args) {
        add(std::make_unique<T>(std::forward<Args>(args)...));
    }

    virtual void start() = 0;
};

std::unique_ptr<PipelineInterface> createPipeline(const unsigned int num_threads = std::thread::hardware_concurrency());
}  // namespace hastings