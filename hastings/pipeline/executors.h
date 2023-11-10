#pragma once

#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>

#include "hastings/pipeline/node.h"

namespace hastings {

class ExecutorInterface : public NodeInterface {
  public:
    ExecutionPolicy executionPolicy() const final { return ExecutionPolicy::Parallel; }
};

class ParallelExecutor final : public ExecutorInterface {
  public:
    explicit ParallelExecutor(Ptr&& node);

    std::string name() const final;

    void process(MultiImageContextInterface& multi_context) final;

  private:
    Ptr node_;
};

class UnorderedExecutor final : public ExecutorInterface {
  public:
    explicit UnorderedExecutor(Ptr&& node);

    std::string name() const final;

    void process(MultiImageContextInterface& multi_context) final;

  private:
    std::mutex mutex_;
    Ptr node_;
};

class OrderedExecutor final : public ExecutorInterface {
  public:
    explicit OrderedExecutor(Ptr&& node);

    std::string name() const final;

    void process(MultiImageContextInterface& multi_context) final;

  private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::size_t frame_id_ = 0;
    Ptr node_;
};
}  // namespace hastings