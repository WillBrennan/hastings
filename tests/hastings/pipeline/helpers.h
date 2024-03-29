#pragma once

#include <hastings/pipeline/context.h>
#include <hastings/pipeline/node.h>

namespace hastings {
struct OrderedNode final : NodeInterface {
    ExecutionPolicy executionPolicy() const override final;
    std::string name() const override final;

    void process(MultiImageContextInterface& multi_context) override final;

  private:
    std::vector<int> frame_ordering_;
};

struct UnorderedNode final : NodeInterface {
  public:
    ExecutionPolicy executionPolicy() const override final;
    std::string name() const override final;

    void process(MultiImageContextInterface& multi_context) override final;

  private:
    std::vector<int> frame_ordering_;
};

struct ParallelNode final : NodeInterface {
    ExecutionPolicy executionPolicy() const override final;
    std::string name() const override final;

    void process(MultiImageContextInterface& multi_context) override final;
};
}  // namespace hastings