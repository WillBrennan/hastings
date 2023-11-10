#pragma once

#include <hastings/pipeline/context.h>
#include <hastings/pipeline/node.h>

namespace hastings {
struct OrderedNode final : NodeInterface {
    ExecutionPolicy executionPolicy() const override final;
    std::string name() const override final;

    void process(MultiImageContextInterface& multi_context) override final;
};

struct UnorderedNode final : NodeInterface {
  public:
    ExecutionPolicy executionPolicy() const override final;
    std::string name() const override final;

    void process(MultiImageContextInterface& multi_context) override final;
};

struct ParallelNode final : NodeInterface {
    ExecutionPolicy executionPolicy() const override final;
    std::string name() const override final;

    void process(MultiImageContextInterface& multi_context) override final;
};
}  // namespace hastings