#pragma once

#include <memory>
#include <string>

#include "hastings/pipeline/context.h"

namespace hastings {

enum class ExecutionPolicy {
    Ordered,
    Unordered,
    Parallel,
};

class NodeInterface {
  public:
    using Ptr = std::unique_ptr<NodeInterface>;

    NodeInterface() = default;
    virtual ~NodeInterface() = default;

    virtual ExecutionPolicy executionPolicy() const = 0;
    virtual std::string name() const = 0;

    // todo(will) - handle fetching & adjusting settings...
    virtual void process(MultiImageContextInterface& multi_context) = 0;
};
}  // namespace hastings