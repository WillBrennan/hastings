#include "helpers.h"

namespace hastings {

ExecutionPolicy OrderedNode::executionPolicy() const { return ExecutionPolicy::Ordered; }
std::string OrderedNode::name() const { return "OrderedNode"; }

void OrderedNode::process(MultiImageContextInterface& multi_context){};

ExecutionPolicy UnorderedNode::executionPolicy() const { return ExecutionPolicy::Unordered; }
std::string UnorderedNode::name() const { return "UnorderedNode"; }

void UnorderedNode::process(MultiImageContextInterface& multi_context){};

ExecutionPolicy ParallelNode::executionPolicy() const { return ExecutionPolicy::Parallel; }
std::string ParallelNode::name() const { return "ParallelNode"; }

void ParallelNode::process(MultiImageContextInterface& multi_context){};
}  // namespace hastings