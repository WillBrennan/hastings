#include "hastings/pipeline/executors.h"

namespace hastings {

ParallelExecutor::ParallelExecutor(Ptr&& node) : node_(std::move(node)) {
    if (node_->executionPolicy() != ExecutionPolicy::Parallel) {
        throw std::invalid_argument("requires a parallel processor");
    }
}

std::string ParallelExecutor::name() const { return "ParallelExecutor"; }

void ParallelExecutor::process(MultiImageContextInterface& multi_context) { node_->process(multi_context); }

UnorderedExecutor::UnorderedExecutor(Ptr&& node) : node_(std::move(node)) {
    if (node_->executionPolicy() != ExecutionPolicy::Unordered) {
        throw std::invalid_argument("requires an unordered processor");
    }
}

std::string UnorderedExecutor::name() const { return "UnorderedExecutor"; }

void UnorderedExecutor::process(MultiImageContextInterface& multi_context) {
    std::lock_guard lock(mutex_);
    node_->process(multi_context);
}

OrderedExecutor::OrderedExecutor(Ptr&& node) : node_(std::move(node)) {
    if (node_->executionPolicy() != ExecutionPolicy::Ordered) {
        throw std::invalid_argument("requires an ordered processor");
    }
};

std::string OrderedExecutor::name() const { return "OrderedExecutor"; }

void OrderedExecutor::process(MultiImageContextInterface& multi_context) {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [&] { return multi_context.frameId() == frame_id_; });
    node_->process(multi_context);

    frame_id_ += 1;

    lock.unlock();
    cv_.notify_all();
};
}  // namespace hastings