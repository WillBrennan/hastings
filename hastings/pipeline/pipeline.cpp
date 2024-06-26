#include "hastings/pipeline/pipeline.h"

#include <glog/logging.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "hastings/helpers/profile_marker.h"
#include "hastings/pipeline/executors.h"

namespace hastings {
class Pipeline final : public PipelineInterface {
  public:
    explicit Pipeline(const unsigned int num_threads) : num_threads_(num_threads) {}

    ~Pipeline() = default;

    void add(NodeInterface::Ptr&& node) override final {
        const auto policy = node->executionPolicy();
        Executor executor;

        switch (policy) {
            case ExecutionPolicy::Ordered:
                executor = std::make_unique<OrderedExecutor>(std::move(node));
                break;
            case ExecutionPolicy::Unordered:
                executor = std::make_unique<UnorderedExecutor>(std::move(node));
                break;
            case ExecutionPolicy::Parallel:
                executor = std::make_unique<ParallelExecutor>(std::move(node));
                break;
            default:
                throw std::invalid_argument("unsupported policy");
        }

        executors_.emplace_back(std::move(executor));
    };

    void start(const std::uint64_t num_frames) override final {
        ProfilerConnection profiler;

        std::vector<std::thread> threads;
        threads.reserve(num_threads_);

        for (int idx = 0; idx < num_threads_; ++idx) {
            threads.emplace_back([this, num_frames] {
                const auto context = createMultiImageContext();

                while (true) {
                    const auto frame_id = frame_id_++;
                    if (frame_id >= num_frames) {
                        return;
                    }

                    context->clear();
                    context->frameId(frame_id);

                    ProfilerFunctionMarker marker_frame("frame");

                    for (auto& executor : executors_) {
                        executor->process(*context);
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    };

  private:
    using Executor = std::unique_ptr<ExecutorInterface>;

    unsigned int num_threads_;
    std::vector<Executor> executors_;
    std::atomic<std::uint64_t> frame_id_ = 0;
};

std::unique_ptr<PipelineInterface> createPipeline(const unsigned int num_threads) { return std::make_unique<Pipeline>(num_threads); }
}  // namespace hastings