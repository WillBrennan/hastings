#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <hastings/pipeline/executors.h>

#include <numeric>
#include <thread>

#include "helpers.h"

std::vector<std::vector<int>> process(const int num_threads, hastings::ExecutorInterface* executor) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::mutex mutex;
    std::vector<std::vector<int>> results;

    for (auto idx = 0; idx < num_threads; ++idx) {
        threads.emplace_back([&mutex, &results, executor, idx] {
            const auto context = hastings::createMultiImageContext();
            context->frameId(idx);
            executor->process(*context);

            {
                std::lock_guard lock(mutex);
                results.emplace_back(context->result<std::vector<int>>("frameOrdering"));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return results;
}

TEST(OrderedExecutor, Constructor) {
    using hastings::OrderedExecutor;
    using hastings::OrderedNode;

    auto executor = OrderedExecutor(std::make_unique<OrderedNode>());
}

TEST(OrderedExecutor, ConstructorBadNode) {
    using hastings::OrderedExecutor;
    using hastings::ParallelNode;
    using hastings::UnorderedNode;

    EXPECT_THROW(OrderedExecutor(std::make_unique<UnorderedNode>()), std::invalid_argument);
    EXPECT_THROW(OrderedExecutor(std::make_unique<ParallelNode>()), std::invalid_argument);
}

TEST(OrderedExecutor, Name) {
    using hastings::OrderedExecutor;
    using hastings::OrderedNode;

    auto executor = OrderedExecutor(std::make_unique<OrderedNode>());
    EXPECT_EQ(executor.name(), "OrderedExecutor");
}

TEST(OrderedExecutor, Process) {
    using hastings::OrderedExecutor;
    using hastings::OrderedNode;

    auto executor = OrderedExecutor(std::make_unique<OrderedNode>());

    const auto num_threads = 100;
    const auto frame_ordering = process(num_threads, &executor);

    std::vector<int> expected_ordering(num_threads);
    std::iota(expected_ordering.begin(), expected_ordering.end(), 0);

    EXPECT_THAT(frame_ordering.at(99), testing::ElementsAreArray(expected_ordering));
}

TEST(UnorderedExecutor, Constructor) {
    using hastings::UnorderedExecutor;
    using hastings::UnorderedNode;

    auto executor = UnorderedExecutor(std::make_unique<UnorderedNode>());
}

TEST(UnorderedExecutor, ConstructorBadNode) {
    using hastings::OrderedNode;
    using hastings::ParallelNode;
    using hastings::UnorderedExecutor;

    EXPECT_THROW(UnorderedExecutor(std::make_unique<OrderedNode>()), std::invalid_argument);
    EXPECT_THROW(UnorderedExecutor(std::make_unique<ParallelNode>()), std::invalid_argument);
}

TEST(UnorderedExecutor, Name) {
    using hastings::UnorderedExecutor;
    using hastings::UnorderedNode;

    auto executor = UnorderedExecutor(std::make_unique<UnorderedNode>());
    EXPECT_EQ(executor.name(), "UnorderedExecutor");
}

TEST(UnorderedExecutor, Process) {
    using hastings::UnorderedExecutor;
    using hastings::UnorderedNode;
    // NOTE(will): relying on tsan to find data races.

    auto executor = UnorderedExecutor(std::make_unique<UnorderedNode>());

    const auto num_threads = 100;
    const auto frame_ordering = process(num_threads, &executor);

    std::vector<int> expected_ordering(num_threads);
    std::iota(expected_ordering.begin(), expected_ordering.end(), 0);

    EXPECT_THAT(frame_ordering.at(99), testing::UnorderedElementsAreArray(expected_ordering));
}

TEST(ParallelExecutor, Constructor) {
    using hastings::ParallelExecutor;
    using hastings::ParallelNode;

    auto executor = ParallelExecutor(std::make_unique<ParallelNode>());
}

TEST(ParallelExecutor, ConstructorBadNode) {
    using hastings::OrderedNode;
    using hastings::ParallelExecutor;
    using hastings::UnorderedNode;

    EXPECT_THROW(ParallelExecutor(std::make_unique<OrderedNode>()), std::invalid_argument);
    EXPECT_THROW(ParallelExecutor(std::make_unique<UnorderedNode>()), std::invalid_argument);
}

TEST(ParallelExecutor, Name) {
    using hastings::ParallelExecutor;
    using hastings::ParallelNode;

    auto executor = ParallelExecutor(std::make_unique<ParallelNode>());
    EXPECT_EQ(executor.name(), "ParallelExecutor");
}

TEST(ParallelExecutor, Process) {
    using hastings::ParallelExecutor;
    using hastings::ParallelNode;

    auto executor = ParallelExecutor(std::make_unique<ParallelNode>());

    const auto num_threads = 100;
    const auto frame_ordering = process(num_threads, &executor);
}