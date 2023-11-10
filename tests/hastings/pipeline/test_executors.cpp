#include <gtest/gtest.h>
#include <hastings/pipeline/executors.h>

#include "helpers.h"

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
    // todo - add frame ids to a vector... they should be in order...
    FAIL();
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

TEST(UnorderedExecutor, Process) { FAIL(); }

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

TEST(ParallelExecutor, Process) { FAIL(); }