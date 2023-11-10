#include <gtest/gtest.h>
#include <hastings/pipeline/node.h>
#include <hastings/pipeline/pipeline.h>

#include "helpers.h"

TEST(Pipeline, Construction) {
    using hastings::createPipeline;

    const auto pipeline = createPipeline();
    EXPECT_NE(pipeline, nullptr);
}

TEST(Pipeline, AddOrdered) {
    using hastings::createPipeline;
    using hastings::OrderedNode;

    const auto pipeline = createPipeline();
    pipeline->add<OrderedNode>();
}

TEST(Pipeline, AddUnordered) {
    using hastings::createPipeline;
    using hastings::UnorderedNode;

    const auto pipeline = createPipeline();
    pipeline->add<UnorderedNode>();
}

TEST(Pipeline, AddParallel) {
    using hastings::createPipeline;
    using hastings::ParallelNode;

    const auto pipeline = createPipeline();
    pipeline->add<ParallelNode>();
}

TEST(Pipeline, Start) { FAIL(); }