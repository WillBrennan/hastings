#include <gtest/gtest.h>
#include <hastings/pipeline/node.h>
#include <hastings/pipeline/pipeline.h>

#include <functional>

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

void testPipeline(const int num_frames, std::function<void(hastings::MultiImageContextInterface&)>&& fn) {
    using hastings::createPipeline;
    using hastings::ExecutionPolicy;
    using hastings::NodeInterface;
    using FnCallback = std::function<void(hastings::MultiImageContextInterface&)>;

    struct LambdaNode final : NodeInterface {
        explicit LambdaNode(FnCallback fn) : fn_(fn) {}

        ExecutionPolicy executionPolicy() const override final { return ExecutionPolicy::Ordered; }

        std::string name() const override final { return "LambdaNode"; }

        void process(hastings::MultiImageContextInterface& multi_context) override final { fn_(multi_context); }

        FnCallback fn_;
    };

    const auto pipeline = createPipeline(1);
    pipeline->add<LambdaNode>(fn);

    pipeline->start(num_frames);
}

TEST(Pipeline, startAndFrameId) {
    using hastings::MultiImageContextInterface;

    auto frame_id = 0;
    testPipeline(1234, [&frame_id](MultiImageContextInterface& multi_context) {
        ASSERT_EQ(multi_context.frameId(), frame_id);
        frame_id += 1;
    });

    EXPECT_EQ(frame_id, 1234);
}

TEST(Pipeline, time) {
    using hastings::MultiImageContextInterface;
    using Time = MultiImageContextInterface::Time;

    testPipeline(100, [&](MultiImageContextInterface& multi_context) { ASSERT_EQ(multi_context.time(), Time()); });
}

TEST(Pipeline, clears) {
    using hastings::MultiImageContextInterface;

    testPipeline(100, [&](MultiImageContextInterface& multi_context) {
        ASSERT_FALSE(multi_context.result("a").has_value());
        ASSERT_FALSE(multi_context.result("b").has_value());

        multi_context.vectorGraphic("image", {{}, {}});
        ASSERT_EQ(multi_context.vectorGraphic("image").size(), 2);

        multi_context.result("a") = 12345;
        multi_context.result("b") = "helloWorld";
    });
}