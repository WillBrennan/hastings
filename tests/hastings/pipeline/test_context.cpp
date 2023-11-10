#include <gtest/gtest.h>
#include <hastings/pipeline/context.h>

TEST(ImageContext, Construction) {
    using hastings::createImageContext;

    const auto context = createImageContext();
    EXPECT_NE(context, nullptr);
}

TEST(ImageContext, ResultMissingValue) {
    using hastings::createImageContext;

    const auto context = createImageContext();
    const auto& value_a = context->result("value_a");
    EXPECT_FALSE(value_a.has_value());
}

TEST(ImageContext, ResultExistingValue) {
    using hastings::createImageContext;

    const auto context = createImageContext();

    context->result("value_a") = 100;
    const auto& value = context->result("value_a");

    EXPECT_TRUE(value.has_value());
    EXPECT_EQ(value.type(), typeid(int));
    EXPECT_EQ(std::any_cast<int>(value), 100);
}

TEST(ImageContext, ResultWithType) {
    using hastings::createImageContext;

    const auto context = createImageContext();

    context->result("value_a") = 100;
    EXPECT_EQ(context->result<int>("value_a"), 100);
}

TEST(ImageContext, ResultWithTypeRef) {
    using hastings::createImageContext;

    const auto context = createImageContext();

    context->result("value_a") = 100;

    auto& value_a = context->result<int>("value_a");
    EXPECT_EQ(value_a, 100);

    value_a = 150;
    EXPECT_EQ(context->result<int>("value_a"), 150);
}

TEST(ImageContext, Clear) {
    using hastings::createImageContext;

    const auto context = createImageContext();

    context->result("value_a") = 100;

    context->clear();
    EXPECT_FALSE(context->result("value_a").has_value());
}

TEST(ImageContext, frameId) { FAIL(); }

TEST(ImageContext, time) { FAIL(); }

TEST(MultiImageContext, Construction) {
    using hastings::createMultiImageContext;

    const auto context = createMultiImageContext();
    EXPECT_NE(context, nullptr);
}

TEST(MultiImageContext, getCamera) {
    using hastings::createMultiImageContext;

    const auto context = createMultiImageContext();
    auto camera_ptr_a = context->cameras("camera_a");
    auto camera_ptr_b = context->cameras("camera_a");

    EXPECT_NE(camera_ptr_a, nullptr);
    EXPECT_EQ(camera_ptr_a, camera_ptr_b);
}

TEST(MultiImageContext, cameras) {
    using hastings::createMultiImageContext;
    using Camera = hastings::MultiImageContextInterface::Camera;
    using Cameras = hastings::MultiImageContextInterface::Cameras;

    const auto context = createMultiImageContext();
    auto camera_ptr_a = context->cameras("camera_a");
    auto camera_ptr_b = context->cameras("camera_b");

    const auto& cameras = context->cameras();
    EXPECT_EQ(cameras.size(), 2);

    EXPECT_EQ(std::get<0>(cameras[0]), "camera_a");
    EXPECT_EQ(std::get<0>(cameras[1]), "camera_b");

    EXPECT_EQ(std::get<1>(cameras[0]).get(), camera_ptr_a);
    EXPECT_EQ(std::get<1>(cameras[1]).get(), camera_ptr_b);
}

TEST(MultiImageContext, frameId) {
    using hastings::createMultiImageContext;
    const auto context = createMultiImageContext();

    context->frameId(12345);
    EXPECT_EQ(context->frameId(), 12345);
}

TEST(MultiImageContext, time) {
    using hastings::createMultiImageContext;
    const auto context = createMultiImageContext();

    context->time(23456);
    EXPECT_EQ(context->time(), 23456);
}

TEST(MultiImageContext, result) {
    using hastings::createMultiImageContext;
    const auto context = createMultiImageContext();

    context->result("test-it") = std::string("hello-world");
    EXPECT_EQ(context->result<std::string>("test-it"), "hello-world");
}

TEST(MultiImageContext, clear) {
    using hastings::createMultiImageContext;
    const auto context = createMultiImageContext();

    context->result("test-it") = std::string("hello-world");
    EXPECT_EQ(context->result<std::string>("test-it"), "hello-world");

    context->clear();

    EXPECT_FALSE(context->result("test-it").has_value());
}