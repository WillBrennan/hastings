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

TEST(ImageContext, frameId) {
    using hastings::createImageContext;
    const auto context = createImageContext();

    context->frameId(12345);
    EXPECT_EQ(context->frameId(), 12345);
}

TEST(ImageContext, time) {
    using hastings::createImageContext;
    const auto context = createImageContext();

    context->time(12345);
    EXPECT_EQ(context->time(), 12345);
}

TEST(ImageContext, ImageNew) {
    using hastings::createImageContext;

    const auto context = createImageContext();
    auto image_bgr = context->image("BGR");

    EXPECT_EQ(image_bgr.rows, 0);
    EXPECT_EQ(image_bgr.cols, 0);
    EXPECT_EQ(image_bgr.type(), CV_8UC1);
}

TEST(ImageContext, ImageExisting) {
    using hastings::createImageContext;
    const auto context = createImageContext();

    context->image("BGR") = cv::Mat::zeros({10, 12}, CV_8UC3);

    auto& other_image = context->image("BGR");
    EXPECT_EQ(other_image.rows, 12);
    EXPECT_EQ(other_image.cols, 10);
    EXPECT_EQ(other_image.type(), CV_8UC3);
}

TEST(ImageContext, ImageGet) {
    using hastings::createImageContext;
    const auto context = createImageContext();

    context->image("BGR") = cv::Mat::zeros({10, 12}, CV_8UC3);
    context->image("Y") = cv::Mat::zeros({20, 24}, CV_8UC1);

    context->images([](const std::string& name, cv::Mat& image) {
        if (name == "BGR") {
            EXPECT_EQ(image.cols, 10);
            EXPECT_EQ(image.rows, 12);
            EXPECT_EQ(image.type(), CV_8UC3);
        } else if (name == "Y") {
            EXPECT_EQ(image.cols, 20);
            EXPECT_EQ(image.rows, 24);
            EXPECT_EQ(image.type(), CV_8UC1);
        } else {
            FAIL();
        }
    });
}

TEST(ImageContext, ImageConstGet) {
    using hastings::createImageContext;
    const auto context_ptr = createImageContext();

    {
        context_ptr->image("BGR") = cv::Mat::zeros({10, 12}, CV_8UC3);
        context_ptr->image("Y") = cv::Mat::zeros({20, 24}, CV_8UC1);
    }

    const auto& context = *context_ptr;

    context.images([](const std::string& name, const cv::Mat& image) {
        if (name == "BGR") {
            EXPECT_EQ(image.cols, 10);
            EXPECT_EQ(image.rows, 12);
            EXPECT_EQ(image.type(), CV_8UC3);
        } else if (name == "Y") {
            EXPECT_EQ(image.cols, 20);
            EXPECT_EQ(image.rows, 24);
            EXPECT_EQ(image.type(), CV_8UC1);
        } else {
            FAIL();
        }
    });
}

TEST(ImageContext, VectorGraphicsExistingImage) {
    using hastings::createImageContext;
    using hastings::VectorGraphics;
    const auto context = createImageContext();

    context->image("BGR") = cv::Mat::zeros({10, 12}, CV_8UC3);

    VectorGraphics graphics(10);
    context->vectorGraphic("BGR", std::move(graphics));

    EXPECT_EQ(context->vectorGraphic("BGR").size(), 10);
}

TEST(ImageContext, VectorGraphicsNoImage) {
    using hastings::createImageContext;
    using hastings::VectorGraphics;
    const auto context = createImageContext();

    VectorGraphics graphics(10);
    context->vectorGraphic("BGR", std::move(graphics));

    EXPECT_EQ(context->vectorGraphic("BGR").size(), 10);
}

TEST(ImageContext, VectorGraphicsAppend) {
    using hastings::createImageContext;
    using hastings::VectorGraphics;
    const auto context = createImageContext();

    context->vectorGraphic("BGR", VectorGraphics(10));
    context->vectorGraphic("BGR", VectorGraphics(13));
    context->vectorGraphic("Y", VectorGraphics(7));
    context->vectorGraphic("Y", VectorGraphics(2));

    EXPECT_EQ(context->vectorGraphic("BGR").size(), 23);
    EXPECT_EQ(context->vectorGraphic("Y").size(), 9);
}

TEST(ImageContext, VectorGraphicsNoCamera) {
    using hastings::createImageContext;
    const auto context = createImageContext();

    ASSERT_THROW({ context->vectorGraphic("BGR"); }, std::out_of_range);
}

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

TEST(MultiImageContext, clearResult) {
    using hastings::createMultiImageContext;
    const auto context = createMultiImageContext();

    context->result("test-it") = std::string("hello-world");
    EXPECT_EQ(context->result<std::string>("test-it"), "hello-world");

    context->clear();
    EXPECT_FALSE(context->result("test-it").has_value());
}

TEST(MultiImageContext, clearCameraResult) {
    using hastings::createMultiImageContext;
    const auto context = createMultiImageContext();

    context->cameras("camera")->result("test-other") = 1234;
    EXPECT_EQ(context->cameras("camera")->result<int>("test-other"), 1234);

    context->clear();
    EXPECT_FALSE(context->cameras("camera")->result("test-other").has_value());
}

TEST(MultiImageContext, clearGraphics) {
    using hastings::createMultiImageContext;
    using hastings::VectorGraphics;

    const auto context = createMultiImageContext();
    context->vectorGraphic("BGR", VectorGraphics(123));

    EXPECT_EQ(context->vectorGraphic("BGR").size(), 123);

    context->clear();
    EXPECT_EQ(context->vectorGraphic("BGR").size(), 0);
}

TEST(MultiImageContext, clearCameraGraphics) {
    using hastings::createMultiImageContext;
    using hastings::VectorGraphics;

    const auto context = createMultiImageContext();
    const auto cameraContext = context->cameras("camera_a");
    cameraContext->vectorGraphic("BGR", VectorGraphics(123));

    EXPECT_EQ(cameraContext->vectorGraphic("BGR").size(), 123);

    context->clear();
    EXPECT_EQ(cameraContext->vectorGraphic("BGR").size(), 0);
}