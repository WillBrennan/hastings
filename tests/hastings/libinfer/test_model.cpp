#include <gtest/gtest.h>

#include <filesystem>

#include "hastings/libinfer/model.h"
#include "hastings/libinfer/tensor.h"

TEST(Model, ConstructionFP32) {
    using libinfer::Model;

    const auto onnx_path = std::filesystem::path(__FILE__).parent_path() / "FastSCNN.onnx";
    ASSERT_TRUE(std::filesystem::exists(onnx_path));

    const auto model = Model(onnx_path, false);
}

TEST(Model, ConstructionFP16) {
    using libinfer::Model;

    const auto onnx_path = std::filesystem::path(__FILE__).parent_path() / "FastSCNN.onnx";
    ASSERT_TRUE(std::filesystem::exists(onnx_path));

    const auto model = Model(onnx_path, true);
}

TEST(Model, ForwardPassFP32) {
    using libinfer::Device;
    using libinfer::Model;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;
    using libinfer::Type;

    const auto onnx_path = std::filesystem::path(__FILE__).parent_path() / "FastSCNN.onnx";
    ASSERT_TRUE(std::filesystem::exists(onnx_path));

    auto model = Model(onnx_path, false);

    const auto expt_output_shape = Shape{Ordering::NCHW, 1, 1, 32, 64};
    const auto expt_input_shape = Shape{Ordering::NCHW, 1, 3, 256, 512};

    std::vector<Tensor> images, outputs;
    images.emplace_back(expt_input_shape, Device::CUDA, Type::FLOAT32);

    for (auto i = 0; i < 20; ++i) {
        model.forward(images, outputs);

        ASSERT_EQ(outputs.size(), 1);
        EXPECT_EQ(outputs[0].device(), Device::CUDA);
        EXPECT_EQ(outputs[0].shape(), expt_output_shape);

        ASSERT_EQ(images.size(), 1);
        EXPECT_EQ(images[0].device(), Device::CUDA);
        EXPECT_EQ(images[0].shape(), expt_input_shape);
    }
}

TEST(Model, ForwardPassFP16) {
    using libinfer::Device;
    using libinfer::Model;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;
    using libinfer::Type;

    const auto onnx_path = std::filesystem::path(__FILE__).parent_path() / "FastSCNN.onnx";
    ASSERT_TRUE(std::filesystem::exists(onnx_path));

    auto model = Model(onnx_path, true);

    const auto expt_output_shape = Shape{Ordering::NCHW, 1, 1, 32, 64};
    const auto expt_input_shape = Shape{Ordering::NCHW, 1, 3, 256, 512};

    std::vector<Tensor> images, outputs;
    images.emplace_back(expt_input_shape, Device::CUDA, Type::FLOAT32);

    for (auto i = 0; i < 20; ++i) {
        model.forward(images, outputs);

        ASSERT_EQ(outputs.size(), 1);
        EXPECT_EQ(outputs[0].device(), Device::CUDA);
        EXPECT_EQ(outputs[0].shape(), expt_output_shape);

        ASSERT_EQ(images.size(), 1);
        EXPECT_EQ(images[0].device(), Device::CUDA);
        EXPECT_EQ(images[0].shape(), expt_input_shape);
    }
}