#include <glog/logging.h>
#include <gtest/gtest.h>
#include <hastings/libinfer/tensor.h>

TEST(Shape, Construction) {
    using libinfer::Ordering;
    using libinfer::Shape;

    const Shape shape = {Ordering::NCHW, 10, 4, 3, 2};
}

TEST(Shape, NumElem) {
    using libinfer::Ordering;
    using libinfer::Shape;

    const Shape shape = {Ordering::NCHW, 10, 4, 3, 2};

    EXPECT_EQ(shape.numElems(), 240);
};

TEST(Shape, Equality) {
    using libinfer::Ordering;
    using libinfer::Shape;

    const Shape shape_a = {Ordering::NCHW, 10, 4, 3, 2};
    const Shape shape_b = {Ordering::NHWC, 10, 4, 3, 2};
    const Shape shape_c = {Ordering::NCHW, 5, 4, 3, 2};

    EXPECT_EQ(shape_a, shape_a);

    EXPECT_NE(shape_a, shape_b);
    EXPECT_NE(shape_b, shape_a);
    EXPECT_NE(shape_a, shape_c);
    EXPECT_NE(shape_c, shape_a);
};

TEST(Tensor, ConstructEmpty) {
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;

    const auto shape = Shape{Ordering::NCHW, 0, 0, 0, 0};

    const auto tensor = Tensor();

    ASSERT_EQ(tensor.data(), nullptr);
    ASSERT_EQ(tensor.isView(), false);
    ASSERT_EQ(tensor.device(), Device::Empty);
    ASSERT_EQ(tensor.numBytes(), 0);
    ASSERT_EQ(tensor.shape(), shape);
}

TEST(Tensor, ConstructCPU_Float32) {
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;
    using libinfer::Type;

    const auto shape = Shape{Ordering::NCHW, 2, 10, 20, 30};

    auto tensor = Tensor(shape, Device::CPU, Type::FLOAT32);

    EXPECT_NE(tensor.data(), nullptr);
    EXPECT_EQ(tensor.isView(), false);
    EXPECT_EQ(tensor.device(), Device::CPU);
    EXPECT_EQ(tensor.numBytes(), 48000);
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.type(), Type::FLOAT32);
}

TEST(Tensor, ConstructCPU_UINT8) {
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;
    using libinfer::Type;

    const auto shape = Shape{Ordering::NCHW, 2, 10, 20, 30};

    auto tensor = Tensor(shape, Device::CPU, Type::UINT8);

    EXPECT_NE(tensor.data(), nullptr);
    EXPECT_EQ(tensor.isView(), false);
    EXPECT_EQ(tensor.device(), Device::CPU);
    EXPECT_EQ(tensor.numBytes(), 12000);
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.type(), Type::UINT8);
}

TEST(Tensor, ConstructCUDA_Float32) {
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;
    using libinfer::Type;

    const auto shape = Shape{Ordering::NCHW, 2, 10, 20, 30};

    auto tensor = Tensor(shape, Device::CUDA, Type::FLOAT32);

    EXPECT_NE(tensor.data(), nullptr);
    EXPECT_EQ(tensor.isView(), false);
    EXPECT_EQ(tensor.device(), Device::CUDA);
    EXPECT_EQ(tensor.numBytes(), 48000);
    EXPECT_EQ(tensor.shape(), shape);
}

TEST(Tensor, ConstructCUDA_UINT8) {
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;
    using libinfer::Type;

    const auto shape = Shape{Ordering::NCHW, 2, 10, 20, 30};

    auto tensor = Tensor(shape, Device::CUDA, Type::UINT8);

    EXPECT_NE(tensor.data(), nullptr);
    EXPECT_EQ(tensor.isView(), false);
    EXPECT_EQ(tensor.device(), Device::CUDA);
    EXPECT_EQ(tensor.numBytes(), 12000);
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.type(), Type::UINT8);
}

TEST(Tensor, ConstructCPUZeroShape) {
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;
    using libinfer::Type;

    const auto shape = Shape{Ordering::NCHW, 0, 10, 20, 30};

    auto tensor = Tensor(shape, Device::CPU, Type::FLOAT32);

    EXPECT_EQ(tensor.data(), nullptr);
    EXPECT_EQ(tensor.isView(), false);
    EXPECT_EQ(tensor.device(), Device::CPU);
    EXPECT_EQ(tensor.numBytes(), 0);
    EXPECT_EQ(tensor.shape(), shape);
}

TEST(Tensor, ConstructCUDAZeroShape) {
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;
    using libinfer::Type;

    const auto shape = Shape{Ordering::NCHW, 0, 10, 20, 30};

    auto tensor = Tensor(shape, Device::CUDA, Type::FLOAT32);

    EXPECT_EQ(tensor.data(), nullptr);
    EXPECT_EQ(tensor.isView(), false);
    EXPECT_EQ(tensor.device(), Device::CUDA);
    EXPECT_EQ(tensor.numBytes(), 0);
    EXPECT_EQ(tensor.shape(), shape);
}

TEST(Tensor, ConstructViewFloat32) {
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;
    using libinfer::Type;

    const auto shape = Shape{Ordering::NCHW, 1, 1, 1, 3};
    float data[3] = {1.0f, 2.0f, 3.0f};

    auto tensor = Tensor(shape, Device::CUDA, data);

    EXPECT_NE(tensor.data(), nullptr);
    EXPECT_EQ(tensor.isView(), true);
    EXPECT_EQ(tensor.device(), Device::CUDA);
    EXPECT_EQ(tensor.numBytes(), 12);
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.type(), Type::FLOAT32);
}

TEST(Tensor, ConstructViewUINT8) {
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;
    using libinfer::Type;

    const auto shape = Shape{Ordering::NCHW, 1, 1, 1, 3};
    unsigned char data[3] = {12, 0, 244};

    auto tensor = Tensor(shape, Device::CUDA, data);

    EXPECT_NE(tensor.data(), nullptr);
    EXPECT_EQ(tensor.isView(), true);
    EXPECT_EQ(tensor.device(), Device::CUDA);
    EXPECT_EQ(tensor.numBytes(), 3);
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.type(), Type::UINT8);
}

TEST(Tensor, MoveConstruct_Float32) {
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;
    using libinfer::Type;

    const auto shape = Shape{Ordering::NCHW, 2, 10, 20, 30};
    auto tensor_a = Tensor(shape, Device::CPU, Type::FLOAT32);

    auto tensor_b = std::move(tensor_a);

    EXPECT_EQ(tensor_a.data(), nullptr);
    EXPECT_NE(tensor_b.data(), nullptr);

    EXPECT_EQ(tensor_a.device(), Device::Empty);
    EXPECT_EQ(tensor_b.device(), Device::CPU);
    EXPECT_EQ(tensor_a.numBytes(), 0);
    EXPECT_EQ(tensor_b.numBytes(), 48000);
    EXPECT_EQ(tensor_b.type(), Type::FLOAT32);
}

TEST(Tensor, MoveConstruct_UINT8) {
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;
    using libinfer::Type;

    const auto shape = Shape{Ordering::NCHW, 2, 10, 20, 30};
    auto tensor_a = Tensor(shape, Device::CPU, Type::UINT8);

    auto tensor_b = std::move(tensor_a);

    EXPECT_EQ(tensor_a.data(), nullptr);
    EXPECT_NE(tensor_b.data(), nullptr);

    EXPECT_EQ(tensor_a.device(), Device::Empty);
    EXPECT_EQ(tensor_b.device(), Device::CPU);
    EXPECT_EQ(tensor_a.numBytes(), 0);
    EXPECT_EQ(tensor_b.numBytes(), 12000);
    EXPECT_EQ(tensor_b.type(), Type::UINT8);
}

TEST(Tensor, MoveAssignment_FLOAT32) {
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;
    using libinfer::Type;

    const auto shape = Shape{Ordering::NCHW, 2, 10, 20, 30};

    auto tensor = Tensor();
    EXPECT_EQ(tensor.data(), nullptr);
    EXPECT_EQ(tensor.device(), Device::Empty);
    EXPECT_EQ(tensor.numBytes(), 0);

    tensor = Tensor(shape, Device::CPU, Type::FLOAT32);
    EXPECT_NE(tensor.data(), nullptr);
    EXPECT_EQ(tensor.device(), Device::CPU);
    EXPECT_EQ(tensor.numBytes(), 48000);
    EXPECT_EQ(tensor.shape(), shape);
}

TEST(Tensor, MoveAssignment_UINT8) {
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Shape;
    using libinfer::Tensor;
    using libinfer::Type;

    const auto shape = Shape{Ordering::NCHW, 2, 10, 20, 30};

    auto tensor = Tensor();
    EXPECT_EQ(tensor.data(), nullptr);
    EXPECT_EQ(tensor.device(), Device::Empty);
    EXPECT_EQ(tensor.numBytes(), 0);

    tensor = Tensor(shape, Device::CPU, Type::UINT8);
    EXPECT_NE(tensor.data(), nullptr);
    EXPECT_EQ(tensor.device(), Device::CPU);
    EXPECT_EQ(tensor.numBytes(), 12000);
    EXPECT_EQ(tensor.shape(), shape);
}

TEST(TensorCopy, GoodCopy) {
    using libinfer::copy;
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Tensor;
    using libinfer::Type;

    auto host_a = Tensor({Ordering::NCHW, 1, 1, 2, 2}, Device::CPU, Type::UINT8);
    auto device_a = Tensor({Ordering::NCHW, 1, 1, 2, 2}, Device::CUDA, Type::UINT8);

    host_a.data<unsigned char>()[0] = 10;
    host_a.data<unsigned char>()[1] = 2;
    host_a.data<unsigned char>()[2] = 4;
    host_a.data<unsigned char>()[3] = 8;

    copy(host_a, device_a);

    for (int i = 0; i < 4; ++i) {
        host_a.data<unsigned char>()[i] = 0;
    }

    copy(device_a, host_a);

    ASSERT_EQ(host_a.data<unsigned char>()[0], 10);
    ASSERT_EQ(host_a.data<unsigned char>()[1], 2);
    ASSERT_EQ(host_a.data<unsigned char>()[2], 4);
    ASSERT_EQ(host_a.data<unsigned char>()[3], 8);
}

TEST(TensorCopy, BadSize) {
    using libinfer::copy;
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Tensor;
    using libinfer::Type;

    auto host_a = Tensor({Ordering::NCHW, 1, 1, 2, 2}, Device::CPU, Type::UINT8);
    auto device_a = Tensor({Ordering::NCHW, 1, 1, 3, 2}, Device::CUDA, Type::UINT8);

    EXPECT_THROW({ copy(host_a, device_a); }, std::invalid_argument);
}

TEST(TensorCopy, BadType) {
    using libinfer::copy;
    using libinfer::Device;
    using libinfer::Ordering;
    using libinfer::Tensor;
    using libinfer::Type;

    auto host_a = Tensor({Ordering::NCHW, 1, 1, 2, 2}, Device::CPU, Type::UINT8);
    auto device_a = Tensor({Ordering::NCHW, 1, 1, 2, 2}, Device::CUDA, Type::FLOAT32);

    EXPECT_THROW({ copy(host_a, device_a); }, std::invalid_argument);
}
