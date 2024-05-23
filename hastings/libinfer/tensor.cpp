#include "hastings/libinfer/tensor.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <algorithm>
#include <numeric>
#include <ostream>
#include <stdexcept>

namespace libinfer {

Tensor::Tensor() : device_(Device::Empty){};

Tensor::Tensor(Shape shape, Device device, Type type) : shape_(shape), device_(device), type_(type) {
    // perform an allocation here... either CPU or GPU
    if (device_ == Device::Empty || shape_.numElems() == 0) {
        // do nothing
    } else if (device_ == Device::CUDA) {
        // todo(will.brennan): add cudaStreams & go async
        cudaMalloc(&data_, numBytes());
    } else if (device_ == Device::CPU && type_ == Type::FLOAT32) {
        data_ = new float[shape_.numElems()];
    } else if (device_ == Device::CPU && type_ == Type::UINT8) {
        data_ = new unsigned char[shape_.numElems()];
    } else {
        throw std::invalid_argument("unsupported device in tensor allocation");
    }
}

Tensor::Tensor(Shape shape, Device device, float* data)
    : shape_(shape), device_(device), is_view_(true), data_(data), type_(Type::FLOAT32) {}

Tensor::Tensor(Shape shape, Device device, unsigned char* data)
    : shape_(shape), device_(device), is_view_(true), data_(data), type_(Type::UINT8) {}

Tensor::Tensor(Tensor&& other) {
    std::swap(data_, other.data_);
    std::swap(shape_, other.shape_);
    std::swap(device_, other.device_);
    std::swap(is_view_, other.is_view_);
    std::swap(type_, other.type_);

    other.reset();
}

Tensor& Tensor::operator=(Tensor&& other) {
    std::swap(data_, other.data_);
    std::swap(shape_, other.shape_);
    std::swap(device_, other.device_);
    std::swap(is_view_, other.is_view_);
    std::swap(type_, other.type_);

    other.reset();
    return *this;
}

void Tensor::reset() {
    if (data_ != nullptr && !is_view_) {
        if (device_ == Device::CUDA) {
            // todo(will.brennan): add cudaStreams & go async
            cudaFree(data_);
        } else if (device_ == Device::CPU) {
            delete data_;
        }
    }

    data_ = nullptr;
    device_ = Device::Empty;
    shape_ = Shape{Ordering::NCHW, 0, 0, 0, 0};
    is_view_ = false;
}

Tensor::~Tensor() { reset(); }

void copy(const Tensor& from, Tensor& to) {
    if (from.numBytes() != to.numBytes() || to.type() != to.type()) {
        // note(will.brennan): being sneaky here to let you mangle shapes
        throw std::invalid_argument("tensors must have the same number of bytes && types");
    }

    // todo(will.brennan): add cudaStreams & go async
    cudaMemcpy(to.data(), from.data(), from.numBytes(), cudaMemcpyDefault);

    cudaDeviceSynchronize();
}

std::ostream& operator<<(std::ostream& stream, const Shape& shape) {
    stream << "{ n: " << shape.n;
    stream << ", c: " << shape.c;
    stream << ", h: " << shape.h;
    stream << ", w: " << shape.w;
    stream << ", order: " << (shape.order == Ordering::NCHW ? "NCHW" : "NHWC");
    stream << " }";
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
    stream << "{ shape: " << tensor.shape();
    stream << ", device: " << (tensor.device() == Device::CPU ? "CPU" : "CUDA");
    stream << ", is view: " << tensor.isView();
    stream << " }";
    return stream;
}

void debug(const std::string& name, const Tensor& x) {
    auto cpu = Tensor(x.shape(), Device::CPU, x.type());
    copy(x, cpu);

    const auto num_elems = cpu.shape().numElems();

    if (cpu.type() == Type::FLOAT32) {
        auto* iter = cpu.data<float>();

        const auto [min_iter, max_iter] = std::minmax_element(iter, iter + num_elems);
        const auto min = *min_iter;
        const auto max = *max_iter;

        const auto nth_iter = iter + (num_elems / 2);
        std::nth_element(iter, nth_iter, iter + num_elems);
        const auto median = *nth_iter;

        LOG(INFO) << "debug - " << name << " - " << x;
        LOG(INFO) << "min: " << min << " median: " << median << " max: " << max;
    } else if (cpu.type() == Type::UINT8) {
        auto* iter = cpu.data<unsigned char>();

        const auto [min_iter, max_iter] = std::minmax_element(iter, iter + num_elems);
        const auto min = *min_iter;
        const auto max = *max_iter;

        const auto nth_iter = iter + (num_elems / 2);
        std::nth_element(iter, nth_iter, iter + num_elems);
        const auto median = *nth_iter;

        LOG(INFO) << "debug - " << name << " - " << x;
        LOG(INFO) << "min: " << int(min) << " median: " << int(median) << " max: " << int(max);
    }
}

}  // namespace libinfer