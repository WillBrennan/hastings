#pragma once

#include <cstdint>
#include <iosfwd>
#include <variant>

namespace libinfer {

enum class Device { CPU, CUDA, Empty };
enum class Ordering { NCHW, NHWC };
enum class Type { UINT8, FLOAT32 };

struct Shape {
    using Index = std::int64_t;

    Ordering order = Ordering::NCHW;
    Index n = 0;
    Index c = 0;
    Index h = 0;
    Index w = 0;

    Index numElems() const { return n * c * h * w; }

    bool operator==(const Shape& other) const {
        bool is_equal = order == other.order;
        is_equal &= n == other.n;
        is_equal &= c == other.c;
        is_equal &= h == other.h;
        is_equal &= w == other.w;
        return is_equal;
    }

    bool operator!=(const Shape& other) const { return !operator==(other); }
};

class Tensor {
  public:
    using Index = std::size_t;

    explicit Tensor();
    Tensor(Shape shape, Device device, Type type);
    ~Tensor();

    // note(will.brennan): not using cudaPointerGetAttributes for speed
    Tensor(Shape shape, Device device, float* data);
    Tensor(Shape shape, Device device, unsigned char* data);

    Tensor(Tensor&& other);
    Tensor& operator=(Tensor&& other);

    template <class T>
    T* data() {
        return reinterpret_cast<T*>(data_);
    }

    template <class T>
    const T* data() const {
        return reinterpret_cast<const T*>(data_);
    }

    void* data() { return data_; }
    const void* data() const { return data_; }

    const Shape& shape() const { return shape_; }
    Device device() const { return device_; }
    Type type() const { return type_; }
    bool isView() const { return is_view_; }

    Index numBytes() const {
        const auto bytesPerElem = type_ == Type::UINT8 ? sizeof(unsigned char) : sizeof(float);
        return shape_.numElems() * bytesPerElem;
    }

    void reset();

  private:
    void* data_ = nullptr;
    Shape shape_;
    Device device_;
    Type type_;
    bool is_view_ = false;
};

void copy(const Tensor& from, Tensor& to);

std::ostream& operator<<(std::ostream& stream, const Shape& shape);
std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);

void debug(const std::string& name, const Tensor& x);
}  // namespace libinfer
