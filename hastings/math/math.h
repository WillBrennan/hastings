#pragma once

#include <cmath>
#include <cstdio>
#include <ostream>

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace hastings {
namespace detail {
template <typename Scalar, std::size_t Rows, std::size_t Cols>
struct Storage {
    CUDA_HOST_DEVICE Scalar* rawData() { return values; }
    CUDA_HOST_DEVICE const Scalar* rawData() const { return values; }

    Scalar values[Rows * Cols];
};

template <typename Scalar>
struct Storage<Scalar, 2, 1> {
    Scalar x, y;

    CUDA_HOST_DEVICE Scalar* rawData() { return &x; }
    CUDA_HOST_DEVICE const Scalar* rawData() const { return &x; }
};

template <typename Scalar>
struct Storage<Scalar, 3, 1> {
    Scalar x, y, z;

    CUDA_HOST_DEVICE Scalar* rawData() { return &x; }
    CUDA_HOST_DEVICE const Scalar* rawData() const { return &x; }
};

template <typename Scalar>
struct Storage<Scalar, 4, 1> {
    Scalar x, y, z, w;

    CUDA_HOST_DEVICE Scalar* rawData() { return &x; }
    CUDA_HOST_DEVICE const Scalar* rawData() const { return &x; }
};
}  // namespace detail

template <typename ScalarT, std::size_t Rows, std::size_t Cols>
struct Mat : detail::Storage<ScalarT, Rows, Cols> {
    using Scalar = ScalarT;
    using Index = std::size_t;

    CUDA_HOST_DEVICE Scalar& operator[](const Index idx) { return data()[idx]; }
    CUDA_HOST_DEVICE const Scalar& operator[](const Index idx) const { return data()[idx]; }

    CUDA_HOST_DEVICE Scalar& operator()(const Index row, const Index col) { return operator[](flatten(row, col)); }
    CUDA_HOST_DEVICE const Scalar& operator()(const Index row, const Index col) const { return operator[](flatten(row, col)); }

    CUDA_HOST_DEVICE static constexpr Index rows() { return Rows; }
    CUDA_HOST_DEVICE static constexpr Index cols() { return Cols; }
    CUDA_HOST_DEVICE static constexpr Index size() { return rows() * cols(); }

    CUDA_HOST_DEVICE Scalar* data() { return this->rawData(); }
    CUDA_HOST_DEVICE const Scalar* data() const { return this->rawData(); }

    CUDA_HOST_DEVICE Scalar* begin() { return data(); }
    CUDA_HOST_DEVICE Scalar* end() { return begin() + size(); }

    CUDA_HOST_DEVICE const Scalar* begin() const { return data(); }
    CUDA_HOST_DEVICE const Scalar* end() const { return begin() + size(); }

  private:
    // NOTE(will.brennan): we store column major!
    CUDA_HOST_DEVICE static constexpr Index flatten(const Index row, const Index col) { return col * Rows + row; }
};

template <typename Scalar, std::size_t Dim>
using Vec = Mat<Scalar, Dim, 1>;

template <typename Scalar>
using Vec2 = Vec<Scalar, 2>;

template <typename Scalar>
using Vec3 = Vec<Scalar, 3>;

template <typename Scalar>
using Mat2 = Mat<Scalar, 2, 2>;

template <typename Scalar>
using Mat3 = Mat<Scalar, 3, 3>;

using Vec2d = Vec2<double>;
using Vec2f = Vec2<float>;
using Vec2i = Vec2<int>;
using Vec2u = Vec2<unsigned char>;
using Vec3d = Vec3<double>;
using Vec3f = Vec3<float>;
using Vec3i = Vec3<int>;
using Vec3u = Vec3<unsigned char>;
using Mat3d = Mat3<double>;

namespace detail {
template <typename Scalar, std::size_t Rows, std::size_t Cols, class Fn>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> apply(const Mat<Scalar, Rows, Cols>& lhs, const Mat<Scalar, Rows, Cols>& rhs, Fn&& fn) {
    auto result = Mat<Scalar, Rows, Cols>();

    for (auto idx = 0; idx < lhs.size(); ++idx) {
        result[idx] = fn(lhs[idx], rhs[idx]);
    }

    return result;
}

template <typename Scalar, std::size_t Rows, std::size_t Cols, class Fn>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> apply(const Mat<Scalar, Rows, Cols>& lhs, const Scalar& rhs, Fn&& fn) {
    auto result = Mat<Scalar, Rows, Cols>();

    for (auto idx = 0; idx < lhs.size(); ++idx) {
        result[idx] = fn(lhs[idx], rhs);
    }

    return result;
}

template <typename Scalar, std::size_t Rows, std::size_t Cols, class Fn>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> apply(const Scalar& lhs, const Mat<Scalar, Rows, Cols>& rhs, Fn&& fn) {
    auto result = Mat<Scalar, Rows, Cols>();

    for (auto idx = 0; idx < rhs.size(); ++idx) {
        result[idx] = fn(lhs, rhs[idx]);
    }

    return result;
}
}  // namespace detail

// binary operators - vector

template <typename Scalar, std::size_t Rows, std::size_t Cols>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> operator+(const Mat<Scalar, Rows, Cols>& lhs, const Mat<Scalar, Rows, Cols>& rhs) {
    return detail::apply(lhs, rhs, [](const auto& a, const auto& b) { return a + b; });
}

template <typename Scalar, std::size_t Rows, std::size_t Cols>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> operator-(const Mat<Scalar, Rows, Cols>& lhs, const Mat<Scalar, Rows, Cols>& rhs) {
    return detail::apply(lhs, rhs, [](const auto& a, const auto& b) { return a - b; });
}

template <typename Scalar, std::size_t Rows, std::size_t Cols>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> operator*(const Mat<Scalar, Rows, Cols>& lhs, const Mat<Scalar, Rows, Cols>& rhs) {
    return detail::apply(lhs, rhs, [](const auto& a, const auto& b) { return a * b; });
}

template <typename Scalar, std::size_t Rows, std::size_t Cols>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> operator/(const Mat<Scalar, Rows, Cols>& lhs, const Mat<Scalar, Rows, Cols>& rhs) {
    return detail::apply(lhs, rhs, [](const auto& a, const auto& b) { return a / b; });
}

// binary operators - scalar

template <typename Scalar, std::size_t Rows, std::size_t Cols>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> operator+(const Mat<Scalar, Rows, Cols>& lhs, const Scalar& rhs) {
    return detail::apply(lhs, rhs, [](const auto& a, const auto& b) { return a + b; });
}

template <typename Scalar, std::size_t Rows, std::size_t Cols>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> operator+(const Scalar& lhs, const Mat<Scalar, Rows, Cols>& rhs) {
    return rhs + lhs;
}

template <typename Scalar, std::size_t Rows, std::size_t Cols>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> operator*(const Mat<Scalar, Rows, Cols>& lhs, const Scalar& rhs) {
    return detail::apply(lhs, rhs, [](const auto& a, const auto& b) { return a * b; });
}

template <typename Scalar, std::size_t Rows, std::size_t Cols>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> operator*(const Scalar& lhs, const Mat<Scalar, Rows, Cols>& rhs) {
    return rhs * lhs;
}

template <typename Scalar, std::size_t Rows, std::size_t Cols>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> operator-(const Mat<Scalar, Rows, Cols>& lhs, const Scalar& rhs) {
    return detail::apply(lhs, rhs, [](const auto& a, const auto& b) { return a - b; });
}

template <typename Scalar, std::size_t Rows, std::size_t Cols>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> operator-(const Scalar& lhs, const Mat<Scalar, Rows, Cols>& rhs) {
    return detail::apply(lhs, rhs, [](const auto& a, const auto& b) { return a - b; });
}

template <typename Scalar, std::size_t Rows, std::size_t Cols>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> operator/(const Mat<Scalar, Rows, Cols>& lhs, const Scalar& rhs) {
    return detail::apply(lhs, rhs, [](const auto& a, const auto& b) { return a / b; });
}

template <typename Scalar, std::size_t Rows, std::size_t Cols>
CUDA_HOST_DEVICE Mat<Scalar, Rows, Cols> operator/(const Scalar& lhs, const Mat<Scalar, Rows, Cols>& rhs) {
    return detail::apply(lhs, rhs, [](const auto& a, const auto& b) { return a / b; });
}

// special operators

template <typename Scalar, std::size_t Dim>
CUDA_HOST_DEVICE Scalar squaredLength(const Vec<Scalar, Dim>& vec) {
    auto result = Scalar(0.0);

    for (auto idx = 0; idx < Dim; ++idx) {
        result += vec[idx] * vec[idx];
    }

    return result;
}

template <typename Scalar, std::size_t Dim>
CUDA_HOST_DEVICE Scalar dot(const Vec<Scalar, Dim>& lhs, const Vec<Scalar, Dim>& rhs) {
    auto result = Scalar(0.0);

    for (auto idx = 0; idx < Dim; ++idx) {
        result += lhs[idx] * rhs[idx];
    }

    return result;
}

template <typename Scalar>
CUDA_HOST_DEVICE Vec3<Scalar> cross(const Vec3<Scalar>& lhs, const Vec3<Scalar>& rhs) {
    auto result = Vec3<Scalar>();
    result[0] = lhs[1] * rhs[2] - lhs[2] * rhs[1];
    result[1] = lhs[2] * rhs[0] - lhs[0] * rhs[2];
    result[2] = lhs[0] * rhs[1] - lhs[1] * rhs[0];
    return result;
}

template <typename Scalar, std::size_t DimI, std::size_t DimJ, std::size_t DimK>
CUDA_HOST_DEVICE Mat<Scalar, DimI, DimK> matmul(const Mat<Scalar, DimI, DimJ>& lhs, const Mat<Scalar, DimJ, DimK>& rhs) {
    auto result = Mat<Scalar, DimI, DimK>({});

    for (auto i = 0; i < DimI; ++i) {
        for (auto j = 0; j < DimJ; ++j) {
            for (auto k = 0; k < DimK; ++k) {
                result(i, k) += lhs(i, j) * rhs(j, k);
            }
        }
    }

    return result;
}

template <typename Scalar, std::size_t DimRows, std::size_t DimCols>
CUDA_HOST_DEVICE Mat<Scalar, DimRows, DimCols> abs(const Mat<Scalar, DimRows, DimCols>& mat) {
    auto result = Mat<Scalar, DimRows, DimCols>();

    for (int i = 0; i < DimRows * DimCols; ++i) {
        result[i] = std::abs(mat[i]);
    }

    return result;
}

template <typename Scalar, std::size_t DimRows, std::size_t DimCols>
CUDA_HOST_DEVICE Mat<Scalar, DimRows, DimCols> transpose(const Mat<Scalar, DimRows, DimCols>& mat) {
    auto result = Mat<Scalar, DimRows, DimCols>();
    for (auto row = 0; row < DimRows; ++row) {
        for (auto col = 0; col < DimCols; ++col) {
            result(col, row) = mat(row, col);
        }
    }

    return result;
}

template <typename Scalar, std::size_t DimRows, std::size_t DimCols>
CUDA_HOST_DEVICE bool near(const Mat<Scalar, DimRows, DimCols>& lhs, const Mat<Scalar, DimRows, DimCols>& rhs, const Scalar& epsilon) {
    const auto delta = abs(lhs - rhs);
    Scalar max = delta[0];

    for (int i = 0; i < DimRows * DimCols; ++i) {
        max = std::max(max, delta[i]);
    }

    return max < epsilon;
}

template <typename Scalar, std::size_t DimRows, std::size_t DimCols>
std::ostream& operator<<(std::ostream& stream, const Mat<Scalar, DimRows, DimCols>& mat) {
    stream << "[";
    for (auto idx_row = 0; idx_row < DimRows; ++idx_row) {
        if (DimCols != 1) {
            stream << "[";
        }

        for (auto idx_col = 0; idx_col < DimCols; ++idx_col) {
            stream << mat(idx_row, idx_col) << " ";
        }
        if (DimCols != 1) {
            stream << "]";
        }
    }
    stream << "]";
    return stream;
}
}  // namespace hastings