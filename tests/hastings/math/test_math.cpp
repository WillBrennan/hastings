#include <gtest/gtest.h>

#include <vector>

#include "hastings/math/math.h"

TEST(Vec, is_trivial) {
    static_assert(std::is_trivial_v<hastings::Vec<double, 1>>);
    static_assert(std::is_trivial_v<hastings::Vec<double, 2>>);
    static_assert(std::is_trivial_v<hastings::Vec<double, 3>>);
    static_assert(std::is_trivial_v<hastings::Vec<double, 4>>);
    static_assert(std::is_trivial_v<hastings::Vec<double, 5>>);
}

TEST(Vec2, construction) {
    const auto a = hastings::Vec2d({0.1, 0.2});
    EXPECT_DOUBLE_EQ(a.x, 0.1);
    EXPECT_DOUBLE_EQ(a.y, 0.2);
    EXPECT_DOUBLE_EQ(a[0], 0.1);
    EXPECT_DOUBLE_EQ(a[1], 0.2);
}

TEST(Vec3, construction) {
    using hastings::Vec3d;

    const auto a = Vec3d({0.1, 0.2, 0.3});
    EXPECT_DOUBLE_EQ(a.x, 0.1);
    EXPECT_DOUBLE_EQ(a.y, 0.2);
    EXPECT_DOUBLE_EQ(a.z, 0.3);
    EXPECT_DOUBLE_EQ(a[0], 0.1);
    EXPECT_DOUBLE_EQ(a[1], 0.2);
    EXPECT_DOUBLE_EQ(a[2], 0.3);
}

TEST(Mat, size) {
    using Mat = hastings::Mat<double, 5, 3>;
    EXPECT_EQ(Mat::size(), 15);
}

TEST(Mat, ConstIterator) {
    const auto a = hastings::Mat2<int>({1, 2, 3, 4});
    const auto b = std::vector<int>(a.begin(), a.end());

    EXPECT_EQ(b, std::vector<int>({1, 2, 3, 4}));
}

TEST(Mat, Iterator) {
    auto a = hastings::Mat2<int>({1, 2, 3, 4});
    const auto b = std::vector<int>(a.begin(), a.end());

    EXPECT_EQ(b, std::vector<int>({1, 2, 3, 4}));
}

TEST(Mat, addition) {
    const auto a = hastings::Vec2d({1.1, 2.2});
    const auto b = hastings::Vec2d({1.3, 0.2});

    const auto c = a + b;

    EXPECT_DOUBLE_EQ(c.x, 2.4);
    EXPECT_DOUBLE_EQ(c.y, 2.4);
}

TEST(Mat, subtraction) {
    const auto a = hastings::Vec2d({1.1, 2.2});
    const auto b = hastings::Vec2d({1.3, 0.2});

    const auto c = a - b;

    EXPECT_DOUBLE_EQ(c.x, -0.2);
    EXPECT_DOUBLE_EQ(c.y, 2.0);
}

TEST(Mat, multiply) {
    const auto a = hastings::Vec2d({1.1, 2.2});
    const auto b = hastings::Vec2d({1.3, 0.2});

    const auto c = a * b;

    EXPECT_DOUBLE_EQ(c.x, 1.43);
    EXPECT_DOUBLE_EQ(c.y, 0.44);
}

TEST(Mat, divide) {
    const auto a = hastings::Vec2d({1.1, 2.2});
    const auto b = hastings::Vec2d({1.3, 0.2});

    const auto c = a / b;

    EXPECT_DOUBLE_EQ(c.x, 11.0 / 13.0);
    EXPECT_DOUBLE_EQ(c.y, 11);
}

TEST(Mat, scalarAddition) {
    const auto a = hastings::Vec2d({0.1, 0.2});

    const auto b = a + 1.2;
    const auto c = 1.2 + a;

    EXPECT_DOUBLE_EQ(b.x, c.x);
    EXPECT_DOUBLE_EQ(b.x, 1.3);
    EXPECT_DOUBLE_EQ(b.y, c.y);
    EXPECT_DOUBLE_EQ(c.y, 1.4);
}

TEST(Mat, scalarMultiply) {
    const auto a = hastings::Vec2d({0.1, 0.2});

    const auto b = a * 1.2;
    const auto c = 1.2 * a;

    EXPECT_DOUBLE_EQ(b.x, c.x);
    EXPECT_DOUBLE_EQ(b.x, 0.12);
    EXPECT_DOUBLE_EQ(b.y, c.y);
    EXPECT_DOUBLE_EQ(c.y, 0.24);
}

TEST(Mat, scalarSubtractLhs) {
    const auto a = hastings::Vec2d({0.1, 0.2});

    const auto b = 1.2 - a;
    EXPECT_DOUBLE_EQ(b.x, 1.1);
    EXPECT_DOUBLE_EQ(b.y, 1.0);
}

TEST(Mat, scalarSubtractRhs) {
    const auto a = hastings::Vec2d({0.1, 0.2});

    const auto b = a - 1.2;
    EXPECT_DOUBLE_EQ(b.x, -1.1);
    EXPECT_DOUBLE_EQ(b.y, -1.0);
}

TEST(Mat, scalarDivideLhs) {
    const auto a = hastings::Vec2d({0.1, 0.2});

    const auto b = 1.2 / a;
    EXPECT_DOUBLE_EQ(b.x, 12.0);
    EXPECT_DOUBLE_EQ(b.y, 6.0);
}

TEST(Mat, scalarDivideRhs) {
    const auto a = hastings::Vec2d({0.1, 0.2});

    const auto b = a / 1.2;
    EXPECT_DOUBLE_EQ(b.x, 1.0 / 12.0);
    EXPECT_DOUBLE_EQ(b.y, 1.0 / 6.0);
}

TEST(Mat, dot) {
    const auto a = hastings::Vec2d({1.1, 2.2});
    const auto b = hastings::Vec2d({1.3, 0.2});

    const auto c = dot(a, b);
    const auto d = dot(b, a);

    EXPECT_DOUBLE_EQ(c, 1.87);
    EXPECT_DOUBLE_EQ(d, c);
}

TEST(Vec, squaredLength) {
    const auto a = hastings::Vec2d({3.0, 4.0});
    const auto b = hastings::squaredLength(a);

    EXPECT_NEAR(b, 25.0, 1e-4);
}

TEST(Mat, isColumnMajor) {
    const auto a = hastings::Mat<int, 3, 4>({8, 9, 3, 2, 1, 5, 2, 4, 4, 9, 4, 5});
    EXPECT_EQ(a(0, 0), 8);
    EXPECT_EQ(a(0, 1), 2);
    EXPECT_EQ(a(0, 2), 2);
    EXPECT_EQ(a(0, 3), 9);
    EXPECT_EQ(a(1, 0), 9);
    EXPECT_EQ(a(1, 1), 1);
    EXPECT_EQ(a(1, 2), 4);
    EXPECT_EQ(a(1, 3), 4);
    EXPECT_EQ(a(2, 0), 3);
    EXPECT_EQ(a(2, 1), 5);
    EXPECT_EQ(a(2, 2), 4);
    EXPECT_EQ(a(2, 3), 5);
}

TEST(Mat, cross) {
    const auto a = hastings::Vec3d({3.0, -3.0, 1.0});
    const auto b = hastings::Vec3d({4.0, 9.0, 2.0});
    const auto c = hastings::cross(a, b);

    EXPECT_NEAR(c.x, -15.0, 1e-6);
    EXPECT_NEAR(c.y, -2.0, 1e-6);
    EXPECT_NEAR(c.z, 39.0, 1e-6);
}

TEST(Mat, crossParallel) {
    const auto a = hastings::Vec3d({1.0, 2.0, 3.0});
    const auto b = hastings::Vec3d({-2.0, -4.0, -6.0});

    const auto c = hastings::cross(a, b);
    EXPECT_NEAR(c.x, 0.0, 1e-6);
    EXPECT_NEAR(c.y, 0.0, 1e-6);
    EXPECT_NEAR(c.z, 0.0, 1e-6);
}

TEST(Mat, matmul) {
    const auto a = hastings::Mat<double, 2, 2>({1.0, 2.0, 3.0, 4.0});
    const auto b = hastings::Mat<double, 2, 2>({2.0, 4.0, 6.0, 8.0});

    const auto c = hastings::matmul(a, b);

    EXPECT_DOUBLE_EQ(c[0], 14.0);
    EXPECT_DOUBLE_EQ(c[1], 20.0);
    EXPECT_DOUBLE_EQ(c[2], 30.0);
    EXPECT_DOUBLE_EQ(c[3], 44.0);
}

TEST(Mat, matmul_vec) {
    const auto a = hastings::Mat<int, 2, 2>({1, 2, 3, 4});
    const auto b = hastings::Vec2i({-1, 2});

    // [1, 3]   [-1]   [5]
    // [2, 4] * [2]  = [6]

    const auto c = hastings::matmul(a, b);
    ASSERT_EQ(c.cols(), 1);
    ASSERT_EQ(c.rows(), 2);

    EXPECT_EQ(c(0, 0), 5);
    EXPECT_EQ(c(1, 0), 6);
}

TEST(Mat, abs) {
    const auto a = hastings::Mat<int, 2, 2>({1, -2, -3, 4});
    const auto b = hastings::abs(a);

    EXPECT_EQ(b[0], 1);
    EXPECT_EQ(b[1], 2);
    EXPECT_EQ(b[2], 3);
    EXPECT_EQ(b[3], 4);
}

TEST(Mat, transpose) {
    const auto a = hastings::Mat<int, 2, 2>({1, -2, -3, 4});
    const auto b = hastings::transpose(a);

    EXPECT_EQ(b[0], 1);
    EXPECT_EQ(b[1], -3);
    EXPECT_EQ(b[2], -2);
    EXPECT_EQ(b[3], 4);
}