#include <gtest/gtest.h>

#include "hastings/math/calib.h"

TEST(RadialDistortion, NoDistortion) {
    using hastings::RadialDistortion;
    using hastings::Vec2d;

    const Vec2d pixels[] = {Vec2d({0.3, 0.4}), Vec2d({-0.3, 0.1}), Vec2d({-0.1, -0.2}), Vec2d({0.3, -0.5})};

    const auto distortion = RadialDistortion<double>{0.0};

    for (const auto& pixel : pixels) {
        const auto distorted = distortion.distort(pixel);
        EXPECT_TRUE(near(pixel, distorted, 1e-4));
    }
}

TEST(RadialDistortion, Barrel) {
    using hastings::RadialDistortion;
    using hastings::Vec2d;

    const auto distortion = RadialDistortion<double>{-0.5};

    const auto distorted = Vec2d({0.5, 0.6});
    const auto undistorted = distortion.undistort(distorted);
    const auto redistorted = distortion.distort(undistorted);

    EXPECT_TRUE(near(distorted, redistorted, 1e-4));
}

TEST(RadialDistortion, PinCushion) {
    using hastings::RadialDistortion;
    using hastings::Vec2d;

    const auto distortion = RadialDistortion<double>{0.5};

    const auto distorted = Vec2d({0.5, 0.6});
    const auto undistorted = distortion.undistort(distorted);
    const auto redistorted = distortion.distort(undistorted);

    EXPECT_TRUE(near(distorted, redistorted, 1e-4));
}

TEST(Calib, Construction) {
    using Calib = hastings::Calib<double>;
    const auto calib = Calib({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1500, 1500}, 3000, 1.0, {0.0});
}

TEST(Calib, project) {
    using Calib = hastings::Calib<double>;
    using Vec2d = hastings::Vec2<double>;

    auto calib = Calib({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1500, 1500}, 3000, 1.0, {0.0});

    EXPECT_TRUE(near(calib.project({-0.5, -0.5, 1.0}), Vec2d({0.0, 0.0}), 1e-4));
    EXPECT_TRUE(near(calib.project({0.0, 0.0, 1.0}), Vec2d({1500, 1500}), 1e-4));
    EXPECT_TRUE(near(calib.project({0.5, 0.5, 1.0}), Vec2d({3000, 3000}), 1e-4));
}

TEST(Calib, direction) {
    using Calib = hastings::Calib<double>;
    using Vec3d = hastings::Vec3<double>;

    const auto calib = Calib({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1500, 1500}, 3000, 1.0, {0.0});

    EXPECT_TRUE(near(calib.direction({0.0, 0.0}), Vec3d({-0.5, -0.5, 1.0}), 1e-4));
    EXPECT_TRUE(near(calib.direction({1500, 1500}), Vec3d({0.0, 0.0, 1.0}), 1e-4));
    EXPECT_TRUE(near(calib.direction({3000, 3000}), Vec3d({0.5, 0.5, 1.0}), 1e-4));
}

TEST(Calib, pixelInZ) {
    using Calib = hastings::Calib<double>;
    using Vec2d = hastings::Vec2<double>;
    using Vec3d = hastings::Vec3<double>;
    using hastings::pixelInZ;

    const auto calib = Calib({0.0, 0.0, 3.0}, {4.71238898, 0.0, 0.0}, {1500, 1500}, 3000, 1.0, {0.0});

    const auto pixel = Vec2d({1500, 0.0});
    const auto point = pixelInZ(calib, pixel, 0.0);

    EXPECT_TRUE(near(point, Vec3d({0.0, -6.0, 0.0}), 1e-4));
}