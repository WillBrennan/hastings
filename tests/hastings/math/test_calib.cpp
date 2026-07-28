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

TEST(Calib, directionInvertsProject) {
    // Every other test here uses an ideal lens, where undistort is the identity — so none of them
    // can tell whether direction() undoes what project() did. With a real k1 it did not.
    using Calib = hastings::Calib<double>;
    using Vec3d = hastings::Vec3<double>;

    const Vec3d points[] = {Vec3d({-0.4, -0.3, 1.0}), Vec3d({0.0, 0.0, 1.0}), Vec3d({0.45, 0.35, 1.0}),
                            Vec3d({0.2, -0.45, 1.0})};

    for (const auto k1 : {0.0, -0.27, 0.2}) {
        // Identity pose, so a camera-space ray with z = 1 is the point it passes through.
        const auto calib = Calib({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1500, 1500}, 3000, 1.0, {k1});

        for (const auto& point : points) {
            const auto ray = calib.direction(calib.project(point));
            EXPECT_TRUE(near(ray, point, 1e-4)) << "k1 = " << k1;
        }
    }
}

TEST(Calib, pixelInZDistorted) {
    using Calib = hastings::Calib<double>;
    using Vec3d = hastings::Vec3<double>;
    using hastings::pixelInZ;

    const auto calib = Calib({0.0, 0.0, 3.0}, {4.71238898, 0.0, 0.0}, {1500, 1500}, 3000, 1.0, {-0.27});

    // Project a point on the ground, then ask where that pixel meets the ground again.
    const auto point = Vec3d({1.5, -6.0, 0.0});
    const auto found = pixelInZ(calib, calib.project(point), 0.0);

    EXPECT_TRUE(near(found, point, 1e-3));
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