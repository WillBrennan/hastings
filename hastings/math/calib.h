#pragma once

#include <ceres/rotation.h>

#include <cmath>

#include "hastings/math/math.h"

namespace hastings {
template <typename ScalarT>
struct RadialDistortion {
    using Scalar = ScalarT;
    using Vec2s = Vec2<Scalar>;

    Scalar k1 = Scalar(0.0);
    Scalar k2 = Scalar(0.0);
    Scalar k3 = Scalar(0.0);

    Vec2s distort(const Vec2s& pixel) const {
        const auto r2 = squaredLength(pixel);
        const auto r4 = r2 * r2;
        const auto r6 = r4 * r2;
        const auto scale = Scalar(1.0) + k1 * r2 + k2 * r4 + k3 * r6;

        return pixel * scale;
    }

    Vec2s undistort(const Vec2s& pixel) const { throw std::runtime_error("error not implemented!"); }
};

template <typename ScalarT>
class Calib {
  public:
    using Scalar = ScalarT;
    using Vec2s = Vec2<Scalar>;
    using Vec3s = Vec3<Scalar>;
    using Mat3s = Mat3<Scalar>;
    using Distortion = RadialDistortion<Scalar>;

    // NOTE(will.brennan): by default; cameras point up along the z-axis

    Calib(const Vec3s& trans, const Vec3s& _angle_axis, const Vec2s& center, Scalar focal_length, Scalar aspect_ratio,
          const Distortion& distortion)
        : trans_(trans),
          angle_axis_(_angle_axis),
          center_(center),
          focal_length_(focal_length),
          aspect_ratio_(aspect_ratio),
          distortion_(distortion) {}

    const Vec3s& trans() const { return trans_; }
    const Vec3s& angleAxis() const { return angle_axis_; }
    const Scalar& focalLength() const { return focal_length_; }
    const Scalar& aspectRatio() const { return aspect_ratio_; }
    const Vec2s& center() const { return center_; }
    const Distortion& distortion() const { return distortion_; }

    Vec3s& trans() { return trans_; }
    Vec3s& angleAxis() { return angle_axis_; }
    Scalar& focalLength() { return focal_length_; }
    Scalar& aspectRatio() { return aspect_ratio_; }
    Vec2s& center() { return center_; }
    Distortion& distortion() { return distortion_; }

    Vec2s project(Vec3s point) const {
        point = point - trans_;

        Vec3s point_in_cam;
        ceres::AngleAxisRotatePoint(angle_axis_.data(), point.data(), point_in_cam.data());

        const auto pixel_homogenous = Vec2s({point_in_cam.x, point_in_cam.y}) / point_in_cam.z;

        auto pixel = focal_length_ * distortion_.distort(pixel_homogenous);
        pixel.y *= aspect_ratio_;
        pixel = pixel + center_;

        return pixel;
    }

    Vec3s direction(const Vec2s pixel) const {
        auto pixel_homogenous = distortion_.undistort(pixel - center_) / focal_length_;
        pixel_homogenous.y /= aspect_ratio_;

        auto direction_cam_coords = Vec3s({pixel_homogenous.x, pixel_homogenous.y, Scalar(1.0)});

        const auto inverse_angle_axis = Scalar(-1.0) * angle_axis_;

        Vec3s point_in_world;
        ceres::AngleAxisRotatePoint(inverse_angle_axis.data(), direction_cam_coords.data(), point_in_world.data());
        return point_in_world;
    }

  private:
    // NOTE(will.brennan): we care about memory ordering due to ceres optimization
    Vec3s trans_;
    Vec3s angle_axis_;
    Vec2s center_;
    Scalar focal_length_;
    Scalar aspect_ratio_;

    Distortion distortion_;
};

using Calibd = Calib<double>;

template <typename Scalar>
Vec3<Scalar> pixelInZ(const Calib<Scalar>& calib, const Vec2<Scalar>& pixel, const Scalar& z) {
    const auto direction = calib.direction(pixel);
    const auto t = (z - calib.trans().z) / direction.z;
    return calib.trans() + direction * t;
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& stream, const Calib<Scalar>& calib) {
    stream << "{ trans: " << calib.trans();
    stream << ", angle-axis: " << calib.angleAxis();
    stream << ", center: " << calib.center();
    stream << ", focal length: " << calib.focalLength();
    stream << ", aspect ratio: " << calib.aspectRatio();
    stream << ", k1: " << calib.distortion().k1;
    stream << ", k2: " << calib.distortion().k2;
    stream << ", k3: " << calib.distortion().k3;
    stream << "}";
    return stream;
}
}  // namespace hastings