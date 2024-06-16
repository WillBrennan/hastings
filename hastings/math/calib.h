#pragma once

#include <cmath>

#include "hastings/math/math.h"

namespace hastings {
template <typename ScalarT>
struct RadialDistortion {
    using Scalar = ScalarT;
    using Vec2s = Vec2<Scalar>;

    Scalar k1 = Scalar(0.0);

    CUDA_HOST_DEVICE Vec2s distort(const Vec2s& pixel) const {
        // https://www.robots.ox.ac.uk/~lav/Papers/tordoff_murray_cviu2004/tordoff_murray_cviu2004.pdf
        const Scalar dis2 = squaredLength(pixel);
        const Scalar ratio = sqrt(Scalar(1.0) - Scalar(2.0) * k1 * dis2);

        return pixel / ratio;
    }

    CUDA_HOST_DEVICE Vec2s undistort(const Vec2s& pixel) const {
        // https://www.robots.ox.ac.uk/~lav/Papers/tordoff_murray_cviu2004/tordoff_murray_cviu2004.pdf

        const Scalar dis2 = squaredLength(pixel);
        const Scalar ratio = sqrt(Scalar(1.0) + Scalar(2.0) * k1 * dis2);
        return pixel / ratio;
    }
};

template <typename T>
CUDA_HOST_DEVICE Vec3<T> RollPitchYawRotatePoint(const Vec3<T> rpy, const Vec3<T> pt, const bool inverse) {
    T cosRoll = cos(rpy.x);
    T sinRoll = sin(rpy.x);
    T cosPitch = cos(rpy.y);
    T sinPitch = sin(rpy.y);
    T cosYaw = cos(rpy.z);
    T sinYaw = sin(rpy.z);

    T R11 = cosYaw * cosPitch;
    T R12 = cosYaw * sinPitch * sinRoll - sinYaw * cosRoll;
    T R13 = cosYaw * sinPitch * cosRoll + sinYaw * sinRoll;

    T R21 = sinYaw * cosPitch;
    T R22 = sinYaw * sinPitch * sinRoll + cosYaw * cosRoll;
    T R23 = sinYaw * sinPitch * cosRoll - cosYaw * sinRoll;

    T R31 = -sinPitch;
    T R32 = cosPitch * sinRoll;
    T R33 = cosPitch * cosRoll;

    if (inverse) {
        std::swap(R12, R21);
        std::swap(R13, R31);
        std::swap(R23, R32);
    }

    Vec3<T> result;
    result.x = R11 * pt.x + R12 * pt.y + R13 * pt.z;
    result.y = R21 * pt.x + R22 * pt.y + R23 * pt.z;
    result.z = R31 * pt.x + R32 * pt.y + R33 * pt.z;

    return result;
}

template <typename ScalarT>
class Calib {
  public:
    using Scalar = ScalarT;
    using Vec2s = Vec2<Scalar>;
    using Vec3s = Vec3<Scalar>;
    using Mat3s = Mat3<Scalar>;
    using Distortion = RadialDistortion<Scalar>;

    // NOTE(will.brennan): by default; cameras point up along the z-axis

    CUDA_HOST_DEVICE Calib(const Vec3s& trans, const Vec3s& _rpy, const Vec2s& center, Scalar focal_length, Scalar aspect_ratio,
                           const Distortion& distortion)
        : trans_(trans), rpy_(_rpy), center_(center), focal_length_(focal_length), aspect_ratio_(aspect_ratio), distortion_(distortion) {}

    CUDA_HOST_DEVICE const Vec3s& trans() const { return trans_; }
    CUDA_HOST_DEVICE const Vec3s& rollPitchYaw() const { return rpy_; }
    CUDA_HOST_DEVICE const Scalar& focalLength() const { return focal_length_; }
    CUDA_HOST_DEVICE const Scalar& aspectRatio() const { return aspect_ratio_; }
    CUDA_HOST_DEVICE const Vec2s& center() const { return center_; }
    CUDA_HOST_DEVICE const Distortion& distortion() const { return distortion_; }

    CUDA_HOST_DEVICE Vec3s& trans() { return trans_; }
    CUDA_HOST_DEVICE Vec3s& rollPitchYaw() { return rpy_; }
    CUDA_HOST_DEVICE Scalar& focalLength() { return focal_length_; }
    CUDA_HOST_DEVICE Scalar& aspectRatio() { return aspect_ratio_; }
    CUDA_HOST_DEVICE Vec2s& center() { return center_; }
    CUDA_HOST_DEVICE Distortion& distortion() { return distortion_; }

    CUDA_HOST_DEVICE Vec2s project(Vec3s point) const {
        point = point - trans_;
        const auto point_in_cam = RollPitchYawRotatePoint(rpy_, point, false);

        const auto pixel_homogenous = Vec2s({point_in_cam.x, point_in_cam.y}) / point_in_cam.z;

        auto pixel = focal_length_ * distortion_.distort(pixel_homogenous);
        pixel.y *= aspect_ratio_;
        pixel = pixel + center_;

        return pixel;
    }

    CUDA_HOST_DEVICE Vec3s direction(Vec2s pixel) const {
        pixel = pixel - center_;
        pixel.y /= aspect_ratio_;

        auto pixel_homogenous = distortion_.undistort(pixel) / focal_length_;

        auto direction_cam_coords = Vec3s({pixel_homogenous.x, pixel_homogenous.y, Scalar(1.0)});
        const auto point_in_world = RollPitchYawRotatePoint(rpy_, direction_cam_coords, true);
        return point_in_world;
    }

  private:
    // NOTE(will.brennan): we care about memory ordering due to ceres optimization
    Vec3s trans_;
    Vec3s rpy_;
    Vec2s center_;
    Scalar focal_length_;
    Scalar aspect_ratio_;

    Distortion distortion_;
};

using Calibd = Calib<double>;

template <typename Scalar>
CUDA_HOST_DEVICE Vec3<Scalar> pixelInZ(const Calib<Scalar>& calib, const Vec2<Scalar>& pixel, const Scalar& z) {
    const auto direction = calib.direction(pixel);
    const auto t = (z - calib.trans().z) / direction.z;
    return calib.trans() + direction * t;
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& stream, const Calib<Scalar>& calib) {
    stream << "{ trans: " << calib.trans();
    stream << ", rpy: " << calib.rollPitchYaw();
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