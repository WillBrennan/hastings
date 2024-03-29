#pragma once

#include <any>
#include <chrono>
#include <functional>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "hastings/pipeline/vector_graphic.h"

namespace hastings {
class ImageContextInterface {
  public:
    using Clock = std::chrono::steady_clock;
    using Time = Clock::time_point;
    using Ptr = std::unique_ptr<ImageContextInterface>;
    using FnImage = std::function<void(const std::string&, cv::Mat& image)>;
    using FnConstImage = std::function<void(const std::string&, const cv::Mat& image)>;

    ImageContextInterface() = default;
    virtual ~ImageContextInterface() = default;

    virtual void clear() = 0;

    void timeFromClock() { time(Clock::now()); }
    virtual void time(Time t) = 0;
    virtual Time time() const = 0;

    virtual void frameId(const std::size_t& id) = 0;
    virtual std::size_t frameId() const = 0;

    virtual std::any& result(const std::string& name) = 0;

    virtual cv::Mat& image(const std::string& name) = 0;
    virtual void images(const FnImage& fn_image) = 0;
    virtual void images(const FnConstImage& fn_image) const = 0;

    virtual void vectorGraphic(const std::string& image_name, std::vector<VectorGraphic>&& graphics) = 0;
    virtual const std::vector<VectorGraphic>& vectorGraphic(const std::string& image_name) const = 0;

    template <class T>
    T& result(const std::string& name) {
        return std::any_cast<T&>(result(name));
    }
};

class MultiImageContextInterface : public ImageContextInterface {
  public:
    using Ptr = std::unique_ptr<MultiImageContextInterface>;
    using Camera = std::tuple<std::string, ImageContextInterface::Ptr>;
    using Cameras = std::vector<Camera>;

    MultiImageContextInterface() = default;
    virtual ~MultiImageContextInterface() = default;

    virtual ImageContextInterface* cameras(const std::string& name) = 0;
    virtual const Cameras& cameras() const = 0;
};

ImageContextInterface::Ptr createImageContext();
MultiImageContextInterface::Ptr createMultiImageContext();
}  // namespace hastings