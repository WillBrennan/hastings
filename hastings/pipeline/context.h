#pragma once

#include <any>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace hastings {
class ImageContextInterface {
  public:
    using Ptr = std::unique_ptr<ImageContextInterface>;

    ImageContextInterface() = default;
    virtual ~ImageContextInterface() = default;

    virtual void clear() = 0;

    virtual void time(const std::size_t t) = 0;
    virtual std::size_t time() const = 0;

    virtual void frameId(const std::size_t& id) = 0;
    virtual std::size_t frameId() const = 0;

    virtual std::any& result(const std::string& name) = 0;

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
    virtual const Cameras& cameras() = 0;
};

ImageContextInterface::Ptr createImageContext();
MultiImageContextInterface::Ptr createMultiImageContext();
}  // namespace hastings