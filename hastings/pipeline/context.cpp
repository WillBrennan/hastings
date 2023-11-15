#include "hastings/pipeline/context.h"

#include <map>

namespace hastings {

class ImageContext : public ImageContextInterface {
  public:
    void clear() final {
        for (auto& [key, value] : data_) {
            value.reset();
        }
    }

    void time(const std::size_t t) override final { time_ = t; }
    std::size_t time() const override final { return time_; }

    void frameId(const std::size_t& id) override final { id_ = id; }
    std::size_t frameId() const override final { return id_; };

    std::any& result(const std::string& name) override final { return data_[name]; }

    cv::Mat& image(const std::string& name) override final { return images_[name]; }

    void images(const FnImage& fn_image) override final {
        for (auto& [name, image] : images_) {
            fn_image(name, image);
        }
    }

  private:
    std::size_t time_;
    std::size_t id_;
    std::map<std::string, std::any> data_;
    std::map<std::string, cv::Mat> images_;
};

class MultiImageContext final : public MultiImageContextInterface {
  public:
    const Cameras& cameras() override final { return cameras_; }

    ImageContextInterface* cameras(const std::string& name) override final {
        auto iter = std::find_if(cameras_.begin(), cameras_.end(), [&name](const Camera& camera) { return std::get<0>(camera) == name; });
        if (iter != cameras_.end()) {
            return std::get<1>(*iter).get();
        }

        auto& value = cameras_.emplace_back(Camera{name, createImageContext()});
        return std::get<1>(value).get();
    }

    void clear() override final { context_.clear(); }

    void time(const std::size_t t) override final { context_.time(t); }
    std::size_t time() const override final { return context_.time(); }

    void frameId(const std::size_t& id) override final { context_.frameId(id); }
    std::size_t frameId() const override final { return context_.frameId(); };

    std::any& result(const std::string& name) override final { return context_.result(name); }

    cv::Mat& image(const std::string& name) override final { return context_.image(name); }

    void images(const FnImage& fn_image) override final { context_.images(fn_image); }

  private:
    Cameras cameras_;
    ImageContext context_;
};

ImageContextInterface::Ptr createImageContext() { return std::make_unique<ImageContext>(); }
MultiImageContextInterface::Ptr createMultiImageContext() { return std::make_unique<MultiImageContext>(); }

}  // namespace hastings