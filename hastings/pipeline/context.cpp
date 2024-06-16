#include "hastings/pipeline/context.h"

#include <map>

namespace hastings {

class ImageContext : public ImageContextInterface {
  public:
    void clear() final {
        for (auto& [key, value] : data_) {
            value.reset();
        }

        for (auto& [key, data] : images_) {
            data.image = cv::Mat();
            data.graphics.clear();
        }
    }

    void time(Time t) override final { time_ = t; }
    Time time() const override final { return time_; }

    void frameId(const std::size_t& id) override final { id_ = id; }
    std::size_t frameId() const override final { return id_; };

    std::any& result(const std::string& name) override final { return data_[name]; }

    cv::Mat& image(const std::string& name) override final { return images_[name].image; }

    void images(const FnImage& fn_image) override final {
        for (auto& [name, image] : images_) {
            fn_image(name, image.image);
        }
    }

    void images(const FnConstImage& fn_image) const override final {
        for (auto& [name, image] : images_) {
            fn_image(name, image.image);
        }
    }

    void vectorGraphic(const std::string& image_name, std::vector<VectorGraphic>&& graphics) override final {
        auto& image_data = images_[image_name];
        image_data.graphics.insert(image_data.graphics.end(), graphics.begin(), graphics.end());
    }

    const std::vector<VectorGraphic>& vectorGraphic(const std::string& image_name) const override final {
        return images_.at(image_name).graphics;
    }

  private:
    struct ImageData {
        cv::Mat image;
        std::vector<VectorGraphic> graphics;
    };

    Time time_;
    std::size_t id_;

    std::map<std::string, std::any> data_;
    std::map<std::string, ImageData> images_;
};

class MultiImageContext final : public MultiImageContextInterface {
  public:
    const Cameras& cameras() const override final { return cameras_; }

    ImageContextInterface* cameras(const std::string& name) override final {
        auto iter = std::find_if(cameras_.begin(), cameras_.end(), [&name](const Camera& camera) { return std::get<0>(camera) == name; });
        if (iter != cameras_.end()) {
            return std::get<1>(*iter).get();
        }

        auto& value = cameras_.emplace_back(Camera{name, createImageContext()});
        return std::get<1>(value).get();
    }

    void clear() override final {
        context_.clear();
        for (auto& camera : cameras_) {
            std::get<1>(camera)->clear();
        }
    }

    void time(const Time t) override final {
        context_.time(t);
        for (auto& camera : cameras_) {
            std::get<1>(camera)->time(t);
        }
    }

    Time time() const override final { return context_.time(); }

    void frameId(const std::size_t& id) override final {
        context_.frameId(id);
        for (auto& camera : cameras_) {
            std::get<1>(camera)->frameId(id);
        }
    }
    std::size_t frameId() const override final { return context_.frameId(); };

    std::any& result(const std::string& name) override final { return context_.result(name); }

    cv::Mat& image(const std::string& name) override final { return context_.image(name); }

    void images(const FnImage& fn_image) override final { context_.images(fn_image); }

    void images(const FnConstImage& fn_image) const override final { context_.images(fn_image); }

    void vectorGraphic(const std::string& image_name, std::vector<VectorGraphic>&& graphics) override final {
        context_.vectorGraphic(image_name, std::move(graphics));
    }

    const std::vector<VectorGraphic>& vectorGraphic(const std::string& image_name) const override final {
        return context_.vectorGraphic(image_name);
    }

  private:
    Cameras cameras_;
    ImageContext context_;
};

ImageContextInterface::Ptr createImageContext() { return std::make_unique<ImageContext>(); }
MultiImageContextInterface::Ptr createMultiImageContext() { return std::make_unique<MultiImageContext>(); }

}  // namespace hastings