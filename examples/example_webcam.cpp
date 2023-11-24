#include <gflags/gflags.h>
#include <glog/logging.h>

#include <map>
#include <opencv2/opencv.hpp>
#include <string>

#include "hastings/pipeline/context.h"
#include "hastings/pipeline/node.h"
#include "hastings/pipeline/pipeline.h"
#include "hastings/pipeline/visualizer.h"

namespace hastings {
class VideoCaptureNode final : public NodeInterface {
  public:
    explicit VideoCaptureNode(const int index) : source_(index) { CHECK(source_.isOpened()) << "failed to open webcam"; }

    explicit VideoCaptureNode(const std::string& filename) : source_(filename) { CHECK(source_.isOpened()) << "failed to open video"; }

    ExecutionPolicy executionPolicy() const override final { return ExecutionPolicy::Ordered; }
    std::string name() const override final { return "VideoCaptureNode"; }

    void process(MultiImageContextInterface& multi_context) override final {
        auto context = multi_context.cameras("camera");
        source_.read(context->image("BGR"));

        auto other_context = multi_context.cameras("dummy camera");
        context->image("BGR").copyTo(other_context->image("BGR"));
        cv::flip(other_context->image("BGR"), other_context->image("flipped BGR"), 0);
    }

  private:
    cv::VideoCapture source_;
};

class FrameDiffNode final : public NodeInterface {
  public:
    ExecutionPolicy executionPolicy() const override final { return ExecutionPolicy::Ordered; }
    std::string name() const override final { return "FrameDiffNode"; }

    void process(MultiImageContextInterface& multi_context) override final {
        for (auto& [cameraName, context] : multi_context.cameras()) {
            const auto& image = context->image("BGR");

            auto previous_image = previous_images_[cameraName];
            if (previous_image.empty()) {
                previous_image = image;
            }

            cv::absdiff(image, previous_image, context->image("diff"));
            image.copyTo(previous_images_[cameraName]);
        }
    }

  private:
    std::map<std::string, cv::Mat> previous_images_;
};

class ConvertBGR2Y final : public NodeInterface {
    ExecutionPolicy executionPolicy() const override final { return ExecutionPolicy::Parallel; }
    std::string name() const override final { return "ConvertBGR2Y"; }

    void process(MultiImageContextInterface& multi_context) override final {
        for (const auto& [camera, context] : multi_context.cameras()) {
            cv::cvtColor(context->image("BGR"), context->image("Y"), cv::COLOR_BGR2GRAY);
        }
    }
};

class OpticalFlowNode final : public NodeInterface {
  public:
    OpticalFlowNode(int maxCorners, double quality, double minDistance)
        : maxCorners_(maxCorners), quality_(quality), minDistance_(minDistance) {}

    ExecutionPolicy executionPolicy() const override final { return ExecutionPolicy::Ordered; }
    std::string name() const override final { return "OpticalFlowNode"; }

    void process(MultiImageContextInterface& multi_context) override final {
        // NOTE(will): add better support for single image operations...
        const auto context = multi_context.cameras("camera");

        std::vector<cv::Mat> nextPyramid;
        std::vector<cv::Point2f> nextPts;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::Mat& image_y = context->image("Y");
        cv::buildOpticalFlowPyramid(image_y, nextPyramid, cv::Size(21, 21), 3);

        if (!prevPyramid_.empty()) {
            std::vector<cv::Point2f> prevCVPts;
            prevCVPts.reserve(prevPoints_.size());
            std::transform(prevPoints_.begin(), prevPoints_.end(), std::back_inserter(prevCVPts),
                           [](const Point& point) { return point.point; });

            cv::calcOpticalFlowPyrLK(prevPyramid_, nextPyramid, prevCVPts, nextPts, status, err);

            VectorGraphics graphics;
            graphics.reserve(prevPoints_.size() * 2);
            for (size_t i = 0; i < prevPoints_.size(); ++i) {
                if (status[i]) {
                    const auto next = Vec2f{nextPts[i].x, nextPts[i].y};
                    const auto prev = Vec2f{prevPoints_[i].point.x, prevPoints_[i].point.y};
                    graphics.emplace_back(LineGraphic{{0, 255, 0}, next, prev});
                    graphics.emplace_back(PointGraphic{{0, 255, 0}, next});
                }
            }

            graphics.emplace_back(TextGraphic{{0, 255, 255}, {50, 60}, "hello world"});
            graphics.emplace_back(RectangleGraphic{{255, 0, 255}, {50, 60}, {100, 170}});

            context->vectorGraphic("BGR", std::move(graphics));

            for (auto i = 0; i < prevPoints_.size(); ++i) {
                prevPoints_[i].good = bool(status[i]);
                prevPoints_[i].point = nextPts[i];
            }

            prevPoints_.erase(std::remove_if(prevPoints_.begin(), prevPoints_.end(), [](const Point& point) { return !point.good; }),
                              prevPoints_.end());
        }

        if (prevPoints_.size() < maxCorners_) {
            std::vector<cv::Point2f> pts;
            const auto numCorners = maxCorners_ - prevPoints_.size();
            cv::goodFeaturesToTrack(image_y, pts, numCorners, quality_, minDistance_);

            for (auto& pt : pts) {
                prevPoints_.emplace_back(Point{idx_uuid_++, pt});
            }
        }

        std::swap(nextPyramid, prevPyramid_);
    }

  private:
    struct Point {
        std::size_t idx;
        cv::Point2f point;
        bool good = true;
    };

    int maxCorners_;
    double quality_;
    double minDistance_;
    std::size_t idx_uuid_ = 0;

    std::vector<Point> prevPoints_;
    std::vector<cv::Mat> prevPyramid_;
};

}  // namespace hastings

int main(int argc, char** argv) {
    using hastings::ConvertBGR2Y;
    using hastings::createPipeline;
    using hastings::FrameDiffNode;
    using hastings::OpticalFlowNode;
    using hastings::VideoCaptureNode;
    using hastings::VisualizerStreamerNode;

    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    LOG(INFO) << "creating pipeline";
    auto pipeline = createPipeline(5);

    pipeline->add<VideoCaptureNode>(0);
    pipeline->add<FrameDiffNode>();
    pipeline->add<ConvertBGR2Y>();
    pipeline->add<OpticalFlowNode>(50, 0.01, 30);
    pipeline->add<VisualizerStreamerNode>();

    LOG(INFO) << "starting pipeline";
    pipeline->start();
}