#include <gflags/gflags.h>
#include <glog/logging.h>

#include <map>
#include <opencv2/opencv.hpp>
#include <string>

#include "hastings/pipeline/context.h"
#include "hastings/pipeline/node.h"
#include "hastings/pipeline/pipeline.h"

namespace hastings {
class VideoCaptureNode final : public NodeInterface {
  public:
    explicit VideoCaptureNode(const int index) : source_(index) { CHECK(source_.isOpened()) << "failed to open webcam"; }

    explicit VideoCaptureNode(const std::string& filename) : source_(filename) { CHECK(source_.isOpened()) << "failed to open video"; }

    ExecutionPolicy executionPolicy() const override final { return ExecutionPolicy::Ordered; }
    std::string name() const override final { return "VideoCaptureNode"; }

    // todo(will) - handle fetching & adjusting settings...
    void process(MultiImageContextInterface& multi_context) override final {
        LOG(INFO) << "creating context";

        cv::Mat image;
        source_.read(image);

        auto context = multi_context.cameras("camera");
        context->result("image") = image.clone();

        LOG(INFO) << "num cameras: " << multi_context.cameras().size();

        LOG(INFO) << "complete!";
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
            const auto& image = context->result<cv::Mat>("image");

            auto previous_image = previous_images_[cameraName];
            if (previous_image.empty()) {
                previous_image = image;
            }

            cv::Mat diff_image;
            // this is causing an allocation...
            cv::absdiff(image, previous_image, diff_image);

            context->result("diff") = diff_image;

            image.copyTo(previous_images_[cameraName]);
        }
    }

  private:
    std::map<std::string, cv::Mat> previous_images_;
};

class DisplayImageNode final : public NodeInterface {
  public:
    ExecutionPolicy executionPolicy() const override final { return ExecutionPolicy::Ordered; }
    std::string name() const override final { return "DisplayImageNode"; }

    void process(MultiImageContextInterface& multi_context) override final {
        LOG(INFO) << "displaying frame " << multi_context.frameId();

        for (auto& [cameraName, context] : multi_context.cameras()) {
            for (const auto& imageName : {"image", "diff"}) {
                cv::imshow(cameraName + " " + imageName, context->result<cv::Mat>(imageName));
            }

            cv::waitKey(1);
        }
    }
};

}  // namespace hastings

int main(int argc, char** argv) {
    using hastings::createPipeline;
    using hastings::DisplayImageNode;
    using hastings::FrameDiffNode;
    using hastings::VideoCaptureNode;

    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    LOG(INFO) << "creating pipeline";

    auto pipeline = createPipeline(1);

    pipeline->add<VideoCaptureNode>(0);
    pipeline->add<FrameDiffNode>();
    pipeline->add<DisplayImageNode>();

    LOG(INFO) << "starting pipeline";
    pipeline->start();
}