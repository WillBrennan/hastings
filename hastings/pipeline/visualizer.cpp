#include "hastings/pipeline/visualizer.h"

#include <glog/logging.h>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

#include "hastings/helpers/profile_marker.h"
#include "hastings/helpers/websocket.h"

using nlohmann::json;

namespace hastings {
VisualizerStreamerNode::VisualizerStreamerNode(const Port port) : server_(WebSocketServer::make(port)) {
    server_->messageHandler([this](const std::string& data) {
        const auto decoded = json::from_msgpack(data);

        std::lock_guard lock(mutex_);
        stream_config_ = StreamConfig{decoded["camera"], decoded["image"]};
    });
    server_->start();
}

VisualizerStreamerNode::~VisualizerStreamerNode() {}

ExecutionPolicy VisualizerStreamerNode::executionPolicy() const { return ExecutionPolicy::Ordered; }
std::string VisualizerStreamerNode::name() const { return "VisualizerStreamerNode"; }

void VisualizerStreamerNode::process(MultiImageContextInterface& multi_context) {
    std::optional<StreamConfig> stream_config;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stream_config = stream_config_;
    }

    json json;
    {
        ProfilerFunctionMarker marker("serialization");
        std::map<std::string, std::vector<std::string>> cameras;
        for (const auto& [camera, context] : multi_context.cameras()) {
            auto& images = cameras[camera];

            context->images([&images, &camera, &stream_config](const std::string& name, const cv::Mat& image) {
                images.emplace_back(name);

                if (!stream_config.has_value()) {
                    stream_config = {camera, name};
                }
            });
        }

        json["cameras"] = cameras;
        json["current"] = nullptr;
        json["image"] = nullptr;

        if (stream_config.has_value()) {
            const auto config = stream_config.value();

            const auto image = multi_context.cameras(config.camera)->image(config.image);
            std::vector<std::uint8_t> buffer;
            buffer.reserve(image.cols * image.rows);
            cv::imencode(".bmp", image, buffer);

            json["current"] = {{"camera", config.camera}, {"image", config.image}};
            json["image"] = json::binary_t(std::move(buffer));
        }
    }

    Buffer buffer;
    {
        ProfilerFunctionMarker marker2("to_msgpack");
        buffer = json::to_msgpack(json);
    }

    {
        ProfilerFunctionMarker marker("send on websocket");
        server_->write(std::move(buffer));
    }
}
}  // namespace hastings