#include "hastings/pipeline/visualizer.h"

#include <glog/logging.h>

#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

#include "hastings/helpers/profile_marker.h"

using tcp = boost::asio::ip::tcp;
using boost::asio::io_context;
using nlohmann::json;

namespace websocket = boost::beast::websocket;

namespace hastings {
namespace detail {
template <>
void serialize(const ImageContextInterface& context, nlohmann::json& json) {
    json["frameId"] = context.frameId();
    json["time"] = context.time();

    context.images([&json](const std::string& imageName, const cv::Mat& image) {
        std::vector<std::uint8_t> buffer;
        buffer.reserve(image.cols * image.rows);
        cv::imencode(".bmp", image, buffer);
        json["images"][imageName] = json::binary_t(std::move(buffer));
    });
}

template <>
void deserialize(const nlohmann::json& json, ImageContextInterface& context) {
    context.frameId(json.at("frameId"));
    context.time(json.at("time"));
}

template <>
void serialize(const MultiImageContextInterface& multi_context, nlohmann::json& json) {
    serialize<ImageContextInterface>(multi_context, json);

    for (auto& [cameraName, context] : multi_context.cameras()) {
        serialize<ImageContextInterface>(*context, json["cameras"][cameraName]);
    }
}

template <>
void deserialize(const nlohmann::json& json, MultiImageContextInterface& multi_context) {
    deserialize<ImageContextInterface>(json, multi_context);
}

}  // namespace detail

VisualizerStreamerNode::VisualizerStreamerNode(const Port port)
    : worker_([this, port] {
          while (true) {
              try {
                  websocketThread(port);
              } catch (const std::exception& e) {
                  LOG(WARNING) << "Error: " << e.what();
              }
          }
      }) {}

VisualizerStreamerNode::~VisualizerStreamerNode() {}

ExecutionPolicy VisualizerStreamerNode::executionPolicy() const { return ExecutionPolicy::Ordered; }
std::string VisualizerStreamerNode::name() const { return "VisualizerStreamerNode"; }

void VisualizerStreamerNode::process(MultiImageContextInterface& multi_context) {
    json json;
    {
        ProfilerFunctionMarker marker("serialization");
        detail::serialize(multi_context, json);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);

        Buffer buffer;
        {
            ProfilerFunctionMarker marker2("to_msgpack");
            buffer = json::to_msgpack(json);
        }
        {
            ProfilerFunctionMarker marker2("on buffer");
            msg_queue_.push(std::move(buffer));
        }
    }

    cv_.notify_one();
}

void VisualizerStreamerNode::websocketThread(const Port port) {
    io_context context;
    tcp::acceptor acceptor(context, {tcp::v4(), port});

    // TCP handshake
    auto socket = acceptor.accept();
    // Upgrade to websocket
    auto ws = websocket::stream<tcp::socket>(std::move(socket));
    ws.accept();

    while (ws.is_open()) {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return !msg_queue_.empty(); });

        ProfilerFunctionMarker marker("send on websocket");

        auto data = std::move(msg_queue_.front());
        msg_queue_.pop();

        ws.binary(true);
        ws.write(boost::asio::buffer(data));
    }
}
}  // namespace hastings