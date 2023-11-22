#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "hastings/pipeline/node.h"

namespace hastings {

class WebSocketServer;
class VisualizerStreamerNode final : public NodeInterface {
  public:
    using Port = short unsigned int;

    explicit VisualizerStreamerNode(const Port port = 8080);
    ~VisualizerStreamerNode();

    ExecutionPolicy executionPolicy() const override final;
    std::string name() const override final;

    void process(MultiImageContextInterface& multi_context) override final;

  private:
    using Buffer = std::vector<std::uint8_t>;

    struct StreamConfig {
        std::string camera;
        std::string image;
    };

    std::mutex mutex_;
    std::optional<StreamConfig> stream_config_;
    std::shared_ptr<WebSocketServer> server_;
};
}  // namespace hastings