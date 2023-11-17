#pragma once

#include <condition_variable>
#include <mutex>
#include <nlohmann/json_fwd.hpp>
#include <queue>
#include <thread>

#include "hastings/pipeline/node.h"

namespace hastings {

namespace detail {
template <class T>
void serialize(const T& value, nlohmann::json& json) = delete;

template <class T>
void deserialize(const nlohmann::json& json, T& value) = delete;

template <>
void serialize(const ImageContextInterface& context, nlohmann::json& json);

template <>
void deserialize(const nlohmann::json& json, ImageContextInterface& context);

template <>
void serialize(const MultiImageContextInterface& multi_context, nlohmann::json& json);

template <>
void deserialize(const nlohmann::json& json, MultiImageContextInterface& multi_context);

}  // namespace detail

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

    std::jthread worker_;
    std::condition_variable cv_;
    std::mutex mutex_;
    std::queue<Buffer> msg_queue_;

    void websocketThread(const Port port);
};
}  // namespace hastings