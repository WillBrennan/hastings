#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace hastings {
class WebSocketSession;
class WebSocketServer : public std::enable_shared_from_this<WebSocketServer> {
  public:
    using FnMessageHandler = std::function<void(const std::string&)>;
    using Ptr = std::shared_ptr<WebSocketServer>;
    using Port = short;
    // todo
    // 1. thread safety

    static Ptr make(const Port port);

    void start();

    void messageHandler(FnMessageHandler handler);

    void write(std::string&& message);
    void write(std::vector<std::uint8_t>&& buffer);

  private:
    struct Storage;
    std::unique_ptr<Storage> storage_;
    std::recursive_mutex mutex_;

    WebSocketServer(const Port port);

    void accept();
};
}  // namespace hastings