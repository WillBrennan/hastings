#include <gflags/gflags.h>
#include <glog/logging.h>

#include <functional>
#include <memory>
#include <thread>

#include "hastings/helpers/websocket.h"

int main() {
    using hastings::WebSocketServer;
    using namespace std::chrono_literals;

    auto server = WebSocketServer::make(8080);

    server->messageHandler([server](const std::string& message) {
        LOG(INFO) << "Received message: " << message;
        server->write("this works - " + message);
    });
    server->start();

    std::this_thread::sleep_for(10s);

    return 0;
}