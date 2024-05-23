#include "hastings/helpers/websocket.h"

#include <utility>

#include <glog/logging.h>

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace websocket = beast::websocket;

namespace hastings {
class WebSocketSession : public std::enable_shared_from_this<WebSocketSession> {
  public:
    using FnMessageHandler = WebSocketServer::FnMessageHandler;

    explicit WebSocketSession(boost::asio::ip::tcp::socket socket);

    void accept();

    bool write(const std::string& message);
    bool write(const std::vector<std::uint8_t>& buffer);

    void messageHandler(FnMessageHandler handler);

  private:
    void read();

    boost::beast::websocket::stream<boost::asio::ip::tcp::socket> ws_;
    boost::asio::strand<boost::asio::any_io_executor> strand_;
    boost::beast::flat_buffer buffer_;
    FnMessageHandler messageHandler_;
};

struct WebSocketServer::Storage {
    Storage(const Port port)
        : acceptor_(ioContext_, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port)), socket_(ioContext_) {}

    boost::asio::io_context ioContext_;
    boost::asio::ip::tcp::acceptor acceptor_;
    boost::asio::ip::tcp::socket socket_;
    std::vector<std::shared_ptr<WebSocketSession>> sessions_;
    FnMessageHandler messageHandler_;
    std::thread thread_io_;

    ~Storage() { thread_io_.join(); }
};

WebSocketSession::WebSocketSession(boost::asio::ip::tcp::socket socket)
    : ws_(std::move(socket)), strand_(boost::asio::make_strand(ws_.get_executor())) {}

void WebSocketSession::accept() {
    // Set suggested timeout settings for the websocket
    ws_.set_option(boost::beast::websocket::stream_base::timeout::suggested(boost::beast::role_type::server));

    // Set a decorator to change the Server of the handshake
    ws_.set_option(boost::beast::websocket::stream_base::decorator(
        [](boost::beast::websocket::response_type& res) { res.set(boost::beast::http::field::server, "WebSocketServer"); }));

    // Accept the websocket handshake
    ws_.async_accept([self = shared_from_this()](boost::system::error_code ec) {
        if (ec) {
            LOG(WARNING) << "websocket accept error: " << ec.message();
            return;
        }

        self->read();
    });
}

bool WebSocketSession::write(const std::string& message) {
    auto self = shared_from_this();

    if (!self->ws_.is_open()) {
        return false;
    }

    boost::asio::post(strand_, [self, message]() {
        self->ws_.text(true);
        self->ws_.write(boost::asio::buffer(message));
    });

    return true;
}

bool WebSocketSession::write(const std::vector<std::uint8_t>& buffer) {
    auto self = shared_from_this();

    if (!self->ws_.is_open()) {
        LOG(INFO) << "writing to a closed socket";
        return true;
    }

    boost::asio::post(strand_, [self, buffer]() {
        self->ws_.binary(true);
        self->ws_.write(boost::asio::buffer(buffer));
    });

    return false;
}

void WebSocketSession::messageHandler(FnMessageHandler handler) {
    messageHandler_ = std::move(handler);
    read();
}

void WebSocketSession::read() {
    auto self = shared_from_this();
    ws_.async_read(buffer_, [self](boost::system::error_code ec, std::size_t bytes_transferred) {
        if (!ec) {
            if (self->messageHandler_) {
                const auto message = boost::beast::buffers_to_string(self->buffer_.data());
                self->buffer_.clear();
                self->messageHandler_(message);
            }

            self->read();
        }
    });
}

WebSocketServer::Ptr WebSocketServer::make(const Port port) { return Ptr(new WebSocketServer(port)); }

void WebSocketServer::start() {
    std::lock_guard lock(mutex_);

    auto self = shared_from_this();
    storage_->thread_io_ = std::thread([self] {
        self->accept();
        self->storage_->ioContext_.run();
    });
}

void WebSocketServer::messageHandler(WebSocketSession::FnMessageHandler handler) {
    std::lock_guard lock(mutex_);

    storage_->messageHandler_ = std::move(handler);
    for (auto& session : storage_->sessions_) {
        session->messageHandler(storage_->messageHandler_);
    }
}

void WebSocketServer::write(std::string&& message) {
    std::lock_guard lock(mutex_);

    const auto iter = std::remove_if(storage_->sessions_.begin(), storage_->sessions_.end(),
                                     [message = std::move(message)](auto& session) { return session->write(message); });
    storage_->sessions_.erase(iter, storage_->sessions_.end());
}

void WebSocketServer::write(std::vector<std::uint8_t>&& buffer) {
    std::lock_guard lock(mutex_);

    const auto iter = std::remove_if(storage_->sessions_.begin(), storage_->sessions_.end(), [buffer = std::move(buffer)](auto& session) {
        const auto is_closed = session->write(buffer);
        return is_closed;
    });
    storage_->sessions_.erase(iter, storage_->sessions_.end());
}

WebSocketServer::WebSocketServer(const Port port) : storage_(std::make_unique<WebSocketServer::Storage>(port)) {}

void WebSocketServer::accept() {
    std::lock_guard lock(mutex_);

    storage_->acceptor_.async_accept(storage_->socket_, [self = shared_from_this()](boost::system::error_code ec) {
        if (!ec) {
            std::lock_guard lock(self->mutex_);

            auto session = std::make_shared<WebSocketSession>(std::move(self->storage_->socket_));
            session->messageHandler(self->storage_->messageHandler_);
            session->accept();
            self->storage_->sessions_.push_back(session);
        }
        self->accept();
    });
}

}  // namespace hastings