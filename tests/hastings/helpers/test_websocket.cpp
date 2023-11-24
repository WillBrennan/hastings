#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "hastings/helpers/websocket.h"

// Mock class for FnMessageHandler
class MockMessageHandler {
  public:
    MOCK_METHOD(void, handle, (const std::string&), (const));
};

// Test fixture for WebSocketServer tests
class WebSocketServerTest : public testing::Test {
  public:
    using WebSocketServer = hastings::WebSocketServer;

    void SetUp() override {
        // Set up any common resources or actions needed for the tests
    }

    void TearDown() override {
        // Clean up any common resources or actions needed for the tests
    }

  protected:
    WebSocketServer::Port testPort = 12345;
    WebSocketServer::Ptr server;
};

// Tests for WebSocketServer class
TEST_F(WebSocketServerTest, Construction) {
    ASSERT_NO_THROW(server = WebSocketServer::make(testPort));
    ASSERT_TRUE(server != nullptr);
}

TEST_F(WebSocketServerTest, Start) {
    server = WebSocketServer::make(testPort);
    ASSERT_NO_THROW(server->start());
}

TEST_F(WebSocketServerTest, MessageHandler) {
    GTEST_SKIP() << "test-case is bad";
    server = WebSocketServer::make(testPort);
    ASSERT_NO_THROW(server->start());
    MockMessageHandler mockHandler;

    FAIL();

    EXPECT_CALL(mockHandler, handle(testing::_)).Times(1);

    server->messageHandler([&](const std::string& message) { mockHandler.handle(message); });
}
