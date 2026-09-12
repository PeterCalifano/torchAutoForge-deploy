/**
 * @file test_inference_lifecycle_node.cpp
 * @brief Target-owned lifecycle, inference, and error-response tests.
 */

#include "ptafdeploy_ros/CInferenceLifecycleNode.h"

#include "ptafdeploy_interfaces/msg/model_status.hpp"
#include "ptafdeploy_interfaces/srv/infer_float_tensors.hpp"

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#ifndef PTAFDEPLOY_TEST_MODEL_CONFIG
#error "PTAFDEPLOY_TEST_MODEL_CONFIG must identify the real ONNX fixture manifest"
#endif

namespace
{
    using namespace std::chrono_literals;
    using InferFloatTensors = ptafdeploy_interfaces::srv::InferFloatTensors;
    using ModelStatus = ptafdeploy_interfaces::msg::ModelStatus;

    class CRclcppContextGuard
    {
      public:
        CRclcppContextGuard()
        {
            if (!rclcpp::ok())
            {
                rclcpp::init(0, nullptr);
            }
        }

        ~CRclcppContextGuard()
        {
            if (rclcpp::ok())
            {
                rclcpp::shutdown();
            }
        }
    };

    TEST(InferenceLifecycleNode, RejectsConfigurationWithoutModelConfig)
    {
        CRclcppContextGuard context;
        const auto node = std::make_shared<ptafdeploy_ros::CInferenceLifecycleNode>();

        const auto state = node->configure();
        EXPECT_EQ(state.label(), "unconfigured");
    }

    TEST(InferenceLifecycleNode, RunsFixtureInferenceAndReportsFailures)
    {
        CRclcppContextGuard context;

        rclcpp::NodeOptions options;
        options.append_parameter_override("model_config_path",
                                          std::string{PTAFDEPLOY_TEST_MODEL_CONFIG});
        const auto inference_node =
            std::make_shared<ptafdeploy_ros::CInferenceLifecycleNode>(options);
        const auto client_node = std::make_shared<rclcpp::Node>("ptafdeploy_lifecycle_test_client");

        std::vector<ModelStatus> status_messages;
        const auto status_subscription = client_node->create_subscription<ModelStatus>(
            "/ptafdeploy_inference/status", rclcpp::QoS(10),
            [&status_messages](const ModelStatus& message) { status_messages.push_back(message); });

        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(inference_node->get_node_base_interface());
        executor.add_node(client_node);

        const auto configured_state = inference_node->configure();
        ASSERT_EQ(configured_state.label(), "inactive");

        const auto client =
            client_node->create_client<InferFloatTensors>("/ptafdeploy_inference/infer");
        ASSERT_TRUE(client->wait_for_service(2s));

        auto request = std::make_shared<InferFloatTensors::Request>();
        ptafdeploy_interfaces::msg::FloatTensor input;
        input.shape = {1, 11};
        input.values = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F};
        request->inputs.push_back(input);

        auto inactive_future = client->async_send_request(request);
        ASSERT_EQ(executor.spin_until_future_complete(inactive_future, 2s),
                  rclcpp::FutureReturnCode::SUCCESS);
        EXPECT_FALSE(inactive_future.get()->success);

        const auto active_state = inference_node->activate();
        ASSERT_EQ(active_state.label(), "active");

        auto inference_future = client->async_send_request(request);
        ASSERT_EQ(executor.spin_until_future_complete(inference_future, 5s),
                  rclcpp::FutureReturnCode::SUCCESS);
        const auto inference_response = inference_future.get();
        ASSERT_TRUE(inference_response->success) << inference_response->status;
        ASSERT_EQ(inference_response->outputs.size(), 1U);
        ASSERT_EQ(inference_response->outputs.front().values.size(), 2U);
        EXPECT_NEAR(inference_response->outputs.front().values[0], 1.686690331F, 1.0e-5F);
        EXPECT_NEAR(inference_response->outputs.front().values[1], 4.697796822F, 1.0e-5F);

        auto invalid_request = std::make_shared<InferFloatTensors::Request>();
        ptafdeploy_interfaces::msg::FloatTensor invalid_input;
        invalid_input.shape = {1, 2};
        invalid_input.values = {1.0F};
        invalid_request->inputs.push_back(invalid_input);

        auto invalid_future = client->async_send_request(invalid_request);
        ASSERT_EQ(executor.spin_until_future_complete(invalid_future, 2s),
                  rclcpp::FutureReturnCode::SUCCESS);
        const auto invalid_response = invalid_future.get();
        EXPECT_FALSE(invalid_response->success);
        EXPECT_FALSE(invalid_response->status.empty());

        executor.spin_some(200ms);
        ASSERT_FALSE(status_messages.empty());
        const auto& latest_status = status_messages.back();
        EXPECT_EQ(latest_status.lifecycle_state, "active");
        EXPECT_TRUE(latest_status.model_loaded);
        EXPECT_EQ(latest_status.model_role, "centroiding");
        EXPECT_FALSE(latest_status.backend_detail.empty());
        EXPECT_EQ(latest_status.inference_count, 1U);
        EXPECT_GE(latest_status.last_inference_ms, 0.0);
        EXPECT_FALSE(latest_status.last_error.empty());

        EXPECT_EQ(inference_node->deactivate().label(), "inactive");
        EXPECT_EQ(inference_node->cleanup().label(), "unconfigured");
    }
} // namespace
