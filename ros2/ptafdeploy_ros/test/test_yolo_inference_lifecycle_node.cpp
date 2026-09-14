/**
 * @file test_yolo_inference_lifecycle_node.cpp
 * @brief Real-fixture YOLO inference through the ROS 2 lifecycle service.
 */

#include "ptafdeploy_ros/CInferenceLifecycleNode.h"

#include "ptafdeploy_ros/conversions.h"

#include "ptafdeploy_interfaces/srv/infer_float_tensors.hpp"

#include <inference/task_adapters.h>

#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef PTAFDEPLOY_TEST_YOLO_CONFIG
#error "PTAFDEPLOY_TEST_YOLO_CONFIG must identify the YOLO model manifest"
#endif

#ifndef PTAFDEPLOY_TEST_YOLO_MODEL
#error "PTAFDEPLOY_TEST_YOLO_MODEL must identify the external YOLO ONNX artifact"
#endif

#ifndef PTAFDEPLOY_TEST_YOLO_IMAGE
#error "PTAFDEPLOY_TEST_YOLO_IMAGE must identify the external YOLO sample image"
#endif

namespace
{
    namespace fs = std::filesystem;
    namespace infer = ptafdeploy::inference;
    using namespace std::chrono_literals;
    using InferFloatTensors = ptafdeploy_interfaces::srv::InferFloatTensors;

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

    [[nodiscard]] infer::SFloatTensor PrepareYoloInput(const fs::path& image_path)
    {
        constexpr size_t InputHeight = 640U;
        constexpr size_t InputWidth = 640U;

        const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw std::runtime_error("Could not read the YOLO integration image.");
        }

        cv::Mat resized_image;
        cv::resize(image, resized_image,
                   cv::Size{static_cast<int>(InputWidth), static_cast<int>(InputHeight)}, 0.0, 0.0,
                   cv::INTER_LINEAR);
        if (!resized_image.isContinuous())
        {
            resized_image = resized_image.clone();
        }

        const auto* pixels = resized_image.ptr<std::uint8_t>();
        return infer::MakeNchwFloatTensorFromHwcAccessor(
            "images", InputHeight, InputWidth, 3U, 1.0F / 255.0F, true,
            [pixels](const size_t index) { return pixels[index]; });
    }

    [[nodiscard]] infer::SDetectionRowSchema MakeYoloSchema()
    {
        infer::SDetectionRowSchema schema;
        schema.box_encoding = infer::EBoundingBoxEncoding::center_xywh;
        schema.box_coordinate_0_index = 0U;
        schema.box_coordinate_1_index = 1U;
        schema.box_coordinate_2_index = 2U;
        schema.box_coordinate_3_index = 3U;
        schema.objectness_index = 4;
        schema.first_class_score_index = 5U;
        schema.class_score_count = 80U;
        return schema;
    }

    TEST(InferenceLifecycleNode, RunsRealYoloThroughFloatTensorService)
    {
        const fs::path config_path{PTAFDEPLOY_TEST_YOLO_CONFIG};
        const fs::path model_path{PTAFDEPLOY_TEST_YOLO_MODEL};
        const fs::path image_path{PTAFDEPLOY_TEST_YOLO_IMAGE};
        if (!fs::is_regular_file(config_path) || !fs::is_regular_file(model_path) ||
            !fs::is_regular_file(image_path))
        {
            GTEST_SKIP() << "External YOLO model/image fixture is unavailable.";
        }

        CRclcppContextGuard context;
        rclcpp::NodeOptions options;
        options.append_parameter_override("model_config_path", config_path.string());
        const auto inference_node =
            std::make_shared<ptafdeploy_ros::CInferenceLifecycleNode>(options);
        const auto client_node = std::make_shared<rclcpp::Node>("ptafdeploy_yolo_test_client");

        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(inference_node->get_node_base_interface());
        executor.add_node(client_node);

        ASSERT_EQ(inference_node->configure().label(), "inactive");
        ASSERT_EQ(inference_node->activate().label(), "active");

        const auto client =
            client_node->create_client<InferFloatTensors>("/ptafdeploy_inference/infer");
        ASSERT_TRUE(client->wait_for_service(2s));

        auto request = std::make_shared<InferFloatTensors::Request>();
        request->inputs.push_back(ptafdeploy_ros::ToRosFloatTensor(PrepareYoloInput(image_path)));
        auto future = client->async_send_request(request);
        ASSERT_EQ(executor.spin_until_future_complete(future, 60s),
                  rclcpp::FutureReturnCode::SUCCESS);

        const auto response = future.get();
        ASSERT_TRUE(response->success) << response->status;
        ASSERT_EQ(response->outputs.size(), 1U);
        const infer::SFloatTensor output =
            ptafdeploy_ros::ToCoreFloatTensor(response->outputs.front());
        EXPECT_EQ(output.shape, std::vector<int64_t>({1, 25200, 85}));

        const std::vector<infer::SDetection2D> detections =
            infer::DecodeDetectionRows(output, MakeYoloSchema(), 0.25F, 5U);
        ASSERT_EQ(detections.size(), 5U);
        EXPECT_EQ(detections.front().classification.class_id, 17U);
        EXPECT_GT(detections.front().classification.score, 0.8F);
        for (size_t index = 1; index < detections.size(); ++index)
        {
            EXPECT_GE(detections[index - 1U].classification.score,
                      detections[index].classification.score);
        }

        EXPECT_EQ(inference_node->deactivate().label(), "inactive");
        EXPECT_EQ(inference_node->cleanup().label(), "unconfigured");
    }
} // namespace
