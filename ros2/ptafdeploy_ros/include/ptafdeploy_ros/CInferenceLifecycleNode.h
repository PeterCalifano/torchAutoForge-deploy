/**
 * @file CInferenceLifecycleNode.h
 * @brief Public lifecycle-node bridge for model-facade inference.
 */

#pragma once

#include "inference/model_facade.h"

#include "ptafdeploy_interfaces/msg/model_status.hpp"
#include "ptafdeploy_interfaces/srv/infer_float_tensors.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace ptafdeploy_ros
{
    /**
     * @brief Lifecycle-managed ROS 2 bridge to CModelFacade.
     *
     * The model manifest is loaded during configure. Inference is accepted only
     * while active and serialized so one facade instance owns all backend calls.
     * The inference mutex protects the facade and every status field.
     */
    class CInferenceLifecycleNode final : public rclcpp_lifecycle::LifecycleNode
    {
      public:
        /**
         * @brief Construct an unconfigured inference lifecycle node.
         * @param options ROS node options, including an optional
         * `model_config_path` parameter override.
         */
        explicit CInferenceLifecycleNode(
            const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

        /**
         * @brief Load the required model manifest and create ROS endpoints.
         * @param previous_state Lifecycle state that initiated configuration.
         * @return Success after the facade and endpoints are ready; failure for
         * missing/invalid configuration or model-load errors.
         */
        CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

        /**
         * @brief Enable inference requests and activate status publication.
         * @param previous_state Lifecycle state that initiated activation.
         * @return Success only when a model and status publisher are configured.
         */
        CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

        /**
         * @brief Reject new inference requests and deactivate status publication.
         * @param previous_state Lifecycle state that initiated deactivation.
         * @return Success after the active gate is closed.
         */
        CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

        /**
         * @brief Release the facade, endpoints, and accumulated status state.
         * @param previous_state Lifecycle state that initiated cleanup.
         * @return Success after owned inference resources are released.
         */
        CallbackReturn on_cleanup(const rclcpp_lifecycle::State& previous_state) override;

      private:
        using InferFloatTensors = ptafdeploy_interfaces::srv::InferFloatTensors;
        using ModelStatus = ptafdeploy_interfaces::msg::ModelStatus;

        /** Convert, execute, and translate one service request under the state lock. */
        void HandleInference(const std::shared_ptr<InferFloatTensors::Request> request,
                             std::shared_ptr<InferFloatTensors::Response> response);

        /** Publish the current status when its lifecycle publisher is active. */
        void PublishStatusLocked(const std::string& lifecycle_state);

        std::mutex inference_mutex_{};
        std::unique_ptr<ptafdeploy::inference::CModelFacade> model_facade_{};
        rclcpp_lifecycle::LifecyclePublisher<ModelStatus>::SharedPtr status_publisher_{};
        rclcpp::Service<InferFloatTensors>::SharedPtr inference_service_{};
        bool accepting_requests_{false};
        bool model_loaded_{false};
        std::string model_role_{};
        std::string backend_detail_{};
        std::uint64_t inference_count_{0U};
        double last_inference_ms_{0.0};
        std::string last_error_{};
    };
} // namespace ptafdeploy_ros
