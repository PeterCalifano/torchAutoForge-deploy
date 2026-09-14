/**
 * @file CInferenceLifecycleNode.cpp
 * @brief Lifecycle-managed ROS 2 bridge to the model-role facade.
 */

#include "ptafdeploy_ros/CInferenceLifecycleNode.h"

#include "ptafdeploy_ros/conversions.h"

#include <rclcpp_components/register_node_macro.hpp>

#include <chrono>
#include <exception>
#include <stdexcept>
#include <utility>

namespace ptafdeploy_ros
{
    CInferenceLifecycleNode::CInferenceLifecycleNode(const rclcpp::NodeOptions& options)
        : rclcpp_lifecycle::LifecycleNode("ptafdeploy_inference", options)
    {
        declare_parameter<std::string>("model_config_path", "");
    }

    CInferenceLifecycleNode::CallbackReturn
    CInferenceLifecycleNode::on_configure(const rclcpp_lifecycle::State&)
    {
        std::lock_guard lock{inference_mutex_};

        accepting_requests_ = false;
        model_loaded_ = false;
        model_role_.clear();
        backend_detail_.clear();
        inference_count_ = 0U;
        last_inference_ms_ = 0.0;
        last_error_.clear();

        try
        {
            const std::string model_config_path = get_parameter("model_config_path").as_string();
            if (model_config_path.empty())
            {
                throw std::invalid_argument("Parameter 'model_config_path' is required.");
            }

            auto facade = std::make_unique<ptafdeploy::inference::CModelFacade>();
            facade->LoadModelConfig(model_config_path);
            const auto contract = facade->GetContract();

            status_publisher_ = create_publisher<ModelStatus>("~/status", rclcpp::QoS(10));
            inference_service_ = create_service<InferFloatTensors>(
                "~/infer", [this](const std::shared_ptr<InferFloatTensors::Request> request,
                                  std::shared_ptr<InferFloatTensors::Response> response)
                { HandleInference(request, std::move(response)); });

            model_role_ = contract.role;
            backend_detail_ = contract.backend_detail;
            model_facade_ = std::move(facade);
            model_loaded_ = true;

            RCLCPP_INFO(get_logger(), "Configured model role '%s' from '%s'.", model_role_.c_str(),
                        model_config_path.c_str());
            return CallbackReturn::SUCCESS;
        }
        catch (const std::exception& exception)
        {
            last_error_ = exception.what();
            inference_service_.reset();
            status_publisher_.reset();
            model_facade_.reset();
            RCLCPP_ERROR(get_logger(), "Could not configure inference node: %s",
                         last_error_.c_str());
            return CallbackReturn::FAILURE;
        }
    }

    CInferenceLifecycleNode::CallbackReturn
    CInferenceLifecycleNode::on_activate(const rclcpp_lifecycle::State&)
    {
        std::lock_guard lock{inference_mutex_};
        if (!model_facade_ || !status_publisher_)
        {
            last_error_ = "Cannot activate before a model is configured.";
            RCLCPP_ERROR(get_logger(), "%s", last_error_.c_str());
            return CallbackReturn::FAILURE;
        }

        status_publisher_->on_activate();
        accepting_requests_ = true;
        last_error_.clear();
        PublishStatusLocked("active");
        RCLCPP_INFO(get_logger(), "Activated inference service.");
        return CallbackReturn::SUCCESS;
    }

    CInferenceLifecycleNode::CallbackReturn
    CInferenceLifecycleNode::on_deactivate(const rclcpp_lifecycle::State&)
    {
        std::lock_guard lock{inference_mutex_};
        accepting_requests_ = false;
        if (status_publisher_)
        {
            status_publisher_->on_deactivate();
        }
        RCLCPP_INFO(get_logger(), "Deactivated inference service.");
        return CallbackReturn::SUCCESS;
    }

    CInferenceLifecycleNode::CallbackReturn
    CInferenceLifecycleNode::on_cleanup(const rclcpp_lifecycle::State&)
    {
        std::lock_guard lock{inference_mutex_};
        accepting_requests_ = false;
        inference_service_.reset();
        status_publisher_.reset();
        model_facade_.reset();
        model_loaded_ = false;
        model_role_.clear();
        backend_detail_.clear();
        inference_count_ = 0U;
        last_inference_ms_ = 0.0;
        last_error_.clear();
        RCLCPP_INFO(get_logger(), "Cleaned up inference resources.");
        return CallbackReturn::SUCCESS;
    }

    void CInferenceLifecycleNode::HandleInference(
        const std::shared_ptr<InferFloatTensors::Request> request,
        std::shared_ptr<InferFloatTensors::Response> response)
    {
        const auto start_time = std::chrono::steady_clock::now();
        std::lock_guard lock{inference_mutex_};

        try
        {
            if (!accepting_requests_ || !model_facade_)
            {
                throw std::runtime_error("Inference node is not active.");
            }

            const auto core_inputs = ToCoreFloatTensors(request->inputs);
            const auto core_outputs = model_facade_->InferFloatTensors(core_inputs);
            response->outputs = ToRosFloatTensors(core_outputs);
            response->success = true;
            response->status = "ok";
            ++inference_count_;
            last_error_.clear();
        }
        catch (const std::exception& exception)
        {
            response->success = false;
            response->outputs.clear();
            response->status = exception.what();
            last_error_ = response->status;
            RCLCPP_WARN(get_logger(), "Inference request failed: %s", last_error_.c_str());
        }

        const auto elapsed = std::chrono::steady_clock::now() - start_time;
        last_inference_ms_ = std::chrono::duration<double, std::milli>(elapsed).count();
        PublishStatusLocked(get_current_state().label());
    }

    void CInferenceLifecycleNode::PublishStatusLocked(const std::string& lifecycle_state)
    {
        if (!status_publisher_ || !status_publisher_->is_activated())
        {
            return;
        }

        ModelStatus status;
        status.stamp = get_clock()->now();
        status.lifecycle_state = lifecycle_state;
        status.model_loaded = model_loaded_;
        status.model_role = model_role_;
        status.backend_detail = backend_detail_;
        status.inference_count = inference_count_;
        status.last_inference_ms = last_inference_ms_;
        status.last_error = last_error_;
        status_publisher_->publish(status);
    }
} // namespace ptafdeploy_ros

RCLCPP_COMPONENTS_REGISTER_NODE(ptafdeploy_ros::CInferenceLifecycleNode)
