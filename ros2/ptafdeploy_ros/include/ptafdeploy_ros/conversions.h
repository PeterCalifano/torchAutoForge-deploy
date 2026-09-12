/**
 * @file conversions.h
 * @brief ROS/core float-tensor conversion contracts.
 */

#pragma once

#include "inference/inference_common.h"

#include "ptafdeploy_interfaces/msg/float_tensor.hpp"

#include <vector>

namespace ptafdeploy_ros
{
    /**
     * @brief Convert and validate a ROS float tensor message.
     *
     * @param message Wrapper-safe ROS tensor message.
     * @return Core-owned float tensor with the same name, shape, and values.
     * @throws std::invalid_argument when the shape is dynamic or its element
     * count does not match the value count.
     * @throws std::overflow_error when the shape element count overflows.
     */
    [[nodiscard]] ptafdeploy::inference::SFloatTensor
    ToCoreFloatTensor(const ptafdeploy_interfaces::msg::FloatTensor& message);

    /**
     * @brief Convert and validate a wrapper-safe core tensor for ROS transport.
     *
     * @param tensor Core-owned float tensor.
     * @return ROS tensor message with the same name, shape, and values.
     * @throws std::invalid_argument when the shape is dynamic or its element
     * count does not match the value count.
     * @throws std::overflow_error when the shape element count overflows.
     */
    [[nodiscard]] ptafdeploy_interfaces::msg::FloatTensor
    ToRosFloatTensor(const ptafdeploy::inference::SFloatTensor& tensor);

    /**
     * @brief Convert a sequence of ROS tensor messages to core tensors.
     * @param messages ROS input messages.
     * @return Core-owned tensors in input order.
     * @throws std::invalid_argument or std::overflow_error when any message is invalid.
     */
    [[nodiscard]] std::vector<ptafdeploy::inference::SFloatTensor>
    ToCoreFloatTensors(const std::vector<ptafdeploy_interfaces::msg::FloatTensor>& messages);

    /**
     * @brief Convert a sequence of core tensors to ROS tensor messages.
     * @param tensors Core output tensors.
     * @return ROS messages in input order.
     * @throws std::invalid_argument or std::overflow_error when any tensor is invalid.
     */
    [[nodiscard]] std::vector<ptafdeploy_interfaces::msg::FloatTensor>
    ToRosFloatTensors(const std::vector<ptafdeploy::inference::SFloatTensor>& tensors);
} // namespace ptafdeploy_ros
