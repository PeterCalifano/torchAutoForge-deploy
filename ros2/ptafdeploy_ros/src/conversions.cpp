/**
 * @file conversions.cpp
 * @brief Validated conversion between ROS float tensors and core value types.
 */

#include "ptafdeploy_ros/conversions.h"

#include <stdexcept>
#include <string>

namespace ptafdeploy_ros
{
    namespace
    {
        void ValidateTensorCardinality(const std::string& name, const std::vector<int64_t>& shape,
                                       const size_t value_count)
        {
            const size_t expected_count = ptafdeploy::inference::ComputeElementCount(shape);
            if (expected_count != value_count)
            {
                throw std::invalid_argument(
                    "Float tensor '" + name + "' contains " + std::to_string(value_count) +
                    " values, but its shape requires " + std::to_string(expected_count) + ".");
            }
        }
    } // namespace

    ptafdeploy::inference::SFloatTensor
    ToCoreFloatTensor(const ptafdeploy_interfaces::msg::FloatTensor& message)
    {
        ValidateTensorCardinality(message.name, message.shape, message.values.size());
        return {message.name, message.shape, message.values};
    }

    ptafdeploy_interfaces::msg::FloatTensor
    ToRosFloatTensor(const ptafdeploy::inference::SFloatTensor& tensor)
    {
        ValidateTensorCardinality(tensor.name, tensor.shape, tensor.values.size());

        ptafdeploy_interfaces::msg::FloatTensor message;
        message.name = tensor.name;
        message.shape = tensor.shape;
        message.values = tensor.values;
        return message;
    }

    std::vector<ptafdeploy::inference::SFloatTensor>
    ToCoreFloatTensors(const std::vector<ptafdeploy_interfaces::msg::FloatTensor>& messages)
    {
        std::vector<ptafdeploy::inference::SFloatTensor> tensors;
        tensors.reserve(messages.size());
        for (const auto& message : messages)
        {
            tensors.push_back(ToCoreFloatTensor(message));
        }
        return tensors;
    }

    std::vector<ptafdeploy_interfaces::msg::FloatTensor>
    ToRosFloatTensors(const std::vector<ptafdeploy::inference::SFloatTensor>& tensors)
    {
        std::vector<ptafdeploy_interfaces::msg::FloatTensor> messages;
        messages.reserve(tensors.size());
        for (const auto& tensor : tensors)
        {
            messages.push_back(ToRosFloatTensor(tensor));
        }
        return messages;
    }
} // namespace ptafdeploy_ros
