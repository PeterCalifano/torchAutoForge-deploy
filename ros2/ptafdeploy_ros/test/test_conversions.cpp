/**
 * @file test_conversions.cpp
 * @brief Target-owned tests for ROS/core float-tensor conversion contracts.
 */

#include "ptafdeploy_ros/conversions.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
    namespace core = ptafdeploy::inference;
    using ptafdeploy_interfaces::msg::FloatTensor;
    using ptafdeploy_ros::ToCoreFloatTensor;
    using ptafdeploy_ros::ToRosFloatTensor;

    TEST(FloatTensorConversions, RoundTripsWrapperSafeTensorValues)
    {
        FloatTensor message;
        message.name = "input";
        message.shape = {1, 3};
        message.values = {1.0F, 2.0F, 3.0F};

        const core::SFloatTensor tensor = ToCoreFloatTensor(message);
        EXPECT_EQ(tensor.name, "input");
        EXPECT_EQ(tensor.shape, std::vector<int64_t>({1, 3}));
        EXPECT_EQ(tensor.values, std::vector<float>({1.0F, 2.0F, 3.0F}));

        const FloatTensor round_trip = ToRosFloatTensor(tensor);
        EXPECT_EQ(round_trip.name, message.name);
        EXPECT_EQ(round_trip.shape, message.shape);
        EXPECT_EQ(round_trip.values, message.values);
    }

    TEST(FloatTensorConversions, RejectsDynamicRuntimeDimensions)
    {
        FloatTensor message;
        message.shape = {1, -1};
        message.values = {1.0F};

        EXPECT_THROW(ToCoreFloatTensor(message), std::invalid_argument);
    }

    TEST(FloatTensorConversions, RejectsShapeValueCardinalityMismatch)
    {
        FloatTensor message;
        message.shape = {1, 3};
        message.values = {1.0F, 2.0F};

        EXPECT_THROW(ToCoreFloatTensor(message), std::invalid_argument);
    }

    TEST(FloatTensorConversions, RejectsElementCountOverflow)
    {
        FloatTensor message;
        message.shape = {std::numeric_limits<int64_t>::max(), 3};

        EXPECT_THROW(ToCoreFloatTensor(message), std::overflow_error);
    }
} // namespace
