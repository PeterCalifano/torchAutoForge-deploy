/**
 * @file ptafdeploy_node_main.cpp
 * @brief Standalone process entry point for the inference lifecycle node.
 */

#include "ptafdeploy_ros/CInferenceLifecycleNode.h"

#include <rclcpp/rclcpp.hpp>

#include <memory>

/**
 * @brief Initialize ROS and spin the standalone lifecycle inference node.
 * @return Zero after a clean ROS shutdown.
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(
        std::make_shared<ptafdeploy_ros::CInferenceLifecycleNode>()->get_node_base_interface());
    rclcpp::shutdown();
    return 0;
}
