"""Launch the lifecycle inference node standalone or composed."""

from ament_index_python.packages import get_package_share_directory
from launch import Action, LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LifecycleNode, Node
from launch_ros.descriptions import ComposableNode

import os


def _as_bool(value: str) -> bool:
    """Parse a ROS launch boolean string or reject an ambiguous value."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean launch value: {value}")


def _make_standalone_node(
    node_name: str, namespace: str, model_config_path: str
) -> Action:
    """Run the lifecycle node in its own process."""

    return LifecycleNode(
        package="ptafdeploy_ros",
        executable="ptafdeploy_node",
        name=node_name,
        namespace=namespace,
        output="screen",
        parameters=[{"model_config_path": model_config_path}],
    )


def _make_composed_node(
    node_name: str, namespace: str, model_config_path: str
) -> Action:
    """Load the lifecycle node as a component into a new container."""

    return ComposableNodeContainer(
        name="ptafdeploy_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="ptafdeploy_ros",
                plugin="ptafdeploy_ros::CInferenceLifecycleNode",
                name=node_name,
                namespace=namespace,
                parameters=[{"model_config_path": model_config_path}],
            )
        ],
        output="screen",
    )


def _make_autostart_helper(node_name: str, namespace: str) -> Action:
    """Run the helper that configures and activates the node on any distro."""

    return Node(
        package="ptafdeploy_spinup",
        executable="lifecycle_autostart",
        name=f"{node_name}_autostart",
        namespace=namespace,
        output="screen",
        arguments=[node_name],
    )


def _make_launch_entities(context: LaunchContext) -> list[Action]:
    """Create the node in the requested mode and its optional autostart."""

    node_name = LaunchConfiguration("node_name").perform(context)
    namespace = LaunchConfiguration("namespace").perform(context)
    model_config_path = LaunchConfiguration("model_config_path").perform(
        context
    )
    composition = _as_bool(LaunchConfiguration("composition").perform(context))
    make_node = _make_composed_node if composition else _make_standalone_node
    entities = [make_node(node_name, namespace, model_config_path)]

    if _as_bool(LaunchConfiguration("autostart").perform(context)):
        entities.append(_make_autostart_helper(node_name, namespace))

    return entities


def generate_launch_description() -> LaunchDescription:
    """Declare public launch arguments and construct the spinup launch."""

    package_share = get_package_share_directory("ptafdeploy_spinup")
    default_config = os.path.join(
        package_share, "config", "centroiding_traced_sample.ptafmodel"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("composition", default_value="false"),
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument(
                "node_name", default_value="ptafdeploy_inference"
            ),
            DeclareLaunchArgument(
                "model_config_path", default_value=default_config
            ),
            OpaqueFunction(function=_make_launch_entities),
        ]
    )
