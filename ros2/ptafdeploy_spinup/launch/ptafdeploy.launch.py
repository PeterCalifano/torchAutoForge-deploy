"""Launch the standalone lifecycle inference node with optional autostart."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode

import os


def _as_bool(value: str) -> bool:
    """Parse a ROS launch boolean string or reject an ambiguous value."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean launch value: {value}")


def _make_lifecycle_node(context: LaunchContext) -> list[LifecycleNode]:
    """Create the configured standalone lifecycle node for an opaque action."""

    return [
        LifecycleNode(
            package="ptafdeploy_ros",
            executable="ptafdeploy_node",
            name=LaunchConfiguration("node_name"),
            namespace=LaunchConfiguration("namespace"),
            output="screen",
            parameters=[
                {
                    "model_config_path": LaunchConfiguration(
                        "model_config_path"
                    )
                }
            ],
            autostart=_as_bool(
                LaunchConfiguration("autostart").perform(context)
            ),
        )
    ]


def generate_launch_description() -> LaunchDescription:
    """Declare public launch arguments and construct the standalone launch."""

    package_share = get_package_share_directory("ptafdeploy_spinup")
    default_config = os.path.join(
        package_share, "config", "centroiding_traced_sample.ptafmodel"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument(
                "node_name", default_value="ptafdeploy_inference"
            ),
            DeclareLaunchArgument(
                "model_config_path", default_value=default_config
            ),
            OpaqueFunction(function=_make_lifecycle_node),
        ]
    )
