"""Launch the composable lifecycle inference node with namespaced autostart."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.utilities import perform_substitutions
from launch_ros.actions import ComposableNodeContainer, LifecycleTransition
from launch_ros.descriptions import (
    ComposableLifecycleNode as _RosComposableLifecycleNode,
)
from launch_ros.utilities import (
    LifecycleEventManager,
    make_namespace_absolute,
    prefix_namespace,
)
from lifecycle_msgs.msg import Transition

import os


class _LifecycleNodeIdentity:
    """Minimal node identity consumed by Jazzy's lifecycle event manager."""

    def __init__(self, fully_qualified_name: str) -> None:
        """Store the fully qualified lifecycle-node name."""

        self.fully_qualified_name = fully_qualified_name

    @property
    def node_name(self) -> str:
        """Return the fully qualified node name expected by launch_ros."""

        return self.fully_qualified_name


class _ComposableLifecycleNode(_RosComposableLifecycleNode):
    """Jazzy-compatible composed lifecycle node with namespaced autostart."""

    def __init__(self, *, autostart: bool = False, **kwargs: object) -> None:
        """Create a composed node while deferring custom autostart handling."""

        self.autostart_requested = autostart
        self.fully_qualified_name = ""
        super().__init__(autostart=False, **kwargs)

    def _initialize_lifecycle_manager(self, context: LaunchContext) -> None:
        """Resolve the final namespace and initialize lifecycle event routing."""

        node_name = perform_substitutions(context, self.node_name)
        node_namespace = ""
        if self.node_namespace is not None:
            node_namespace = perform_substitutions(
                context, self.node_namespace
            )

        base_namespace = context.launch_configurations.get(
            "ros_namespace", None
        )
        combined_namespace = make_namespace_absolute(
            prefix_namespace(base_namespace, node_namespace)
        )
        self.fully_qualified_name = (
            prefix_namespace(combined_namespace, node_name) or node_name
        )
        if not self.fully_qualified_name.startswith("/"):
            self.fully_qualified_name = f"/{self.fully_qualified_name}"

        self.lifecycle_event_manager = LifecycleEventManager(
            _LifecycleNodeIdentity(self.fully_qualified_name)
        )
        self.lifecycle_event_manager.setup_lifecycle_manager(context)

    def make_autostart_action(self) -> OpaqueFunction:
        """Return the deferred configure/activate action for this node."""

        return OpaqueFunction(function=self._autostart)

    def _autostart(
        self, context: LaunchContext
    ) -> list[LifecycleTransition]:
        """Create configure/activate transitions when autostart is requested."""

        if not self.autostart_requested:
            return []

        self._initialize_lifecycle_manager(context)
        return [
            LifecycleTransition(
                lifecycle_node_names=[self.fully_qualified_name],
                transition_ids=[
                    Transition.TRANSITION_CONFIGURE,
                    Transition.TRANSITION_ACTIVATE,
                ],
            )
        ]


def _as_bool(value: str) -> bool:
    """Parse a ROS launch boolean string or reject an ambiguous value."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean launch value: {value}")


def _make_container(context: LaunchContext) -> list[object]:
    """Create the component container and deferred lifecycle transition."""

    namespace = LaunchConfiguration("namespace").perform(context)
    node_name = LaunchConfiguration("node_name").perform(context)
    model_config_path = LaunchConfiguration("model_config_path").perform(
        context
    )
    lifecycle_node = _ComposableLifecycleNode(
        package="ptafdeploy_ros",
        plugin="ptafdeploy_ros::CInferenceLifecycleNode",
        name=node_name,
        namespace=namespace,
        parameters=[{"model_config_path": model_config_path}],
        autostart=_as_bool(
            LaunchConfiguration("autostart").perform(context)
        ),
    )

    return [
        ComposableNodeContainer(
            name="ptafdeploy_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[lifecycle_node],
            output="screen",
        ),
        lifecycle_node.make_autostart_action(),
    ]


def generate_launch_description() -> LaunchDescription:
    """Declare public launch arguments and construct the composed launch."""

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
            OpaqueFunction(function=_make_container),
        ]
    )
