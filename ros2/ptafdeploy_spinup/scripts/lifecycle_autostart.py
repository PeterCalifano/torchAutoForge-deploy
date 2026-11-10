#!/usr/bin/env python3
"""Configure and activate one lifecycle node once its services are available.

The spinup launch files run this helper for ``autostart:=true``. It only uses
the ``lifecycle_msgs`` services, so it behaves the same on ROS 2 Humble, whose
``LifecycleNode`` has no ``autostart`` argument, and on Jazzy, standalone or
composed. The target name resolves in this helper's namespace.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import rclpy
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
from rclpy.client import Client
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

AUTOSTART_TRANSITIONS = (
    (Transition.TRANSITION_CONFIGURE, "configure"),
    (Transition.TRANSITION_ACTIVATE, "activate"),
)


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the helper options.

    Args:
        argv: Argument vector without the program name and ROS arguments.

    Returns:
        Target node name and the service and transition timeouts in seconds.
    """

    parser = argparse.ArgumentParser(description="Configure and activate a lifecycle node.")
    parser.add_argument("target", help="lifecycle node name, resolved in this node's namespace")
    parser.add_argument(
        "--service-timeout",
        type=float,
        default=30.0,
        help="seconds to wait for the change_state service",
    )
    parser.add_argument(
        "--transition-timeout",
        type=float,
        default=60.0,
        help="seconds to wait for each transition, model loading included",
    )
    return parser.parse_args(argv)


def request_transition(
    node: Node,
    client: Client,
    transition: tuple[int, str],
    timeout_sec: float,
) -> bool:
    """Request one transition and report whether the lifecycle node accepted it."""

    transition_id, label = transition
    request = ChangeState.Request()
    request.transition.id = transition_id
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)

    response = future.result() if future.done() else None
    if response is None:
        node.get_logger().error(f"{label}: no response within {timeout_sec:g} s")
        return False
    if not response.success:
        node.get_logger().error(f"{label}: rejected by {client.srv_name}")
        return False

    node.get_logger().info(f"{label}: done")
    return True


def main() -> int:
    """Run the configure and activate sequence and return a process status."""

    arguments = parse_arguments(remove_ros_args(sys.argv)[1:])

    rclpy.init(args=sys.argv)
    node = rclpy.create_node("lifecycle_autostart")
    try:
        client = node.create_client(ChangeState, f"{arguments.target}/change_state")
        if not client.wait_for_service(timeout_sec=arguments.service_timeout):
            node.get_logger().error(
                f"{client.srv_name} unavailable after {arguments.service_timeout:g} s"
            )
            return 1

        for transition in AUTOSTART_TRANSITIONS:
            if not request_transition(node, client, transition, arguments.transition_timeout):
                return 1

        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
