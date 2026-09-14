"""Exercise installed standalone/composed lifecycle inference launch modes."""

import time
import unittest

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_testing
from launch_testing.actions import ReadyToTest
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from ptafdeploy_interfaces.msg import FloatTensor, ModelStatus
from ptafdeploy_interfaces.srv import InferFloatTensors
import pytest
import rclpy
from rclpy.node import Node


@pytest.mark.launch_test
@launch_testing.parametrize(
    "launch_file, namespace",
    [
        ("ptafdeploy.launch.py", ""),
        ("ptafdeploy_composition.launch.py", ""),
        ("ptafdeploy.launch.py", "integration"),
        ("ptafdeploy_composition.launch.py", "integration"),
    ],
)
def generate_test_description(
    launch_file: str,
    namespace: str,
) -> LaunchDescription:
    """Launch one standalone/composed and root/namespaced test case."""

    package_share = get_package_share_directory("ptafdeploy_spinup")
    launch_source = PythonLaunchDescriptionSource(
        f"{package_share}/launch/{launch_file}"
    )
    launch_action = IncludeLaunchDescription(
        launch_source,
        launch_arguments={
            "autostart": "true",
            "namespace": namespace,
            "node_name": "ptafdeploy_inference",
        }.items(),
    )
    return LaunchDescription([launch_action, ReadyToTest()])


class TestSpinupLaunch(unittest.TestCase):
    """Validate lifecycle activation, real inference, and status publication."""

    node: Node

    @classmethod
    def setUpClass(cls) -> None:
        """Create the shared ROS test node."""

        rclpy.init()
        cls.node = rclpy.create_node("ptafdeploy_spinup_launch_test")

    @classmethod
    def tearDownClass(cls) -> None:
        """Destroy the shared node and stop the ROS client library."""

        cls.node.destroy_node()
        rclpy.shutdown()

    def _wait_for_active(
        self,
        node_path: str,
        case: str,
        timeout_sec: float = 15.0,
    ) -> None:
        """Wait until the launched lifecycle node reaches the active state."""

        state_client = self.node.create_client(
            GetState, f"{node_path}/get_state"
        )
        self.assertTrue(
            state_client.wait_for_service(timeout_sec=timeout_sec),
            f"Lifecycle state service was unavailable for {case}",
        )

        deadline = time.monotonic() + timeout_sec
        last_state = State.PRIMARY_STATE_UNKNOWN
        while time.monotonic() < deadline:
            future = state_client.call_async(GetState.Request())
            rclpy.spin_until_future_complete(
                self.node, future, timeout_sec=1.0
            )
            if future.done() and future.exception() is None:
                response = future.result()
                self.assertIsNotNone(response)
                last_state = response.current_state.id
                if last_state == State.PRIMARY_STATE_ACTIVE:
                    return
            time.sleep(0.1)

        self.fail(
            f"Lifecycle node did not become active for {case}; "
            f"last state was {last_state}"
        )

    def test_real_fixture_inference(
        self,
        launch_file: str,
        namespace: str,
    ) -> None:
        """Run installed fixture inference and validate its status metadata."""

        namespace_prefix = f"/{namespace}" if namespace else ""
        node_path = f"{namespace_prefix}/ptafdeploy_inference"
        case = f"launch={launch_file}, namespace={namespace or '<root>'}"
        self._wait_for_active(node_path, case)

        client = self.node.create_client(
            InferFloatTensors, f"{node_path}/infer"
        )
        self.assertTrue(
            client.wait_for_service(timeout_sec=5.0),
            f"Inference service was unavailable for {case}",
        )

        status_messages: list[ModelStatus] = []
        status_subscription = self.node.create_subscription(
            ModelStatus,
            f"{node_path}/status",
            status_messages.append,
            10,
        )
        try:
            discovery_deadline = time.monotonic() + 5.0
            while (
                status_subscription.get_publisher_count() == 0
                and time.monotonic() < discovery_deadline
            ):
                rclpy.spin_once(self.node, timeout_sec=0.1)
            self.assertGreater(
                status_subscription.get_publisher_count(),
                0,
                f"Status publisher was undiscovered for {case}",
            )

            settle_deadline = time.monotonic() + 0.5
            while time.monotonic() < settle_deadline:
                rclpy.spin_once(self.node, timeout_sec=0.05)

            request = InferFloatTensors.Request()
            input_tensor = FloatTensor()
            input_tensor.shape = [1, 11]
            input_tensor.values = [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
            ]
            request.inputs.append(input_tensor)
            future = client.call_async(request)
            response_deadline = time.monotonic() + 10.0
            while (
                (not future.done() or not status_messages)
                and time.monotonic() < response_deadline
            ):
                rclpy.spin_once(self.node, timeout_sec=0.1)

            self.assertTrue(future.done(), f"Inference timed out for {case}")
            self.assertIsNone(future.exception(), case)
            response = future.result()
            self.assertIsNotNone(response, case)
            self.assertTrue(response.success, response.status)
            self.assertEqual(len(response.outputs), 1, case)
            self.assertEqual(len(response.outputs[0].values), 2, case)
            self.assertAlmostEqual(
                response.outputs[0].values[0],
                1.686690331,
                places=5,
                msg=case,
            )
            self.assertAlmostEqual(
                response.outputs[0].values[1],
                4.697796822,
                places=5,
                msg=case,
            )

            self.assertTrue(status_messages, f"Status timed out for {case}")
            status = status_messages[-1]
            self.assertEqual(status.lifecycle_state, "active", case)
            self.assertTrue(status.model_loaded, case)
            self.assertEqual(status.model_role, "centroiding", case)
            self.assertTrue(status.backend_detail, case)
            self.assertEqual(status.inference_count, 1, case)
            self.assertGreaterEqual(status.last_inference_ms, 0.0, case)
            self.assertEqual(status.last_error, "", case)
            self.assertTrue(
                status.stamp.sec > 0 or status.stamp.nanosec > 0,
                f"Status timestamp was not populated for {case}",
            )
        finally:
            self.node.destroy_subscription(status_subscription)
