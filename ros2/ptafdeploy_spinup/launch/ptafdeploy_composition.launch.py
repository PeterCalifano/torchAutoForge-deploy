"""Launch the lifecycle inference node composed into a component container."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

import os


def generate_launch_description() -> LaunchDescription:
    """Include the spinup launch with composition enabled."""

    spinup_launch = os.path.join(
        get_package_share_directory("ptafdeploy_spinup"),
        "launch",
        "ptafdeploy.launch.py",
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(spinup_launch),
                launch_arguments={"composition": "true"}.items(),
            )
        ]
    )
