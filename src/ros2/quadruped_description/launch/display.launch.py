"""
Launch the quadruped URDF in RViz2 with an interactive joint-state slider GUI.

Usage (after colcon build + source install/setup.bash):
    ros2 launch quadruped_description display.launch.py

What starts:
  1. robot_state_publisher  — publishes /robot_description + TF tree
  2. joint_state_publisher_gui — slider GUI so you can manually drive each joint
  3. rviz2                  — visualises the robot model and TF frames
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("quadruped_description")

    urdf_path = os.path.join(pkg_share, "urdf", "quadruped.urdf")
    rviz_path = os.path.join(pkg_share, "rviz", "display.rviz")

    with open(urdf_path, "r") as f:
        robot_description = f.read()

    use_gui_arg = DeclareLaunchArgument(
        "use_gui",
        default_value="true",
        description="Launch joint_state_publisher_gui (true) or headless (false)",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        condition=None,  # always launch; swap to IfCondition(LaunchConfiguration("use_gui")) to make optional
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_path],
        output="screen",
    )

    return LaunchDescription([
        use_gui_arg,
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz,
    ])
