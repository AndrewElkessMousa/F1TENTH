from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_params = PathJoinSubstitution([FindPackageShare('frenet_tenth_planner'), 'config', 'planner_params.yaml'])
    params_file = LaunchConfiguration('params_file')
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to planner parameter file'
        ),
        Node(
            package='frenet_tenth_planner',
            executable='frenet_planner_node',
            name='frenet_planner_node',
            output='screen',
            parameters=[params_file]
        )
    ])
