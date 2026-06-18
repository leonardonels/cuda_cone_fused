from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():

    config_node = os.path.join(
        get_package_share_directory('cuda_cone_fused'),
        'config',
        'config.yaml'
        )

    is_skidpad_mission = DeclareLaunchArgument(
        'is_skidpad_mission',
        default_value="false",
        description='Set to true if skidpad mission is selected'
    )


    node=Node(
            package='cuda_cone_fused',
            # namespace='cuda_cone_fused',
            name='cuda_cone_fused_node',
            executable='cuda_cone_fused_node',
            output = 'screen',
            # prefix=["gdbserver localhost:3000"],
            parameters=[config_node, {'is_skidpad_mission': LaunchConfiguration('is_skidpad_mission')}]

        )

    return LaunchDescription(
        [           
            is_skidpad_mission,
            node
        ]
    )
